import logging
import os
import time
import numpy as np
from PIL import Image
import glob
from donkeycar.utils import rgb2gray
import cv2
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CameraError(Exception):
    pass

class BaseCamera:

    def run_threaded(self):
        return self.frame

class PiCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, vflip=False, hflip=False):
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        resolution = (image_w, image_h)
        # initialize the camera and stream
        self.camera = PiCamera() #PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True
        self.image_d = image_d

        # get the first frame or timeout
        logger.info('PiCamera loaded...')
        if self.stream is not None:
            logger.info('PiCamera opened...')
            warming_time = time.time() + 5  # quick after 5 seconds
            while self.frame is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("Unable to start PiCamera.")
        else:
            raise CameraError("Unable to open PiCamera.")
        logger.info("PiCamera ready.")

    def run(self):
        # grab the frame from the stream and clear the stream in
        # preparation for the next frame
        if self.stream is not None:
            f = next(self.stream)
            if f is not None:
                self.frame = f.array
                self.rawCapture.truncate(0)
                if self.image_d == 1:
                    self.frame = rgb2gray(self.frame)

        return self.frame

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.on:
            self.run()

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        logger.info('Stopping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()
        self.stream = None
        self.rawCapture = None
        self.camera = None


class Webcam(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate = 20, camera_index = 0):
        #
        # pygame is not installed by default.
        # Installation on RaspberryPi (with env activated):
        #
        # sudo apt-get install libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0
        # pip install pygame
        #
        super().__init__()
        self.cam = None
        self.framerate = framerate

        # initialize variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h

        self.init_camera(image_w, image_h, image_d, camera_index)
        self.on = True

    def init_camera(self, image_w, image_h, image_d, camera_index=0):
        try:
            import pygame
            import pygame.camera
        except ModuleNotFoundError as e:
            logger.error("Unable to import pygame.  Try installing it:\n"
                         "    sudo apt-get install libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0\n"
                         "    pip install pygame")
            raise e

        logger.info('Opening Webcam...')

        self.resolution = (image_w, image_h)

        try:
            pygame.init()
            pygame.camera.init()
            l = pygame.camera.list_cameras()

            if len(l) == 0:
                raise CameraError("There are no cameras available")

            logger.info(f'Available cameras {l}')
            if camera_index < 0 or camera_index >= len(l):
                raise CameraError(f"The 'CAMERA_INDEX={camera_index}' configuration in myconfig.py is out of range.")

            self.cam = pygame.camera.Camera(l[camera_index], self.resolution, "RGB")
            self.cam.start()

            logger.info(f'Webcam opened at {l[camera_index]} ...')
            warming_time = time.time() + 5  # quick after 5 seconds
            while self.frame is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("Unable to start Webcam.\n"
                                   "If more than one camera is available then"
                                   " make sure your 'CAMERA_INDEX' is correct in myconfig.py")

        except CameraError:
            raise
        except Exception as e:
            raise CameraError("Unable to open Webcam.\n"
                               "If more than one camera is available then"
                               " make sure your 'CAMERA_INDEX' is correct in myconfig.py") from e
        logger.info("Webcam ready.")

    def run(self):
        import pygame.image
        if self.cam.query_image():
            snapshot = self.cam.get_image()
            if snapshot is not None:
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)
                self.frame = pygame.surfarray.pixels3d(pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), 90))
                if self.image_d == 1:
                    self.frame = rgb2gray(frame)

        return self.frame

    def update(self):	
        from datetime import datetime, timedelta
        while self.on:
            start = datetime.now()
            self.run()
            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)


    def run_threaded(self):
        return self.frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        if self.cam:
            logger.info('stopping Webcam')
            self.cam.stop()
            self.cam = None
        time.sleep(.5)


class CSICamera(BaseCamera):
    '''
    Camera for Jetson Nano IMX219 based camera
    Credit: https://github.com/feicccccccc/donkeycar/blob/dev/donkeycar/parts/camera.py
    gstreamer init string from https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/jetbot/camera.py
    '''
    def gstreamer_pipeline(self, capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21, flip_method=0) :   
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                capture_width, capture_height, framerate, flip_method, output_width, output_height)
    
    def __init__(self, image_w=160, image_h=120, image_d=3, capture_width=3280, capture_height=2464, framerate=60, gstreamer_flip=0):
        '''
        gstreamer_flip = 0 - no flip
        gstreamer_flip = 1 - rotate CCW 90
        gstreamer_flip = 2 - flip vertically
        gstreamer_flip = 3 - rotate CW 90
        '''
        self.w = image_w
        self.h = image_h
        self.flip_method = gstreamer_flip
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.framerate = framerate
        self.frame = None
        self.init_camera()
        self.running = True

    def init_camera(self):
        import cv2

        # initialize the camera and stream
        self.camera = cv2.VideoCapture(
            self.gstreamer_pipeline(
                capture_width=self.capture_width,
                capture_height=self.capture_height,
                output_width=self.w,
                output_height=self.h,
                framerate=self.framerate,
                flip_method=self.flip_method),
            cv2.CAP_GSTREAMER)

        if self.camera and self.camera.isOpened():
            logger.info('CSICamera opened...')
            warming_time = time.time() + 5  # quick after 5 seconds
            while self.frame is None and time.time() < warming_time:
                logger.info("...warming camera")
                self.poll_camera()
                time.sleep(0.2)

            if self.frame is None:
                raise RuntimeError("Unable to start CSICamera.")
        else:
            raise RuntimeError("Unable to open CSICamera.")
        logger.info("CSICamera ready.")

    def update(self):
        while self.running:
            self.poll_camera()

    def poll_camera(self):
        import cv2
        self.ret , frame = self.camera.read()
        if frame is not None:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def run(self):
        self.poll_camera()
        return self.frame

    def run_threaded(self):
        return self.frame
    
    def shutdown(self):
        self.running = False
        logger.info('Stopping CSICamera')
        time.sleep(.5)
        del(self.camera)


class V4LCamera(BaseCamera):
    '''
    uses the v4l2capture library from this fork for python3 support: https://github.com/atareao/python3-v4l2capture
    sudo apt-get install libv4l-dev
    cd python3-v4l2capture
    python setup.py build
    pip install -e .
    '''
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, dev_fn="/dev/video0", fourcc='MJPG'):

        self.running = True
        self.frame = None
        self.image_w = image_w
        self.image_h = image_h
        self.dev_fn = dev_fn
        self.fourcc = fourcc

    def init_video(self):
        import v4l2capture

        self.video = v4l2capture.Video_device(self.dev_fn)

        # Suggest an image size to the device. The device may choose and
        # return another size if it doesn't support the suggested one.
        self.size_x, self.size_y = self.video.set_format(self.image_w, self.image_h, fourcc=self.fourcc)

        logger.info("V4L camera granted %d, %d resolution." % (self.size_x, self.size_y))

        # Create a buffer to store image data in. This must be done before
        # calling 'start' if v4l2capture is compiled with libv4l2. Otherwise
        # raises IOError.
        self.video.create_buffers(30)

        # Send the buffer to the device. Some devices require this to be done
        # before calling 'start'.
        self.video.queue_all_buffers()

        # Start the device. This lights the LED if it's a camera that has one.
        self.video.start()

    def update(self):
        import select
        from donkeycar.parts.image import JpgToImgArr

        self.init_video()
        jpg_conv = JpgToImgArr()

        while self.running:
            # Wait for the device to fill the buffer.
            select.select((self.video,), (), ())
            image_data = self.video.read_and_queue()
            self.frame = jpg_conv.run(image_data)

    def shutdown(self):
        self.running = False
        time.sleep(0.5)


class MockCamera(BaseCamera):
    '''
    Fake camera. Returns only a single static frame
    '''
    def __init__(self, image_w=160, image_h=120, image_d=3, image=None):
        if image is not None:
            self.frame = image
        else:
            self.frame = np.array(Image.new('RGB', (image_w, image_h)))

    def update(self):
        pass

    def shutdown(self):
        pass


class ImageListCamera(BaseCamera):
    '''
    Use the images from a tub as a fake camera output
    '''
    def __init__(self, path_mask='~/mycar/data/**/images/*.jpg'):
        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)
    
        def get_image_index(fnm):
            sl = os.path.basename(fnm).split('_')
            return int(sl[0])

        '''
        I feel like sorting by modified time is almost always
        what you want. but if you tared and moved your data around,
        sometimes it doesn't preserve a nice modified time.
        so, sorting by image index works better, but only with one path.
        '''
        self.image_filenames.sort(key=get_image_index)
        #self.image_filenames.sort(key=os.path.getmtime)
        self.num_images = len(self.image_filenames)
        logger.info('%d images loaded.' % self.num_images)
        logger.info( self.image_filenames[:10])
        self.i_frame = 0
        self.frame = None
        self.update()

    def update(self):
        pass

    def run_threaded(self):        
        if self.num_images > 0:
            self.i_frame = (self.i_frame + 1) % self.num_images
            self.frame = Image.open(self.image_filenames[self.i_frame]) 

        return np.asarray(self.frame)

    def shutdown(self):
        pass

class OAKDCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, frame_rate=20, with_record = True, with_stereo=True, with_nn=False, nn_model_path="", with_imu=True, imu_accelerometer_freq = 500, imu_gyroscope_freq=400, imu_batch_report_threshold = 1, imu_max_batch_reports=10):
        import depthai as dai
        self.device = dai.Device()
        self.frame = None
        self.res = dai.ColorCameraProperties.SensorResolution.THE_720_P
        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(image_w, image_h)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setResolution(self.res)
        self.cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam_rgb.setFps(frame_rate)
        if with_stereo:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                self.cam_rgb.initialControl.setManualFocus(lensPosition)
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        self.with_record = with_record
        self.with_stereo = with_stereo
        self.with_nn = with_nn
        self.with_imu = with_imu
        self.rgb_frame = None
        self.depth_frame = None
        self.nn_output = None
        self.imu_output = None
        self.frame_rate = frame_rate
        self.output_names = ["rgb"]
        if with_record:
            self.videoEnc = self.pipeline.create(dai.node.VideoEncoder)
            self.videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.H265_MAIN)
            self.cam_rgb.preview.link(self.videoEnc.input)
            self.xout_vid = self.pipeline.create(dai.node.XLinkOut)
            self.xout_vid.setStreamName("record")
            self.videoEnc.bitstream.link(self.xout_vid.input)
            self.output_names.append("record")
            self.current_date = datetime.datetime.now()
        if with_stereo:
            self.left = self.pipeline.create(dai.node.MonoCamera)
            self.left.setPreviewSize(image_w, image_h)
            self.left.setInterleaved(False)
            self.left.setResolution(self.res)
            self.left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            self.left.setFps(frame_rate)

            self.right = self.pipeline.create(dai.node.MonoCamera)
            self.right.setPreviewSize(image_w, image_h)
            self.right.setInterleaved(False)
            self.right.setResolution(self.res)
            self.right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            self.right.setFps(frame_rate)

            self.stereo = self.pipeline.create(dai.node.StereoDepth)
            self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            self.stereo.setLeftRightCheck(True)
            self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            self.left.out.link(self.stereo.left)
            self.right.out.link(self.stereo.right)
            self.xout_depth = self.pipeline.create(dai.node.XLinkOut)
            self.xout_depth.setStreamName("depth")
            self.stereo.disparity.link(self.xout_depth.input)
            self.output_names.append("depth")
        if with_nn:
            self.detection_nn = self.pipeline.create(dai.node.NeuralNetwork)
            self.detection_nn.setBlobPath(str(Path(nn_model_path).resolve().absolute()))
            self.cam_rgb.preview.link(self.detection_nn.input)
            self.xout_nn = self.pipeline.create(dai.node.XLinkOut)
            self.xout_nn.setStreamName("nn")
            self.detection_nn.preview.link(self.xout_nn.input)
            self.output_names.append("nn")
        if with_imu:
            self.imu = self.pipeline.create(dai.node.IMU)
            self.imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_accelerometer_freq)
            self.imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, imu_gyroscope_freq)
            self.imu.setBatchReportThreshold(imu_batch_report_threshold)
            self.imu.setMaxBatchReports(imu_max_batch_reports)
            self.xout_imu = self.pipeline.create(dai.node.XLinkOut)
            self.xout_imu.setStreamName("imu")
            self.imu.preview.link(self.xout_imu.input)
            self.output_names.append("imu")
        self.device.startPipeline(self.pipeline)
    
    def update(self):
        for name in self.output_names:
            msg = self.device.getOutputQueue(name).get()
            if msg is not None:
                if name == "rgb":
                    self.rgb_frame = cv2.cvtColor(msg.getFrame(), cv2.COLOR_BGR2RGB)
                    cv2.imshow(self.rgb_frame, name)
                elif name == "record":
                    with open(f'rgb_data_{self.current_date.strftime("%m_%d_%Y_%H_%M_%S")}.h265', 'wb') as recorded_data:
                        msg.getData().tofile(recorded_data)
                elif name == "depth":
                    self.depth_frame = msg.getFrame()
                    cv2.imshow(self.depth_frame, name)
                elif name == "nn":
                    pass
                elif name == "imu":
                    for packet in msg.packets():
                        acceleroValues = packet.acceleroMeter
                        gyroValues = packet.gyroscope
                        imuF = "{:.06f}"
                        print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
                        print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")
                        self.imu_output = [acceleroValues.x, acceleroValues.y, acceleroValues.z, gyroValues.x, gyroValues.y, gyroValues.z]

    def run(self):
        return []

    def run_threaded(self):
        return []