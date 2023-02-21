#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
from docopt import docopt

#
# import cv2 early to avoid issue with importing after tensorflow
# see https://github.com/opencv/opencv/issues/14884#issuecomment-599852128
#
try:
    import cv2
except:
    pass


import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, WebFpv, JoystickController
from donkeycar.parts.path import CsvThrottlePath, PathPlot, CTE, PID_Pilot, \
    PlotCircle, PImage, OriginOffset
from donkeycar.parts.transform import PIDController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
    
def enable_logging(cfg):
    logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
    logger.addHandler(ch)

def enable_telemetry(V, cfg, inputs, types):
    from donkeycar.parts.telemetry import MqttTelemetry
    tel = MqttTelemetry(cfg)
    telem_inputs, _ = tel.add_step_inputs(inputs, types)
    V.add(tel, inputs=telem_inputs, outputs=["tub/queue_size"], threaded=True)

def add_simulator(V, cfg):
    #the simulator will use cuda and then we usually run out of resources
    #if we also try to use cuda. so disable for donkey_gym.
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    logger.info("Disabling CUDA for Donkey Gym")
    # Donkey gym part will output position information if it is configured
    # TODO: the simulation outputs conflict with imu, odometry, kinematics pose estimation and T265 outputs; make them work together.
    from donkeycar.parts.dgym import DonkeyGymEnv
    # rbx
    gym = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF,
                        record_location=cfg.SIM_RECORD_LOCATION, record_gyroaccel=cfg.SIM_RECORD_GYROACCEL,
                        record_velocity=cfg.SIM_RECORD_VELOCITY, record_lidar=cfg.SIM_RECORD_LIDAR,
                    #    record_distance=cfg.SIM_RECORD_DISTANCE, record_orientation=cfg.SIM_RECORD_ORIENTATION,
                        delay=cfg.SIM_ARTIFICIAL_LATENCY)
    threaded = True
    inputs = ['angle', 'throttle']
    outputs = ['cam/image_array']

    if cfg.SIM_RECORD_LOCATION:
        outputs += ['pos/x', 'pos/z', 'pos/y', 'pos/speed', 'pos/cte']
    if cfg.SIM_RECORD_GYROACCEL:
        outputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
    if cfg.SIM_RECORD_VELOCITY:
        outputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
    if cfg.SIM_RECORD_LIDAR:
        outputs += ['lidar/dist_array']
    # if cfg.SIM_RECORD_DISTANCE:
    #     outputs += ['dist/left', 'dist/right']
    # if cfg.SIM_RECORD_ORIENTATION:
    #     outputs += ['roll', 'pitch', 'yaw']

    V.add(gym, inputs=inputs, outputs=outputs, threaded=threaded)

def add_odometry(V, cfg):
    """
    If the configuration support odometry, then
    add encoders, odometry and kinematics to the vehicle pipeline
    :param V: the vehicle pipeline.
              On output this may be modified.
    :param cfg: the configuration (from myconfig.py)
    """
    logger.info("No supported encoder found in this template")

def add_camera(V, cfg):
    """
    Add the configured camera to the vehicle pipeline.

    :param V: the vehicle pipeline.
              On output this will be modified.
    :param cfg: the configuration (from myconfig.py)
    """
    logger.info("cfg.CAMERA_TYPE %s"%cfg.CAMERA_TYPE)
    if cfg.CAMERA_TYPE == "OAKD LITE":
        from donkeycar.parts.camera import OAKDCamera
        cam = OAKDCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, frame_rate=cfg.IMAGE_FPS, with_record=cfg.RECORD_IMAGE, with_stereo=cfg.OAKD_LITE_STEREO_ENABLED, nn_model_path=cfg.OAKD_LITE_NN_MODEL_PATH, with_imu=(cfg.HAVE_IMU and cfg.IMU_TYPE == "OAKD LITE"))
        V.add(cam, inputs=[], outputs=cam.get_outputs(), threaded=True)
    elif cfg.CAMERA_TYPE == "MOCK":
        from donkeycar.parts.camera import MockCamera
        cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
    else:
        logger.info("This camera is not yet supported in this template")

def add_lidar(V, cfg):
    logger.info("cfg.LIDAR_TYPE %s"%cfg.LIDAR_TYPE)
    if cfg.LIDAR_TYPE == 'SICK TIM 571':
        logger.info('SICK Lidar to be supported')
    else:
        logger.info("This lidar is not yet supported in this template")

def add_imu(V, cfg):
    logger.info("cfg.IMU_TYPE %s"%cfg.IMU_TYPE)
    if cfg.IMU_TYPE == "SPARKFUN 9DOF IMU":
        logger.info("SPARKFUN 9DOF IMU to be supported")
    else:
        logger.info("This IMU is not yet supported in this template")

def add_gps(V, cfg):
    if cfg.GPS_TYPE == "P1 Fusion Engine":
        pass
    else:
        from donkeycar.parts.serial_port import SerialPort, SerialLineReader
        from donkeycar.parts.gps import GpsNmeaPositions, GpsLatestPosition, GpsPlayer
        from donkeycar.parts.pipe import Pipe
        from donkeycar.parts.text_writer import CsvLogger

        #
        # parts to
        # - read nmea lines from serial port
        # - OR play from recorded file
        # - convert nmea lines to positions
        # - retrieve the most recent position
        #
        serial_port = SerialPort(cfg.GPS_SERIAL, cfg.GPS_SERIAL_BAUDRATE)
        nmea_reader = SerialLineReader(serial_port)
        V.add(nmea_reader, outputs=['gps/nmea'], threaded=True)

        gps_positions = GpsNmeaPositions(debug=cfg.GPS_DEBUG)
        V.add(gps_positions, inputs=['gps/nmea'], outputs=['gps/positions'])
        gps_latest_position = GpsLatestPosition(debug=cfg.GPS_DEBUG)
        V.add(gps_latest_position, inputs=['gps/positions'], outputs=['gps/timestamp', 'gps/utm/longitude', 'gps/utm/latitude'])

        # rename gps utm position to pose values
        V.add(Pipe(), inputs=['gps/utm/longitude', 'gps/utm/latitude'], outputs=['pos/x', 'pos/y'])

def enable_fps(V, cfg):
    from donkeycar.parts.fps import FrequencyLogger
    V.add(FrequencyLogger(cfg.FPS_DEBUG_INTERVAL), outputs=["fps/current", "fps/fps_list"])

def enable_perfmon(V, cfg):
    from donkeycar.parts.perfmon import PerfMonitor
    mon = PerfMonitor(cfg)
    perfmon_outputs = ['perf/cpu', 'perf/mem', 'perf/freq']
    inputs += perfmon_outputs
    types += ['float', 'float', 'float']
    V.add(mon, inputs=[], outputs=perfmon_outputs, threaded=True)

def add_user_controller(V, cfg, input_image='cam/image_array'):
    """
    Add the web controller and any other
    configured user input controller.
    :param V: the vehicle pipeline.
              On output this will be modified.
    :param cfg: the configuration (from myconfig.py)
    :return: the controller
    """

    #
    # This web controller will create a web server that is capable
    # of managing steering, throttle, and modes, and more.
    #
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(ctr,
          inputs=[input_image, 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    #
    # also add a physical controller if one is configured
    #
    if cfg.HAVE_JOYSTICK:
        #
        # custom game controller mapping created with
        # `donkey createjs` command
        #
        if cfg.CONTROLLER_TYPE == "custom":  # custom controller created with `donkey createjs` command
            from my_joystick import MyJoystickController
            ctr = MyJoystickController(
                throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
            ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
        elif cfg.CONTROLLER_TYPE == "mock":
            from donkeycar.parts.controller import MockController
            ctr = MockController(steering=cfg.MOCK_JOYSTICK_STEERING,
                                    throttle=cfg.MOCK_JOYSTICK_THROTTLE)
        else:
            #
            # game controller
            #
            from donkeycar.parts.controller import get_js_controller
            ctr = get_js_controller(cfg)
        V.add(
            ctr,
            inputs=[input_image, 'user/mode', 'recording'],
            outputs=['user/angle', 'user/throttle',
                        'user/mode', 'recording'],
            threaded=True)
    return ctr, has_input_controller

def add_web_buttons(V):
     #
    # explode the buttons into their own key/values in memory
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # adding a button handler is just adding a part with a run_condition
    # set to the button's name, so it runs when button is pressed.
    #
    V.add(Lambda(lambda v: logger.info(f"web/w1 clicked")), inputs=["web/w1"], run_condition="web/w1")
    V.add(Lambda(lambda v: logger.info(f"web/w2 clicked")), inputs=["web/w2"], run_condition="web/w2")
    V.add(Lambda(lambda v: logger.info(f"web/w3 clicked")), inputs=["web/w3"], run_condition="web/w3")
    V.add(Lambda(lambda v: logger.info(f"web/w4 clicked")), inputs=["web/w4"], run_condition="web/w4")
    V.add(Lambda(lambda v: logger.info(f"web/w5 clicked")), inputs=["web/w5"], run_condition="web/w5")

def add_throttle_reverse(V):
    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

def add_pilot_condition(V):
    #See if we should even run the pilot module.
    #This is only needed because the part run_condition only accepts boolean
    from donkeycar.parts.behavior import PilotCondition

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

def add_record_tracker(V, cfg, ctr):
    from donkeycar.parts.logger import RecordTracker
    rec_tracker_part = RecordTracker(logger, cfg.REC_COUNT_ALERT, cfg.REC_COUNT_ALERT_CYC, cfg.RECORD_ALERT_COLOR_ARR)
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE:
        def show_record_count_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        if isinstance(ctr, JoystickController):
            ctr.set_button_down_trigger('circle', show_record_count_status) #then we are not using the circle button. hijack that to force a record count indication
        else:   
            show_record_count_status()

def enable_fpv(V):
    # Use the FPV preview, which will show the cropped image output, or the full frame.
    V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

def load_model(kl, model_path):
    start = time.time()
    logger.info(f'loading model {model_path}')
    kl.load(model_path)
    logger.info(f'finished loading in {(str(time.time() - start))} sec.')

def load_weights(kl, weights_path):
    start = time.time()
    try:
        logger.info(f'loading model weights {weights_path}')
        kl.model.load_weights(weights_path)
        logger.info(f'finished loading in {(str(time.time() - start))} sec.')
    except Exception as e:
        logger.info(e)
        logger.info(f'ERR>> problems loading weights {weights_path}')

def load_model_json(kl, json_fnm):
    start = time.time()
    logger.info(f'loading model json {json_fnm}')
    from tensorflow.python import keras
    try:
        with open(json_fnm, 'r') as handle:
            contents = handle.read()
            kl.model = keras.models.model_from_json(contents)
        logger.info(f'finished loading json in {(str(time.time() - start))} sec.')
    except Exception as e:
        logger.info(e)
        logger.info(f"ERR>> problems loading model json {json_fnm}")

def add_behavior_cloning_model(V, cfg, ctr):
    # If we have a model, create an appropriate Keras part
    model_path = cfg.BEHAVIOR_CLONE_MODEL_PATH
    if model_path is None:
        return
    model_type = cfg.BEHAVIOR_CLONE_MODEL_TYPE
    kl = dk.utils.get_model_by_type(model_type, cfg)
    model_reload_cb = None
    if '.h5' in model_path or '.trt' in model_path or '.tflite' in \
            model_path or '.savedmodel' in model_path or '.pth':
        # load the whole model with weigths, etc
        load_model(kl, model_path)
        def reload_model(filename):
            load_model(kl, filename)

        model_reload_cb = reload_model

    elif '.json' in model_path:
        # when we have a .json extension
        # load the model from there and look for a matching
        # .wts file with just weights
        load_model_json(kl, model_path)
        weights_path = model_path.replace('.json', '.weights')
        load_weights(kl, weights_path)
        
        def reload_weights(filename):
            weights_path = filename.replace('.json', '.weights')
            load_weights(kl, weights_path)

        model_reload_cb = reload_weights

    else:
        logger.info("ERR>> Unknown extension type on model file!!")
        return

    # these parts will reload the model file, but only when ai is running
    # so we don't interrupt user driving
    V.add(FileWatcher(model_path), outputs=['modelfile/dirty'],
            run_condition="ai_running")
    V.add(DelayedTrigger(100), inputs=['modelfile/dirty'],
            outputs=['modelfile/reload'], run_condition="ai_running")
    V.add(TriggeredCallback(model_path, model_reload_cb),
            inputs=["modelfile/reload"], run_condition="ai_running")

    #
    # collect inputs to model for inference
    #
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]

    elif cfg.USE_LIDAR_IN_MODEL:
        inputs = ['cam/image_array', 'lidar/dist_array']

    elif cfg.USE_ODOM_IN_MODEL:
        inputs = ['cam/image_array', 'enc/speed']

    elif model_type == "imu":
        assert cfg.HAVE_IMU, 'Missing imu parameter in config'

        class Vectorizer:
            def run(self, *components):
                return components

        V.add(Vectorizer, inputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                                    'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
                outputs=['imu_array'])

        inputs = ['cam/image_array', 'imu_array']
    else:
        inputs = ['cam/image_array']

    #
    # collect model inference outputs
    #
    outputs = ['pilot/angle', 'pilot/throttle']

    if cfg.TRAIN_LOCALIZER:
        outputs.append("pilot/loc")

    #
    # Add image transformations like crop or trapezoidal mask
    #
    if hasattr(cfg, 'TRANSFORMATIONS') and cfg.TRANSFORMATIONS:
        from donkeycar.pipeline.augmentations import ImageAugmentation
        V.add(ImageAugmentation(cfg, 'TRANSFORMATIONS'),
                inputs=['cam/image_array'], outputs=['cam/image_array_trans'])
        inputs = ['cam/image_array_trans'] + inputs[1:]

    V.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')

def add_autopilot(V, cfg, ctr):
    #
    # to give the car a boost when starting ai mode in a race.
    # This will also override the stop sign detector so that
    # you can start at a stop sign using launch mode, but
    # will stop when it comes to the stop sign the next time.
    #
    # NOTE: when launch throttle is in effect, pilot speed is set to None
    #
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    V.add(aiLauncher,
          inputs=['user/mode', 'pilot/throttle'],
          outputs=['pilot/throttle'])
    from donkeycar.parts.behavior import DriveMode
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)

    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    # Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

def add_tub_writer(V, cfg):
    # add tub to save data
    if cfg.HAVE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array', 'user/angle', 'user/throttle', 'user/mode']
        types = ['image_array', 'nparray','float', 'float', 'str']
    else:
        inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
        types=['image_array','float', 'float','str']

    if cfg.HAVE_ODOM:
        inputs += ['enc/speed']
        types += ['float']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_DEPTH:
        inputs += ['cam/depth_array']
        types += ['gray16_array']

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    # rbx
    if cfg.DONKEY_GYM:
        if cfg.SIM_RECORD_LOCATION:
            inputs += ['pos/x', 'pos/y', 'pos/z', 'pos/speed', 'pos/cte']
            types  += ['float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_GYROACCEL:
            inputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
            types  += ['float', 'float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_VELOCITY:
            inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
            types  += ['float', 'float', 'float']
        if cfg.SIM_RECORD_LIDAR:
            inputs += ['lidar/dist_array']
            types  += ['nparray']

    # do we want to store new records into own dir or append to existing
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta = []
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
    return inputs, types, tub_writer
#
# Drive train setup
#
def add_drivetrain(V, cfg):
    logger.info("cfg.DRIVE_TRAIN_TYPE %s"%cfg.DRIVE_TRAIN_TYPE)
    if cfg.DRIVE_TRAIN_TYPE == "MOCK":
        return
    elif cfg.DRIVE_TRAIN_TYPE == "VESC":
        from donkeycar.parts.actuator import VESC
        logger.info("Creating VESC at port {}".format(cfg.VESC_SERIAL_PORT))
        if cfg.HAVE_IMU and cfg.IMU_TYPE == "VESC":
            vesc = VESC(cfg.VESC_SERIAL_PORT,
                        cfg.VESC_MAX_SPEED_PERCENT,
                        cfg.VESC_HAS_SENSOR,
                        cfg.VESC_START_HEARTBEAT,
                        cfg.VESC_BAUDRATE,
                        cfg.VESC_TIMEOUT,
                        cfg.VESC_STEERING_SCALE,
                        cfg.VESC_STEERING_OFFSET
                    )
        elif cfg.HAVE_ODOM and cfg.ODOM_TYPE == "VESC":
            pass
        V.add(vesc, inputs=['angle', 'throttle'])
    else:
        logger.info("This Drive Train Type is not yet supported in this template")

def drive(cfg):
    V = dk.Vehicle()
    logger.info(f'PID: {os.getpid()}')
    #Initialize car
    V = dk.vehicle.Vehicle()

    #Initialize logging before anything else to allow console logging
    if cfg.HAVE_CONSOLE_LOGGING:
        enable_logging(cfg)
    if cfg.DONKEY_GYM:
        add_simulator(V, cfg)
    else:
        if cfg.HAVE_ODOM:
            # add odometry
            add_odometry(V, cfg)
        if cfg.HAVE_CAMERA:
            # setup primary camera
            add_camera(V, cfg)
        if cfg.HAVE_LIDAR:
            # add lidar
            add_lidar(V, cfg)
        if cfg.HAVE_IMU:
            # add IMU
            add_imu(V, cfg)
        if cfg.HAVE_GPS:
            # add GPS
            add_gps(V, cfg)
    ctr, has_input_controller = add_user_controller(V, cfg)
    if cfg.DRIVE_TYPE == "behavior_cloning":
        add_web_buttons(V)
        add_throttle_reverse(V)
        add_pilot_condition(V)
        add_record_tracker(V, cfg, ctr)
        add_behavior_cloning_model(V, cfg, ctr)
        add_autopilot(V, cfg, ctr)
        inputs, types, tub_writer = add_tub_writer(V, cfg)
        if cfg.HAVE_MQTT_TELEMETRY:
            enable_telemetry(V, cfg, inputs, types)
    elif cfg.DRIVE_TYPE == "gps_follow":
        #
        # explode the web buttons into their own key/values in memory
        #
        V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

        #
        # This part will reset the car back to the origin. You must put the car in the known origin
        # and push the cfg.RESET_ORIGIN_BTN on your controller. This will allow you to induce an offset
        # in the mapping.
        #
        origin_reset = OriginOffset(cfg.PATH_DEBUG)
        V.add(origin_reset, inputs=['pos/x', 'pos/y', 'cte/closest_pt'], outputs=['pos/x', 'pos/y', 'cte/closest_pt'])


        class UserCondition:
            def run(self, mode):
                if mode == 'user':
                    return True
                else:
                    return False

        V.add(UserCondition(), inputs=['user/mode'], outputs=['run_user'])

        from donkeycar.parts.behavior import PilotCondition

        V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

        # This is the path object. It will record a path when distance changes and it travels
        # at least cfg.PATH_MIN_DIST meters. Except when we are in follow mode, see below...
        path = CsvThrottlePath(min_dist=cfg.PATH_MIN_DIST)
        V.add(path, inputs=['recording', 'pos/x', 'pos/y', 'user/throttle'], outputs=['path', 'throttles'])

        def save_path():
            if path.length() > 0:
                if path.save(cfg.PATH_FILENAME):
                    print("That path was saved to ", cfg.PATH_FILENAME)
                else:
                    print("The path could NOT be saved; check the PATH_FILENAME in myconfig.py to make sure it is a legal path")
            else:
                print("There is no path to save; try recording the path.")

        def load_path():
            if os.path.exists(cfg.PATH_FILENAME) and path.load(cfg.PATH_FILENAME):
                print("The path was loaded was loaded from ", cfg.PATH_FILENAME)

        def erase_path():
            origin_reset.reset_origin()
            if path.reset():
                print("The origin and the path were reset; you are ready to record a new path.")
            else:
                print("The origin was reset; you are ready to record a new path.")

        def reset_origin():
            """
            Reset effective pose to (0, 0)
            """
            origin_reset.reset_origin()
            print("The origin was reset to the current position.")


        # When a path is loaded, we will be in follow mode. We will not record.
        if os.path.exists(cfg.PATH_FILENAME):
            load_path()

        # Here's an image we can map to.
        img = PImage(clear_each_frame=True)
        V.add(img, outputs=['map/image'])

        # This PathPlot will draw path on the image

        plot = PathPlot(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET)
        V.add(plot, inputs=['map/image', 'path'], outputs=['map/image'])

        # This will use path and current position to output cross track error
        cte = CTE(look_ahead=cfg.PATH_LOOK_AHEAD, look_behind=cfg.PATH_LOOK_BEHIND, num_pts=cfg.PATH_SEARCH_LENGTH)
        V.add(cte, inputs=['path', 'pos/x', 'pos/y', 'cte/closest_pt'], outputs=['cte/error', 'cte/closest_pt'], run_condition='run_pilot')

        # This will use the cross track error and PID constants to try to steer back towards the path.
        pid = PIDController(p=cfg.PID_P, i=cfg.PID_I, d=cfg.PID_D)
        pilot = PID_Pilot(pid, cfg.PID_THROTTLE, cfg.USE_CONSTANT_THROTTLE, min_throttle=cfg.PID_THROTTLE)
        V.add(pilot, inputs=['cte/error', 'throttles', 'cte/closest_pt'], outputs=['pilot/angle', 'pilot/throttle'], run_condition="run_pilot")

        def dec_pid_d():
            pid.Kd -= cfg.PID_D_DELTA
            logging.info("pid: d- %f" % pid.Kd)

        def inc_pid_d():
            pid.Kd += cfg.PID_D_DELTA
            logging.info("pid: d+ %f" % pid.Kd)

        def dec_pid_p():
            pid.Kp -= cfg.PID_P_DELTA
            logging.info("pid: p- %f" % pid.Kp)

        def inc_pid_p():
            pid.Kp += cfg.PID_P_DELTA
            logging.info("pid: p+ %f" % pid.Kp)


        class ToggleRecording:
            def __init__(self, auto_record_on_throttle):
                self.auto_record_on_throttle = auto_record_on_throttle
                self.recording_latch:bool = None
                self.toggle_latch:bool = False
                self.last_recording = None

            def set_recording(self, recording:bool):
                self.recording_latch = recording

            def toggle_recording(self):
                self.toggle_latch = True

            def run(self, mode:str, recording:bool):
                recording_in = recording
                if recording_in != self.last_recording:
                    logging.info(f"Recording Change = {recording_in}")

                if self.toggle_latch:
                    if self.auto_record_on_throttle:
                        logger.info('auto record on throttle is enabled; ignoring toggle of manual mode.')
                    else:
                        recording = not self.last_recording
                    self.toggle_latch = False

                if self.recording_latch is not None:
                    recording = self.recording_latch
                    self.recording_latch = None

                if recording and mode != 'user':
                    logging.info("Ignoring recording in auto-pilot mode")
                    recording = False

                if self.last_recording != recording:
                    logging.info(f"Setting Recording = {recording}")

                self.last_recording = recording

                return recording


        recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE)
        V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])


        #
        # Add buttons for handling various user actions
        # The button names are in configuration.
        # They may refer to game controller (joystick) buttons OR web ui buttons
        #
        # There are 5 programmable webui buttons, "web/w1" to "web/w5"
        # adding a button handler for a webui button
        # is just adding a part with a run_condition set to
        # the button's name, so it runs when button is pressed.
        #
        have_joystick = ctr is not None and isinstance(ctr, JoystickController)

        # Here's a trigger to save the path. Complete one circuit of your course, when you
        # have exactly looped, or just shy of the loop, then save the path and shutdown
        # this process. Restart and the path will be loaded.
        if cfg.SAVE_PATH_BTN:
            print(f"Save path button is {cfg.SAVE_PATH_BTN}")
            if cfg.SAVE_PATH_BTN.startswith("web/w"):
                V.add(Lambda(lambda: save_path()), run_condition=cfg.SAVE_PATH_BTN)
            elif have_joystick:
                ctr.set_button_down_trigger(cfg.SAVE_PATH_BTN, save_path)

        # allow controller to (re)load the path
        if cfg.LOAD_PATH_BTN:
            print(f"Load path button is {cfg.LOAD_PATH_BTN}")
            if cfg.LOAD_PATH_BTN.startswith("web/w"):
                V.add(Lambda(lambda: load_path()), run_condition=cfg.LOAD_PATH_BTN)
            elif have_joystick:
                ctr.set_button_down_trigger(cfg.LOAD_PATH_BTN, load_path)

        # Here's a trigger to erase a previously saved path.
        # This erases the path in memory; it does NOT erase any saved path file
        if cfg.ERASE_PATH_BTN:
            print(f"Erase path button is {cfg.ERASE_PATH_BTN}")
            if cfg.ERASE_PATH_BTN.startswith("web/w"):
                V.add(Lambda(lambda: erase_path()), run_condition=cfg.ERASE_PATH_BTN)
            elif have_joystick:
                ctr.set_button_down_trigger(cfg.ERASE_PATH_BTN, erase_path)

        # Here's a trigger to reset the origin based on the current position
        if cfg.RESET_ORIGIN_BTN:
            print(f"Reset origin button is {cfg.RESET_ORIGIN_BTN}")
            if cfg.RESET_ORIGIN_BTN.startswith("web/w"):
                V.add(Lambda(lambda: reset_origin()), run_condition=cfg.RESET_ORIGIN_BTN)
            elif have_joystick:
                ctr.set_button_down_trigger(cfg.RESET_ORIGIN_BTN, reset_origin)

        # button to toggle recording
        if cfg.TOGGLE_RECORDING_BTN:
            print(f"Toggle recording button is {cfg.TOGGLE_RECORDING_BTN}")
            if cfg.TOGGLE_RECORDING_BTN.startswith("web/w"):
                V.add(Lambda(lambda: recording_control.toggle_recording()), run_condition=cfg.TOGGLE_RECORDING_BTN)
            elif have_joystick:
                ctr.set_button_down_trigger(cfg.TOGGLE_RECORDING_BTN, recording_control.toggle_recording)


        #Choose what inputs should change the car.
        class DriveMode:
            def run(self, mode, 
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle):
                if mode == 'user':
                    return user_angle, user_throttle
                elif mode == 'local_angle':
                    return pilot_angle, user_throttle
                else:
                    return pilot_angle, pilot_throttle

        V.add(DriveMode(), 
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle'], 
            outputs=['angle', 'throttle'])

        #
        # draw a map image as the vehicle moves
        #
        loc_plot = PlotCircle(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET, color = "blue")
        V.add(loc_plot, inputs=['map/image', 'pos/x', 'pos/y'], outputs=['map/image'], run_condition='run_pilot')

        loc_plot = PlotCircle(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET, color = "green")
        V.add(loc_plot, inputs=['map/image', 'pos/x', 'pos/y'], outputs=['map/image'], run_condition='run_user')
    else:
        logger.info("Unsupported drive type passed")
    
    
    if not cfg.DONKEY_GYM and cfg.HAVE_DRIVETRAIN:
            add_drivetrain(V, cfg)
    if cfg.USE_FPV:
        enable_fpv(V)
    if cfg.HAVE_FPS_COUNTER:
        # enable FPS counter
        enable_fps(V, cfg)
    if cfg.HAVE_PERFMON:
        enable_perfmon(V, cfg)
    if cfg.DONKEY_GYM:
        logger.info(f"You can now go to http://localhost:{cfg.WEB_CONTROL_PORT} to drive your car.")
    else:
        logger.info(f"You can now go to <your hostname.local>:{cfg.WEB_CONTROL_PORT} to drive your car.")
    if has_input_controller:
        logger.info("You can now move your controller to drive your car.")
        if isinstance(ctr, JoystickController):
            if cfg.DRIVE_TYPE == "behavior_cloning":
                ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)
        
        

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])
    drive(cfg)
