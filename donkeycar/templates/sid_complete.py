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

def enable_telemetry(cfg):
    from donkeycar.parts.telemetry import MqttTelemetry
    tel = MqttTelemetry(cfg)

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
        outputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
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
        if cfg.HAVE_IMU and cfg.IMU_TYPE == "OAKD LITE":
            pass
        elif cfg.OAKD_LITE_STEREO_ENABLED:
            pass
        else:
            pass
        logger.info("OAKD LITE to be supported")
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
        elif cfg.HAVE_ODOMETRY and cfg.ODOM_TYPE == "VESC":
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

    if cfg.HAVE_MQTT_TELEMETRY:
        enable_telemetry(cfg)

    logger.info(f'PID: {os.getpid()}')
    if cfg.DONKEY_GYM:
        add_simulator(V, cfg)
    else:
        if cfg.HAVE_ODOM:
            # add odometry
            add_odometry(V, cfg)
        if cfg.HAVE_CAMERA:
            # setup primary camera
            add_camera(V, cfg, camera_type)
        if cfg.HAVE_LIDAR:
            # add lidar
            add_lidar(V, cfg)
        if cfg.HAVE_IMU:
            # add IMU
            add_imu(V, cfg)
    ctr, has_input_controller = add_user_controller(V, cfg)
    add_web_buttons(V)
    add_throttle_reverse(V)
    add_pilot_condition(V)
    if cfg.HAVE_FPS_COUNTER:
        # enable FPS counter
        enable_fps(V, cfg)
    if cfg.HAVE_PERFMON:
        enable_perfmon(V, cfg)
    
    if not cfg.DONKEY_GYM:
        if cfg.HAVE_DRIVETRAIN:
            add_drivetrain(V, cfg)
        
        

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])
    drive(cfg, args)
    if args['behavior_clone_drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])
    elif args['gps_follow_drive']:
        pass
    else:
        logger.info('Unsupported arg passed\n')


def drive(cfg, model_path=None, use_joystick=False, model_type=None,
          camera_type='single', meta=[]):
    """
    Construct a working robotic vehicle from many parts. Each part runs as a
    job in the Vehicle loop, calling either it's run or run_threaded method
    depending on the constructor flag `threaded`. All parts are updated one
    after another at the framerate given in cfg.DRIVE_LOOP_HZ assuming each
    part finishes processing in a timely manner. Parts may have named outputs
    and inputs. The framework handles passing named outputs to parts
    requesting the same named input.
    """

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE:
        def show_record_count_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):  # these controllers don't use the joystick class
            if isinstance(ctr, JoystickController):
                ctr.set_button_down_trigger('circle', show_record_count_status) #then we are not using the circle button. hijack that to force a record count indication
        else:
            
            show_record_count_status()


    # Use the FPV preview, which will show the cropped image output, or the full frame.
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    def load_model(kl, model_path):
        start = time.time()
        logger.info('loading model', model_path)
        kl.load(model_path)
        logger.info('finished loading in %s sec.' % (str(time.time() - start)) )

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            logger.info('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            logger.info('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            logger.info(e)
            logger.info('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        logger.info('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            logger.info('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            logger.info(e)
            logger.info("ERR>> problems loading model json", json_fnm)

    #
    # load and configure model for inference
    #
    if model_path:
        # If we have a model, create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        #
        # get callback function to reload the model
        # for the configured model format
        #
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

        # this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True),
              outputs=['modelfile/modified'])

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

        elif cfg.USE_LIDAR:
            inputs = ['cam/image_array', 'lidar/dist_array']

        elif cfg.HAVE_ODOM:
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

    # Choose what inputs should change the car.
    class DriveMode:
        def run(self, mode,
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle):
            if mode == 'user':
                return user_angle, user_throttle

            elif mode == 'local_angle':
                return pilot_angle if pilot_angle else 0.0, user_throttle

            else:
                return pilot_angle if pilot_angle else 0.0, \
                       pilot_throttle * cfg.AI_THROTTLE_MULT \
                           if pilot_throttle else 0.0

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])



    if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
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

    #
    # Setup drivetrain
    #
    add_drivetrain(V, cfg)


    # OLED setup
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_ROTATION, cfg.SSD1306_RESOLUTION, auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)

    # add tub to save data

    if cfg.USE_LIDAR:
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
            inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
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
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # Telemetry (we add the same metrics added to the TubHandler
    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)
        telem_inputs, _ = tel.add_step_inputs(inputs, types)
        V.add(tel, inputs=telem_inputs, outputs=["tub/queue_size"], threaded=True)

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])


    if cfg.DONKEY_GYM:
        logger.info("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    else:
        logger.info("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        logger.info("You can now move your controller to drive your car.")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])
    drive(cfg)
