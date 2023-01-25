"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20      # the vehicle loop will pause if faster than this speed.
MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.

HAVE_DRIVETRAIN = False
DRIVE_TRAIN_TYPE = "VESC"

#VESC controller, primarily need to change VESC_SERIAL_PORT  and VESC_MAX_SPEED_PERCENT
VESC_MAX_SPEED_PERCENT = 0.2  # Max speed as a percent of the actual speed
VESC_SERIAL_PORT= "/dev/ttyACM0" # Serial device to use for communication. Can check with ls /dev/tty*
VESC_HAS_SENSOR= True # Whether or not the bldc motor is using a hall effect sensor
VESC_START_HEARTBEAT= True # Whether or not to automatically start the heartbeat thread that will keep commands alive.
VESC_BAUDRATE= 115200 # baudrate for the serial communication. Shouldn't need to change this.
VESC_TIMEOUT= 0.05 # timeout for the serial communication
VESC_STEERING_SCALE= 0.5 # VESC accepts steering inputs from 0 to 1. Joystick is usually -1 to 1. This changes it to -0.5 to 0.5
VESC_STEERING_OFFSET = 0.5 # VESC accepts steering inputs from 0 to 1. Coupled with above change we move Joystick to 0 to 1

#ODOMETRY
HAVE_ODOM = False                   # Do you have an odometer/encoder
USE_ODOM_IN_MODEL = False
ODOM_TYPE = "None"

# #LIDAR
HAVE_LIDAR = False
LIDAR_TYPE = 'None' #(RP|YD)
LIDAR_LOWER_LIMIT = 90 # angles that will be recorded. Use this to block out obstructed areas on your car, or looking backwards. Note that for the RP A1M8 Lidar, "0" is in the direction of the motor
LIDAR_UPPER_LIMIT = 270
USE_LIDAR_IN_MODEL = False

#TRAINING
# The default AI framework to use. Choose from (tensorflow|pytorch)
DEFAULT_AI_FRAMEWORK = 'tensorflow'

# The DEFAULT_MODEL_TYPE will choose which model will be created at training
# time. This chooses between different neural network designs. You can
# override this setting by passing the command line parameter --type to the
# python manage.py train and drive commands.
# tensorflow models: (linear|categorical|tflite_linear|tensorrt_linear)
# pytorch models: (resnet18)
BATCH_SIZE = 128                #how many records to use when doing one pass of gradient decent. Use a smaller number if your gpu is running out of memory.
TRAIN_TEST_SPLIT = 0.8          #what percent of records to use for training. the remaining used for validation.
MAX_EPOCHS = 100                #how many times to visit all records of your data
SHOW_PLOT = True                #would you like to see a pop up display of final loss?
VERBOSE_TRAIN = True            #would you like to see a progress bar with text during training?
USE_EARLY_STOP = True           #would you like to stop the training if we see it's not improving fit?
EARLY_STOP_PATIENCE = 5         #how many epochs to wait before no improvement
MIN_DELTA = .0005               #early stop will want this much loss change before calling it improved.
PRINT_MODEL_SUMMARY = True      #print layers and weights to stdout
OPTIMIZER = None                #adam, sgd, rmsprop, etc.. None accepts default
LEARNING_RATE = 0.001           #only used when OPTIMIZER specified
LEARNING_RATE_DECAY = 0.0       #only used when OPTIMIZER specified
SEND_BEST_MODEL_TO_PI = False   #change to true to automatically send best model during training
CREATE_TF_LITE = True           # automatically create tflite model in training
CREATE_TENSOR_RT = False        # automatically create tensorrt model in training

PRUNE_CNN = False               #This will remove weights from your model. The primary goal is to increase performance.
PRUNE_PERCENT_TARGET = 75       # The desired percentage of pruning.
PRUNE_PERCENT_PER_ITERATION = 20 # Percenge of pruning that is perform per iteration.
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # The max amout of validation loss that is permitted during pruning.
PRUNE_EVAL_PERCENT_OF_DATASET = .05  # percent of dataset used to perform evaluation of model.

# Augmentations and Transformations
AUGMENTATIONS = []
TRANSFORMATIONS = []
# Settings for brightness and blur, use 'MULTIPLY' and/or 'BLUR' in
# AUGMENTATIONS
AUG_MULTIPLY_RANGE = (0.5, 3.0)
AUG_BLUR_RANGE = (0.0, 3.0)
# Region of interest cropping, requires 'CROP' in TRANSFORMATIONS to be set
# If these crops values are too large, they will cause the stride values to
# become negative and the model with not be valid.
ROI_CROP_TOP = 45               # the number of rows of pixels to ignore on the top of the image
ROI_CROP_BOTTOM = 0             # the number of rows of pixels to ignore on the bottom of the image
ROI_CROP_RIGHT = 0              # the number of rows of pixels to ignore on the right of the image
ROI_CROP_LEFT = 0               # the number of rows of pixels to ignore on the left of the image
# For trapezoidal see explanation in augmentations.py. Requires 'TRAPEZE' in
# TRANSFORMATIONS to be set
ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = 160
ROI_TRAPEZE_UL = 20
ROI_TRAPEZE_UR = 140
ROI_TRAPEZE_MIN_Y = 60
ROI_TRAPEZE_MAX_Y = 120

#Model transfer options
#When copying weights during a model transfer operation, should we freeze a certain number of layers
#to the incoming weights and not allow them to change during training?
FREEZE_LAYERS = False               #default False will allow all layers to be modified by training
NUM_LAST_LAYERS_TO_TRAIN = 7        #when freezing layers, how many layers from the last should be allowed to train?

#WEB CONTROL
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # which port to listen on when making a web controller
WEB_INIT_MODE = "user"              # which control mode to start in. one of user|local_angle|local. Setting local will start in ai mode.

#JOYSTICK
HAVE_JOYSTICK = False      #when starting the manage.py, when True, will not require a --js option to use the joystick
JOYSTICK_MAX_THROTTLE = 0.5         #this scalar is multiplied with the -1 to 1 throttle value to limit the maximum throttle. This can help if you drop the controller or just don't need the full speed available.
JOYSTICK_STEERING_SCALE = 1.0       #some people want a steering that is less sensitve. This scalar is multiplied with the steering -1 to 1. It can be negative to reverse dir.
AUTO_RECORD_ON_THROTTLE = True      #if true, we will record whenever throttle is not zero. if false, you must manually toggle recording with some other trigger. Usually circle button on joystick.
CONTROLLER_TYPE = 'xbox'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the my_joystick.py controller written by the `donkey createjs` command
USE_NETWORKED_JS = False            #should we listen for remote joystick control over the network?
NETWORK_JS_SERVER_IP = None         #when listening for network joystick control, which ip is serving this information
JOYSTICK_DEADZONE = 0.01            # when non zero, this is the smallest throttle before recording triggered.
JOYSTICK_THROTTLE_DIR = -1.0         # use -1.0 to flip forward/backward, use 1.0 to use joystick's natural forward/backward
USE_FPV = False                     # send camera data to FPV webserver
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # this is the unix file use to access the joystick.

#IMU
HAVE_IMU = False                #when true, this add a Mpu6050 part and records the data. Can be used with a
IMU_TYPE = 'mpu6050'          # (mpu6050|mpu9250)
IMU_ADDRESS = 0x68              # if AD0 pin is pulled high them address is 0x69, otherwise it is 0x68
IMU_DLP_CONFIG = 0              # Digital Lowpass Filter setting (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)

#LOGGING
HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'          # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects

#TELEMETRY
HAVE_MQTT_TELEMETRY = False
TELEMETRY_DONKEY_NAME = 'my_robot1234'
TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
TELEMETRY_MQTT_JSON_ENABLE = False
TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
TELEMETRY_MQTT_BROKER_PORT = 1883
TELEMETRY_PUBLISH_PERIOD = 1
TELEMETRY_LOGGING_ENABLE = True
TELEMETRY_LOGGING_LEVEL = 'INFO' # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
TELEMETRY_LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
TELEMETRY_DEFAULT_TYPES = 'float,float'

# PERF MONITOR
HAVE_PERFMON = False

#RECORD OPTIONS
RECORD_DURING_AI = False        #normally we do not record during ai mode. Set this to true to get image and steering records for your Ai. Be careful not to use them to train.
AUTO_CREATE_NEW_TUB = False     #create a new tub (tub_YY_MM_DD) directory when recording or append records to data directory directly

#LED Color for record count indicator
REC_COUNT_ALERT = 1000          #how many records before blinking alert
REC_COUNT_ALERT_CYC = 15        #how many cycles of 1/20 of a second to blink per REC_COUNT_ALERT records
REC_COUNT_ALERT_BLINK_RATE = 0.4 #how fast to blink the led in seconds on/off
#first number is record count, second tuple is color ( r, g, b) (0-100)
#when record count exceeds that number, the color will be used
RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]

#DonkeyGym
#Only on Ubuntu linux, you can use the simulator as a virtual donkey and
#issue the same python manage.py drive command as usual, but have them control a virtual car.
#This enables that, and sets the path to the simualator and the environment.
#You will want to download the simulator binary from: https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip
#then extract that and modify DONKEY_SIM_PATH.
DONKEY_GYM = True
DONKEY_SIM_PATH = "remote" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
DONKEY_GYM_ENV_NAME = "donkey-avc-sparkfun-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "car01", "body_rgb" : (255, 205, 0), "car_name" : "UCSD-DSC-178-YourName", "font_size" : 30} # body style(donkey|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "UCSD-DSC-178-YourName"
GYM_CONF["country"] = "USA"
GYM_CONF["bio"] = "Representing HDSI"
GYM_CONF["cam_config"] = {
    "img_w": 160, # image width
    "img_h": 120, # image height
    "img_d": 3, # depth in dimensions
    "img_enc": "JPG", # encoding format to send images in
    "fov": 70.0, # The field of view
    "fish_eye_x": 0.0, # controls distortion
    "fish_eye_y": 0.0, # controls distortion
    "offset_x": 0.0, #offset_x moves camera left/right
    "offset_y": 0.0, #offset_y moves camera up/down
    "offset_z": 0.0, #offset_z moves camera forward/back
    "rot_x": 0.0, #rot_x will rotate the camera laterally
    "rot_y": 0.0, #rot_y will rotate the camera up / down
    "rot_z": 0.0, #rot_z will rotate the camera around forward vector
}
GYM_CONF["lidar_config"] = {
    "deg_per_sweep_inc": 2.0, # Angular resolution
    "deg_ang_down": 10,
    "deg_ang_delta": -1.0,
    "num_sweeps_levels": 1, # Number of sweeps
    "max_range": 25.0,
    "noise": 0.0,
    "offset_x": 0.0,
    "offset_y": 0.0,
    "offset_z": 0.0,
    "rot_x": 0.0,
}

SIM_HOST = "127.0.0.1"              # when racing on virtual-race-league use host "trainmydonkey.com"
SIM_ARTIFICIAL_LATENCY = 0          # this is the millisecond latency in controls. Can use useful in emulating the delay when useing a remote server. values of 100 to 400 probably reasonable.

# Save info from Simulator (pln)
SIM_RECORD_LOCATION = False
SIM_RECORD_GYROACCEL= False
SIM_RECORD_VELOCITY = False
SIM_RECORD_LIDAR = False

#When racing, to give the ai a boost, configure these values.
AI_LAUNCH_DURATION = 0.0            # the ai will output throttle for this many seconds
AI_LAUNCH_THROTTLE = 0.0            # the ai will output this throttle value
AI_LAUNCH_ENABLE_BUTTON = 'R2'      # this keypress will enable this boost. It must be enabled before each use to prevent accidental trigger.
AI_LAUNCH_KEEP_ENABLED = False      # when False ( default) you will need to hit the AI_LAUNCH_ENABLE_BUTTON for each use. This is safest. When this True, is active on each trip into "local" ai mode.

#Scale the output of the throttle of the ai pilot for all model types.
AI_THROTTLE_MULT = 1.0              # this multiplier will scale every throttle value for all output from NN models

#Path following
PATH_FILENAME = "donkey_path.pkl"   # the path will be saved to this filename
PATH_SCALE = 5.0                    # the path display will be scaled by this factor in the web page
PATH_OFFSET = (0, 0)                # 255, 255 is the center of the map. This offset controls where the origin is displayed.
PATH_MIN_DIST = 0.3                 # after travelling this distance (m), save a path point
PID_P = -10.0                       # proportional mult for PID path follower
PID_I = 0.000                       # integral mult for PID path follower
PID_D = -0.2                        # differential mult for PID path follower
PID_THROTTLE = 0.2                  # constant throttle value during path following
SAVE_PATH_BTN = "cross"             # joystick button to save path
RESET_ORIGIN_BTN = "triangle"       # joystick button to press to move car back to origin

# FPS counter# FPS counter
HAVE_FPS_COUNTER = False
FPS_DEBUG_INTERVAL = 10    # the interval in seconds for printing the frequency info into the shell

# Behavior cloning model
BEHAVIOR_CLONE_MODEL_TYPE = 'linear'
BEHAVIOR_CLONE_MODEL_PATH = None
#BEHAVIORS
#When training the Behavioral Neural Network model, make a list of the behaviors,
#Set the TRAIN_BEHAVIORS = True, and use the BEHAVIOR_LED_COLORS to give each behavior a color
TRAIN_BEHAVIORS = False
BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]  #RGB tuples 0-100 per chanel

#Localizer
#The localizer is a neural network that can learn to predict its location on the track.
#This is an experimental feature that needs more developement. But it can currently be used
#to predict the segement of the course, where the course is divided into NUM_LOCATIONS segments.
TRAIN_LOCALIZER = False
NUM_LOCATIONS = 10
BUTTON_PRESS_NEW_TUB = False #when enabled, makes it easier to divide our data into one tub per track length if we make a new tub on each X button press.

HAVE_CAMERA = False
CAMERA_TYPE = "None"
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3 

DRIVE_TYPE = "behavior_cloning"