import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_PORT = "/dev/ttyS0"
BAUD_RATE = 57600

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_H_FOV_DEG = 78
CALIBRATION_FILE = os.path.join(BASE_DIR, "calib.npz")

SAVE_DIR = os.environ.get("VIENTO_SAVE_DIR", os.path.join(BASE_DIR, "outputs"))
DEBUG_DIR = os.path.join(SAVE_DIR, "debug_logs")

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best2.pt"),
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), MODEL_CANDIDATES[0])

TARGET_CLASS_NAMES = {"blue_square", "red_square"}
PRED_CONF_THRESHOLD = 0.15
CLASS_MIN_CONF = {"blue_square": 0.20, "red_square": 0.40}
CLASS_MIN_AREA = {"blue_square": 80, "red_square": 160}
CLASS_REQUIRED_STREAK = {"blue_square": 1, "red_square": 1}
DRAW_DETECTIONS = True

CENTER_PRIORITY_ENABLED = True
PER_CLASS_CENTER_PICK = True
COLOR_VERIFICATION_ENABLED = True
MIN_COLOR_RATIO = {"blue_square": 0.10, "red_square": 0.25}
SQUARE_ASPECT_TOLERANCE = 0.35

SERVO1_PIN = 17
SERVO2_PIN = 27
SERVO_PWM_FREQ = 50
SERVO_PERIOD_US = 20000
SERVO1_INIT_US = 1540
SERVO1_RELEASE_US = 1882
SERVO2_INIT_US = 1547
SERVO2_RELEASE_US = 1882

MANUAL_WPS = [
    (37.0301885, 37.3120460, 20),
    (37.0309679, 37.3124027, 15),
    (37.0310450, 37.3119280, 10),
    (37.0304518, 37.3116329, 6),
]
MANUAL_LAND = (37.0300664, 37.3114425, 0)
MID_TURN_WP = (37.0291714, 37.3109959, 10)

HEADING_TOLERANCE_DEG = 30.0
MIN_RELEASE_DISTANCE_M = 20.0
MAX_RELEASE_DISTANCE_M = 25.0

WINDOW_TITLE = "HEDEF TESPITI"
DETECT_EVERY_N = 2
VIDEO_FPS = 20
FRAME_HISTORY_LEN = 8


def ensure_runtime_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
