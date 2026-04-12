#!/usr/bin/env python3
# -- coding: utf-8 --

import cv2
import numpy as np
from ultralytics import YOLO
import math
from pymavlink import mavutil
import time
import threading
import os
from geo_utils import (
    calculate_release_distance as calculate_release_distance_util,
    compute_target_gps as compute_target_gps_util,
    compute_target_gps_with_attitude as compute_target_gps_with_attitude_util,
    compute_target_gps_with_attitude_and_heading as compute_target_gps_with_attitude_and_heading_util,
    distance_2d as distance_2d_util,
)
from logging_utils import create_debug_logger
from pixhawk_connection import connect_pixhawk as pixhawk_connect
from runtime_config import (
    BAUD_RATE,
    CALIBRATION_FILE,
    CAMERA_H_FOV_DEG,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CENTER_PRIORITY_ENABLED,
    CLASS_MIN_AREA,
    CLASS_MIN_CONF,
    CLASS_REQUIRED_STREAK,
    DEBUG_DIR,
    DETECT_EVERY_N,
    DRAW_DETECTIONS,
    FRAME_HISTORY_LEN,
    HEADING_TOLERANCE_DEG,
    MANUAL_LAND,
    MANUAL_WPS,
    MID_TURN_WP,
    MIN_RELEASE_DISTANCE_M,
    MAX_RELEASE_DISTANCE_M,
    MODEL_PATH,
    PER_CLASS_CENTER_PICK,
    PRED_CONF_THRESHOLD,
    SAVE_DIR,
    SERIAL_PORT,
    SERVO1_INIT_US,
    SERVO1_PIN,
    SERVO1_RELEASE_US,
    SERVO2_INIT_US,
    SERVO2_PIN,
    SERVO2_RELEASE_US,
    SERVO_PERIOD_US,
    SERVO_PWM_FREQ,
    TARGET_CLASS_NAMES,
    VIDEO_FPS,
    WINDOW_TITLE,
    ensure_runtime_dirs,
)
from servo_controller import ServoController
from vision_utils import (
    ratio_of_target_color_and_square_shape as target_box_ok,
    undistort_point as undistort_point_util,
)
from datetime import datetime
import queue
from collections import deque
import RPi.GPIO as GPIO
try:
    import torch  # noqa: F401
    _torch_available = True
except Exception:
    _torch_available = False
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setmode(GPIO.BCM)  # Pin numaralandırması için BCM kullan
GPIO.setwarnings(False)  # Uyarıları kapat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERIAL_PORT = "/dev/ttyS0"  # Raspberry Pi GPIO UART portu (pin 8=TX, pin 10=RX)
BAUD_RATE = 57600
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_H_FOV_DEG = 78
CALIBRATION_FILE = os.path.join(BASE_DIR, "calib.npz")

SAVE_DIR = os.environ.get("VIENTO_SAVE_DIR", os.path.join(BASE_DIR, "outputs"))
DEBUG_DIR = os.path.join(SAVE_DIR, "debug_logs")  # Debug logları için klasör
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

debug_log_file = os.path.join(DEBUG_DIR, f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def write_debug_log(message, print_to_console=True):
    """Debug mesajını hem dosyaya hem de (istenirse) konsola yazar"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_message = f"[{timestamp}] {message}\n"
    
    try:
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)
    except Exception as e:
        print(f"Log dosyası yazma hatası: {e}")
    
    if print_to_console:
        print(message)


ensure_runtime_dirs()
debug_log_file, write_debug_log = create_debug_logger(DEBUG_DIR)


if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(f"Kamera kalibrasyon dosyası bulunamadı: {CALIBRATION_FILE}")
calib_data = np.load(CALIBRATION_FILE)
mtx = calib_data['mtx']
dist = calib_data['dist']
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (CAMERA_WIDTH, CAMERA_HEIGHT), 5)

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(BASE_DIR, "best2.pt"),
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), MODEL_CANDIDATES[0])
model = YOLO(MODEL_PATH)
print("Model etiketleri:", model.names)

try:
    model.fuse()
except Exception:
    pass
device = "cpu"
if _torch_available:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
try:
    model.to(device)
except Exception:
    pass

TARGET_CLASS_NAMES = {"blue_square", "red_square"}
try:
    TARGET_CLASS_IDS = [i for i, n in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names)) if n in TARGET_CLASS_NAMES]
except Exception:
    TARGET_CLASS_IDS = []

PRED_CONF_THRESHOLD = 0.15
CLASS_MIN_CONF = {"blue_square": 0.20, "red_square": 0.40}  # Mavi için güven eşiğini düşürdük
CLASS_MIN_AREA = {"blue_square": 80, "red_square": 160}  # Mavi için minimum alanı düşürdük
CLASS_REQUIRED_STREAK = {"blue_square": 1, "red_square": 1}
DRAW_DETECTIONS = True  # Tespit kutularını göster

CENTER_PRIORITY_ENABLED = True  # Merkeze en yakın hedefi seç
PER_CLASS_CENTER_PICK = True  # Her sınıf için ayrı merkez kontrolü yap
MAX_CENTER_DISTANCE_RATIO = 0.90

COLOR_VERIFICATION_ENABLED = True
MIN_COLOR_RATIO = {"blue_square": 0.10, "red_square": 0.25}  # Mavi için renk oranı eşiğini daha da düşürdük - genişletilmiş eşikler için
SQUARE_ASPECT_TOLERANCE = 0.35

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass
try:
    cap.set(cv2.CAP_PROP_FPS, 20)
except Exception:
    pass
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
except Exception:
    pass
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı")

def connect_pixhawk():
    """Pixhawk'a bağlanmayı dene ve bağlantı nesnesini döndür"""
    import serial
    import serial.tools.list_ports
    max_attempts = 3
    
    def _find_pixhawk_port():
        """Pixhawk'ın bağlı olduğu portu bulmaya çalış"""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if any(id in port.hwid.lower() for id in ['2341', '26ac', '0483']):
                return port.device
        return None
    
    def _configure_serial_port(port):
        """Seri portu yapılandır"""
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = BAUD_RATE
            ser.bytesize = serial.EIGHTBITS
            ser.parity = serial.PARITY_NONE
            ser.stopbits = serial.STOPBITS_ONE
            ser.timeout = 1
            ser.write_timeout = 1
            ser.xonxoff = False
            ser.rtscts = False
            ser.dsrdtr = False
            
            ser.open()
            
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            ser.setDTR(False)
            ser.setRTS(False)
            time.sleep(0.3)  # Daha uzun bekleme
            ser.setDTR(True)
            ser.setRTS(True)
            
            ser.close()
            time.sleep(0.3)  # Daha uzun bekleme
            return True
        except Exception as e:
            print(f"Port yapılandırma hatası: {e}")
            try:
                ser.close()
            except:
                pass
            return False
    
    def _wait_for_heartbeat(mavlink_conn, timeout=10):
        """Heartbeat için bekle ve bağlantı durumunu kontrol et"""
        print("Heartbeat bekleniyor...")
        start_wait = time.time()
        heartbeat_count = 0
        required_heartbeats = 3  # En az 3 heartbeat al
        
        while time.time() - start_wait < timeout:
            try:
                msg = mavlink_conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if msg:
                    heartbeat_count += 1
                    print(f"Heartbeat alındı ({heartbeat_count}/{required_heartbeats})")
                    if heartbeat_count >= required_heartbeats:
                        return True
            except Exception as e:
                print(f"Heartbeat okuma hatası: {e}")
                continue
        return False
    
    auto_port = _find_pixhawk_port()
    if auto_port:
        print(f"Pixhawk tespit edildi: {auto_port}")
        port_to_use = auto_port
    else:
        print(f"Pixhawk otomatik tespit edilemedi, varsayılan port kullanılacak: {SERIAL_PORT}")
        port_to_use = SERIAL_PORT
    
    for attempt in range(max_attempts):
        try:
            print(f"\nPixhawk bağlantısı deneniyor... (Deneme {attempt + 1}/{max_attempts})")
            
            if not _configure_serial_port(port_to_use):
                print("⚠️ Port yapılandırılamadı, yine de bağlanmaya çalışılacak...")
            
            mavlink_conn = mavutil.mavlink_connection(
                port_to_use,
                baud=BAUD_RATE,
                autoreconnect=True,
                source_system=1,
                source_component=1,
                force_connected=True,
                write_timeout=1,
                timeout=1
            )
            
            if hasattr(mavlink_conn.port, 'setBufferSize'):
                mavlink_conn.port.setBufferSize(rx_size=4096, tx_size=4096)  # Buffer boyutunu artır
            
            if hasattr(mavlink_conn.port, 'setLatencyTimer'):
                mavlink_conn.port.setLatencyTimer(milliseconds=1)  # Gecikmeyi azalt
            
            if _wait_for_heartbeat(mavlink_conn):
                print(f"✅ Pixhawk {port_to_use} üzerinden {BAUD_RATE} baud ile bağlandı")
                time.sleep(2)
                return mavlink_conn
            else:
                print("❌ Yeterli heartbeat alınamadı")
                try:
                    mavlink_conn.close()
                except:
                    pass
            
        except Exception as e:
            print(f"⚠️ Bağlantı hatası: {e}")
            if attempt < max_attempts - 1:
                print("Yeniden deneniyor...")
                time.sleep(2)
            continue
    
    print("\n❌ Pixhawk bağlantısı kurulamadı!")
    print("⚠️ Lütfen kontrol edin:")
    print("  1. USB bağlantısını çıkarıp yeniden takın")
    print("  2. Pixhawk'ı yeniden başlatın")
    print("  3. Başka MAVLink programlarını kapatın")
    print("  4. QGroundControl veya Mission Planner'ı kapatın")
    print(f"  5. Port doğru mu? (şu an: {port_to_use})")
    raise SystemExit("Bağlantı kurulamadı")

master = pixhawk_connect(SERIAL_PORT, BAUD_RATE)

print("\nMevcut görevler kontrol ediliyor...")
master.mav.mission_request_list_send(master.target_system, master.target_component)
msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=2)
if msg:
    print(f"Sistemde {msg.count} görev mevcut")

SERVO1_PIN = 17  # Kırmızı yük
SERVO2_PIN = 27  # Mavi yük
GPIO.setup(SERVO1_PIN, GPIO.OUT)
GPIO.setup(SERVO2_PIN, GPIO.OUT)

SERVO_PWM_FREQ = 50  # 50 Hz → 20,000 µs periyot
SERVO_PERIOD_US = 20000

SERVO1_INIT_US = 1540
SERVO1_RELEASE_US = 1882

SERVO2_INIT_US = 1547
SERVO2_RELEASE_US = 1882

def _us_to_duty(pulse_width_us):
    pulse = max(500, min(2500, int(pulse_width_us)))
    return (pulse / SERVO_PERIOD_US) * 100.0

servo1_pwm = None
servo2_pwm = None

def servo_move_to_us(pin, pulse_us, duration_sec=0.6, keep_on=False):
    """Servo kontrolü - Normal GPIO PWM kullanarak"""
    try:
        pwm = GPIO.PWM(pin, SERVO_PWM_FREQ)
        duty_cycle = _us_to_duty(pulse_us)  # Mikrosaniyeyi duty cycle'a çevir
        pwm.start(duty_cycle)
        
        time.sleep(max(0.05, float(duration_sec)))
        
        if keep_on:
            global servo1_pwm, servo2_pwm
            if pin == SERVO1_PIN:
                servo1_pwm = pwm
            elif pin == SERVO2_PIN:
                servo2_pwm = pwm
        else:
            pwm.stop()
    except Exception as e:
        print(f"Servo kontrol hatası (pin {pin}): {e}")
        pass

def initialize_servos_to_neutral():
    print(f"GPIO{SERVO1_PIN} başlangıç PWM: {SERVO1_INIT_US} µs")
    servo_move_to_us(SERVO1_PIN, SERVO1_INIT_US, duration_sec=0.6, keep_on=False)
    print(f"GPIO{SERVO2_PIN} başlangıç PWM: {SERVO2_INIT_US} µs")
    servo_move_to_us(SERVO2_PIN, SERVO2_INIT_US, duration_sec=0.6, keep_on=False)

def release_servo(pin):
    if pin == SERVO1_PIN:
        print(f"GPIO{SERVO1_PIN} bırakma PWM: {SERVO1_RELEASE_US} µs")
        servo_move_to_us(SERVO1_PIN, SERVO1_RELEASE_US, duration_sec=0.6, keep_on=False)
    elif pin == SERVO2_PIN:
        print(f"GPIO{SERVO2_PIN} bırakma PWM: {SERVO2_RELEASE_US} µs")
        servo_move_to_us(SERVO2_PIN, SERVO2_RELEASE_US, duration_sec=0.6, keep_on=False)

initialize_servos_to_neutral()

servo_controller = ServoController(
    GPIO,
    SERVO1_PIN,
    SERVO2_PIN,
    SERVO_PWM_FREQ,
    SERVO_PERIOD_US,
    SERVO1_INIT_US,
    SERVO1_RELEASE_US,
    SERVO2_INIT_US,
    SERVO2_RELEASE_US,
)
servo_controller.setup()
release_servo = servo_controller.release

def calculate_release_distance(altitude, velocity):
    g = 9.81
    if altitude is None or altitude <= 0:
        return None
    t = math.sqrt(2 * altitude / g)
    d = velocity * t
    return d

def distance_2d(lat1, lon1, lat2, lon2):
    """İki GPS koordinatı arasındaki mesafeyi metre cinsinden hesaplar (Equirectangular yaklaşımı)"""
    try:
        lat1, lon1 = float(lat1), float(lon1)
        lat2, lon2 = float(lat2), float(lon2)
        
        if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90) or \
           not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
            print(f"⚠️ Geçersiz koordinatlar: ({lat1}, {lon1}) -> ({lat2}, {lon2})")
            return None
            
        if abs(lat1 - lat2) < 0.0000001 and abs(lon1 - lon2) < 0.0000001:
            return 0.0
            
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        avg_lat = (lat1_rad + lat2_rad) / 2
        
        x = dlon * math.cos(avg_lat)
        y = dlat
        
        R = 6378137.0  # WGS84 ekvator yarıçapı (metre)
        d = R * math.sqrt(x*x + y*y)
        
        print(f"\n=== MESAFE HESAPLAMA ===")
        print(f"  Nokta 1: {lat1:.7f}, {lon1:.7f}")
        print(f"  Nokta 2: {lat2:.7f}, {lon2:.7f}")
        print(f"  Farklar: {math.degrees(dlat):.7f}°N, {math.degrees(dlon):.7f}°E")
        print(f"  Ort. Enlem: {math.degrees(avg_lat):.7f}°")
        print(f"  X-Y metre: {x*R:.1f}m E, {y*R:.1f}m N")
        print(f"  Mesafe: {d:.1f}m")
        
        if d < 0.01:  # 1 cm'den küçük mesafeleri hassas göster
            return round(d, 3)
        return round(d, 1)
        
    except Exception as e:
        print(f"⚠️ Mesafe hesaplama hatası: {e}")
        print(f"  Girdi değerleri: ({lat1}, {lon1}) -> ({lat2}, {lon2})")
        return None

red_released = False
blue_released = False
red_wp_seq = None
blue_wp_seq = None
red_mission_index = None
blue_mission_index = None

MANUAL_WPS = [
    (37.0301885, 37.3120460, 20),
    (37.0309679, 37.3124027, 15),
    (37.0310450, 37.3119280, 10),
    (37.0304518, 37.3116329, 6)
]
MANUAL_LAND = (37.0300664, 37.3114425, 0)

MID_TURN_WP = (37.0291714, 37.3109959, 10)

HEADING_TOLERANCE_DEG = 30.0  # Yönelim toleransı (derece) - Rüzgarlı hava için daha geniş tolerans

RELEASE_EXTRA_DELAY_SEC = 0.25  # Servo gecikmesi telafisi (saniye)
CROSS_TRACK_TOLERANCE_M = 10.0  # Yan sapma toleransı (metre)

CAM_ROLL_OFFSET_DEG = 0.0
CAM_PITCH_OFFSET_DEG = 0.0
CAM_YAW_OFFSET_DEG = 0.0
RELEASE_DEBUG = True  # Debug mesajlarını her zaman göster
RELEASE_DEBUG_INTERVAL = 10.0  # Debug mesajlarını 10 saniyede bir göster
MIN_RELEASE_DISTANCE_M = 20.0  # En az 20 metre kala
MAX_RELEASE_DISTANCE_M = 25.0  # En fazla 25 metre kala

def compute_target_gps(cx, cy, center_x, center_y, current_lat, current_lon, current_alt):
    fov_x_rad = math.radians(CAMERA_H_FOV_DEG)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (CAMERA_HEIGHT / CAMERA_WIDTH))
    angle_per_pixel_x = fov_x_rad / CAMERA_WIDTH
    angle_per_pixel_y = fov_y_rad / CAMERA_HEIGHT
    dx = cx - center_x
    dy = cy - center_y
    angle_x = dx * angle_per_pixel_x
    angle_y = dy * angle_per_pixel_y
    h = current_alt
    target_dx = h * math.tan(angle_x)
    target_dy = h * math.tan(angle_y)
    delta_lat = target_dy / 111_000
    delta_lon = target_dx / (111_000 * math.cos(math.radians(current_lat)))
    target_lat = current_lat + delta_lat
    target_lon = current_lon + delta_lon
    return target_lat, target_lon

def compute_target_gps_with_attitude(cx, cy, center_x, center_y, current_lat, current_lon, current_alt, roll, pitch):
    fov_x_rad = math.radians(CAMERA_H_FOV_DEG)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (CAMERA_HEIGHT / CAMERA_WIDTH))
    angle_per_pixel_x = fov_x_rad / CAMERA_WIDTH
    angle_per_pixel_y = fov_y_rad / CAMERA_HEIGHT
    dx = cx - center_x
    dy = cy - center_y
    angle_x = dx * angle_per_pixel_x
    angle_y = dy * angle_per_pixel_y
    total_angle_x = angle_x - roll
    total_angle_y = angle_y - pitch
    h = current_alt
    target_dx = h * math.tan(total_angle_x)
    target_dy = h * math.tan(total_angle_y)
    delta_lat = target_dy / 111_000
    delta_lon = target_dx / (111_000 * math.cos(math.radians(current_lat)))
    target_lat = current_lat + delta_lat
    target_lon = current_lon + delta_lon
    return target_lat, target_lon

def compute_target_gps_with_attitude_and_heading(cx, cy, center_x, center_y, current_lat, current_lon, current_alt, roll, pitch, heading_deg):
    print(f"\nHedef Hesaplama Debug:")
    print(f"  Piksel: ({cx}, {cy})")
    print(f"  Merkez: ({center_x}, {center_y})")
    print(f"  İHA: {current_lat:.7f}, {current_lon:.7f}, {current_alt:.1f}m")
    print(f"  Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°")
    print(f"  Heading: {heading_deg if heading_deg is not None else 'Bilinmiyor'}°")

    fx = float(mtx[0, 0])  # Yatay odak uzaklığı
    fy = float(mtx[1, 1])  # Dikey odak uzaklığı
    
    fov_x_rad = math.radians(CAMERA_H_FOV_DEG)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * (CAMERA_HEIGHT / CAMERA_WIDTH))
    
    dx = cx - center_x
    dy = cy - center_y
    angle_x = math.atan2(dx, fx)  # Yatay açı
    angle_y = math.atan2(dy, fy)  # Dikey açı
    
    if heading_deg is not None:
        heading_rad = math.radians((90 - heading_deg) % 360)
    else:
        heading_rad = 0.0
    
    effective_x = angle_x - roll
    effective_y = angle_y - pitch
    
    h = float(current_alt if current_alt is not None else 0.0)
    
    ground_distance = h * math.tan(math.sqrt(effective_x**2 + effective_y**2))
    
    bearing = heading_rad + math.atan2(effective_x, effective_y)
    
    dx = ground_distance * math.sin(bearing)
    dy = ground_distance * math.cos(bearing)
    
    print(f"  Hesaplanan yer mesafesi: {ground_distance:.1f}m")
    print(f"  Hesaplanan ofset: Doğu={dx:.1f}m, Kuzey={dy:.1f}m")
    
    R = 6378137.0  # WGS84 ekvator yarıçapı (metre)
    
    delta_lat = (dy / R) * (180.0 / math.pi)
    
    delta_lon = (dx / R) * (180.0 / math.pi) / math.cos(math.radians(current_lat))
    
    target_lat = current_lat + delta_lat
    target_lon = current_lon + delta_lon
    
    print("\n=== GPS KOORDİNAT DÖNÜŞÜMÜ ===")
    print(f"  Ofsetler: {dx:.1f}m Doğu, {dy:.1f}m Kuzey")
    print(f"  Derece değişimi: {delta_lat:.7f}° Enlem, {delta_lon:.7f}° Boylam")
    print(f"  Mevcut konum: {current_lat:.7f}, {current_lon:.7f}")
    print(f"  Hedef konum: {target_lat:.7f}, {target_lon:.7f}")
    
    final_distance = distance_2d(current_lat, current_lon, target_lat, target_lon)
    print(f"  Final mesafe: {final_distance:.1f}m")
    
    if final_distance > 100.0:  # 100 metre üstü şüpheli
        print("⚠️ Mesafe çok büyük, basit hesaplama deneniyor...")
        return compute_target_gps_with_attitude(cx, cy, center_x, center_y, current_lat, current_lon, current_alt, roll, pitch)
    
    return target_lat, target_lon


def upload_mission_to_pixhawk(wps):
    """Waypoint listesini Pixhawk'a yükle (mevcut + yeni tüm listeyi gönderir)"""
    print("\nGörev yükleme başlıyor...")
    global master
    existing_wps = []
    with mavlink_lock:
        try:
            print("Mevcut görevler okunuyor...")
            master.mav.mission_request_list_send(master.target_system, master.target_component)
            msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=2)
            if msg and getattr(msg, 'count', 0) > 0:
                print(f"Mevcut görev sayısı: {msg.count}")
                for i in range(int(msg.count)):
                    master.mav.mission_request_int_send(master.target_system, master.target_component, i)
                    wp = master.recv_match(type='MISSION_ITEM_INT', blocking=True, timeout=2)
                    if wp:
                        existing_wps.append((
                            wp.x / 1e7,
                            wp.y / 1e7,
                            wp.z,
                            wp.command,
                            wp.param1,
                            wp.param2,
                            wp.param3,
                            wp.param4
                        ))
                print(f"✅ {len(existing_wps)} mevcut görev okundu")
            else:
                print("ℹ️ Mevcut görev bulunamadı")

            all_wps = existing_wps + wps
            existing_count = len(existing_wps)
            total = len(all_wps)
            print(f"\nToplam görev sayısı: {total} (Mevcut: {existing_count}, Yeni: {len(wps)})")

            time.sleep(0.2)
            master.mav.mission_count_send(master.target_system, master.target_component, total)

            uploaded = set()
            start_time = time.time()
            timeout = 30
            while len(uploaded) < total and (time.time() - start_time) < timeout:
                req = master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=2)
                if not req:
                    continue
                seq = int(getattr(req, 'seq', 0))
                if seq < 0 or seq >= total or seq in uploaded:
                    continue
                lat, lon, alt = all_wps[seq][:3]
                cmd = all_wps[seq][3] if len(all_wps[seq]) > 3 else mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
                p1 = all_wps[seq][4] if len(all_wps[seq]) > 4 else 0
                p2 = all_wps[seq][5] if len(all_wps[seq]) > 5 else 0
                p3 = all_wps[seq][6] if len(all_wps[seq]) > 6 else 0
                p4 = all_wps[seq][7] if len(all_wps[seq]) > 7 else 0
                print(f"WP{seq} yükleniyor: {lat:.7f}, {lon:.7f}, {alt}m")
                master.mav.mission_item_int_send(
                    master.target_system,
                    master.target_component,
                    seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    cmd,
                    0, 1,
                    p1, p2, p3, p4,
                    int(lat * 1e7),
                    int(lon * 1e7),
                    float(alt)
                )
                uploaded.add(seq)
                time.sleep(0.05)

            if len(uploaded) != total:
                print(f"⚠️ Sadece {len(uploaded)}/{total} waypoint yüklenebildi!")
                return False, existing_count, total

            ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
            if ack:
                print('✅ Görev yükleme başarılı!')
                master.mav.mission_request_list_send(master.target_system, master.target_component)
                msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=2)
                if msg and getattr(msg, 'count', 0) == total:
                    print(f"✅ Doğrulama başarılı: {msg.count} waypoint yüklendi")
                    return True, existing_count, total
                else:
                    print("⚠️ Yüklenen görev sayısı doğrulanamadı!")
                    return False, existing_count, total
            else:
                print('⚠️ Görev yükleme onayı (ACK) alınamadı!')
                return False, existing_count, total
        except Exception as e:
            print(f"⚠️ Görev yükleme sırasında hata: {e}")
            return False, 0, 0

def get_gps_from_pixhawk():
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000
        return lat, lon, alt
    return None, None, None

def get_gps_and_attitude_from_pixhawk():
    lat, lon, alt = None, None, None
    roll, pitch = 0.0, 0.0
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
    if msg:
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000
    att = master.recv_match(type='ATTITUDE', blocking=True, timeout=2)
    if att:
        roll = att.roll
        pitch = att.pitch
    return lat, lon, alt, roll, pitch

def get_existing_mission_items():
    master.mav.mission_request_list_send(master.target_system, master.target_component)
    mission_items = []
    mission_count = 0
    while True:
        msg = master.recv_match(type=['MISSION_COUNT', 'MISSION_ITEM_INT'], blocking=True)
        if msg and msg.get_type() == 'MISSION_COUNT':
            mission_count = msg.count
            break
    for i in range(mission_count):
        master.mav.mission_request_int_send(master.target_system, master.target_component, i)
        item = master.recv_match(type='MISSION_ITEM_INT', blocking=True)
        mission_items.append(item)
    return mission_items

red_target_detected = False
blue_target_detected = False
red_wp_uploaded = False
blue_wp_uploaded = False
sabitler_uploaded = False
red_target_lat, red_target_lon = None, None
blue_target_lat, blue_target_lon = None, None
red_mission_index, blue_mission_index = None, None
red_rel_wp_index, blue_rel_wp_index = None, None

detected_targets = []

shared_data = {'lat': None, 'lon': None, 'alt': None, 'roll': 0.0, 'pitch': 0.0, 'groundspeed': None, 'gps_ts': 0.0}
heading_data = {
    'current_heading': None,
    'target_heading': None,
    'heading_ok': False,
    'last_update': 0
}
servo_data = {
    'red_target': {'lat': None, 'lon': None, 'released': False, 'heading': None},
    'blue_target': {'lat': None, 'lon': None, 'released': False, 'heading': None}
}
data_lock = threading.Lock()
heading_lock = threading.Lock()
servo_lock = threading.Lock()
mavlink_lock = threading.Lock()  # MAVLink veri okuma için yeni kilit
mission_queue = queue.Queue(maxsize=1)
mission_read_queue = queue.Queue(maxsize=1)
mission_read_result_queue = queue.Queue(maxsize=1)
mission_upload_event = threading.Event()

rates_configured = False
_last_buf_log_ts = 0.0

def drain_mavlink_queue(max_msgs: int = 200) -> int:
    """Buffer şiştiğinde hızlıca mesaj tüket. İşlem yapmadan sadece boşaltır."""
    global master
    drained = 0
    try:
        while drained < max_msgs:
            msg = master.recv_match(blocking=False, timeout=0)
            if msg is None:
                break
            drained += 1
    except Exception:
        pass
    return drained

def configure_mavlink_rates():
    """Temel mesajların yayın hızlarını düşürerek buffer taşmasını azalt."""
    global master, rates_configured
    if master is None or rates_configured:
        return False
    try:
        def _set(msg_id, hz):
            interval_us = int(1_000_000 / hz) if hz > 0 else -1
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                float(msg_id),
                float(interval_us),
                0, 0, 0, 0, 0
            )
            time.sleep(0.05)

        _set(33, 10)  # Pozisyon 10 Hz
        _set(30, 10)  # Attitude 10 Hz
        _set(74, 5)   # VFR_HUD 5 Hz
        _set(24, 5)   # GPS_RAW_INT 5 Hz
        _set(42, 2)   # MISSION_CURRENT 2 Hz

        rates_configured = True
        print("✅ MAVLink yayın hızları ayarlandı")
        return True
    except Exception as e:
        print(f"⚠️ MAVLink hız ayarlama başarısız: {e}")
        return False

def read_mavlink_data(msg_type, blocking=False, timeout=None):
    """Thread-safe MAVLink veri okuma fonksiyonu"""
    global master
    with mavlink_lock:
        try:
            if master is None:
                print("⚠️ MAVLink bağlantısı yok!")
                return None
            
            if not hasattr(master, 'port') or master.port is None:
                print("⚠️ Port bağlantısı kopmuş!")
                master = None
                return None
            
            try:
                if hasattr(master.port, 'is_open') and not master.port.is_open:
                    print("⚠️ Port kapalı, yeniden açılıyor...")
                    master.port.open()
            except Exception as e:
                print(f"Port açma hatası: {e}")
                master = None
                return None
            
            try:
                if hasattr(master.port, 'in_waiting'):
                    waiting = master.port.in_waiting
                    if waiting > 1024:
                        drained = drain_mavlink_queue(max_msgs=500)
                        now = time.time()
                        global _last_buf_log_ts
                        if now - _last_buf_log_ts > 2.0:
                            print(f"⚠️ Buffer dolu (~{waiting} byte). {drained} mesaj boşaltıldı")
                            _last_buf_log_ts = now
                        time.sleep(0.02)
            except Exception:
                pass
            
            try:
                msg = master.recv_match(type=msg_type, blocking=blocking, timeout=timeout)
                if msg is not None:
                    return msg
                
                if not blocking:
                    return None
                
                if hasattr(master.port, 'in_waiting') and master.port.in_waiting > 0:
                    drained = drain_mavlink_queue(max_msgs=300)
                    print(f"⚠️ Buffer okuma tıkandı. {drained} mesaj boşaltıldı")
                    time.sleep(0.02)
                
                return None
                
            except Exception as e:
                error_str = str(e).lower()
                if "device reports readiness" in error_str:
                    print("⚠️ Port hazır ama veri okunamıyor, buffer temizleniyor...")
                    if hasattr(master.port, 'reset_input_buffer'):
                        master.port.reset_input_buffer()
                    time.sleep(0.1)
                    return None
                elif any(err in error_str for err in ["timeout", "connection", "broken", "disconnected"]):
                    print(f"⚠️ Bağlantı hatası: {e}")
                    master = None
                    return None
                else:
                    raise  # Diğer hataları yukarı ilet
                
        except Exception as e:
            print(f"MAVLink veri okuma hatası ({msg_type}): {e}")
            try:
                master = pixhawk_connect(SERIAL_PORT, BAUD_RATE)
            except:
                master = None
            return None

HEADING_TOLERANCE_DEG = 30.0  # ±30 derece tolerans

def heading_update_thread():
    """Heading verisini sürekli güncelleyen thread"""
    print("Heading güncelleme thread'i başladı")
    
    def calculate_heading(curr_lat, curr_lon, target_lat, target_lon):
        """İki nokta arasındaki heading'i hesapla (derece cinsinden, kuzeyden saat yönünde)"""
        d_lon = math.radians(target_lon - curr_lon)
        lat1 = math.radians(curr_lat)
        lat2 = math.radians(target_lat)
        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        heading = math.degrees(math.atan2(y, x))
        return (heading + 360.0) % 360.0
    
    while True:
        try:
            with data_lock:
                current_lat = shared_data['lat']
                current_lon = shared_data['lon']
            
            if current_lat is None or current_lon is None:
                time.sleep(0.1)
                continue
            
            with servo_lock:
                red_data = servo_data['red_target']
                blue_data = servo_data['blue_target']
                
                if red_data['lat'] is not None and not red_data['released']:
                    red_data['heading'] = calculate_heading(
                        current_lat, current_lon,
                        red_data['lat'], red_data['lon']
                    )
                
                if blue_data['lat'] is not None and not blue_data['released']:
                    blue_data['heading'] = calculate_heading(
                        current_lat, current_lon,
                        blue_data['lat'], blue_data['lon']
                    )
            
            msg = read_mavlink_data('VFR_HUD', blocking=False)
            if msg and hasattr(msg, 'heading'):
                try:
                    current_heading = float(msg.heading)
                    current_heading = (current_heading + 360.0) % 360.0
                    with heading_lock:
                        heading_data['current_heading'] = current_heading
                        heading_data['last_update'] = time.time()
                except (ValueError, TypeError) as e:
                    print(f"Heading veri dönüşüm hatası: {e}")
                    pass
            
            time.sleep(0.1)  # CPU yükünü azalt
            
        except Exception as e:
            print(f"Heading güncelleme thread hatası: {e}")
            time.sleep(0.5)  # Hata durumunda biraz daha uzun bekle

def servo_control_thread():
    """Servo kontrol ve mesafe hesaplama thread'i"""
    print("Servo kontrol thread'i başladı")
    
    def check_heading(current_heading, target_heading):
        """Heading kontrolü - Geniş tolerans ile"""
        if current_heading is None or target_heading is None:
            return True  # Heading verisi yoksa geçerli say
        diff = abs((current_heading - target_heading + 540.0) % 360.0 - 180.0)
        return diff <= HEADING_TOLERANCE_DEG
    
    while True:
        try:
            with data_lock:
                current_lat = shared_data['lat']
                current_lon = shared_data['lon']
                altitude = shared_data['alt']
            
            with heading_lock:
                current_heading = heading_data['current_heading']
                heading_age = time.time() - heading_data.get('last_update', 0)
                if heading_age > 2.0:  # 2 saniyeden eski heading verisi varsa güncelle
                    msg = read_mavlink_data('VFR_HUD', blocking=False)
                    if msg and hasattr(msg, 'heading'):
                        try:
                            current_heading = float(msg.heading)
                            current_heading = (current_heading + 360.0) % 360.0
                            heading_data['current_heading'] = current_heading
                            heading_data['last_update'] = time.time()
                        except (ValueError, TypeError):
                            pass
            
            if current_lat is None or current_lon is None:
                time.sleep(0.1)
                continue
            
            with servo_lock:
                red_data = servo_data['red_target']
                if (not red_data['released'] and 
                    red_data['lat'] is not None and 
                    red_data['lon'] is not None):
                    
                    snap_curr_lat = float(current_lat)
                    snap_curr_lon = float(current_lon)
                    snap_tgt_lat = float(red_data['lat'])
                    snap_tgt_lon = float(red_data['lon'])
                    snap_gps_ts = shared_data.get('gps_ts', 0.0)

                    write_debug_log(f"\nMesafe Hesaplama Debug (Kırmızı):")
                    write_debug_log(f"  Mevcut Konum: {snap_curr_lat:.7f}, {snap_curr_lon:.7f} (ts={snap_gps_ts:.0f})")
                    write_debug_log(f"  Hedef Konum: {snap_tgt_lat:.7f}, {snap_tgt_lon:.7f}")
                    
                    dist = distance_2d_util(snap_curr_lat, snap_curr_lon, snap_tgt_lat, snap_tgt_lon)
                    
                    if dist is not None:
                        write_debug_log(f"  Hesaplanan Mesafe: {dist:.1f}m")
                    
                    heading_ok = check_heading(current_heading, red_data['heading'])
                    
                    now_ts = time.time()
                    last = globals().get('_last_red_dbg_ts', 0)
                    if now_ts - last >= 1.0:
                        debug_msg = "\n[KIRMIZI HEDEF DURUM]"
                        if current_lat is None or current_lon is None:
                            debug_msg += "\n  GPS VERİSİ YOK!"
                            debug_msg += "\n  Mesafe hesaplanamıyor"
                        else:
                            debug_msg += f"\n  Mesafe: {dist:.1f}m"
                            debug_msg += f"\n  Mesafe Limitleri: {MIN_RELEASE_DISTANCE_M}-{MAX_RELEASE_DISTANCE_M}m"
                            debug_msg += f"\n  Mesafe Uygun mu: {'✅' if MIN_RELEASE_DISTANCE_M <= dist <= MAX_RELEASE_DISTANCE_M else '❌'}"
                        if current_heading is not None and red_data['heading'] is not None:
                            debug_msg += f"\n  Mevcut Heading: {current_heading:.1f}°"
                            debug_msg += f"\n  Hedef Heading: {red_data['heading']:.1f}°"
                            debug_msg += f"\n  Heading Uygun mu: {'✅' if heading_ok else '❌'}"
                        write_debug_log(debug_msg)
                        globals()['_last_red_dbg_ts'] = now_ts
                    
                    if (MIN_RELEASE_DISTANCE_M <= dist <= MAX_RELEASE_DISTANCE_M and heading_ok):
                        write_debug_log("\n[KIRMIZI] Servo bırakılıyor!")
                        release_servo(SERVO1_PIN)
                        red_data['released'] = True
                        write_debug_log("Kırmızı yük atışı yapıldı")
            
            with servo_lock:
                blue_data = servo_data['blue_target']
                if (not blue_data['released'] and 
                    blue_data['lat'] is not None and 
                    blue_data['lon'] is not None):
                    
                    snap_curr_lat = float(current_lat)
                    snap_curr_lon = float(current_lon)
                    snap_tgt_lat = float(blue_data['lat'])
                    snap_tgt_lon = float(blue_data['lon'])
                    snap_gps_ts = shared_data.get('gps_ts', 0.0)

                    write_debug_log(f"\nMesafe Hesaplama Debug (Mavi):")
                    write_debug_log(f"  Mevcut Konum: {snap_curr_lat:.7f}, {snap_curr_lon:.7f} (ts={snap_gps_ts:.0f})")
                    write_debug_log(f"  Hedef Konum: {snap_tgt_lat:.7f}, {snap_tgt_lon:.7f}")
                    
                    dist = distance_2d_util(snap_curr_lat, snap_curr_lon, snap_tgt_lat, snap_tgt_lon)
                    
                    if dist is not None:
                        write_debug_log(f"  Hesaplanan Mesafe: {dist:.1f}m")
                    
                    heading_ok = check_heading(current_heading, blue_data['heading'])
                    
                    now_ts = time.time()
                    last = globals().get('_last_blue_dbg_ts', 0)
                    if now_ts - last >= 1.0:
                        debug_msg = "\n[MAVİ HEDEF DURUM]"
                        if current_lat is None or current_lon is None:
                            debug_msg += "\n  GPS VERİSİ YOK!"
                            debug_msg += "\n  Mesafe hesaplanamıyor"
                        else:
                            debug_msg += f"\n  Mesafe: {dist:.1f}m"
                            debug_msg += f"\n  Mesafe Limitleri: {MIN_RELEASE_DISTANCE_M}-{MAX_RELEASE_DISTANCE_M}m"
                            debug_msg += f"\n  Mesafe Uygun mu: {'✅' if MIN_RELEASE_DISTANCE_M <= dist <= MAX_RELEASE_DISTANCE_M else '❌'}"
                        if current_heading is not None and blue_data['heading'] is not None:
                            debug_msg += f"\n  Mevcut Heading: {current_heading:.1f}°"
                            debug_msg += f"\n  Hedef Heading: {blue_data['heading']:.1f}°"
                            debug_msg += f"\n  Heading Uygun mu: {'✅' if heading_ok else '❌'}"
                        write_debug_log(debug_msg)
                        globals()['_last_blue_dbg_ts'] = now_ts
                    
                    if (MIN_RELEASE_DISTANCE_M <= dist <= MAX_RELEASE_DISTANCE_M and heading_ok):
                        write_debug_log("\n[MAVİ] Servo bırakılıyor!")
                        release_servo(SERVO2_PIN)
                        blue_data['released'] = True
                        write_debug_log("Mavi yük atışı yapıldı")
            
            time.sleep(0.1)  # CPU yükünü azalt
            
        except Exception as e:
            print(f"Servo kontrol thread hatası: {e}")
            time.sleep(0.1)

def undistort_point(x, y, mtx, dist):
    pts = np.array([[[x, y]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    return undistorted[0, 0, 0], undistorted[0, 0, 1]

WINDOW_TITLE = "HEDEF TESPITI"
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(4)
except Exception:
    pass

fps_counter = 0
fps_timer = time.time()
frame_index = 0
DETECT_EVERY_N = 2  # Her N karede bir tespit yap  # Tespiti her N karede bir yap
INFER_IMGSZ = max(CAMERA_WIDTH, CAMERA_HEIGHT)

VIDEO_FPS = 20
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = None
video_path = os.path.join(SAVE_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")

infer_queue = queue.Queue(maxsize=1)
infer_lock = threading.Lock()
latest_detections = {"boxes": [], "timestamp": 0.0}

FRAME_HISTORY_LEN = 8
recent_frames = deque(maxlen=FRAME_HISTORY_LEN)

def _ratio_of_target_color_and_square_shape(frame, bbox, label):
    if not COLOR_VERIFICATION_ENABLED:
        return True
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    aspect = bw / max(1.0, bh)
    if not (1.0 - SQUARE_ASPECT_TOLERANCE <= aspect <= 1.0 + SQUARE_ASPECT_TOLERANCE):
        return False

    if label == "blue_square":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 50:  # Çok küçük contour'ları reddet
            return False

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        if solidity < 0.7 or solidity > 1.0:
            return False

        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

        if compactness < 0.6 or compactness > 0.95:
            return False

        if len(contours) > 3:  # Çok fazla küçük contour varsa şüpheli
            return False

    elif label == "red_square":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 100:  # Daha büyük alan kontrolü - küçük üçgenleri de reddet
            return False

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        if solidity < 0.85 or solidity > 1.0:  # Üçgenler genellikle daha düşük solidity'ye sahip
            return False

        perimeter = cv2.arcLength(largest_contour, True)
        compactness = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

        if compactness < 0.75 or compactness > 0.95:  # Üçgenleri dışlamak için daha sıkı sınır
            return False


        if len(contours) != 1:  # Kesinlikle tek contour olmalı
            return False

        if bw > 0 and bh > 0:
            aspect_ratio = max(bw, bh) / min(bw, bh)
            if aspect_ratio > 1.3:  # Çok daha sıkı aspect ratio - üçgenler genellikle daha farklı oranlara sahip
                return False

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        corners = len(approx)

        if corners != 4:
            return False

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            angles = []
            for i in range(4):
                pt1 = pts[i]
                pt2 = pts[(i + 1) % 4]
                pt3 = pts[(i + 2) % 4]

                v1 = pt1 - pt2
                v2 = pt3 - pt2

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(np.degrees(angle))

            for angle in angles:
                if not (80 <= angle <= 100):  # 80-100 derece arası kabul et
                    return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if label == "blue_square":
        hsv_mean = np.mean(hsv, axis=(0, 1))

        base_hue_min, base_hue_max = 85, 140
        base_sat_min, base_val_min = 50, 30

        brightness = hsv_mean[2]  # V kanalı ortalaması
        if brightness < 60:  # Karanlık ortam
            val_min = max(15, base_val_min - 15)
            sat_min = max(30, base_sat_min - 20)
        elif brightness > 180:  # Parlak ortam
            val_min = min(80, base_val_min + 30)
            sat_min = min(70, base_sat_min + 20)
        else:  # Normal aydınlatma
            val_min = base_val_min
            sat_min = base_sat_min

        hue_center = hsv_mean[0]
        if hue_center < 100:  # Daha yeşil-mavi
            hue_min = max(75, base_hue_min - 10)
            hue_max = min(150, base_hue_max + 10)
        elif hue_center > 120:  # Daha mor-mavi
            hue_min = max(80, base_hue_min - 5)
            hue_max = min(155, base_hue_max + 15)
        else:  # Standart mavi
            hue_min, hue_max = base_hue_min, base_hue_max

        lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
        upper = np.array([hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Gaussian blur ile gürültü azaltma
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Delikleri doldurma
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Gürültü temizleme
    else:  # red_square
        lower1 = np.array([0, 80, 50], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 80, 50], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Gaussian blur ile gürültü azaltma
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Delikleri doldurma
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Gürültü temizleme
    color_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    ratio = (color_pixels / max(1, total_pixels))

    if label in ["blue_square", "red_square"]:
        debug_info = f"\n[{label.upper()} Debug] Konum:({x1},{y1},{x2},{y2}) Boyut:{bw:.0f}x{bh:.0f}"

        if label == "blue_square":
            debug_info += f"\n  HSV Mean: H={hsv_mean[0]:.1f}, S={hsv_mean[1]:.1f}, V={hsv_mean[2]:.1f}"
            debug_info += f"\n  Dinamik Eşik: [{hue_min},{sat_min},{val_min}] - [{hue_max},255,255]"
        else:  # red_square
            debug_info += f"\n  HSV Eşik: [0-10,160-179,80-255]"

        debug_info += f"\n  Renk Pikselleri: {color_pixels}/{total_pixels} = {ratio:.3f}"
        debug_info += f"\n  Gerekli Oran: {MIN_COLOR_RATIO.get(label, 0.25):.3f}"
        debug_info += f"\n  Geçti mi: {'✅' if ratio >= MIN_COLOR_RATIO.get(label, 0.25) else '❌'}"

        if 'contours' in locals() and contours:
            debug_info += f"\n  Contour Sayısı: {len(contours)}"
            debug_info += f"\n  Solidity: {solidity:.3f}, Compactness: {compactness:.3f}"
            debug_info += f"\n  Contour Alan: {contour_area:.1f}"
            if label == "red_square":
                debug_info += f"\n  Aspect Ratio: {max(bw, bh) / min(bw, bh):.2f}"
                if 'approx' in locals():
                    debug_info += f"\n  Köşe Sayısı: {len(approx)}"
                    if len(approx) == 4 and 'angles' in locals():
                        debug_info += f"\n  Köşe Açıları: {[f'{a:.1f}°' for a in angles]}"

        write_debug_log(debug_info)

    return ratio >= MIN_COLOR_RATIO.get(label, 0.25)

def _label_name(cls_id):
    return model.names[cls_id] if isinstance(model.names, dict) else model.names[cls_id]

def yolo_inference_thread():
    print("Inference thread başladı")
    while True:
        try:
            frame_in = infer_queue.get()
            predict_kwargs = {"imgsz": INFER_IMGSZ, "verbose": False, "conf": PRED_CONF_THRESHOLD}
            if TARGET_CLASS_IDS:
                predict_kwargs["classes"] = TARGET_CLASS_IDS
            res = model(frame_in, **predict_kwargs)[0]
            detected = []
            for b in res.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = _label_name(cls_id)
                area = max(0, (x2 - x1) * (y2 - y1))
                min_conf = CLASS_MIN_CONF.get(label, 0.6)
                min_area = CLASS_MIN_AREA.get(label, 300)
                if conf < min_conf or area < min_area:
                    continue
                if not target_box_ok(frame_in, (x1, y1, x2, y2), label, write_debug_log):
                    continue
                detected.append({"cls": cls_id, "conf": conf, "xyxy": (x1, y1, x2, y2), "frame_ts": time.time()})
            with infer_lock:
                latest_detections["boxes"] = detected
                latest_detections["timestamp"] = time.time()
        except Exception as e:
            with infer_lock:
                latest_detections["boxes"] = []
                latest_detections["timestamp"] = time.time()

def pixhawk_thread():
    """Pixhawk ile iletişimi yöneten ana thread"""
    print("🔗 Pixhawk thread başladı")
    global master
    connection_retry_count = 0
    last_heartbeat_time = time.time()
    
    def check_connection():
        """Bağlantı durumunu kontrol et ve gerekirse yeniden bağlan"""
        nonlocal connection_retry_count, last_heartbeat_time
        global master
        
        try:
            if master is None:
                print("⚠️ MAVLink bağlantısı yok, yeniden bağlanmaya çalışılıyor...")
                master = pixhawk_connect(SERIAL_PORT, BAUD_RATE)
                if master is None:
                    return False
            
            msg = read_mavlink_data('HEARTBEAT', blocking=False)
            if msg:
                last_heartbeat_time = time.time()
                connection_retry_count = 0
                return True
            
            if time.time() - last_heartbeat_time > HEARTBEAT_TIMEOUT:
                connection_retry_count += 1
                print(f"\n⚠️ Pixhawk bağlantısı koptu! ({connection_retry_count}/{MAX_RETRIES})")
                
                if connection_retry_count > MAX_RETRIES:
                    print("❌ Maksimum yeniden bağlanma denemesi aşıldı!")
                    print("Program sonlandırılıyor...")
                    raise SystemExit("Bağlantı kurulamadı")
                
                try:
                    print("Bağlantı yenileniyor...")
                    if master:
                        try:
                            master.close()
                        except:
                            pass
                        master = None
                    
                    time.sleep(2)  # Biraz daha uzun bekle
                    
                    master = pixhawk_connect(SERIAL_PORT, BAUD_RATE)
                    if master is None:
                        return False
                    
                    start_wait = time.time()
                    while time.time() - start_wait < 5.0:
                        msg = read_mavlink_data('HEARTBEAT', blocking=True, timeout=1)
                        if msg:
                            print("✅ Bağlantı başarıyla yenilendi!")
                            last_heartbeat_time = time.time()
                            return True
                    
                    print("❌ Heartbeat alınamadı!")
                    master = None
                    return False
                        
                except Exception as e:
                    print(f"⚠️ Yeniden bağlanma hatası: {e}")
                    master = None
                    time.sleep(1)
                    return False
            
            return True  # Timeout olmadıysa bağlantı hala aktif
            
        except Exception as e:
            print(f"⚠️ Bağlantı kontrol hatası: {e}")
            time.sleep(0.1)
            return False
    
    try:
        while True:
            if not check_connection():
                print("⚠️ Pixhawk bağlantısı yok, bekleniyor...")
                time.sleep(1)
                continue  # Bağlantı yoksa diğer işlemleri atla
            if not rates_configured:
                configure_mavlink_rates()
            
            if not mission_queue.empty():
                wps = mission_queue.get()
                mission_upload_event.clear()
                total = len(wps)
                print(f"\n🚀 Görev yükleme başlıyor: toplam {total} öğe")
                try:
                    master.mav.mission_clear_all_send(master.target_system, master.target_component)
                    time.sleep(0.2)
                except Exception:
                    pass

                master.mav.mission_count_send(master.target_system, master.target_component, total)

                sent = set()
                start_time = time.time()
                while len(sent) < total and (time.time() - start_time) < 20:
                    req = master.recv_match(type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'], blocking=True, timeout=2)
                    if not req:
                        continue
                    seq = int(getattr(req, 'seq', 0))
                    if seq < 0 or seq >= total:
                        continue
                    if seq in sent:
                        continue
                    entry = wps[seq]
                    if len(entry) >= 8:
                        lat, lon, alt, cmd, p1, p2, p3, p4 = entry[:8]
                    else:
                        lat, lon, alt, cmd = entry[:4]
                        p1 = p2 = p3 = p4 = 0.0
                    current_flag = 1 if seq == 0 else 0
                    master.mav.mission_item_int_send(
                        master.target_system,
                        master.target_component,
                        seq,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        cmd,
                        current_flag, 1, p1, p2, p3, p4,
                        int(lat * 1e7),
                        int(lon * 1e7),
                        alt
                    )
                    sent.add(seq)
                    print(f"WP#{seq}: {lat:.7f}, {lon:.7f}, {alt}m, cmd={cmd}")

                ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
                if ack:
                    print('✅ Pixhawk görev yükleme tamamlandı.')
                    try:
                        master.mav.mission_set_current_send(master.target_system, master.target_component, 0)
                    except Exception:
                        pass
                    mission_upload_event.set()
                else:
                    print('⚠️ Pixhawk görev yükleme ACK alınamadı!')
            if not mission_read_queue.empty():
                _ = mission_read_queue.get()
                master.mav.mission_request_list_send(master.target_system, master.target_component)
                mission_items = []
                mission_count = 0
                while True:
                    msg = master.recv_match(type=['MISSION_COUNT', 'MISSION_ITEM_INT'], blocking=True)
                    if msg and msg.get_type() == 'MISSION_COUNT':
                        mission_count = msg.count
                        break
                for i in range(mission_count):
                    master.mav.mission_request_int_send(master.target_system, master.target_component, i)
                    item = master.recv_match(type='MISSION_ITEM_INT', blocking=True)
                    mission_items.append(item)
                mission_read_result_queue.put(mission_items)
            gps_msg = read_mavlink_data('GPS_RAW_INT', blocking=False)
            if gps_msg:
                fix_type = getattr(gps_msg, 'fix_type', 0)
                satellites = getattr(gps_msg, 'satellites_visible', 0)
                hdop = getattr(gps_msg, 'eph', 0) / 100.0 if hasattr(gps_msg, 'eph') else 0
                
                gps_status = "❌ Yok"
                if fix_type == 1:
                    gps_status = "⚠️ Sadece GPS"
                elif fix_type == 2:
                    gps_status = "✅ 2D Fix"
                elif fix_type == 3:
                    gps_status = "✅ 3D Fix"
                elif fix_type == 4:
                    gps_status = "✅ DGPS"
                elif fix_type == 5:
                    gps_status = "✅ RTK Float"
                elif fix_type == 6:
                    gps_status = "✅ RTK Fixed"
                
                print(f"\nGPS Durum:")
                print(f"  Fix: {gps_status}")
                print(f"  Uydu: {satellites}")
                print(f"  HDOP: {hdop:.1f}")
            
            pos_msg = read_mavlink_data('GLOBAL_POSITION_INT', blocking=False)
            if pos_msg:
                with data_lock:
                    new_lat = pos_msg.lat / 1e7
                    new_lon = pos_msg.lon / 1e7
                    
                    old_lat = shared_data.get('lat')
                    old_lon = shared_data.get('lon')
                    
                    if old_lat is not None and old_lon is not None:
                        dist = distance_2d_util(old_lat, old_lon, new_lat, new_lon)
                        if dist is not None and dist > 0.1:  # 10cm'den fazla değişim varsa
                            print(f"\nKonum değişimi: {dist:.1f}m")
                            print(f"  Önceki: {old_lat:.7f}, {old_lon:.7f}")
                            print(f"  Yeni   : {new_lat:.7f}, {new_lon:.7f}")
                    
                    shared_data['lat'] = new_lat
                    shared_data['lon'] = new_lon
                    shared_data['gps_ts'] = time.time()
                    
                    alt = abs(pos_msg.relative_alt) / 1000.0  # Mutlak değer al
                    
                    vfr = read_mavlink_data('VFR_HUD', blocking=False)
                    if vfr and hasattr(vfr, 'alt'):
                        try:
                            vfr_alt = abs(float(vfr.alt))
                            if vfr_alt < 100.0:  # Makul bir değer ise
                                alt = vfr_alt
                                if abs(pos_msg.relative_alt/1000.0 - vfr_alt) > 5.0:
                                    print(f"ℹ️ VFR irtifa kullanılıyor: {vfr_alt:.1f}m")
                        except (ValueError, TypeError):
                            pass  # VFR okunamazsa GPS irtifasını kullan
                    
                    shared_data['alt'] = alt
                    print(f"İrtifa: {alt:.1f}m")
            
            att = read_mavlink_data('ATTITUDE', blocking=False)
            if att:
                with data_lock:
                    shared_data['roll'] = att.roll
                    shared_data['pitch'] = att.pitch
            
            vfr = read_mavlink_data('VFR_HUD', blocking=False)
            if vfr:
                with data_lock:
                    try:
                        if hasattr(vfr, 'groundspeed'):
                            shared_data['groundspeed'] = float(vfr.groundspeed)
                        
                        if hasattr(vfr, 'alt'):
                            shared_data['vfr_alt'] = float(vfr.alt)
                            print(f"VFR İrtifa: {shared_data['vfr_alt']:.1f}m")
                        
                        if hasattr(vfr, 'heading'):
                            heading = float(vfr.heading)
                            heading = (heading + 360.0) % 360.0
                            shared_data['heading'] = heading
                            print(f"Heading güncellendi: {heading:.1f}°")
                    except (ValueError, TypeError) as e:
                        print(f"VFR_HUD veri dönüşüm hatası: {e}")
                        pass
            mc = master.recv_match(type='MISSION_CURRENT', blocking=False)
            if mc:
                with data_lock:
                    current_seq = int(mc.seq)
                    if current_seq != shared_data.get('mission_current'):
                        print(f"\nAktif waypoint değişti: WP{current_seq}")
                        if red_mission_index is not None and blue_mission_index is not None:
                            print(f"Hedef WP'ler: Kırmızı={red_mission_index}, Mavi={blue_mission_index}")
                            if current_seq < min(red_mission_index, blue_mission_index):
                                print("⚠️ Aktif WP hedeflerden önce!")
                            elif current_seq > max(red_mission_index, blue_mission_index):
                                print("⚠️ Hedefler geçildi!")
                    shared_data['mission_current'] = current_seq
            time.sleep(0.05)
    except Exception as e:
        print("Pixhawk thread hatası:", e)

threading.Thread(target=pixhawk_thread, daemon=True).start()
threading.Thread(target=yolo_inference_thread, daemon=True).start()
threading.Thread(target=heading_update_thread, daemon=True).start()
threading.Thread(target=servo_control_thread, daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        recent_frames.append(frame.copy())
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2
        if video_writer is None:
            try:
                video_writer = cv2.VideoWriter(video_path, VIDEO_FOURCC, VIDEO_FPS, (width, height))
            except Exception:
                video_writer = None
        frame_index += 1
        fps_counter += 1
        if time.time() - fps_timer >= 20.0:
            print(f"Gerçek FPS: {fps_counter / (time.time() - fps_timer):.2f}")
            fps_counter = 0
            fps_timer = time.time()
        if frame_index % DETECT_EVERY_N == 0 and infer_queue.empty():
            try:
                frame_to_process = recent_frames[-1] if len(recent_frames) > 0 else frame.copy()
                infer_queue.put_nowait(frame_to_process)
                print("\nYOLO tespiti için frame gönderildi")
            except Exception as e:
                print(f"Frame gönderme hatası: {e}")
                pass

        with infer_lock:
            detections_snapshot = list(latest_detections["boxes"]) if latest_detections["boxes"] else []
            if detections_snapshot:
                print(f"\nTespit edilen nesneler: {len(detections_snapshot)}")
                for det in detections_snapshot:
                    cls_id = det["cls"]
                    label = _label_name(cls_id)
                    conf = det.get("conf", 0.0)
                    x1, y1, x2, y2 = det["xyxy"]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    print(f"  {label}: merkez=({cx}, {cy}), güven={conf:.2f}")

        detections_to_process = detections_snapshot
        if CENTER_PRIORITY_ENABLED and detections_snapshot:
            def _center_distance_px(det):
                x1, y1, x2, y2 = det["xyxy"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                dist = math.hypot(cx - center_x, cy - center_y)
                print(f"  Merkeze uzaklık: {dist:.1f}px")
                return dist
                
            if PER_CLASS_CENTER_PICK:
                by_class = {}
                for d in detections_snapshot:
                    label = _label_name(d["cls"])
                    dist = _center_distance_px(d)
                    if label not in by_class or dist < _center_distance_px(by_class[label]):
                        by_class[label] = d
                        print(f"  {label} için en yakın aday seçildi (uzaklık: {dist:.1f}px)")
                detections_to_process = list(by_class.values())
            else:
                best = min(detections_snapshot, key=_center_distance_px)
                print(f"  En yakın aday seçildi: {_label_name(best['cls'])}")
                detections_to_process = [best]

        if DRAW_DETECTIONS and detections_to_process:
            for det in detections_to_process:
                x1, y1, x2, y2 = det["xyxy"]
                cls_id = det["cls"]
                conf = det.get("conf", 0.0)
                label = _label_name(cls_id)
                color = (255, 0, 0) if label == "blue_square" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(10, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if 'streak' not in globals():
            streak = {"blue_square": 0, "red_square": 0}
        if 'last_center' not in globals():
            last_center = {"blue_square": None, "red_square": None}

        for det in detections_to_process:
            cls_id = det["cls"]
            label = _label_name(cls_id)
            if label not in TARGET_CLASS_NAMES:
                continue
            x1, y1, x2, y2 = det["xyxy"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            undist_cx, undist_cy = undistort_point_util(cx, cy, mtx, dist)

            with data_lock:
                lat = shared_data['lat']
                lon = shared_data['lon']
                alt = shared_data['alt']
                roll = shared_data['roll']
                pitch = shared_data['pitch']
                heading_for_geo = shared_data.get('heading')
            if lat is None or lon is None or alt is None:
                print(f"⚠️ GPS verisi yok ({label} hedefi için), varsayılan koordinatlar kullanılacak")
                lat, lon, alt = 37.0287, 37.3117, 10.0
                roll, pitch = 0.0, 0.0
                heading_for_geo = 0.0

            streak[label] = streak.get(label, 0) + 1
            last_center[label] = (undist_cx, undist_cy)

            required_streak = CLASS_REQUIRED_STREAK.get(label, 3)
            if streak[label] >= required_streak and label not in [t[0] for t in detected_targets]:
                if heading_for_geo is not None:
                    target_lat, target_lon = compute_target_gps_with_attitude_and_heading_util(
                        undist_cx, undist_cy, center_x, center_y, lat, lon, alt, roll, pitch, heading_for_geo, mtx,
                        CAMERA_H_FOV_DEG, CAMERA_WIDTH, CAMERA_HEIGHT)
                else:
                    target_lat, target_lon = compute_target_gps_with_attitude_util(
                        undist_cx, undist_cy, center_x, center_y, lat, lon, alt, roll, pitch,
                        CAMERA_H_FOV_DEG, CAMERA_WIDTH, CAMERA_HEIGHT)
                print(f"\n{label.capitalize()} hedef tespit edildi:")
                print(f"  GPS: {target_lat:.7f}, {target_lon:.7f}")
                print(f"  Piksel koordinatları: ({undist_cx}, {undist_cy})")
                print(f"  İHA konumu: {lat:.7f}, {lon:.7f}, {alt:.1f}m")
                print(f"  Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°")
                print(f"  Heading: {heading_for_geo if heading_for_geo is not None else 'Bilinmiyor'}°")
                detected_targets.append((label, target_lat, target_lon))
                
                with servo_lock:
                    if label == "red_square":
                        servo_data['red_target']['lat'] = target_lat
                        servo_data['red_target']['lon'] = target_lon
                    elif label == "blue_square":
                        servo_data['blue_target']['lat'] = target_lat
                        servo_data['blue_target']['lon'] = target_lon

                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                best_img = None
                best_score = -1.0
                x1, y1, x2, y2 = det["xyxy"]
                for cand in list(recent_frames)[-3:][::-1]:  # en yeni → daha eski
                    ok = target_box_ok(cand, (x1, y1, x2, y2), label, write_debug_log)
                    score = 1.0 if ok else 0.0
                    if score > best_score:
                        best_score = score
                        best_img = cand
                if best_img is None:
                    best_img = frame
                filename = f"{SAVE_DIR}/{label}_{timestamp}.jpg"
                cv2.imwrite(filename, best_img)
                print(f"{label} için fotoğraf kaydedildi: {filename}")

        for name in list(TARGET_CLASS_NAMES):
            if not any(_label_name(d["cls"]) == name for d in detections_to_process):
                streak[name] = 0
        for idx, (label, lat, lon) in enumerate(detected_targets):
            if label == "red_square":
                red_wp_seq = idx
                red_target_lat, red_target_lon = lat, lon
            if label == "blue_square":
                blue_wp_seq = idx
                blue_target_lat, blue_target_lon = lat, lon
        with data_lock:
            current_lat = shared_data['lat']
            current_lon = shared_data['lon']
            altitude = shared_data['alt']
        with data_lock:
            velocity = shared_data.get('groundspeed') if isinstance(shared_data, dict) else None
            heading_deg = shared_data.get('heading') if isinstance(shared_data, dict) else None
            mission_current = shared_data.get('mission_current') if isinstance(shared_data, dict) else None
        if velocity is None:
            velocity = 10.0
        def _bearing_deg(lat1, lon1, lat2, lon2):
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dlambda = math.radians(lon2 - lon1)
            y = math.sin(dlambda) * math.cos(phi2)
            x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
            brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
            return brng

        def _is_heading_towards(current_heading_deg, desired_bearing_deg, tolerance_deg):
            if current_heading_deg is None:
                return True
            diff = abs((current_heading_deg - desired_bearing_deg + 540.0) % 360.0 - 180.0)
            return diff <= tolerance_deg

        def _meters_east_north(lat1, lon1, lat2, lon2):
            d_north = (lat2 - lat1) * 111_000.0
            d_east = (lon2 - lon1) * 111_000.0 * math.cos(math.radians(lat1))
            return d_east, d_north

        def _validate_mission_sequence(current_wp_index, red_wp_index, blue_wp_index):
            """Görev sırasını kontrol et - yanlış hedef için bırakma yapılmasını engelle"""
            print(f"\nWaypoint Kontrol:")
            print(f"  Mevcut WP: {current_wp_index}")
            print(f"  Kırmızı WP: {red_wp_index}")
            print(f"  Mavi WP: {blue_wp_index}")
            
            if current_wp_index is None:
                return False, "Mevcut WP bilinmiyor"
            
            if red_wp_index is None and blue_wp_index is None:
                return False, "Hedefler henüz yüklenmedi"
            
            if red_wp_index is not None:
                if abs(current_wp_index - red_wp_index) <= 1:  # Hedef WP'ye yakınız
                    return True, "Kırmızı hedef için uygun"
                elif current_wp_index > red_wp_index + 1:  # Hedefi geçtik
                    return False, "Kırmızı hedef geçildi"
            
            if blue_wp_index is not None:
                if abs(current_wp_index - blue_wp_index) <= 1:  # Hedef WP'ye yakınız
                    return True, "Mavi hedef için uygun"
                elif current_wp_index > blue_wp_index + 1:  # Hedefi geçtik
                    return False, "Mavi hedef geçildi"
            
            next_wp = min(wp for wp in [red_wp_index, blue_wp_index] if wp is not None)
            if current_wp_index < next_wp:
                return False, f"Hedef WP'ye henüz ulaşılmadı (Mevcut: {current_wp_index}, Hedef: {next_wp})"
            
            return False, "WP sırası uygun değil"

        def _release_ok(curr_lat, curr_lon, target_lat, target_lon, altitude_m, velocity_ms, heading_deg_val, current_wp, target_wp):
            dist_to_target = distance_2d_util(curr_lat, curr_lon, target_lat, target_lon)
            
            distance_ok = MIN_RELEASE_DISTANCE_M <= dist_to_target <= MAX_RELEASE_DISTANCE_M
                
            print(f"  Mesafe durumu: {dist_to_target:.1f}m (Limit: {MIN_RELEASE_DISTANCE_M}-{MAX_RELEASE_DISTANCE_M}m)")
            
            return distance_ok, dist_to_target, {
                "distance_ok": distance_ok,
                "dist": dist_to_target,
                "alt": altitude_m
            }

        if len(detected_targets) == 2 and not sabitler_uploaded:
            print("\n=== EKLENECEK YENİ GÖREV LİSTESİ OLUŞTURULUYOR ===")
            new_wps = []
            base_count = 0
            
            print("\n=== TESPİT EDİLEN HEDEFLER ===")
            if len(detected_targets) < 2:
                print("⚠️ Henüz yeterli hedef tespit edilmedi!")
                print(f"  Tespit edilen: {len(detected_targets)}/2")
                continue
                
            first_label, first_lat, first_lon = detected_targets[0]
            print(f"\n1. HEDEF:")
            print(f"  Tip: {first_label}")
            print(f"  Konum: {first_lat:.7f}, {first_lon:.7f}")
            print("  Yükseklik: 20.0m")
            new_wps.append((first_lat, first_lon, 20.0, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 0.0, 0.0, 0.0))
            
            print(f"\nARA DÖNÜŞ NOKTASI:")
            print(f"  Konum: {MID_TURN_WP[0]:.7f}, {MID_TURN_WP[1]:.7f}")
            print(f"  Yükseklik: {MID_TURN_WP[2]}m")
            new_wps.append((MID_TURN_WP[0], MID_TURN_WP[1], MID_TURN_WP[2], mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 0.0, 0.0, 0.0))
            
            second_label, second_lat, second_lon = detected_targets[1]
            print(f"\n2. HEDEF:")
            print(f"  Tip: {second_label}")
            print(f"  Konum: {second_lat:.7f}, {second_lon:.7f}")
            print("  Yükseklik: 20.0m")
            new_wps.append((second_lat, second_lon, 20.0, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 0.0, 0.0, 0.0))
            
            print("\n=== GÖREV İNDEKSLERİ AYARLANIYOR ===")
            try:
                if first_label == "red_square":
                    red_mission_index = base_count
                    red_rel_wp_index = 0
                    print(f"✅ Kırmızı hedef -> WP{red_mission_index}")
                    print(f"   Konum: {first_lat:.7f}, {first_lon:.7f}")
                else:
                    blue_mission_index = base_count
                    print(f"✅ Mavi hedef -> WP{blue_mission_index}")
                    print(f"   Konum: {first_lat:.7f}, {first_lon:.7f}")
                
                if second_label == "red_square":
                    red_mission_index = base_count + 2
                    red_rel_wp_index = base_count + 2
                    print(f"✅ Kırmızı hedef -> WP{red_mission_index}")
                    print(f"   Konum: {second_lat:.7f}, {second_lon:.7f}")
                else:
                    blue_mission_index = base_count + 2
                    blue_rel_wp_index = 2
                    print(f"✅ Mavi hedef -> WP{blue_mission_index}")
                    print(f"   Konum: {second_lat:.7f}, {second_lon:.7f}")
                
                first_target = min(wp for wp in [red_mission_index, blue_mission_index] if wp is not None)
                print(f"\nİlk hedef: WP{first_target}")
                print(f"Görev sırası: Kırmızı=WP{red_mission_index}, Mavi=WP{blue_mission_index}")
            except Exception as e:
                print(f"⚠️ Hedef waypoint ayarlama hatası: {e}")
                print(f"⚠️ Hata detayı:")
                print(f"  {str(e)}")
                mission_upload_event.set()
                continue
            
            print("\nManuel waypoint'ler ekleniyor:")
            for idx, (lat, lon, alt) in enumerate(MANUAL_WPS):
                print(f"Manuel WP{idx + 1}: {lat:.7f}, {lon:.7f}, {alt}m")
                new_wps.append((lat, lon, alt, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0.0, 0.0, 0.0, 0.0))
            
            print(f"\nİniş noktası: {MANUAL_LAND[0]:.7f}, {MANUAL_LAND[1]:.7f}, {MANUAL_LAND[2]}m")
            new_wps.append((MANUAL_LAND[0], MANUAL_LAND[1], MANUAL_LAND[2], mavutil.mavlink.MAV_CMD_NAV_LAND, 0.0, 0.0, 0.0, 0.0))
            
            print("\nToplam görev listesi:")
            for i, wp in enumerate(new_wps):
                lat, lon, alt = wp[:3]
                cmd = wp[3]
                cmd_name = "LAND" if cmd == mavutil.mavlink.MAV_CMD_NAV_LAND else "WP"
                print(f"WP{i}: {cmd_name} - {lat:.7f}, {lon:.7f}, {alt}m")
            
            print("\nGörev Pixhawk'a yükleniyor (append)...")
            try:
                while not mission_queue.empty():
                    try:
                        mission_queue.get_nowait()
                        print("⚠️ Eski görev isteği temizlendi")
                    except:
                        break
            except:
                pass
            mission_queue.put(new_wps, timeout=5)
            waited = 0
            while not mission_upload_event.is_set() and waited < 10:
                time.sleep(0.5)
                waited += 0.5
            if mission_upload_event.is_set():
                print("✅ Görev yükleme sinyali alındı")
                sabitler_uploaded = True
            else:
                print("⚠️ Görev yükleme onay sinyali alınamadı, tekrar denenecek")
        try:
            if video_writer is not None:
                video_writer.write(frame)
        except Exception:
            pass
        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            release_servo(SERVO1_PIN)
            print("[MANUAL] Kırmızı yük manuel olarak serbest bırakıldı (bayraklar değişmedi)")
        elif key == ord('b'):
            release_servo(SERVO2_PIN)
            print("[MANUAL] Mavi yük manuel olarak serbest bırakıldı (bayraklar değişmedi)")
finally:
    cap.release()
    try:
        if video_writer is not None:
            video_writer.release()
    except Exception:
        pass
    try:
        if 'servo_controller' in globals():
            servo_controller.cleanup()
        else:
            GPIO.cleanup()
    except Exception as e:
        print(f"GPIO temizleme hatası: {e}")
        pass
    cv2.destroyAllWindows()
