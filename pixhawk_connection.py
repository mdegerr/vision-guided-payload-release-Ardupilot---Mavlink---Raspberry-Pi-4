import time

from pymavlink import mavutil


def connect_pixhawk(serial_port, baud_rate):
    import serial
    import serial.tools.list_ports

    def find_pixhawk_port():
        for port in serial.tools.list_ports.comports():
            hardware_id = port.hwid.lower()
            if any(identifier in hardware_id for identifier in ["2341", "26ac", "0483"]):
                return port.device
        return None

    def configure_serial_port(port_name):
        ser = serial.Serial()
        try:
            ser.port = port_name
            ser.baudrate = baud_rate
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
            time.sleep(0.3)
            ser.setDTR(True)
            ser.setRTS(True)
            ser.close()
            time.sleep(0.3)
            return True
        except Exception as exc:
            print(f"Serial setup error: {exc}")
            try:
                ser.close()
            except Exception:
                pass
            return False

    def wait_for_heartbeat(mavlink_conn, timeout=10):
        start_time = time.time()
        heartbeat_count = 0
        while time.time() - start_time < timeout:
            try:
                msg = mavlink_conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
                if msg:
                    heartbeat_count += 1
                    if heartbeat_count >= 3:
                        return True
            except Exception:
                continue
        return False

    port_to_use = find_pixhawk_port() or serial_port
    for attempt in range(3):
        try:
            if not configure_serial_port(port_to_use):
                print("Serial port setup skipped, continuing with direct MAVLink connect.")
            mavlink_conn = mavutil.mavlink_connection(
                port_to_use,
                baud=baud_rate,
                autoreconnect=True,
                source_system=1,
                source_component=1,
                force_connected=True,
                write_timeout=1,
                timeout=1,
            )
            if hasattr(mavlink_conn.port, "setBufferSize"):
                mavlink_conn.port.setBufferSize(rx_size=4096, tx_size=4096)
            if hasattr(mavlink_conn.port, "setLatencyTimer"):
                mavlink_conn.port.setLatencyTimer(milliseconds=1)
            if wait_for_heartbeat(mavlink_conn):
                print(f"Connected to Pixhawk on {port_to_use} at {baud_rate} baud")
                time.sleep(2)
                return mavlink_conn
            try:
                mavlink_conn.close()
            except Exception:
                pass
        except Exception as exc:
            print(f"Pixhawk connection attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                time.sleep(2)

    raise SystemExit("Pixhawk connection could not be established")
