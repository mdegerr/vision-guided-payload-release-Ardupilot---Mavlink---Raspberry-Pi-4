# Vision-Guided Autonomous Payload Release for UAVs

A Raspberry Pi and Pixhawk based UAV mission system for visual target detection, target geolocation, dynamic mission updates, and autonomous payload release.

## Overview

This project detects `red_square` and `blue_square` ground targets from a live camera feed, estimates their GPS coordinates using camera calibration and aircraft attitude data, updates the mission flow through MAVLink, and triggers the correct servo when the aircraft enters the release window.

It was built for a fixed-wing UAV workflow where lightweight onboard vision has to work together with autopilot telemetry and real mission timing.

## Key Features

- Real-time target detection with YOLO
- Additional color and shape validation to reduce false positives
- GPS estimation from image coordinates and aircraft attitude
- MAVLink-based Pixhawk communication
- Automatic mission update after target discovery
- Servo-based payload release logic for multiple targets
- Camera test and calibration utilities for field setup

## Repository Structure

```text
.
|- vision_guided_payload_release.py  # Main runtime entry point
|- runtime_config.py                 # Runtime constants and shared configuration
|- geo_utils.py                      # Distance and target geolocation helpers
|- vision_utils.py                   # Vision verification and undistortion helpers
|- pixhawk_connection.py             # Pixhawk connection bootstrap
|- servo_controller.py               # Servo actuation and cleanup logic
|- logging_utils.py                  # Debug log helpers
|- camera_calibration.py             # Chessboard-based camera calibration
|- camera_test.py                    # Camera and frame-drop test script
|- calib_images/                     # Calibration sample images
|- docs/system-overview.md           # Technical architecture summary
|- docs/github-metadata.md           # Suggested GitHub description and topics
|- docs/demo-plan.md                 # Demo screenshot and video production plan
```

## Hardware Requirements

- Raspberry Pi
- USB or CSI camera
- Pixhawk or another MAVLink-compatible flight controller
- Servo outputs for payload release

## Software Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLO
- pymavlink
- pyserial
- `RPi.GPIO` for Raspberry Pi deployment

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Test the camera

```bash
python camera_test.py
```

### 2. Generate or refresh the calibration file

```bash
python camera_calibration.py
```

### 3. Run the main mission script

```bash
python vision_guided_payload_release.py
```

## Runtime Notes

- The script looks for `best.pt` first and falls back to `best2.pt` if needed.
- `calib.npz` is device-specific and should be regenerated for the actual camera used in flight.
- Output files are written to `outputs/` by default.
- You can override the output directory with:

```bash
set VIENTO_SAVE_DIR=C:\path\to\output
```

## Output Artifacts

The runtime can produce:

- session video files
- detected target photos
- debug logs
- updated mission waypoints through MAVLink

## Documentation

- Architecture summary: [docs/system-overview.md](./docs/system-overview.md)
- GitHub description and topics: [docs/github-metadata.md](./docs/github-metadata.md)
- Demo media planning: [docs/demo-plan.md](./docs/demo-plan.md)

## Current Limitations

- Some runtime parameters are still hardcoded.
- The main runtime file is still larger than ideal even after modular cleanup.
- Full functionality depends on real Raspberry Pi, camera, and Pixhawk hardware.

## Roadmap

- Split the main runtime into smaller mission services
- Add a dry-run or simulation mode
- Improve structured logging
- Add automated checks for syntax and packaging
- Document model training and deployment workflow

## License

This project is distributed under the MIT License. See [LICENSE](./LICENSE).
