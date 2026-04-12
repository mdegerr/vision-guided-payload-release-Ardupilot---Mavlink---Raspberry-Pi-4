# System Overview

## Purpose

This project performs vision-guided target detection and autonomous payload release for UAV missions built around Raspberry Pi and Pixhawk.

## Core Pipeline

1. Camera frames are captured in real time.
2. The YOLO model detects `red_square` and `blue_square` targets.
3. Additional color and shape checks reduce false positives.
4. Camera calibration and aircraft attitude data are used to estimate target GPS coordinates.
5. Detected targets are converted into mission waypoints.
6. When the aircraft reaches the release window, the corresponding servo is triggered.

## Main Components

### `vision_guided_payload_release.py`

The main runtime entry point. It coordinates:

- camera capture
- YOLO inference
- target verification
- mission flow
- telemetry-driven release logic

### `runtime_config.py`

Central location for runtime constants and shared configuration values.

### `geo_utils.py`

Distance calculations and target geolocation helpers.

### `vision_utils.py`

Lens correction and target validation helpers for color and shape filtering.

### `pixhawk_connection.py`

Connection bootstrap for Pixhawk over MAVLink.

### `servo_controller.py`

Servo actuation and GPIO cleanup helpers.

### `camera_calibration.py`

Creates `calib.npz` from chessboard images stored in `calib_images/`.

### `camera_test.py`

Checks camera availability, frame flow, and frame drop behavior before flight tests.

## Runtime Threads

The main script uses multiple threads:

- `pixhawk_thread`: reads MAVLink telemetry and mission state
- `yolo_inference_thread`: runs object detection asynchronously
- `heading_update_thread`: tracks heading information toward detected targets
- `servo_control_thread`: decides when to release payloads

## Inputs

- live camera feed
- Pixhawk telemetry over MAVLink
- trained YOLO model weights
- camera calibration file

## Outputs

- target photos
- session video
- debug logs
- updated mission waypoints
- servo release actions

## Current Limitations

- most configuration values are still hardcoded
- the main runtime file is still larger than ideal
- hardware-specific dependencies limit desktop-only testing

## Recommended Next Refactor

- break the remaining runtime flow into smaller services or classes
- add a simulation or dry-run mode
