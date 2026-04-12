# Demo Media Plan

## Goal

Prepare clear media that helps visitors understand the project quickly without reading the full codebase first.

## Recommended Demo Assets

### 1. Main Screenshot

Capture one clean runtime frame that shows:

- live camera image
- detected target bounding box
- confidence label
- a stable and readable scene

Recommended filename:

`docs/media/main-detection-frame.png`

### 2. Field Setup Photo

Take one photo showing the physical setup:

- aircraft or test bench
- Raspberry Pi
- camera placement
- Pixhawk or wiring area if visible

Recommended filename:

`docs/media/hardware-setup.jpg`

### 3. Short Demo Video

Record a 20 to 45 second demo that shows:

1. camera stream running
2. target detection appearing
3. target coordinates or mission behavior updating
4. servo trigger or release simulation

Recommended filename:

`docs/media/demo.mp4`

### 4. Optional GIF Preview

Export a short 5 to 10 second loop from the main video for the README.

Recommended filename:

`docs/media/demo-preview.gif`

## Suggested Video Flow

### Intro

- show the aircraft or bench setup for 2 to 3 seconds

### Detection

- show the runtime screen with red and blue target recognition

### Mission Logic

- show telemetry, target location estimate, or mission update logs

### Release

- show servo actuation or release test result

## Capture Checklist

- remove unnecessary desktop clutter
- use readable terminal or overlay text size
- keep lighting stable
- avoid shaky handheld footage if possible
- record at 720p or 1080p
- trim dead time before publishing

## README Integration Plan

After media is ready:

1. add one main screenshot near the top of the README
2. add one short "Demo" section with a GIF or video link
3. add one "Hardware Setup" image if the physical system is a key strength
