# AI Camera Demo — On-device Neural Vision Driving a Robot

This repository (`edgeai2mcu`) contains early-stage code for a self-designed AI camera board
running on-device neural network inference for real-time vision and direct robot control.

The current demo performs on-device person/hand detection and triggers a simple
GO / STOP control loop without using Jetson, a PC, or cloud services.

## Current Status
- Phase 1 demo
- Code is experimental and under active development
- Focus is on validating the end-to-end vision → control loop on real hardware

## Demo Overview
- Input: camera video stream
- Model: on-device neural network for object detection (TPU-based)
- Output: motor control (GO / STOP)

## Running the Demo (high level)
> Detailed setup instructions will be added later.

Typical steps:
1. Connect the AI camera board and motor controller
2. Start the vision pipeline (on AI-CAM board)
3. Run the control loop script (on ESP32S3 board)

## Notes
- This code is not optimized or cleaned up yet
- APIs and structure may change
- Intended for developers, students, and robotics experiments

## License
TBD
