# Backend Tasks

## Goal

Build the backend around the approved model without changing the training part.

## Approved Model

- `models/approved/weapon_detector_best.pt`

## Main Tasks

1. Connect the IP camera stream to the system.
2. Load the approved model and run detection on incoming frames.
3. Create alerts when a weapon is detected.
4. Save event data and event images.
5. Build API endpoints for frontend use.
6. Generate a simple daily report.
