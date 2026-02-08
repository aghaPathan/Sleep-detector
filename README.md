# Sleep Detector

A computer vision application that detects drowsiness by monitoring eye state using facial recognition.

## Overview

This script uses OpenCV to detect faces and track eye state (open/closed) in real-time video, designed to alert drowsy drivers.

## Features

- 👁️ **Eye State Detection** — Monitors if eyes are open or closed
- 😴 **Drowsiness Alert** — Detects when driver may be falling asleep
- 📹 **Real-time Processing** — Works with live webcam feed
- 🎯 **Face Tracking** — Locates and tracks face in frame

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- dlib (for facial landmarks)
- NumPy

## Installation

```bash
pip install opencv-python dlib numpy
```

## Usage

```bash
python sleep_detector.py
```

## How It Works

1. Captures video from webcam
2. Detects face using Haar Cascade / dlib
3. Identifies eye regions
4. Calculates eye aspect ratio (EAR)
5. Alerts if eyes closed for extended period

## Applications

- Driver drowsiness detection
- Workplace fatigue monitoring
- Safety systems

## License

MIT
