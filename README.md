# Sleep Detector

Real-time drowsiness detection using webcam video and dlib facial landmarks. Monitors eye state via the Eye Aspect Ratio (EAR) algorithm and alerts when eyes remain closed for a configurable duration.

## Features

- **Eye Aspect Ratio (EAR) detection** — Computes EAR from dlib's 68-point facial landmarks to determine if eyes are open or closed
- **Configurable thresholds** — EAR sensitivity, alert duration, and camera index are all CLI arguments
- **Visual feedback** — Green (awake), yellow (eyes closing), red (drowsy) eye contour overlays with live EAR display
- **Audible alert with cooldown** — Terminal bell fires on drowsiness detection with a 3-second cooldown to prevent spam
- **Graceful shutdown** — Handles SIGINT/SIGTERM to properly release camera resources

## Requirements

- Python 3.8+
- Webcam
- dlib shape predictor model (see [Setup](#setup))

## Installation

```bash
# Clone the repository
git clone https://github.com/aghaPathan/Sleep-detector.git
cd Sleep-detector

# Install dependencies
pip install -r requirements.txt

# Download the dlib facial landmark model (~99 MB)
python download_model.py
```

## Usage

```bash
# Run with defaults (threshold=0.2, alert after 5s, camera 0)
python detect_sleep.py

# Custom settings
python detect_sleep.py --threshold 0.25 --duration 3 --camera 1

# Verbose logging
python detect_sleep.py -v
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `shape_predictor_68_face_landmarks.dat` | Path to dlib model file |
| `--threshold` | `0.2` | EAR value below which eyes are considered closed |
| `--duration` | `5` | Seconds eyes must stay closed before alert |
| `--camera` | `0` | Camera device index |
| `-v, --verbose` | off | Enable debug logging |

Press **q** to quit the detector.

## How It Works

1. Captures video frames from the webcam
2. Detects faces using dlib's frontal face detector
3. Extracts 68 facial landmarks and isolates eye regions
4. Computes Eye Aspect Ratio (EAR) from vertical and horizontal eye distances
5. Tracks how long EAR stays below threshold
6. Triggers alert (audible bell + red contour) when eyes are closed beyond the configured duration

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
Sleep-detector/
├── detect_sleep.py       # Main application
├── download_model.py     # Model download helper
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package configuration
├── LICENSE               # MIT license
├── tests/
│   └── test_detect_sleep.py
└── .github/
    └── workflows/
        └── pr-checks.yml # CI pipeline
```

## Applications

- Driver drowsiness detection
- Workplace fatigue monitoring
- Safety-critical monitoring systems

## License

[MIT](LICENSE)

## CI Status

All PRs are checked for:
- Syntax (Python, JS, TS, YAML, JSON, Dockerfile, Shell)
- Secrets (no hardcoded credentials)
- Security (high-severity vulnerabilities)
