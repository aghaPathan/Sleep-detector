# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-03-11

### Added
- `SleepDetector` class encapsulating all detection logic (replaces global state)
- CLI arguments for threshold, duration, camera index, model path, and verbosity
- `download_model.py` script to fetch the required dlib facial landmark model
- Proper `requirements.txt` in pip-installable format with version bounds
- `pyproject.toml` for package configuration and `sleep-detector` console entry point
- MIT `LICENSE` file
- Unit tests in `tests/test_detect_sleep.py` covering EAR computation, argument parsing, and initialization
- Structured logging via Python's `logging` module with timestamps and levels
- Graceful shutdown handling (SIGINT, SIGTERM) to release camera resources
- Alert cooldown (3 seconds) to prevent alert spam on every frame
- Yellow eye contour color for "eyes closing but not yet alerting" state
- Live EAR value overlay on video frame
- Terminal bell (`\a`) for audible drowsiness alert
- Model file entries in `.gitignore`
- `EVALUATION.md` with full readiness assessment

### Changed
- Renamed `eye_aspect_ratio()` to `compute_ear()` (PEP 8 snake_case)
- Renamed `DlibVideoHandler()` / `dlibVideo()` / `dlibVideoAlert()` to class methods `run()` / `_process_frame()` / `_trigger_alert()`
- Rewrote `README.md` with correct filename, complete dependencies, accurate algorithm description, CLI docs, and project structure
- Camera open failure now logs an error and exits cleanly instead of silently looping
- Frame read failures now log a warning and retry instead of silently continuing

### Fixed
- `requirements.txt` was not in pip format — now uses standard `package>=version` syntax
- README referenced `sleep_detector.py` but actual file is `detect_sleep.py`
- README install command was missing `scipy` and `imutils` dependencies
- README incorrectly mentioned Haar Cascade; only dlib is used
- No error handling for missing model file — now raises `FileNotFoundError` with download instructions
- Alert fired on every frame after threshold — now respects 3-second cooldown
- Global mutable variables (`lock`, `start`, `threshold`) replaced with instance attributes

### Removed
- Commented-out FPS debug code
- Non-standard `packages:` / `additional files` format from `requirements.txt`

## [1.0.0] - 2024-01-01

### Added
- Initial drowsiness detection using EAR algorithm
- Real-time webcam processing with OpenCV
- dlib facial landmark detection
- Green/red eye contour visualization
- Basic console alert
