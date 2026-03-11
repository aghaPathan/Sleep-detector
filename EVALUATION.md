# Sleep-Detector Repository — Finished Product Evaluation

## Overall Verdict: NOT ready for production

**Score: 3/10**

---

## CRITICAL Issues (must fix)

| # | Issue | Detail |
|---|-------|--------|
| 1 | **Broken `requirements.txt`** | Uses a non-standard format. Not pip-installable. Lists `time` (stdlib) and `cv2` instead of `opencv-python`. Running `pip install -r requirements.txt` will fail. |
| 2 | **Missing model file** | App requires `shape_predictor_68_face_landmarks.dat` (~99 MB) which is neither included nor downloadable via any setup script. App crashes without it. |
| 3 | **README filename mismatch** | README says `python sleep_detector.py` but actual file is `detect_sleep.py`. |
| 4 | **README lists incomplete deps** | README install command omits `scipy` and `imutils`, both required by imports. |
| 5 | **No error handling** | No handling for missing model file, camera not found, no face detected, or invalid frames. Any of these cause unhandled crashes. |
| 6 | **No tests** | Zero test files. CI checks syntax but nothing validates correctness. |

## MAJOR Issues (should fix)

| # | Issue | Detail |
|---|-------|--------|
| 7 | **Global mutable state** | `lock`, `start`, `threshold` are globals mutated inside `dlibVideo()`. Untestable, non-reentrant, fragile. Should be a class or passed as params. |
| 8 | **Alert is a no-op** | `dlibVideoAlert()` only prints "Alert". No audible alarm, no visual overlay, no system notification. Dangerously insufficient for a drowsiness detector. |
| 9 | **No LICENSE file** | README claims MIT but no LICENSE file exists. Project is not legally licensed. |
| 10 | **No dependency pinning** | No version pins. Builds are not reproducible. |
| 11 | **Naming conventions** | `DlibVideoHandler` is PascalCase (implies class) but is a function. Inconsistent with PEP 8. |
| 12 | **Continuous alert spam** | Once eyes closed >5s, alert fires every frame with no cooldown. |

## MINOR Issues (nice to fix)

| # | Issue | Detail |
|---|-------|--------|
| 13 | **Hardcoded magic numbers** | Threshold `0.2` and duration `5s` buried in source. Should be CLI args or config. |
| 14 | **Commented-out debug code** | FPS tracking on lines 41/44 left commented out. |
| 15 | **No graceful shutdown** | No signal handling. Camera resources may leak. |
| 16 | **No logging** | Uses `print()` instead of `logging` module. |
| 17 | **No packaging** | No `setup.py` or `pyproject.toml`. |
| 18 | **Misleading README** | Mentions "Haar Cascade / dlib" but code only uses dlib. |

## What's Done Well

- Core EAR-based drowsiness detection algorithm is correct and well-established.
- CI pipeline (`pr-checks.yml`) is comprehensive for syntax, secrets, and security.
- Clean `.gitignore` for Python projects.
- Logical git history with conventional commit prefixes.

## Summary

This is a **proof-of-concept script**, not a finished product. The core detection logic is sound, but the project lacks working installation, required model files, correct documentation, error handling, and tests. The critical and major issues must be resolved before this can be considered shippable.
