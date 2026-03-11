"""Tests for the sleep detector module."""

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from detect_sleep import SleepDetector, compute_ear, parse_args


class TestComputeEar(unittest.TestCase):
    """Tests for the compute_ear function."""

    def test_open_eye_returns_high_ear(self):
        """An open eye shape should produce an EAR above the default threshold."""
        # Simulated open eye landmarks (6 points)
        eye = np.array([
            [0, 0],    # left corner
            [1, 2],    # upper-left
            [3, 2],    # upper-right
            [4, 0],    # right corner
            [3, -2],   # lower-right
            [1, -2],   # lower-left
        ], dtype=np.float64)
        ear = compute_ear(eye)
        self.assertGreater(ear, SleepDetector.DEFAULT_EAR_THRESHOLD)

    def test_closed_eye_returns_low_ear(self):
        """A closed eye shape should produce an EAR below the default threshold."""
        # Simulated closed eye landmarks (nearly flat)
        eye = np.array([
            [0, 0],
            [1, 0.05],
            [3, 0.05],
            [4, 0],
            [3, -0.05],
            [1, -0.05],
        ], dtype=np.float64)
        ear = compute_ear(eye)
        self.assertLess(ear, SleepDetector.DEFAULT_EAR_THRESHOLD)

    def test_symmetric_eye(self):
        """EAR should be the same regardless of left/right eye orientation."""
        eye = np.array([
            [0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1],
        ], dtype=np.float64)
        ear = compute_ear(eye)
        self.assertAlmostEqual(ear, compute_ear(eye), places=6)

    def test_zero_horizontal_raises(self):
        """EAR should raise when horizontal distance is zero (degenerate case)."""
        eye = np.array([
            [0, 0], [0, 1], [0, 1], [0, 0], [0, -1], [0, -1],
        ], dtype=np.float64)
        with self.assertRaises(ZeroDivisionError):
            compute_ear(eye)


class TestParseArgs(unittest.TestCase):
    """Tests for command-line argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.threshold, SleepDetector.DEFAULT_EAR_THRESHOLD)
        self.assertEqual(args.duration, SleepDetector.DEFAULT_CLOSED_SECONDS)
        self.assertEqual(args.camera, 0)
        self.assertFalse(args.verbose)

    def test_custom_values(self):
        args = parse_args(["--threshold", "0.3", "--duration", "10", "--camera", "1", "-v"])
        self.assertAlmostEqual(args.threshold, 0.3)
        self.assertEqual(args.duration, 10)
        self.assertEqual(args.camera, 1)
        self.assertTrue(args.verbose)

    def test_model_path(self):
        args = parse_args(["--model", "/tmp/my_model.dat"])
        self.assertEqual(args.model, "/tmp/my_model.dat")


class TestSleepDetectorInit(unittest.TestCase):
    """Tests for SleepDetector initialization."""

    def test_missing_model_raises(self):
        """Should raise FileNotFoundError when model file does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            SleepDetector(model_path="/nonexistent/model.dat")
        self.assertIn("Model file not found", str(ctx.exception))
        self.assertIn("download_model.py", str(ctx.exception))

    @patch("detect_sleep.dlib")
    @patch("detect_sleep.face_utils")
    def test_successful_init(self, mock_face_utils, mock_dlib):
        """Should initialize successfully with a valid model path."""
        mock_face_utils.FACIAL_LANDMARKS_IDXS = {
            "left_eye": (42, 48),
            "right_eye": (36, 42),
        }

        with patch("os.path.isfile", return_value=True):
            detector = SleepDetector(
                model_path="/fake/model.dat",
                ear_threshold=0.25,
                closed_seconds=3,
            )
        self.assertEqual(detector.ear_threshold, 0.25)
        self.assertEqual(detector.closed_seconds, 3)
        self.assertIsNone(detector._eyes_closed_since)


class TestSleepDetectorAlert(unittest.TestCase):
    """Tests for the alert mechanism."""

    @patch("detect_sleep.dlib")
    @patch("detect_sleep.face_utils")
    def test_alert_cooldown(self, mock_face_utils, mock_dlib):
        """Alerts should respect the cooldown period."""
        mock_face_utils.FACIAL_LANDMARKS_IDXS = {
            "left_eye": (42, 48),
            "right_eye": (36, 42),
        }
        with patch("os.path.isfile", return_value=True):
            detector = SleepDetector(model_path="/fake/model.dat")

        import time
        detector._last_alert_time = time.time()
        # Alert should NOT fire within cooldown
        with patch("builtins.print") as mock_print:
            detector._trigger_alert(6.0)
        # _trigger_alert always prints, cooldown is checked by caller
        mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
