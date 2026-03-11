"""Sleep detector using Eye Aspect Ratio (EAR) with dlib facial landmarks."""

import argparse
import logging
import os
import signal
import sys
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


def compute_ear(eye):
    """Compute the Eye Aspect Ratio (EAR) for a given eye.

    Args:
        eye: Array of 6 (x, y) landmark coordinates for one eye.

    Returns:
        The eye aspect ratio as a float.
    """
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class SleepDetector:
    """Real-time drowsiness detector using webcam and dlib facial landmarks."""

    DEFAULT_EAR_THRESHOLD = 0.2
    DEFAULT_CLOSED_SECONDS = 5
    ALERT_COOLDOWN_SECONDS = 3

    def __init__(self, model_path, ear_threshold=None, closed_seconds=None, camera_index=0):
        """Initialize the sleep detector.

        Args:
            model_path: Path to dlib's shape_predictor_68_face_landmarks.dat.
            ear_threshold: EAR value below which eyes are considered closed.
            closed_seconds: Seconds eyes must stay closed before alerting.
            camera_index: Index of the camera device to use.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download it with: python download_model.py\n"
                "Or manually from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        self.ear_threshold = ear_threshold or self.DEFAULT_EAR_THRESHOLD
        self.closed_seconds = closed_seconds or self.DEFAULT_CLOSED_SECONDS
        self.camera_index = camera_index

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self._eyes_closed_since = None
        self._last_alert_time = 0.0
        self._running = False

    def run(self):
        """Start the detection loop using the webcam feed."""
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            logger.error("Cannot open camera at index %d", self.camera_index)
            sys.exit(1)

        self._running = True
        logger.info(
            "Sleep detector started (EAR threshold=%.2f, alert after %ds). Press 'q' to quit.",
            self.ear_threshold,
            self.closed_seconds,
        )

        def _signal_handler(_signum, _frame):
            logger.info("Received shutdown signal, stopping...")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while self._running:
                ret, frame = capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera, retrying...")
                    continue

                annotated = self._process_frame(frame)
                cv2.imshow("Sleep Detector", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            logger.info("Sleep detector stopped.")

    def _process_frame(self, frame):
        """Process a single video frame for drowsiness detection.

        Args:
            frame: BGR image from the webcam.

        Returns:
            Annotated frame with eye contours drawn.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[self.l_start:self.l_end]
            right_eye = shape[self.r_start:self.r_end]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            left_hull = cv2.convexHull(left_eye)
            right_hull = cv2.convexHull(right_eye)

            eyes_closed = avg_ear < self.ear_threshold
            now = time.time()

            if eyes_closed:
                if self._eyes_closed_since is None:
                    self._eyes_closed_since = now

                closed_duration = now - self._eyes_closed_since
                if closed_duration > self.closed_seconds:
                    color = (0, 0, 255)  # Red when drowsy
                    if now - self._last_alert_time > self.ALERT_COOLDOWN_SECONDS:
                        self._trigger_alert(closed_duration)
                        self._last_alert_time = now
                else:
                    color = (0, 255, 255)  # Yellow when eyes closed but below threshold
            else:
                self._eyes_closed_since = None
                color = (0, 255, 0)  # Green when awake

            cv2.drawContours(frame, [left_hull], -1, color, 1)
            cv2.drawContours(frame, [right_hull], -1, color, 1)

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame

    def _trigger_alert(self, duration):
        """Trigger a drowsiness alert.

        Args:
            duration: How many seconds the eyes have been closed.
        """
        logger.warning("DROWSINESS ALERT! Eyes closed for %.1f seconds", duration)
        print(f"\a")  # Terminal bell character for audible alert


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Detect drowsiness using webcam and dlib.")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
        help="Path to dlib shape predictor model file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SleepDetector.DEFAULT_EAR_THRESHOLD,
        help="EAR threshold for closed eyes (default: %(default)s).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=SleepDetector.DEFAULT_CLOSED_SECONDS,
        help="Seconds eyes must stay closed before alert (default: %(default)s).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the sleep detector."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = SleepDetector(
        model_path=args.model,
        ear_threshold=args.threshold,
        closed_seconds=args.duration,
        camera_index=args.camera,
    )
    detector.run()


if __name__ == "__main__":
    main()
