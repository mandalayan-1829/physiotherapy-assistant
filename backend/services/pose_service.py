"""
Pose service — wraps the existing AI core for real-time video analysis.
"""

import sys
import os
import base64
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
from core.pose_detector import PoseDetector
from core.exercise_detector import ExerciseDetector


class SessionManager:
    """Manages a real-time exercise session with pose detection.

    This is stateful — one instance per active session.
    """

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.pose_detector = PoseDetector()
        self.exercise_detector = ExerciseDetector(exercise=exercise)
        self.prev_reps = 0
        self.good_reps = 0
        self.target_reps = 10
        self.consecutive_errors = 0
        self.alarm_active = False

    def process_frame(self, frame) -> dict:
        """Process a single BGR frame and return analysis results.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            dict with pose analysis results
        """
        frame = cv2.flip(frame, 1)
        frame, landmarks = self.pose_detector.find_pose(frame, draw=True)
        result = self.exercise_detector.process(landmarks)

        # Track good reps
        if result["reps"] > self.prev_reps:
            if result["form_ok"]:
                self.good_reps += 1
            self.prev_reps = result["reps"]

        # Track alarm state
        if not result["form_ok"]:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
            if self.alarm_active:
                self.alarm_active = False

        if self.consecutive_errors >= 3 and not self.alarm_active:
            self.alarm_active = True

        # Determine form status
        if landmarks is None:
            form_status = "no_pose"
        elif not result["form_ok"]:
            form_status = "incorrect"
        elif self.alarm_active:
            form_status = "warning"
        else:
            form_status = "good"

        # Exercise-specific form status adjustments
        # Shoulder raises: when arms are raised and movement is incomplete,
        # downgrade from "good" to "warning" so the badge reflects the
        # need to lower arms before the next rep.
        if (
            form_status == "good"
            and self.exercise == "shoulder_raises"
            and result["stage"] == "up"
        ):
            form_status = "warning"

        return {
            "rep_count": result["reps"],
            "angle": result["angle"],
            "stage": result["stage"],
            "feedback": result["feedback"],
            "form_status": form_status,
            "form_errors": result["form_errors"],
            "form_ok": result["form_ok"],
            "exercise": result["exercise"],
            "hold_count": result.get("hold_count", 0),
            "alarm_active": self.alarm_active,
            "consecutive_errors": self.consecutive_errors,
        }

    def get_form_accuracy(self) -> int:
        """Calculate form accuracy as a percentage."""
        total = self.prev_reps
        if total == 0:
            return 0
        return int((self.good_reps / total) * 100)

    def reset(self) -> None:
        """Reset the session state."""
        self.exercise_detector.reset()
        self.prev_reps = 0
        self.good_reps = 0
        self.consecutive_errors = 0
        self.alarm_active = False


def decode_frame_from_base64(data: str) -> np.ndarray | None:
    """Decode a base64-encoded JPEG image to an OpenCV frame.

    Args:
        data: base64 string (with or without data URL prefix)

    Returns:
        numpy array (BGR) or None on failure
    """
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None
