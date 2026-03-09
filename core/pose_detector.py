"""
core/pose_detector.py
---------------------
Wraps MediaPipe Pose into a clean, reusable class.
Used by exercise_detector.py and the Streamlit dashboard.
"""

import cv2
import mediapipe as mp


class PoseDetector:
    """
    Detects body landmarks from a video frame using MediaPipe Pose.

    Usage:
        detector = PoseDetector()
        frame, landmarks = detector.find_pose(frame)
        if landmarks:
            coords = detector.get_position(landmarks, frame)
    """

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def find_pose(self, frame, draw=True):
        """
        Process a BGR frame and optionally draw the skeleton.

        Returns:
            frame      : annotated frame (BGR)
            landmarks  : mediapipe landmark list, or None if no pose found
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        landmarks = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if draw:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

        return frame, landmarks

    def get_position(self, landmarks, frame):
        """
        Convert normalised landmarks to pixel coordinates.

        Returns:
            dict: { landmark_index: (pixel_x, pixel_y) }
        """
        h, w, _ = frame.shape
        positions = {}
        for idx, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            positions[idx] = (px, py)
        return positions

    def landmark_indices(self):
        """Return the PoseLandmark enum for easy reference."""
        return self.mp_pose.PoseLandmark