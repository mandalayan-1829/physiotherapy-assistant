"""
core/angle_calculator.py
------------------------
Calculates the angle between three body landmarks.
This is the mathematical engine used by exercise_detector.py.
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B, formed by points A-B-C.

    Parameters:
        a (list/tuple): [x, y] of the first point  (e.g. hip)
        b (list/tuple): [x, y] of the middle point (e.g. knee)  ← angle measured here
        c (list/tuple): [x, y] of the third point  (e.g. ankle)

    Returns:
        float: Angle in degrees (0–180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors from B to A and from B to C
    ba = a - b
    bc = c - b

    # Dot product and magnitudes
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)

    # Clamp to [-1, 1] to avoid floating-point errors in arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 2)


def get_landmark_coords(landmarks, index):
    """
    Extract (x, y) from a MediaPipe landmark by index.

    Parameters:
        landmarks: mediapipe pose landmark list
        index (int): landmark index (e.g. mp_pose.PoseLandmark.LEFT_KNEE.value)

    Returns:
        list: [x, y]
    """
    lm = landmarks[index]
    return [lm.x, lm.y]


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 90-degree angle test: A=(0,1), B=(0,0), C=(1,0)
    angle = calculate_angle([0, 1], [0, 0], [1, 0])
    print(f"Test 1 – Expected 90°  → Got {angle}°")

    # 180-degree angle test: straight line
    angle = calculate_angle([0, 1], [0, 0], [0, -1])
    print(f"Test 2 – Expected 180° → Got {angle}°")

    # 45-degree angle test
    angle = calculate_angle([1, 1], [0, 0], [1, 0])
    print(f"Test 3 – Expected 45°  → Got {angle}°")