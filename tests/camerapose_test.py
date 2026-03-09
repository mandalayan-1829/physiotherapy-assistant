"""
tests/camera_pose_test.py
--------------------------
Standalone test — no Streamlit needed.
Run this FIRST to verify your camera + MediaPipe + angle logic all work.

Usage:
    python tests/camera_pose_test.py
    python tests/camera_pose_test.py --exercise bicep_curl
    python tests/camera_pose_test.py --exercise shoulder_press

Keys:
    Q  – quit
    R  – reset rep counter
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import argparse
from core.pose_detector import PoseDetector
from core.exercise_detector import ExerciseDetector


def draw_ui(frame, result, target=10):
    h, w, _ = frame.shape

    # Background banner
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)

    # Reps
    cv2.putText(frame, "REPS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"{result['reps']} / {target}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 100), 3)

    # Stage
    cv2.putText(frame, "STAGE", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    stage_color = (0, 255, 255) if result["stage"] == "down" else (255, 255, 255)
    cv2.putText(frame, result["stage"].upper(), (200, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, stage_color, 3)

    # Angle
    cv2.putText(frame, "ANGLE", (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"{result['angle']}deg", (400, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 200, 0), 3)

    # Exercise name top-right
    cv2.putText(frame, result["exercise"].replace("_", " ").upper(),
                (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

    # Feedback bottom
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, result["feedback"], (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="AI Physiotherapy – Camera Test")
    parser.add_argument("--exercise", default="squat",
                        choices=["squat", "bicep_curl", "shoulder_press"])
    parser.add_argument("--target",   default=10, type=int)
    args = parser.parse_args()

    print(f"\n🏋️  Exercise : {args.exercise}")
    print(f"🎯  Target   : {args.target} reps")
    print("📹  Press Q to quit | R to reset\n")

    pose_detector = PoseDetector()
    ex_detector   = ExerciseDetector(exercise=args.exercise)
    cap           = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame, landmarks = pose_detector.find_pose(frame)
        result = ex_detector.process(landmarks)

        frame = draw_ui(frame, result, target=args.target)
        cv2.imshow("AI Physiotherapy – Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            ex_detector.reset()
            print("🔄 Counter reset.")

        if result["reps"] >= args.target:
            print(f"\n🎉 Target of {args.target} reps reached!")
            cv2.waitKey(2000)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete — Total reps: {ex_detector.reps}")


if __name__ == "__main__":
    main()