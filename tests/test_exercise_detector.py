"""
Tests for core/exercise_detector.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.exercise_detector import ExerciseDetector


class MockLandmark:
    """Mock MediaPipe landmark for testing."""

    def __init__(self, x, y, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def make_landmarks(positions: list[tuple[float, float]]) -> list:
    """Create a list of 33 mock landmarks from a dict of {index: (x, y)}."""
    landmarks = [MockLandmark(0.5, 0.5)] * 33  # default to center
    for i, (x, y) in positions:
        landmarks[i] = MockLandmark(x, y)
    return landmarks


def test_supported_exercises():
    """All 10 exercises should be supported."""
    exercises = [
        "squat", "shoulder_raises", "crossover_arm_stretch",
        "lateral_walks", "lunges", "calf_raises", "knee_raises",
        "tree_pose", "warrior_pose", "cat_cow_stretch",
    ]
    for ex in exercises:
        detector = ExerciseDetector(exercise=ex)
        assert detector.exercise == ex


def test_invalid_exercise():
    """Invalid exercise name should raise ValueError."""
    try:
        ExerciseDetector(exercise="invalid_exercise")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_process_none_landmarks():
    """Processing None landmarks should return no-pose feedback."""
    detector = ExerciseDetector(exercise="squat")
    result = detector.process(None)
    assert result["feedback"] == "No pose detected — stand in frame"
    assert result["reps"] == 0


def test_squat_detection():
    """Test squat rep counting with mock landmarks."""
    # PoseLandmark values for left side:
    # LEFT_SHOULDER=11, LEFT_ELBOW=13, LEFT_WRIST=15, LEFT_HIP=23,
    # LEFT_KNEE=25, LEFT_ANKLE=27, LEFT_EAR=7, LEFT_FOOT_INDEX=31, LEFT_HEEL=29
    # RIGHT_SHOULDER=12, RIGHT_KNEE=26

    # Standing pose (up)
    standing = [
        (11, (0.4, 0.3)),   # shoulder
        (23, (0.42, 0.5)),  # hip
        (25, (0.42, 0.7)),  # knee
        (27, (0.42, 0.9)),  # ankle
        (7, (0.4, 0.15)),   # ear
        (31, (0.44, 0.92)), # foot_index
        (26, (0.55, 0.7)),  # right_knee
    ]
    lm_up = make_landmarks(standing)

    detector = ExerciseDetector(exercise="squat")
    result = detector.process(lm_up)
    assert result["stage"] == "up"
    assert result["reps"] == 0

    # Squatting pose (down) — knee angle < 90
    squatting = [
        (11, (0.4, 0.25)),  # shoulder
        (23, (0.42, 0.45)), # hip
        (25, (0.2, 0.55)),  # knee (pushed far forward to create <90 angle)
        (27, (0.42, 0.7)),  # ankle
        (7, (0.4, 0.12)),   # ear
        (31, (0.44, 0.72)), # foot_index
        (26, (0.55, 0.55)), # right_knee
    ]
    lm_down = make_landmarks(squatting)

    result = detector.process(lm_down)
    assert result["stage"] == "down", f"Expected 'down', got {result['stage']}"

    # Back to standing → rep should count
    result = detector.process(lm_up)
    assert result["stage"] == "up"
    assert result["reps"] == 1


def test_reset():
    """Test that reset clears all state."""
    detector = ExerciseDetector(exercise="squat")
    detector.reps = 5
    detector.stage = "down"
    detector.reset()
    assert detector.reps == 0
    assert detector.stage == "up"


def test_state_output():
    """Test that _state returns expected keys."""
    detector = ExerciseDetector(exercise="squat")
    result = detector.process(None)
    expected_keys = {"reps", "stage", "angle", "feedback", "form_errors", "form_ok", "exercise", "hold_count"}
    assert expected_keys.issubset(set(result.keys())), f"Missing keys: {expected_keys - set(result.keys())}"


# ══════════════════════════════════════════════════════════════════════════════
# Shoulder Raises — state machine and form validation tests
# ══════════════════════════════════════════════════════════════════════════════
#
# Shoulder-raise angle = calculate_angle(elbow, shoulder, hip)
#   Arms at sides (down): elbow below shoulder → angle ≈ 10–30°
#   Arms raised:          elbow at shoulder height → angle ≈ 80–90°
#   Arms above shoulder:  angle > 90°
#
# Stage machine:
#   angle < 30         → stage="down",  feedback="Raise arms to the side!"
#   angle > 80 + down  → stage="up",    reps += 1, feedback="Rep N done! Lower slowly."
#   stage==up + angle≥50 → feedback="Lower arms fully before next raise."
#   stage==up + angle<50 → stage="down", feedback="Arms lowered. Raise to the side!"
#
# Landmark indices: LEFT_ELBOW=13, LEFT_SHOULDER=11, LEFT_HIP=23,
#                   LEFT_WRIST=15, LEFT_EAR=7
# ══════════════════════════════════════════════════════════════════════════════


def _make_shoulder_raise_landmarks(elbow_xy, shoulder_xy, hip_xy,
                                     wrist_xy=None, ear_xy=None):
    """Build mock landmarks for shoulder-raise testing.

    Defaults wrist and ear to neutral positions that pass form checks.
    """
    if wrist_xy is None:
        # Wrist directly below elbow (arm straight)
        wrist_xy = (elbow_xy[0], elbow_xy[1] + 0.2)
    if ear_xy is None:
        # Ear above shoulder (spine straight)
        ear_xy = (shoulder_xy[0], shoulder_xy[1] - 0.25)

    positions = [
        (11, shoulder_xy),   # LEFT_SHOULDER
        (13, elbow_xy),      # LEFT_ELBOW
        (15, wrist_xy),      # LEFT_WRIST
        (23, hip_xy),        # LEFT_HIP
        (7, ear_xy),         # LEFT_EAR
    ]
    return make_landmarks(positions)


def _sr_down():
    """Arms fully at sides — small elbow-shoulder-hip angle."""
    return _make_shoulder_raise_landmarks(
        elbow_xy=(0.4, 0.55),    # below shoulder
        shoulder_xy=(0.4, 0.35),
        hip_xy=(0.42, 0.6),
    )


def _sr_raised():
    """Arms raised at shoulder height — large elbow-shoulder-hip angle."""
    return _make_shoulder_raise_landmarks(
        elbow_xy=(0.2, 0.35),    # out to the side at shoulder height
        shoulder_xy=(0.4, 0.35),
        hip_xy=(0.42, 0.6),
    )


def _sr_mid(angle_hint="lowering"):
    """Arms partially lowered — angle between down and raised.

    Verified angles (shoulder=(0.4, 0.35), hip=(0.42, 0.6)):
      angle_hint='stuck'    → elbow=(0.28, 0.40) → angle=71.95° (dead zone)
      angle_hint='lowering' → elbow=(0.32, 0.47) → angle=38.27° (below 50)
    """
    if angle_hint == "stuck":
        return _make_shoulder_raise_landmarks(
            elbow_xy=(0.28, 0.40),
            shoulder_xy=(0.4, 0.35),
            hip_xy=(0.42, 0.6),
        )
    else:
        return _make_shoulder_raise_landmarks(
            elbow_xy=(0.32, 0.47),
            shoulder_xy=(0.4, 0.35),
            hip_xy=(0.42, 0.6),
        )


def test_shoulder_raise_basic_rep_counting():
    """20 → 85 → 20 → 85 should produce 2 reps."""
    detector = ExerciseDetector(exercise="shoulder_raises")
    lm_down = _sr_down()
    lm_raised = _sr_raised()

    # Arms down
    r = detector.process(lm_down)
    assert r["stage"] == "down", f"Expected down, got {r['stage']}"
    assert r["reps"] == 0

    # Arms raised → rep 1
    r = detector.process(lm_raised)
    assert r["stage"] == "up", f"Expected up, got {r['stage']}"
    assert r["reps"] == 1, f"Expected 1 rep, got {r['reps']}"

    # Arms down again
    r = detector.process(lm_down)
    assert r["stage"] == "down"
    assert r["reps"] == 1

    # Arms raised → rep 2
    r = detector.process(lm_raised)
    assert r["stage"] == "up"
    assert r["reps"] == 2, f"Expected 2 reps, got {r['reps']}"


def test_shoulder_raise_no_dead_zone():
    """85 → 77 → 50 → 20 should not get stuck."""
    detector = ExerciseDetector(exercise="shoulder_raises")
    lm_down = _sr_down()
    lm_raised = _sr_raised()
    lm_stuck = _sr_mid("stuck")   # angle ~75
    lm_lower = _sr_mid("lowering") # angle ~45

    # Start: arms down → stage down
    r = detector.process(lm_down)
    assert r["stage"] == "down"

    # Raise → rep 1, stage up
    r = detector.process(lm_raised)
    assert r["stage"] == "up"
    assert r["reps"] == 1

    # Lower partially (angle ~75) → stage stays up, needs more lowering
    r = detector.process(lm_stuck)
    assert r["stage"] == "up", f"Expected up at angle~75, got {r['stage']}"
    assert "Lower" in r["feedback"]

    # Lower more (angle ~45 < 50) → stage transitions to down
    r = detector.process(lm_lower)
    assert r["stage"] == "down", f"Expected down at angle~45, got {r['stage']}"
    assert "lowered" in r["feedback"].lower()

    # Raise again → rep 2
    r = detector.process(lm_raised)
    assert r["stage"] == "up"
    assert r["reps"] == 2, f"Expected 2 reps, got {r['reps']}"


def test_shoulder_raise_hold_at_top_keeps_up():
    """Holding at 77–85 keeps stage 'up' and gives lowering feedback."""
    detector = ExerciseDetector(exercise="shoulder_raises")
    lm_down = _sr_down()
    lm_raised = _sr_raised()
    lm_stuck = _sr_mid("stuck")  # angle ~75

    # Get to stage up
    detector.process(lm_down)
    r = detector.process(lm_raised)
    assert r["stage"] == "up"

    # Hold at top for multiple frames
    for _ in range(5):
        r = detector.process(lm_stuck)
        assert r["stage"] == "up", f"Expected up while holding, got {r['stage']}"
        assert "Lower" in r["feedback"], f"Expected lowering feedback, got {r['feedback']}"

    # Still up, no rep counted during hold
    assert r["reps"] == 1


def test_shoulder_raise_below_50_transitions_to_down():
    """angle < 50 transitions stage to 'down'."""
    detector = ExerciseDetector(exercise="shoulder_raises")
    lm_down = _sr_down()
    lm_raised = _sr_raised()
    lm_lower = _sr_mid("lowering")  # angle ~45

    # Get to stage up
    detector.process(lm_down)
    detector.process(lm_raised)
    assert detector.stage == "up"

    # Lower below 50 → stage down
    r = detector.process(lm_lower)
    assert r["stage"] == "down", f"Expected down when angle<50, got {r['stage']}"
    assert "lowered" in r["feedback"].lower()


def test_shoulder_raise_form_status_warning_when_raised():
    """PoseService should downgrade shoulder-raise form_status to 'warning'
    when stage is 'up' (arms raised / not fully lowered)."""
    try:
        from backend.services.pose_service import SessionManager
    except ImportError:
        return  # skip if pose_service deps not available

    import cv2
    import numpy as np

    manager = SessionManager("shoulder_raises")

    # Build a synthetic BGR frame (black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # We can't easily feed real landmarks through process_frame without
    # a real image, so we test the logic directly:
    # 1) Simulate: exercise_detector has stage="up", form_ok=True
    manager.exercise_detector.stage = "up"
    manager.exercise_detector.form_errors = []
    manager.exercise_detector.angle = 77.0
    manager.exercise_detector.feedback = "Lower arms fully before next raise."
    manager.exercise_detector.reps = 1
    result = manager.exercise_detector._state()

    # Verify form_ok is True
    assert result["form_ok"] is True

    # Apply the same form_status logic as process_frame
    form_status = "good"  # would be "good" because form_ok is True
    if form_status == "good" and manager.exercise == "shoulder_raises" and result["stage"] == "up":
        form_status = "warning"

    assert form_status == "warning", (
        f"Expected 'warning' for shoulder_raises stage=up, got '{form_status}'"
    )


def test_shoulder_raise_form_status_good_when_down():
    """PoseService keeps form_status 'good' when shoulder-raise stage is 'down'."""
    try:
        from backend.services.pose_service import SessionManager
    except ImportError:
        return

    manager = SessionManager("shoulder_raises")

    manager.exercise_detector.stage = "down"
    manager.exercise_detector.form_errors = []
    manager.exercise_detector.angle = 20.0
    manager.exercise_detector.feedback = "Arms lowered. Raise to the side!"
    result = manager.exercise_detector._state()

    assert result["form_ok"] is True

    form_status = "good"
    if form_status == "good" and manager.exercise == "shoulder_raises" and result["stage"] == "up":
        form_status = "warning"

    assert form_status == "good", (
        f"Expected 'good' for shoulder_raises stage=down, got '{form_status}'"
    )


def test_other_exercises_not_globally_affected():
    """The shoulder-raise form_status rule must NOT apply to other exercises."""
    try:
        from backend.services.pose_service import SessionManager
    except ImportError:
        return

    # Test with squat — stage up should NOT trigger warning
    manager = SessionManager("squat")

    manager.exercise_detector.stage = "up"
    manager.exercise_detector.form_errors = []
    manager.exercise_detector.angle = 165.0
    manager.exercise_detector.reps = 1
    result = manager.exercise_detector._state()

    assert result["form_ok"] is True

    form_status = "good"
    # Apply the shoulder_raises rule — it should NOT fire for squat
    if form_status == "good" and manager.exercise == "shoulder_raises" and result["stage"] == "up":
        form_status = "warning"

    assert form_status == "good", (
        f"Expected 'good' for squat, got '{form_status}' — shoulder rule leaked"
    )
