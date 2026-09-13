def __getattr__(name):
    if name == "PoseDetector":
        from core.pose_detector import PoseDetector
        return PoseDetector
    if name == "calculate_angle":
        from core.angle_calculator import calculate_angle
        return calculate_angle
    if name == "get_landmark_coords":
        from core.angle_calculator import get_landmark_coords
        return get_landmark_coords
    if name == "ExerciseDetector":
        from core.exercise_detector import ExerciseDetector
        return ExerciseDetector
    raise AttributeError(f"module 'core' has no attribute '{name}'")