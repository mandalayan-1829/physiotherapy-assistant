"""
core/exercise_detector.py
--------------------------
AI Physiotherapy engine — 6 exercises, rep counting + form detection.
Exercises: Squat, Shoulder Raises, Crossover Arm Stretch,
           Lateral Walks, Lunges, Calf Raises
"""

import mediapipe as mp
from core.angle_calculator import calculate_angle, get_landmark_coords

PL = mp.solutions.pose.PoseLandmark

LEFT_SHOULDER   = PL.LEFT_SHOULDER.value
LEFT_ELBOW      = PL.LEFT_ELBOW.value
LEFT_WRIST      = PL.LEFT_WRIST.value
LEFT_HIP        = PL.LEFT_HIP.value
LEFT_KNEE       = PL.LEFT_KNEE.value
LEFT_ANKLE      = PL.LEFT_ANKLE.value
LEFT_EAR        = PL.LEFT_EAR.value
LEFT_FOOT_INDEX = PL.LEFT_FOOT_INDEX.value
LEFT_HEEL       = PL.LEFT_HEEL.value

RIGHT_SHOULDER   = PL.RIGHT_SHOULDER.value
RIGHT_ELBOW      = PL.RIGHT_ELBOW.value
RIGHT_WRIST      = PL.RIGHT_WRIST.value
RIGHT_HIP        = PL.RIGHT_HIP.value
RIGHT_KNEE       = PL.RIGHT_KNEE.value
RIGHT_ANKLE      = PL.RIGHT_ANKLE.value
RIGHT_FOOT_INDEX = PL.RIGHT_FOOT_INDEX.value


class ExerciseDetector:
    SUPPORTED = [
        "squat", "shoulder_raises", "crossover_arm_stretch",
        "lateral_walks", "lunges", "calf_raises"
    ]

    def __init__(self, exercise: str = "squat"):
        exercise = exercise.lower()
        if exercise not in self.SUPPORTED:
            raise ValueError(f"Exercise '{exercise}' not supported. Choose from {self.SUPPORTED}")
        self.exercise    = exercise
        self.reps        = 0
        self.stage       = "up"
        self.angle       = 0.0
        self.feedback    = "Stand in frame"
        self.form_errors = []

    def reset(self):
        self.reps        = 0
        self.stage       = "up"
        self.angle       = 0.0
        self.feedback    = "Stand in frame"
        self.form_errors = []

    def process(self, landmarks):
        if landmarks is None:
            self.feedback    = "No pose detected — stand in frame"
            self.form_errors = []
            return self._state()

        self.form_errors = []

        dispatch = {
            "squat":                  self._detect_squat,
            "shoulder_raises":        self._detect_shoulder_raises,
            "crossover_arm_stretch":  self._detect_crossover_arm_stretch,
            "lateral_walks":          self._detect_lateral_walks,
            "lunges":                 self._detect_lunges,
            "calf_raises":            self._detect_calf_raises,
        }
        dispatch[self.exercise](landmarks)
        return self._state()

    # ════════════════════════════════════════════════════════════════════════
    # 1. SQUAT
    # ════════════════════════════════════════════════════════════════════════
    def _detect_squat(self, lm):
        hip      = get_landmark_coords(lm, LEFT_HIP)
        knee     = get_landmark_coords(lm, LEFT_KNEE)
        ankle    = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear      = get_landmark_coords(lm, LEFT_EAR)

        self.angle = calculate_angle(hip, knee, ankle)

        if self.angle < 90:
            self.stage    = "down"
            self.feedback = "Good depth! Now stand up."
        elif self.angle > 160 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Go down again."
        elif self.stage == "up":
            self.feedback = "Bend your knees to squat down."

        # Form: back too bent
        back_angle = calculate_angle(shoulder, hip, [hip[0], hip[1] + 0.1])
        if back_angle > 70 and self.stage == "down":
            self.form_errors.append("⚠️ Back too bent — keep chest up!")

        # Form: knees over toes
        if self.stage == "down":
            knee_x = lm[LEFT_KNEE].x
            toe_x  = lm[LEFT_FOOT_INDEX].x
            if abs(knee_x - toe_x) > 0.08:
                self.form_errors.append("⚠️ Knees going past toes — push hips back!")

        # Form: spine alignment
        spine_angle = calculate_angle(ear, shoulder, hip)
        if spine_angle < 150:
            self.form_errors.append("⚠️ Spine not straight — align head with hips!")

    # ════════════════════════════════════════════════════════════════════════
    # 2. SHOULDER RAISES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_shoulder_raises(self, lm):
        elbow    = get_landmark_coords(lm, LEFT_ELBOW)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        hip      = get_landmark_coords(lm, LEFT_HIP)
        wrist    = get_landmark_coords(lm, LEFT_WRIST)
        ear      = get_landmark_coords(lm, LEFT_EAR)

        self.angle = calculate_angle(elbow, shoulder, hip)

        if self.angle < 30:
            self.stage    = "down"
            self.feedback = "Raise arms out to the side!"
        elif self.angle > 80 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Lower slowly."
        elif self.stage == "up":
            self.feedback = "Lower arms fully before next raise."

        # Form: bent elbows
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        if elbow_angle < 150:
            self.form_errors.append("⚠️ Keep arms straight — don't bend elbows!")

        # Form: raising too high
        wrist_y    = lm[LEFT_WRIST].y
        shoulder_y = lm[LEFT_SHOULDER].y
        if wrist_y < shoulder_y - 0.05:
            self.form_errors.append("⚠️ Don't raise above shoulder height!")

        # Form: spine straight
        spine_angle = calculate_angle(ear, shoulder, hip)
        if spine_angle < 150:
            self.form_errors.append("⚠️ Keep spine straight — relax your neck!")

    # ════════════════════════════════════════════════════════════════════════
    # 3. CROSSOVER ARM STRETCH
    # ════════════════════════════════════════════════════════════════════════
    def _detect_crossover_arm_stretch(self, lm):
        """
        Left arm crosses body horizontally toward right shoulder.
        Measures: angle at left shoulder (left elbow - left shoulder - right shoulder)
        DOWN: arm at side (angle > 150)
        UP:   arm crossed across body (angle < 60)
        """
        l_elbow    = get_landmark_coords(lm, LEFT_ELBOW)
        l_shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        r_shoulder = get_landmark_coords(lm, RIGHT_SHOULDER)
        ear        = get_landmark_coords(lm, LEFT_EAR)
        hip        = get_landmark_coords(lm, LEFT_HIP)

        self.angle = calculate_angle(l_elbow, l_shoulder, r_shoulder)

        if self.angle > 150:
            self.stage    = "down"
            self.feedback = "Bring left arm across your chest."
        elif self.angle < 60 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Hold 2 sec, then release."
        elif self.stage == "up":
            self.feedback = "Return arm to side before next stretch."

        # Form: shoulder shrugging (shoulder rising toward ear)
        ear_y      = lm[LEFT_EAR].y
        shoulder_y = lm[LEFT_SHOULDER].y
        if (shoulder_y - ear_y) < 0.08:
            self.form_errors.append("⚠️ Don't shrug — keep shoulders relaxed!")

        # Form: torso rotating
        spine_angle = calculate_angle(ear, l_shoulder, hip)
        if spine_angle < 145:
            self.form_errors.append("⚠️ Keep torso still — don't rotate your body!")

    # ════════════════════════════════════════════════════════════════════════
    # 4. LATERAL WALKS
    # ════════════════════════════════════════════════════════════════════════
    def _detect_lateral_walks(self, lm):
        """
        Tracks hip width changes as person steps side to side.
        Wide stance = step out (DOWN), narrow stance = step in (UP).
        """
        l_hip  = lm[LEFT_HIP].x
        r_hip  = lm[RIGHT_HIP].x
        l_ankle = lm[LEFT_ANKLE].x
        r_ankle = lm[RIGHT_ANKLE].x

        # Foot spread relative to hip width
        hip_width  = abs(l_hip - r_hip)
        foot_spread = abs(l_ankle - r_ankle)
        ratio = foot_spread / (hip_width + 1e-6)

        self.angle = round(ratio * 100, 1)   # show as percentage spread

        # Hip-knee-ankle for form check
        hip   = get_landmark_coords(lm, LEFT_HIP)
        knee  = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        ear   = get_landmark_coords(lm, LEFT_EAR)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)

        if ratio > 1.6:
            self.stage    = "down"
            self.feedback = "Good wide step! Now step back in."
        elif ratio < 1.1 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Step out again."
        elif self.stage == "up":
            self.feedback = "Step out to the side — wider than hip width."

        # Form: staying low (slight knee bend)
        knee_angle = calculate_angle(hip, knee, ankle)
        if knee_angle > 170:
            self.form_errors.append("⚠️ Bend knees slightly — stay low!")

        # Form: torso upright
        spine_angle = calculate_angle(ear, shoulder, hip)
        if spine_angle < 150:
            self.form_errors.append("⚠️ Keep torso upright — don't lean sideways!")

    # ════════════════════════════════════════════════════════════════════════
    # 5. LUNGES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_lunges(self, lm):
        """
        Front knee angle (hip-knee-ankle) on left leg.
        DOWN: knee angle < 100 (deep lunge)
        UP:   knee angle > 160 (standing)
        """
        hip      = get_landmark_coords(lm, LEFT_HIP)
        knee     = get_landmark_coords(lm, LEFT_KNEE)
        ankle    = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear      = get_landmark_coords(lm, LEFT_EAR)

        self.angle = calculate_angle(hip, knee, ankle)

        if self.angle < 100:
            self.stage    = "down"
            self.feedback = "Good lunge! Now push back up."
        elif self.angle > 160 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Lunge again."
        elif self.stage == "up":
            self.feedback = "Step forward and lower your back knee."

        # Form: front knee over toes
        knee_x = lm[LEFT_KNEE].x
        toe_x  = lm[LEFT_FOOT_INDEX].x
        if self.stage == "down" and abs(knee_x - toe_x) > 0.07:
            self.form_errors.append("⚠️ Front knee too far forward — step longer!")

        # Form: torso upright
        torso_angle = calculate_angle(ear, shoulder, hip)
        if torso_angle < 150:
            self.form_errors.append("⚠️ Keep torso upright — don't lean forward!")

        # Form: not going deep enough
        if self.stage == "down" and self.angle > 120:
            self.form_errors.append("⚠️ Go deeper — lower your back knee more!")

    # ════════════════════════════════════════════════════════════════════════
    # 6. CALF RAISES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_calf_raises(self, lm):
        """
        Tracks ankle angle (knee-ankle-foot_index).
        UP:   angle < 80  (on tiptoes)
        DOWN: angle > 100 (flat foot)
        """
        knee       = get_landmark_coords(lm, LEFT_KNEE)
        ankle      = get_landmark_coords(lm, LEFT_ANKLE)
        foot_index = get_landmark_coords(lm, LEFT_FOOT_INDEX)
        hip        = get_landmark_coords(lm, LEFT_HIP)
        ear        = get_landmark_coords(lm, LEFT_EAR)
        shoulder   = get_landmark_coords(lm, LEFT_SHOULDER)

        self.angle = calculate_angle(knee, ankle, foot_index)

        if self.angle > 100:
            self.stage    = "down"
            self.feedback = "Rise up on your tiptoes!"
        elif self.angle < 80 and self.stage == "down":
            self.stage    = "up"
            self.reps    += 1
            self.feedback = f"Rep {self.reps} complete! Lower back down slowly."
        elif self.stage == "up":
            self.feedback = "Lower heels fully before next raise."

        # Form: knees bending
        knee_angle = calculate_angle(hip, knee, ankle)
        if knee_angle < 160:
            self.form_errors.append("⚠️ Keep legs straight — don't bend knees!")

        # Form: leaning forward
        spine_angle = calculate_angle(ear, shoulder, hip)
        if spine_angle < 150:
            self.form_errors.append("⚠️ Stand tall — don't lean forward!")

    # ════════════════════════════════════════════════════════════════════════
    def _state(self):
        return {
            "reps":        self.reps,
            "stage":       self.stage,
            "angle":       self.angle,
            "feedback":    self.feedback,
            "form_errors": self.form_errors,
            "form_ok":     len(self.form_errors) == 0,
            "exercise":    self.exercise,
        }