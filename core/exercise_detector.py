"""
core/exercise_detector.py
--------------------------
AI Physiotherapy engine — 10 exercises with rep counting + form detection.
Physio: Squat, Shoulder Raises, Crossover Arm Stretch,
        Lateral Walks, Lunges, Calf Raises, Knee Raises
Yoga:   Tree Pose, Warrior Pose, Cat-Cow Stretch
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
        "lateral_walks", "lunges", "calf_raises", "knee_raises",
        "tree_pose", "warrior_pose", "cat_cow_stretch",
    ]

    def __init__(self, exercise: str = "squat"):
        exercise = exercise.lower()
        if exercise not in self.SUPPORTED:
            raise ValueError(f"Exercise '{exercise}' not supported.")
        self.exercise    = exercise
        self.reps        = 0
        self.stage       = "up"
        self.angle       = 0.0
        self.feedback    = "Stand in frame"
        self.form_errors = []
        self.hold_count  = 0   # for pose-hold exercises

    def reset(self):
        self.reps = 0; self.stage = "up"; self.angle = 0.0
        self.feedback = "Stand in frame"; self.form_errors = []; self.hold_count = 0

    def process(self, landmarks):
        if landmarks is None:
            self.feedback = "No pose detected — stand in frame"
            self.form_errors = []
            return self._state()
        self.form_errors = []
        dispatch = {
            "squat":                 self._detect_squat,
            "shoulder_raises":       self._detect_shoulder_raises,
            "crossover_arm_stretch": self._detect_crossover_arm_stretch,
            "lateral_walks":         self._detect_lateral_walks,
            "lunges":                self._detect_lunges,
            "calf_raises":           self._detect_calf_raises,
            "knee_raises":           self._detect_knee_raises,
            "tree_pose":             self._detect_tree_pose,
            "warrior_pose":          self._detect_warrior_pose,
            "cat_cow_stretch":       self._detect_cat_cow_stretch,
        }
        dispatch[self.exercise](landmarks)
        return self._state()

    # ════════════════════════════════════════════════════════════════════════
    # 1. SQUAT
    # ════════════════════════════════════════════════════════════════════════
    def _detect_squat(self, lm):
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear = get_landmark_coords(lm, LEFT_EAR)
        self.angle = calculate_angle(hip, knee, ankle)
        if self.angle < 90:
            self.stage = "down"; self.feedback = "Good depth! Now stand up."
        elif self.angle > 160 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} complete! Go down again."
        elif self.stage == "up":
            self.feedback = "Bend your knees to squat down."
        back_angle = calculate_angle(shoulder, hip, [hip[0], hip[1] + 0.1])
        if back_angle > 70 and self.stage == "down":
            self.form_errors.append("⚠️ Back too bent — keep chest up!")
        if self.stage == "down":
            if abs(lm[LEFT_KNEE].x - lm[LEFT_FOOT_INDEX].x) > 0.08:
                self.form_errors.append("⚠️ Knees going past toes — push hips back!")
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep spine straight!")

    # ════════════════════════════════════════════════════════════════════════
    # 2. SHOULDER RAISES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_shoulder_raises(self, lm):
        elbow = get_landmark_coords(lm, LEFT_ELBOW)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        hip = get_landmark_coords(lm, LEFT_HIP)
        wrist = get_landmark_coords(lm, LEFT_WRIST)
        ear = get_landmark_coords(lm, LEFT_EAR)
        self.angle = calculate_angle(elbow, shoulder, hip)
        if self.angle < 30:
            self.stage = "down"; self.feedback = "Raise arms to the side!"
        elif self.angle > 80 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Lower slowly."
        elif self.stage == "up":
            self.feedback = "Lower arms fully before next raise."
        if calculate_angle(shoulder, elbow, wrist) < 150:
            self.form_errors.append("⚠️ Keep arms straight — don't bend elbows!")
        if lm[LEFT_WRIST].y < lm[LEFT_SHOULDER].y - 0.05:
            self.form_errors.append("⚠️ Don't raise above shoulder height!")
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep spine straight!")

    # ════════════════════════════════════════════════════════════════════════
    # 3. CROSSOVER ARM STRETCH
    # ════════════════════════════════════════════════════════════════════════
    def _detect_crossover_arm_stretch(self, lm):
        l_elbow = get_landmark_coords(lm, LEFT_ELBOW)
        l_shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        r_shoulder = get_landmark_coords(lm, RIGHT_SHOULDER)
        ear = get_landmark_coords(lm, LEFT_EAR)
        hip = get_landmark_coords(lm, LEFT_HIP)
        self.angle = calculate_angle(l_elbow, l_shoulder, r_shoulder)
        if self.angle > 150:
            self.stage = "down"; self.feedback = "Bring arm across your chest."
        elif self.angle < 60 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Hold 2 sec then release."
        elif self.stage == "up":
            self.feedback = "Return arm to side before next stretch."
        if (lm[LEFT_SHOULDER].y - lm[LEFT_EAR].y) < 0.08:
            self.form_errors.append("⚠️ Don't shrug — keep shoulders relaxed!")
        if calculate_angle(ear, l_shoulder, hip) < 145:
            self.form_errors.append("⚠️ Keep torso still — don't rotate!")

    # ════════════════════════════════════════════════════════════════════════
    # 4. LATERAL WALKS
    # ════════════════════════════════════════════════════════════════════════
    def _detect_lateral_walks(self, lm):
        hip_w = abs(lm[LEFT_HIP].x - lm[RIGHT_HIP].x)
        foot_spread = abs(lm[LEFT_ANKLE].x - lm[RIGHT_ANKLE].x)
        ratio = foot_spread / (hip_w + 1e-6)
        self.angle = round(ratio * 100, 1)
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        ear = get_landmark_coords(lm, LEFT_EAR)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        if ratio > 1.6:
            self.stage = "down"; self.feedback = "Good wide step! Now step back in."
        elif ratio < 1.1 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Step out again."
        elif self.stage == "up":
            self.feedback = "Step out wider than hip width."
        if calculate_angle(hip, knee, ankle) > 170:
            self.form_errors.append("⚠️ Bend knees slightly — stay low!")
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep torso upright!")

    # ════════════════════════════════════════════════════════════════════════
    # 5. LUNGES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_lunges(self, lm):
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear = get_landmark_coords(lm, LEFT_EAR)
        self.angle = calculate_angle(hip, knee, ankle)
        if self.angle < 100:
            self.stage = "down"; self.feedback = "Good lunge! Push back up."
        elif self.angle > 160 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Lunge again."
        elif self.stage == "up":
            self.feedback = "Step forward and lower your back knee."
        if self.stage == "down" and abs(lm[LEFT_KNEE].x - lm[LEFT_FOOT_INDEX].x) > 0.07:
            self.form_errors.append("⚠️ Front knee too far — step longer!")
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep torso upright!")
        if self.stage == "down" and self.angle > 120:
            self.form_errors.append("⚠️ Go deeper — lower back knee more!")

    # ════════════════════════════════════════════════════════════════════════
    # 6. CALF RAISES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_calf_raises(self, lm):
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        foot_index = get_landmark_coords(lm, LEFT_FOOT_INDEX)
        hip = get_landmark_coords(lm, LEFT_HIP)
        ear = get_landmark_coords(lm, LEFT_EAR)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        self.angle = calculate_angle(knee, ankle, foot_index)
        if self.angle > 100:
            self.stage = "down"; self.feedback = "Rise up on your tiptoes!"
        elif self.angle < 80 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Lower heels slowly."
        elif self.stage == "up":
            self.feedback = "Lower heels fully before next raise."
        if calculate_angle(hip, knee, ankle) < 160:
            self.form_errors.append("⚠️ Keep legs straight!")
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Stand tall — don't lean forward!")

    # ════════════════════════════════════════════════════════════════════════
    # 7. KNEE RAISES
    # ════════════════════════════════════════════════════════════════════════
    def _detect_knee_raises(self, lm):
        """
        Alternating knee raise — measures hip flexion (shoulder-hip-knee).
        DOWN: knee down (angle > 150), UP: knee raised (angle < 90)
        """
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ear = get_landmark_coords(lm, LEFT_EAR)
        self.angle = calculate_angle(shoulder, hip, knee)
        if self.angle > 150:
            self.stage = "down"; self.feedback = "Raise your knee up!"
        elif self.angle < 90 and self.stage == "down":
            self.stage = "up"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Lower and switch sides."
        elif self.stage == "up":
            self.feedback = "Lower leg fully before next raise."
        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep back straight — don't lean back!")
        if self.stage == "up" and self.angle > 100:
            self.form_errors.append("⚠️ Raise knee higher — above hip level!")

    # ════════════════════════════════════════════════════════════════════════
    # 8. TREE POSE (YOGA)
    # ════════════════════════════════════════════════════════════════════════
    def _detect_tree_pose(self, lm):
        """
        Balance on one leg — measures knee angle of raised leg.
        Hold for 5 seconds = 1 rep.
        """
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear = get_landmark_coords(lm, LEFT_EAR)
        r_knee_y = lm[RIGHT_KNEE].y
        l_hip_y  = lm[LEFT_HIP].y
        self.angle = calculate_angle(hip, knee, ankle)

        if r_knee_y < l_hip_y - 0.05:   # right knee raised above hip
            self.hold_count += 1
            self.stage = "holding"
            secs = self.hold_count // 10
            self.feedback = f"Hold... {secs}s — Keep balancing!"
            if self.hold_count >= 50:   # ~5 seconds at 10fps
                self.reps += 1; self.hold_count = 0
                self.feedback = f"Rep {self.reps} done! Switch legs."
        else:
            self.stage = "down"; self.hold_count = 0
            self.feedback = "Raise your right foot to inner left thigh."

        if calculate_angle(ear, shoulder, hip) < 155:
            self.form_errors.append("⚠️ Stand tall — keep spine straight!")
        spine_lateral = abs(lm[LEFT_SHOULDER].x - lm[LEFT_HIP].x)
        if spine_lateral > 0.15:
            self.form_errors.append("⚠️ Don't lean sideways — find your balance!")

    # ════════════════════════════════════════════════════════════════════════
    # 9. WARRIOR POSE (YOGA)
    # ════════════════════════════════════════════════════════════════════════
    def _detect_warrior_pose(self, lm):
        """
        Front knee bent at ~90°, back leg straight.
        Measures front knee angle (hip-knee-ankle).
        """
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ankle = get_landmark_coords(lm, LEFT_ANKLE)
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        ear = get_landmark_coords(lm, LEFT_EAR)
        l_wrist = get_landmark_coords(lm, LEFT_WRIST)
        r_wrist = get_landmark_coords(lm, RIGHT_WRIST)
        self.angle = calculate_angle(hip, knee, ankle)

        if self.angle > 150:
            self.stage = "up"; self.feedback = "Bend front knee into warrior stance."
        elif self.angle < 100 and self.stage == "up":
            self.stage = "down"; self.hold_count += 1
            secs = self.hold_count // 10
            self.feedback = f"Hold warrior pose... {secs}s"
            if self.hold_count >= 30:
                self.reps += 1; self.hold_count = 0; self.stage = "up"
                self.feedback = f"Rep {self.reps} done! Switch sides."
        elif self.stage == "down":
            self.hold_count += 1
            self.feedback = f"Hold... {self.hold_count // 10}s — Stay strong!"

        if calculate_angle(ear, shoulder, hip) < 150:
            self.form_errors.append("⚠️ Keep torso upright!")
        wrist_spread = abs(lm[LEFT_WRIST].x - lm[RIGHT_WRIST].x)
        if wrist_spread < 0.3 and self.stage == "down":
            self.form_errors.append("⚠️ Extend arms wider — shoulder height!")
        if self.stage == "down" and self.angle > 115:
            self.form_errors.append("⚠️ Bend front knee more — aim for 90°!")

    # ════════════════════════════════════════════════════════════════════════
    # 10. CAT-COW STRETCH (YOGA)
    # ════════════════════════════════════════════════════════════════════════
    def _detect_cat_cow_stretch(self, lm):
        """
        Back flexion/extension. Measures spine angle (shoulder-hip-knee).
        Arched back (cow) vs rounded back (cat).
        """
        shoulder = get_landmark_coords(lm, LEFT_SHOULDER)
        hip = get_landmark_coords(lm, LEFT_HIP)
        knee = get_landmark_coords(lm, LEFT_KNEE)
        ear = get_landmark_coords(lm, LEFT_EAR)
        self.angle = calculate_angle(shoulder, hip, knee)

        if self.angle > 160:
            self.stage = "cow"
            self.feedback = "Good cow pose! Now round your back (cat)."
        elif self.angle < 130 and self.stage == "cow":
            self.stage = "cat"; self.reps += 1
            self.feedback = f"Rep {self.reps} done! Now arch back (cow)."
        elif self.stage == "cat":
            self.feedback = "Arch your back into cow pose."
        else:
            self.feedback = "Start on all fours — arch your back first."

        ear_y = lm[LEFT_EAR].y
        hip_y = lm[LEFT_HIP].y
        if self.stage == "cow" and ear_y > hip_y:
            self.form_errors.append("⚠️ Lift your head up in cow pose!")
        if self.stage == "cat" and ear_y < hip_y - 0.1:
            self.form_errors.append("⚠️ Drop your head down in cat pose!")

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
            "hold_count":  self.hold_count,
        }