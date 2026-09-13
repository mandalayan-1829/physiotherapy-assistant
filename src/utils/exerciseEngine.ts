import { ExerciseState, Landmark } from '../types';
import { calculateAngle, getLandmarkCoords, POSE_LANDMARKS } from './angleCalculator';
import { soundManager } from './audio';

export class ExerciseEngine {
  private exercise: string;
  private targetReps: number;
  private reps: number = 0;
  private stage: ExerciseState['stage'] = 'up';
  private angle: number = 0;
  private feedback: string = 'Stand in camera frame';
  private formErrors: string[] = [];
  private holdCount: number = 0;
  private holdSecondsRequired: number = 0;
  private completed: boolean = false;
  private lastRepCount: number = 0;

  constructor(exercise: string, targetReps: number = 10) {
    this.exercise = exercise;
    this.targetReps = targetReps;
    this.reset();
  }

  public reset() {
    this.reps = 0;
    this.stage = this.exercise === 'cat_cow_stretch' ? 'cow' : 'up';
    this.angle = 0;
    this.feedback = 'Stand in frame to begin';
    this.formErrors = [];
    this.holdCount = 0;
    this.completed = false;
    this.lastRepCount = 0;

    if (this.exercise === 'tree_pose') {
      this.holdSecondsRequired = 5;
    } else if (this.exercise === 'warrior_pose') {
      this.holdSecondsRequired = 3;
    } else {
      this.holdSecondsRequired = 0;
    }
  }

  public getState(): ExerciseState {
    return {
      exercise: this.exercise,
      reps: this.reps,
      targetReps: this.targetReps,
      stage: this.stage,
      angle: this.angle,
      feedback: this.feedback,
      formErrors: [...this.formErrors],
      formOk: this.formErrors.length === 0,
      holdCount: this.holdCount,
      holdSecondsRequired: this.holdSecondsRequired,
      isComplete: this.reps >= this.targetReps,
    };
  }

  public process(landmarks: Landmark[] | null): ExerciseState {
    if (!landmarks || landmarks.length === 0) {
      this.feedback = 'No body pose detected — align full body in frame';
      this.formErrors = [];
      return this.getState();
    }

    this.formErrors = [];

    switch (this.exercise) {
      case 'squat':
        this.detectSquat(landmarks);
        break;
      case 'shoulder_raises':
        this.detectShoulderRaises(landmarks);
        break;
      case 'crossover_arm_stretch':
        this.detectCrossoverArmStretch(landmarks);
        break;
      case 'lateral_walks':
        this.detectLateralWalks(landmarks);
        break;
      case 'lunges':
        this.detectLunges(landmarks);
        break;
      case 'calf_raises':
        this.detectCalfRaises(landmarks);
        break;
      case 'knee_raises':
        this.detectKneeRaises(landmarks);
        break;
      case 'tree_pose':
        this.detectTreePose(landmarks);
        break;
      case 'warrior_pose':
        this.detectWarriorPose(landmarks);
        break;
      case 'cat_cow_stretch':
        this.detectCatCowStretch(landmarks);
        break;
      default:
        this.detectSquat(landmarks);
        break;
    }

    // Trigger audio cues on new rep completion
    if (this.reps > this.lastRepCount) {
      this.lastRepCount = this.reps;
      soundManager.playRepSuccess();
      if (this.reps >= this.targetReps && !this.completed) {
        this.completed = true;
        soundManager.playCompleteFanfare();
      }
    }

    return this.getState();
  }

  // 1. SQUAT
  private detectSquat(lm: Landmark[]) {
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);

    this.angle = calculateAngle(hip, knee, ankle);

    if (this.angle < 95) {
      this.stage = 'down';
      this.feedback = 'Good squat depth! Push up through your heels.';
    } else if (this.angle > 160 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} complete! Lower into the next squat.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Bend knees and push hips back into a squat.';
    }

    const backAngle = calculateAngle(shoulder, hip, [hip[0], hip[1] + 0.1]);
    if (backAngle > 70 && this.stage === 'down') {
      this.formErrors.push('Back too bent — keep chest proud and upright');
    }

    const kneeX = lm[POSE_LANDMARKS.LEFT_KNEE].x;
    const footX = lm[POSE_LANDMARKS.LEFT_FOOT_INDEX].x;
    if (this.stage === 'down' && Math.abs(kneeX - footX) > 0.08) {
      this.formErrors.push('Knees extending past toes — shift hips backward');
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Maintain neutral cervical spine');
    }
  }

  // 2. SHOULDER RAISES
  private detectShoulderRaises(lm: Landmark[]) {
    const elbow = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ELBOW);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const wrist = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_WRIST);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);

    this.angle = calculateAngle(elbow, shoulder, hip);

    if (this.angle < 30) {
      this.stage = 'down';
      this.feedback = 'Raise your arms outwards to shoulder height.';
    } else if (this.angle > 80 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} done! Lower with controlled tempo.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Lower arms fully to sides before elevating again.';
    }

    if (calculateAngle(shoulder, elbow, wrist) < 150) {
      this.formErrors.push('Keep arms extended — do not bend elbows');
    }

    if (lm[POSE_LANDMARKS.LEFT_WRIST].y < lm[POSE_LANDMARKS.LEFT_SHOULDER].y - 0.05) {
      this.formErrors.push('Do not lift wrist above shoulder horizontal plane');
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Keep spine straight — avoid swinging back');
    }
  }

  // 3. CROSSOVER ARM STRETCH
  private detectCrossoverArmStretch(lm: Landmark[]) {
    const lElbow = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ELBOW);
    const lShoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const rShoulder = getLandmarkCoords(lm, POSE_LANDMARKS.RIGHT_SHOULDER);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);

    this.angle = calculateAngle(lElbow, lShoulder, rShoulder);

    if (this.angle > 145) {
      this.stage = 'down';
      this.feedback = 'Draw arm smoothly across your chest.';
    } else if (this.angle < 65 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} completed! Relax arm back to side.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Return arm to neutral before the next adduction.';
    }

    if (lm[POSE_LANDMARKS.LEFT_SHOULDER].y - lm[POSE_LANDMARKS.LEFT_EAR].y < 0.08) {
      this.formErrors.push('Relax trap muscles — do not shrug shoulder');
    }

    if (calculateAngle(ear, lShoulder, hip) < 145) {
      this.formErrors.push('Keep thoracic torso squared — avoid twisting');
    }
  }

  // 4. LATERAL WALKS
  private detectLateralWalks(lm: Landmark[]) {
    const hipW = Math.abs(lm[POSE_LANDMARKS.LEFT_HIP].x - lm[POSE_LANDMARKS.RIGHT_HIP].x);
    const footSpread = Math.abs(lm[POSE_LANDMARKS.LEFT_ANKLE].x - lm[POSE_LANDMARKS.RIGHT_ANKLE].x);
    const ratio = footSpread / (hipW + 1e-6);

    this.angle = Math.round(ratio * 100);

    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);

    if (ratio > 1.55) {
      this.stage = 'down';
      this.feedback = 'Great wide lateral step! Now step back in.';
    } else if (ratio < 1.15 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} completed! Step out wide again.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Take a wide athletic step out laterally.';
    }

    if (calculateAngle(hip, knee, ankle) > 170) {
      this.formErrors.push('Keep knees slightly bent in athletic squat stance');
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Keep torso upright — avoid excessive pitch forward');
    }
  }

  // 5. LUNGES
  private detectLunges(lm: Landmark[]) {
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);

    this.angle = calculateAngle(hip, knee, ankle);

    if (this.angle < 100) {
      this.stage = 'down';
      this.feedback = 'Solid depth! Drive upward through the front heel.';
    } else if (this.angle > 155 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} done! Step and lower into next lunge.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Step forward and drop the back knee toward the ground.';
    }

    if (this.stage === 'down' && Math.abs(lm[POSE_LANDMARKS.LEFT_KNEE].x - lm[POSE_LANDMARKS.LEFT_FOOT_INDEX].x) > 0.08) {
      this.formErrors.push('Front knee past toes — lengthen stride distance');
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Keep upper torso perpendicular to the floor');
    }
  }

  // 6. CALF RAISES
  private detectCalfRaises(lm: Landmark[]) {
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const footIndex = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_FOOT_INDEX);
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);

    this.angle = calculateAngle(knee, ankle, footIndex);

    if (this.angle > 100) {
      this.stage = 'down';
      this.feedback = 'Rise up onto your balls of feet and toes!';
    } else if (this.angle < 80 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} done! Slowly lower heels back down.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Lower heels completely to floor before next raise.';
    }

    if (calculateAngle(hip, knee, ankle) < 160) {
      this.formErrors.push('Keep legs fully extended — do not bend knees');
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Stand tall — avoid leaning body forward');
    }
  }

  // 7. KNEE RAISES
  private detectKneeRaises(lm: Landmark[]) {
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);

    this.angle = calculateAngle(shoulder, hip, knee);

    if (this.angle > 150) {
      this.stage = 'down';
      this.feedback = 'Drive your knee upward to hip height!';
    } else if (this.angle < 95 && this.stage === 'down') {
      this.stage = 'up';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} completed! Lower foot with control.`;
    } else if (this.stage === 'up') {
      this.feedback = 'Lower leg fully before elevating next rep.';
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Brace abdominal core — do not lean backward');
    }
  }

  // 8. TREE POSE (YOGA)
  private detectTreePose(lm: Landmark[]) {
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);

    this.angle = calculateAngle(hip, knee, ankle);

    const rKneeY = lm[POSE_LANDMARKS.RIGHT_KNEE].y;
    const lHipY = lm[POSE_LANDMARKS.LEFT_HIP].y;

    // Right knee lifted above hip
    if (rKneeY < lHipY - 0.04) {
      this.holdCount += 1;
      this.stage = 'holding';
      const secondsHeld = Math.floor(this.holdCount / 10);
      this.feedback = `Hold balance... ${secondsHeld}s / 5s — Focus your gaze!`;

      if (this.holdCount % 10 === 0) {
        soundManager.playTick();
      }

      if (this.holdCount >= 50) {
        this.reps += 1;
        this.holdCount = 0;
        this.feedback = `Rep ${this.reps} complete! Rest or alternate leg.`;
      }
    } else {
      this.stage = 'down';
      this.holdCount = 0;
      this.feedback = 'Place right foot against inner calf or thigh and balance.';
    }

    if (calculateAngle(ear, shoulder, hip) < 155) {
      this.formErrors.push('Lengthen axial spine — keep shoulders level');
    }

    const spineLateral = Math.abs(lm[POSE_LANDMARKS.LEFT_SHOULDER].x - lm[POSE_LANDMARKS.LEFT_HIP].x);
    if (spineLateral > 0.14) {
      this.formErrors.push('Avoid lateral torso lean — balance through center');
    }
  }

  // 9. WARRIOR POSE (YOGA)
  private detectWarriorPose(lm: Landmark[]) {
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);
    const ankle = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_ANKLE);
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const ear = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_EAR);
    const lWrist = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_WRIST);
    const rWrist = getLandmarkCoords(lm, POSE_LANDMARKS.RIGHT_WRIST);

    this.angle = calculateAngle(hip, knee, ankle);

    if (this.angle > 145) {
      this.stage = 'up';
      this.feedback = 'Deepen your stance into a 90° warrior bend.';
    } else if (this.angle < 105) {
      this.stage = 'down';
      this.holdCount += 1;
      const secondsHeld = Math.floor(this.holdCount / 10);
      this.feedback = `Hold Warrior II... ${secondsHeld}s / 3s — Stay strong!`;

      if (this.holdCount % 10 === 0) {
        soundManager.playTick();
      }

      if (this.holdCount >= 30) {
        this.reps += 1;
        this.holdCount = 0;
        this.stage = 'up';
        this.feedback = `Rep ${this.reps} done! Straighten lead knee to reset.`;
      }
    }

    if (calculateAngle(ear, shoulder, hip) < 150) {
      this.formErrors.push('Keep torso vertical — do not lean over front thigh');
    }

    const wristSpread = Math.abs(lWrist[0] - rWrist[0]);
    if (wristSpread < 0.3 && this.stage === 'down') {
      this.formErrors.push('Extend arms wide and parallel to the floor');
    }
  }

  // 10. CAT-COW STRETCH (YOGA)
  private detectCatCowStretch(lm: Landmark[]) {
    const shoulder = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_SHOULDER);
    const hip = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_HIP);
    const knee = getLandmarkCoords(lm, POSE_LANDMARKS.LEFT_KNEE);

    this.angle = calculateAngle(shoulder, hip, knee);

    if (this.angle > 158) {
      this.stage = 'cow';
      this.feedback = 'Cow pose: arch back gently and inhale. Now round back (Cat).';
    } else if (this.angle < 132 && this.stage === 'cow') {
      this.stage = 'cat';
      this.reps += 1;
      this.feedback = `Rep ${this.reps} done! Exhale and round spine upward.`;
    } else if (this.stage === 'cat') {
      this.feedback = 'Arch back toward floor into Cow pose.';
    }

    const earY = lm[POSE_LANDMARKS.LEFT_EAR].y;
    const hipY = lm[POSE_LANDMARKS.LEFT_HIP].y;
    if (this.stage === 'cow' && earY > hipY) {
      this.formErrors.push('Lift chin slightly to complete Cow pose extension');
    }
  }
}
