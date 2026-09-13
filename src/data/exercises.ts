import { Exercise, User } from '../types';

export const EXERCISES: Record<string, Exercise> = {
  squat: {
    id: 'squat',
    label: 'Squat',
    icon: '',
    target: 'Knee & Hip Rehab',
    type: 'physio',
    videoId: 'YaXPRqUwItQ',
    formChecks: ['Back angle', 'Knees over toes', 'Spine straight'],
    tip: 'Stand shoulder-width apart, keep chest up and core engaged!',
    limitations: ['knee_pain', 'hip_pain'],
    defaultTargetReps: 10,
    description: 'A fundamental rehabilitation exercise for restoring lower body kinetic chain mobility, quadriceps engagement, and gluteal stabilization.',
    primaryJoint: 'Knee Joint (Hip-Knee-Ankle)',
    idealAngleRange: '90° (at bottom) to 165° (at top)',
  },
  shoulder_raises: {
    id: 'shoulder_raises',
    label: 'Shoulder Raises',
    icon: '',
    target: 'Shoulder Rehab',
    type: 'physio',
    videoId: 'FeGNSMVFBHg',
    formChecks: ['Arms straight', 'Height control', 'Spine upright'],
    tip: 'Move slowly and with deliberate control — avoid swinging your torso!',
    limitations: ['shoulder_injury'],
    defaultTargetReps: 12,
    description: 'Isolates and strengthens the lateral deltoids and supraspinatus without placing excessive compressive stress on the glenohumeral joint.',
    primaryJoint: 'Glenohumeral Joint (Elbow-Shoulder-Hip)',
    idealAngleRange: '20° (down) to 85° (shoulder height)',
  },
  crossover_arm_stretch: {
    id: 'crossover_arm_stretch',
    label: 'Crossover Arm Stretch',
    icon: '',
    target: 'Shoulder Mobility',
    type: 'physio',
    videoId: '5bMBCOgFHug',
    formChecks: ['No shoulder shrug', 'Torso still', 'Hold 2 seconds'],
    tip: 'Hold each gentle stretch for a full 2 seconds across your chest!',
    limitations: ['shoulder_injury'],
    defaultTargetReps: 8,
    description: 'Stretches the posterior capsule and infraspinatus, restoring transverse adduction mobility for stiff or frozen shoulder rehabilitation.',
    primaryJoint: 'Arm Adduction (Elbow-Left Shoulder-Right Shoulder)',
    idealAngleRange: '150° (relaxed) to < 60° (adducted)',
  },
  lateral_walks: {
    id: 'lateral_walks',
    label: 'Lateral Walks',
    icon: '',
    target: 'Hip & Knee Rehab',
    type: 'physio',
    videoId: 'swFjPnGXFxk',
    formChecks: ['Knee bend athletic stance', 'Torso upright', 'Wide step'],
    tip: 'Stay low in an athletic stance throughout — do not stand upright between steps!',
    limitations: ['knee_pain'],
    defaultTargetReps: 10,
    description: 'Activates the gluteus medius and stabilizes the lateral pelvic stabilizers to prevent knee valgus collapse during ambulation.',
    primaryJoint: 'Pelvic Foot Stance (Spread vs Hip Width Ratio)',
    idealAngleRange: 'Ratio > 1.6x (out) to < 1.1x (in)',
  },
  lunges: {
    id: 'lunges',
    label: 'Lunges',
    icon: '',
    target: 'Leg Strength Rehab',
    type: 'physio',
    videoId: 'QOVaHwm-Q6U',
    formChecks: ['Knee alignment', 'Torso upright', 'Depth'],
    tip: 'Keep your front knee stacked directly above your ankle — step long!',
    limitations: ['knee_pain', 'hip_pain'],
    defaultTargetReps: 10,
    description: 'Unilateral rehabilitation movement challenging hip flexor length on the trailing leg while loading the quadriceps and hamstring on the lead leg.',
    primaryJoint: 'Lead Knee (Hip-Knee-Ankle)',
    idealAngleRange: '95° (bottom lunge) to 165° (upright)',
  },
  calf_raises: {
    id: 'calf_raises',
    label: 'Calf Raises',
    icon: '',
    target: 'Ankle & Calf Rehab',
    type: 'physio',
    videoId: 'J0DnG1_S92I',
    formChecks: ['Legs straight', 'No forward lean', 'Full plantarflexion'],
    tip: 'Lower heels all the way down between each rep and pause at top!',
    limitations: ['ankle_injury'],
    defaultTargetReps: 15,
    description: 'Restores gastrocnemius-soleus strength and Achilles tendon elastic stiffness following sprains or immobilization.',
    primaryJoint: 'Ankle Plantarflexion (Knee-Ankle-Foot Index)',
    idealAngleRange: '> 100° (neutral/low) to < 80° (tiptoe elevation)',
  },
  knee_raises: {
    id: 'knee_raises',
    label: 'Knee Raises',
    icon: '',
    target: 'Hip Flexor & Core',
    type: 'physio',
    videoId: 'RHrGLFDRRCY',
    formChecks: ['Back straight', 'Height above hip', 'Controlled tempo'],
    tip: 'Engage your lower abdominal core — avoid leaning backward when elevating!',
    limitations: ['hip_pain'],
    defaultTargetReps: 12,
    description: 'Dynamic hip flexion restoring psoas strength and single-leg standing balance during the gait swing phase.',
    primaryJoint: 'Hip Flexion (Shoulder-Hip-Knee)',
    idealAngleRange: '> 150° (standing) to < 90° (knee lifted)',
  },
  tree_pose: {
    id: 'tree_pose',
    label: 'Tree Pose',
    icon: '',
    target: 'Balance & Stability',
    type: 'yoga',
    videoId: 'wdln9qWYloU',
    formChecks: ['Spine straight', 'No lateral lean', 'Hold 5 seconds'],
    tip: 'Fix your visual gaze on one stationary point straight ahead to stabilize!',
    limitations: ['balance_issues'],
    defaultTargetReps: 4,
    description: 'Vrikshasana improves neuromuscular proprioception, ankle stabilizer endurance, and postural axial alignment.',
    primaryJoint: 'Single Leg Balance + Knee Height Hold',
    idealAngleRange: 'Raised foot to inner thigh, 5s sustained hold',
  },
  warrior_pose: {
    id: 'warrior_pose',
    label: 'Warrior Pose',
    icon: '',
    target: 'Leg & Core Strength',
    type: 'yoga',
    videoId: 'Mn6RSIRCV3w',
    formChecks: ['Torso upright', 'Arms wide at shoulder level', 'Deep knee bend'],
    tip: 'Keep the front knee tracking directly over your middle toes — hold steady!',
    limitations: ['knee_pain', 'hip_pain'],
    defaultTargetReps: 4,
    description: 'Virabhadrasana II develops isometric stamina in the hip abductors, quadriceps, and scapular retractors while expanding chest capacity.',
    primaryJoint: 'Front Knee Flexion (Hip-Knee-Ankle)',
    idealAngleRange: '~90° knee bend with wide arm span, 3s hold',
  },
  cat_cow_stretch: {
    id: 'cat_cow_stretch',
    label: 'Cat-Cow Stretch',
    icon: '',
    target: 'Spine Flexibility',
    type: 'yoga',
    videoId: 'kqnua4rHVVA',
    formChecks: ['Head position', 'Full lumbar arch', 'Full thoracic round'],
    tip: 'Inhale deeply as you arch into Cow; exhale fully as you round into Cat!',
    limitations: ['back_pain'],
    defaultTargetReps: 10,
    description: 'Chakravakasana coordinates spinal flexion and extension with diaphragmatic breathing, relieving facet joint stiffness and muscular spasm.',
    primaryJoint: 'Spinal Curvature (Shoulder-Hip-Knee axis)',
    idealAngleRange: '> 160° (Cow arch) to < 130° (Cat flexion)',
  },
};

export const LIMITATION_KEYWORDS: Record<string, string[]> = {
  knee_pain: ['knee', 'acl', 'pcl', 'meniscus', 'kneecap', 'patella', 'patellofemoral'],
  hip_pain: ['hip', 'groin', 'pelvis', 'bursitis', 'labrum'],
  shoulder_injury: ['shoulder', 'rotator', 'cuff', 'impingement', 'frozen shoulder', 'clavicle'],
  ankle_injury: ['ankle', 'achilles', 'sprain', 'plantar', 'heel'],
  back_pain: ['back', 'spine', 'lumbar', 'disc', 'scoliosis', 'sciatica', 'cervical'],
  balance_issues: ['vertigo', 'balance', 'dizziness', 'vestibular'],
};

export function getUserLimitations(user: Partial<User>): string[] {
  const limitations: string[] = [];
  const text = `${user.medicalConditions || ''} ${user.exerciseLimitations || ''} ${user.painLocation || ''}`.toLowerCase();

  for (const [key, keywords] of Object.entries(LIMITATION_KEYWORDS)) {
    if (keywords.some((kw) => text.includes(kw))) {
      limitations.push(key);
    }
  }
  return limitations;
}

export function isExerciseSafe(exerciseKey: string, user: Partial<User>): { safe: boolean; warning?: string } {
  const exercise = EXERCISES[exerciseKey];
  if (!exercise) return { safe: true };

  const userLim = getUserLimitations(user);
  const conflicting = exercise.limitations.filter((lim) => userLim.includes(lim));

  if (conflicting.length > 0) {
    const formatted = conflicting.map((c) => c.replace('_', ' ')).join(', ');
    return {
      safe: false,
      warning: `Caution: Your medical profile flags ${formatted}. Exercise with caution or consult your physical therapist.`,
    };
  }

  return { safe: true };
}

export function getRecommendedExercises(user: Partial<User>): Exercise[] {
  const userText = `${user.currentProblem || ''} ${user.rehabGoals || ''} ${user.painLocation || ''}`.toLowerCase();
  const all = Object.values(EXERCISES);

  if (!userText.trim()) {
    // Default rehabilitation starter set
    return [EXERCISES.squat, EXERCISES.shoulder_raises, EXERCISES.tree_pose, EXERCISES.cat_cow_stretch];
  }

  // Score exercises based on relevance and safety
  const scored = all.map((ex) => {
    let score = 0;
    const { safe } = isExerciseSafe(ex.id, user);
    if (!safe) score -= 5;

    if (userText.includes('knee') || userText.includes('leg') || userText.includes('squat')) {
      if (['squat', 'lunges', 'lateral_walks', 'calf_raises'].includes(ex.id)) score += 4;
    }
    if (userText.includes('shoulder') || userText.includes('arm') || userText.includes('neck')) {
      if (['shoulder_raises', 'crossover_arm_stretch'].includes(ex.id)) score += 4;
    }
    if (userText.includes('back') || userText.includes('spine') || userText.includes('posture')) {
      if (['cat_cow_stretch', 'tree_pose', 'squat'].includes(ex.id)) score += 4;
    }
    if (userText.includes('balance') || userText.includes('fall') || userText.includes('stability')) {
      if (['tree_pose', 'warrior_pose', 'knee_raises'].includes(ex.id)) score += 4;
    }
    if (userText.includes('walk') || userText.includes('mobility') || userText.includes('hip')) {
      if (['lateral_walks', 'knee_raises', 'squat'].includes(ex.id)) score += 3;
    }

    return { ex, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 4).map((s) => s.ex);
}
