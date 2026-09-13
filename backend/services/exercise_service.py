"""
Exercise metadata — extracted from Streamlit app.py.
Provides exercise definitions, safety checks, and medical limitation logic.
"""

from typing import Optional


EXERCISES = {
    # ── Physiotherapy ──────────────────────────────────────────────────────
    "squat": {
        "id": "squat",
        "label": "Squat",
        "icon": "🦵",
        "target": "Knee & Hip Rehab",
        "type": "physio",
        "video_id": "YaXPRqUwItQ",
        "form_checks": ["Back angle", "Knees over toes", "Spine"],
        "tip": "Stand shoulder-width apart, keep chest up!",
        "limitations": ["knee_pain", "hip_pain"],
        "difficulty": "moderate",
        "recommended_reps": 10,
        "description": "A fundamental lower body exercise targeting the quadriceps, hamstrings, and glutes. Essential for knee and hip rehabilitation.",
    },
    "shoulder_raises": {
        "id": "shoulder_raises",
        "label": "Shoulder Raises",
        "icon": "💪",
        "target": "Shoulder Rehab",
        "type": "physio",
        "video_id": "FeGNSMVFBHg",
        "form_checks": ["Arms straight", "Height control", "Spine"],
        "tip": "Move slowly and in control — no swinging!",
        "limitations": ["shoulder_injury"],
        "difficulty": "easy",
        "recommended_reps": 10,
        "description": "Lateral arm raises to strengthen the shoulder muscles and improve rotator cuff stability.",
    },
    "crossover_arm_stretch": {
        "id": "crossover_arm_stretch",
        "label": "Crossover Arm Stretch",
        "icon": "🤸",
        "target": "Shoulder Mobility",
        "type": "physio",
        "video_id": "5bMBCOgFHug",
        "form_checks": ["No shrug", "Torso still"],
        "tip": "Hold each stretch for a full 2 seconds!",
        "limitations": ["shoulder_injury"],
        "difficulty": "easy",
        "recommended_reps": 10,
        "description": "A cross-body shoulder stretch that improves flexibility and range of motion in the posterior shoulder.",
    },
    "lateral_walks": {
        "id": "lateral_walks",
        "label": "Lateral Walks",
        "icon": "🚶",
        "target": "Hip & Knee Rehab",
        "type": "physio",
        "video_id": "swFjPnGXFxk",
        "form_checks": ["Knee bend", "Torso upright"],
        "tip": "Stay low throughout — don't stand between steps!",
        "limitations": ["knee_pain"],
        "difficulty": "moderate",
        "recommended_reps": 10,
        "description": "Side-stepping exercise that activates hip abductors and improves lateral stability.",
    },
    "lunges": {
        "id": "lunges",
        "label": "Lunges",
        "icon": "🏃",
        "target": "Leg Strength Rehab",
        "type": "physio",
        "video_id": "QOVaHwm-Q6U",
        "form_checks": ["Knee alignment", "Torso", "Depth"],
        "tip": "Keep front knee directly above ankle!",
        "limitations": ["knee_pain", "hip_pain"],
        "difficulty": "moderate",
        "recommended_reps": 10,
        "description": "Forward lunges that build single-leg strength, balance, and coordination.",
    },
    "calf_raises": {
        "id": "calf_raises",
        "label": "Calf Raises",
        "icon": "👟",
        "target": "Ankle & Calf Rehab",
        "type": "physio",
        "video_id": "J0DnG1_S92I",
        "form_checks": ["Legs straight", "No forward lean"],
        "tip": "Lower heels ALL the way down each rep!",
        "limitations": ["ankle_injury"],
        "difficulty": "easy",
        "recommended_reps": 15,
        "description": "Standing calf raises to strengthen the gastrocnemius and soleus muscles for ankle stability.",
    },
    "knee_raises": {
        "id": "knee_raises",
        "label": "Knee Raises",
        "icon": "🦿",
        "target": "Hip Flexor & Core",
        "type": "physio",
        "video_id": "RHrGLFDRRCY",
        "form_checks": ["Back straight", "Height", "Control"],
        "tip": "Engage your core — don't lean backward!",
        "limitations": ["hip_pain"],
        "difficulty": "easy",
        "recommended_reps": 10,
        "description": "Alternating knee raises that strengthen hip flexors and core stability.",
    },
    # ── Yoga ───────────────────────────────────────────────────────────────
    "tree_pose": {
        "id": "tree_pose",
        "label": "Tree Pose",
        "icon": "🌳",
        "target": "Balance & Stability",
        "type": "yoga",
        "video_id": "wdln9qWYloU",
        "form_checks": ["Spine straight", "No lean", "Balance"],
        "tip": "Fix your gaze on one spot to help balance!",
        "limitations": ["balance_issues"],
        "difficulty": "moderate",
        "recommended_reps": 5,
        "description": "A standing balance pose that strengthens the legs, core, and improves overall stability.",
    },
    "warrior_pose": {
        "id": "warrior_pose",
        "label": "Warrior Pose",
        "icon": "⚔️",
        "target": "Leg & Core Strength",
        "type": "yoga",
        "video_id": "Mn6RSIRCV3w",
        "form_checks": ["Torso upright", "Arms wide", "Knee bend"],
        "tip": "Front knee tracks over front foot — not inward!",
        "limitations": ["knee_pain", "hip_pain"],
        "difficulty": "moderate",
        "recommended_reps": 5,
        "description": "A powerful standing pose that builds strength in the legs, opens the hips and chest.",
    },
    "cat_cow_stretch": {
        "id": "cat_cow_stretch",
        "label": "Cat-Cow Stretch",
        "icon": "🐄",
        "target": "Spine Flexibility",
        "type": "yoga",
        "video_id": "kqnua4rHVVA",
        "form_checks": ["Head position", "Full arch", "Full round"],
        "tip": "Breathe in for cow, breathe out for cat!",
        "limitations": ["back_pain"],
        "difficulty": "easy",
        "recommended_reps": 10,
        "description": "A flowing spinal movement that improves flexibility and relieves back tension.",
    },
}

# Limitation keywords to check against user profile
LIMITATION_KEYWORDS = {
    "knee_pain":       ["knee", "acl", "pcl", "meniscus", "kneecap"],
    "hip_pain":        ["hip", "groin", "pelvis"],
    "shoulder_injury": ["shoulder", "rotator", "cuff"],
    "ankle_injury":    ["ankle", "achilles"],
    "back_pain":       ["back", "spine", "lumbar", "disc", "scoliosis"],
    "balance_issues":  ["vertigo", "balance", "dizziness"],
}


def get_all_exercises() -> list[dict]:
    """Return all exercise definitions."""
    return list(EXERCISES.values())


def get_exercise(exercise_id: str) -> Optional[dict]:
    """Return a single exercise definition by ID."""
    return EXERCISES.get(exercise_id)


def get_user_limitations(user: dict) -> list[str]:
    """Check user's medical conditions against known limitation categories."""
    limitations = []
    text = (
        user.get("medical_conditions", "") + " " + user.get("exercise_limitations", "")
    ).lower()
    for key, keywords in LIMITATION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            limitations.append(key)
    return limitations


def is_exercise_safe(exercise_id: str, user: dict) -> tuple[bool, str]:
    """Check if an exercise is safe for the user based on medical profile.
    Returns (safe, warning_message).
    """
    exercise = EXERCISES.get(exercise_id)
    if not exercise:
        return False, f"Exercise '{exercise_id}' not found."

    user_lims = get_user_limitations(user)
    ex_lims = exercise.get("limitations", [])
    conflicts = [l for l in ex_lims if l in user_lims]

    if conflicts:
        conflict_str = ", ".join(c.replace("_", " ") for c in conflicts)
        return (
            False,
            f"⚠️ Based on your medical profile, this exercise may affect: "
            f"**{conflict_str}**. Proceed only if your doctor approves.",
        )
    return True, ""


def generate_guardian_whatsapp_link(user: dict, alert_msg: str) -> tuple[Optional[str], str]:
    """Generate a WhatsApp link to send an alert to the user's guardian."""
    import urllib.parse

    number = (
        user.get("guardian_whatsapp", "")
        .replace("+", "")
        .replace(" ", "")
        .replace("-", "")
    )
    if not number:
        return None, "No guardian WhatsApp number set."
    text = f"🚨 PhysioAI Alert for {user['name']}: {alert_msg}"
    encoded = urllib.parse.quote(text)
    return f"https://wa.me/{number}?text={encoded}", text
