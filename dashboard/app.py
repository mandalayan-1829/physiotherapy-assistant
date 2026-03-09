"""
dashboard/app.py
----------------
AI Physiotherapy Dashboard — Human-friendly UI with interactive flow.
Run with: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import streamlit as st
import mediapipe as mp
from core.pose_detector import PoseDetector
from core.exercise_detector import ExerciseDetector

st.set_page_config(page_title="AI Physiotherapy", page_icon="🏥", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-title    { font-size: 2.8rem; font-weight: 700; text-align: center; margin-bottom: 0; }
    .sub-title    { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 30px; }
    .ex-card      { border: 1px solid #333; border-radius: 14px; padding: 22px 16px;
                    text-align: center; background: #111827; transition: 0.2s; }
    .stat-box     { background: #1a1a2e; border-radius: 12px; padding: 16px 20px;
                    margin-bottom: 12px; border-left: 4px solid #4f8ef7; }
    .stat-label   { font-size: 0.78rem; color: #888; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 4px; }
    .stat-value   { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .good-form    { background: #0d2b1f; border-left: 4px solid #22c55e;
                    border-radius: 10px; padding: 12px 16px; color: #22c55e;
                    font-weight: 600; font-size: 1rem; }
    .bad-form     { background: #2b0d0d; border-left: 4px solid #ef4444;
                    border-radius: 10px; padding: 12px 16px; color: #ef4444;
                    font-weight: 600; font-size: 0.9rem; }
    .feedback-box { background: #1a1f2e; border-radius: 10px; padding: 14px 18px;
                    color: #93c5fd; font-size: 1rem; border-left: 4px solid #3b82f6; }
    .start-btn    { background: #4f8ef7; color: white; border: none; border-radius: 50px;
                    padding: 16px 48px; font-size: 1.2rem; cursor: pointer;
                    font-weight: 600; margin: 20px auto; display: block; }
    .nav-btn      { font-size: 0.85rem; }
    .progress-text { font-size: 0.85rem; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Exercise data ─────────────────────────────────────────────────────────────
EXERCISES = {
    "squat": {
        "label": "Squat", "icon": "🦵", "target": "Knee & Hip Rehab",
        "description": "Strengthens knees and hips. Keep your back straight and knees behind toes.",
        "video_id": "YaXPRqUwItQ",
        "form_checks": ["Back angle", "Knees over toes", "Spine alignment"],
        "tip": "Stand shoulder-width apart, keep chest up!",
    },
    "shoulder_raises": {
        "label": "Shoulder Raises", "icon": "💪", "target": "Shoulder Rehab",
        "description": "Improves shoulder mobility. Raise arms to shoulder height, elbows straight.",
        "video_id": "FeGNSMVFBHg",
        "form_checks": ["Arms straight", "Height control", "Spine straight"],
        "tip": "Move slowly and in control — no swinging!",
    },
    "crossover_arm_stretch": {
        "label": "Crossover Arm Stretch", "icon": "🤸", "target": "Shoulder Mobility",
        "description": "Stretches shoulder muscles. Bring arm across chest, hold for 2 seconds.",
        "video_id": "5bMBCOgFHug",
        "form_checks": ["No shoulder shrug", "Torso still"],
        "tip": "Hold each stretch for a full 2 seconds!",
    },
    "lateral_walks": {
        "label": "Lateral Walks", "icon": "🚶", "target": "Hip & Knee Rehab",
        "description": "Strengthens hips and knees. Step sideways keeping knees slightly bent.",
        "video_id": "swFjPnGXFxk",
        "form_checks": ["Knee bend", "Torso upright"],
        "tip": "Stay low throughout — don't stand up between steps!",
    },
    "lunges": {
        "label": "Lunges", "icon": "🏃", "target": "Leg Strength Rehab",
        "description": "Builds leg strength. Step forward and lower back knee toward the floor.",
        "video_id": "QOVaHwm-Q6U",
        "form_checks": ["Knee alignment", "Torso upright", "Depth"],
        "tip": "Keep your front knee above your ankle!",
    },
    "calf_raises": {
        "label": "Calf Raises", "icon": "👟", "target": "Ankle & Calf Rehab",
        "description": "Strengthens calves and ankles. Rise up on tiptoes, lower slowly.",
        "video_id": "J0DnG1_S92I",
        "form_checks": ["Legs straight", "No forward lean"],
        "tip": "Lower your heels ALL the way down each rep!",
    },
}

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "page": "welcome", "selected_exercise": None,
    "detector": None, "pose_detector": None,
    "history": [], "target_reps": 10,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WELCOME
# ════════════════════════════════════════════════════════════════════════════
def show_welcome():
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown("<p class='big-title'>👋 Hello!</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Welcome to your personal AI Physiotherapy Assistant</p>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem;text-align:center'>📹</div>
            <div style='text-align:center;margin-top:8px;font-weight:600'>Real-time Tracking</div>
            <div style='text-align:center;color:#888;font-size:0.85rem;margin-top:4px'>
            Your camera watches your movements live</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem;text-align:center'>🩺</div>
            <div style='text-align:center;margin-top:8px;font-weight:600'>Posture Correction</div>
            <div style='text-align:center;color:#888;font-size:0.85rem;margin-top:4px'>
            Instant warning if your form is wrong</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem;text-align:center'>🔢</div>
            <div style='text-align:center;margin-top:8px;font-weight:600'>Auto Rep Counter</div>
            <div style='text-align:center;color:#888;font-size:0.85rem;margin-top:4px'>
            Counts your reps automatically</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.15rem;color:#ccc;'>Ready to begin your session?</p>",
                unsafe_allow_html=True)

    col = st.columns([2, 1, 2])
    with col[1]:
        if st.button("🚀  Let's Begin!", use_container_width=True):
            st.session_state.page = "select_exercise"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXERCISE SELECTION
# ════════════════════════════════════════════════════════════════════════════
def show_exercise_selection():
    st.markdown("<p class='big-title' style='font-size:2rem'>Which exercise would you like to do today?</p>",
                unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Pick one below — your AI coach will guide you through it</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    keys   = list(EXERCISES.keys())
    chunks = [keys[i:i+3] for i in range(0, len(keys), 3)]

    for chunk in chunks:
        cols = st.columns(3)
        for col, key in zip(cols, chunk):
            ex = EXERCISES[key]
            with col:
                st.markdown(f"""
                <div class='ex-card'>
                    <div style='font-size:2.8rem'>{ex['icon']}</div>
                    <div style='font-size:1.1rem;font-weight:700;margin:8px 0 2px'>{ex['label']}</div>
                    <div style='font-size:0.8rem;color:#4f8ef7;margin-bottom:8px'>🎯 {ex['target']}</div>
                    <div style='font-size:0.82rem;color:#aaa;margin-bottom:12px'>{ex['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Choose  {ex['icon']}  {ex['label']}", key=f"sel_{key}",
                             use_container_width=True):
                    st.session_state.selected_exercise = key
                    st.session_state.page = "setup"
                    st.rerun()

    st.markdown("---")
    if st.button("← Back to Home"):
        st.session_state.page = "welcome"
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SETUP
# ════════════════════════════════════════════════════════════════════════════
def show_setup():
    key = st.session_state.selected_exercise
    ex  = EXERCISES[key]

    st.markdown(f"<p class='big-title' style='font-size:2rem'>{ex['icon']}  Get Ready for {ex['label']}</p>",
                unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>Watch the video, set your reps, then hit Start!</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    col_vid, col_right = st.columns([2, 1])

    with col_vid:
        st.markdown("#### 🎥 Watch how it's done")
        st.markdown(
            f'<iframe width="100%" height="300" '
            f'src="https://www.youtube.com/embed/{ex["video_id"]}" '
            f'frameborder="0" allowfullscreen></iframe>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class='feedback-box' style='margin-top:12px'>
            💡 <b>Pro tip:</b> {ex['tip']}
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("#### ⚙️ How many reps?")
        target = st.slider("", min_value=5, max_value=50, value=10, step=5,
                           label_visibility="collapsed")
        st.session_state.target_reps = target

        st.markdown(f"""
        <div class='stat-box' style='margin-top:16px'>
            <div class='stat-label'>Your target</div>
            <div class='stat-value'>{target} reps</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🩺 What we'll check")
        for check in ex["form_checks"]:
            st.markdown(f"✅ {check}")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("▶️  Start Exercise", use_container_width=True):
            st.session_state.detector      = ExerciseDetector(exercise=key)
            st.session_state.pose_detector = PoseDetector()
            st.session_state.history       = []
            st.session_state.page          = "tracking"
            st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("← Choose Different Exercise", use_container_width=True):
            st.session_state.page = "select_exercise"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE TRACKING
# ════════════════════════════════════════════════════════════════════════════
def show_tracking():
    key      = st.session_state.selected_exercise
    ex       = EXERCISES[key]
    detector = st.session_state.detector
    pose_det = st.session_state.pose_detector
    target   = st.session_state.target_reps

    # ── Top nav ───────────────────────────────────────────────────────────
    nav1, nav2, nav3, nav4 = st.columns([2, 1, 1, 1])
    with nav1:
        st.markdown(f"### {ex['icon']} {ex['label']}")
    with nav2:
        if st.button("🔄 Reset", use_container_width=True):
            detector.reset()
            st.session_state.history = []
            st.rerun()
    with nav3:
        if st.button("🔀 Switch", use_container_width=True):
            st.session_state.page = "select_exercise"
            st.rerun()
    with nav4:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()

    st.markdown("---")

    col_video, col_stats = st.columns([4, 2])

    # ── Stats panel ───────────────────────────────────────────────────────
    with col_stats:
        st.markdown("#### 📊 Live Stats")
        reps_display     = st.empty()
        progress_display = st.empty()

        c1, c2 = st.columns(2)
        with c1:
            stage_display = st.empty()
        with c2:
            angle_display = st.empty()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("#### 💬 Feedback")
        feedback_display = st.empty()

        st.markdown("#### 🩺 Posture")
        posture_display  = st.empty()

        st.markdown("#### 📋 Rep History")
        history_display  = st.empty()

    # ── Camera panel ──────────────────────────────────────────────────────
    with col_video:
        camera_on = st.button("📸  Click here to Start your Exercise Session",
                              use_container_width=True)
        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False
        if camera_on:
            st.session_state.cam_running = True

        frame_placeholder = st.empty()

        if not st.session_state.cam_running:
            st.markdown("""
            <div style='background:#111827;border-radius:14px;padding:80px 20px;
                        text-align:center;margin-top:10px;border:2px dashed #333'>
                <div style='font-size:3.5rem'>📸</div>
                <div style='font-size:1.2rem;color:#888;margin-top:14px'>
                    Click the button above to start your camera</div>
                <div style='font-size:0.85rem;color:#555;margin-top:8px'>
                    Step back so your <b>full body</b> is visible in frame</div>
            </div>""", unsafe_allow_html=True)

    # ── Camera loop ───────────────────────────────────────────────────────
    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Check your webcam is connected.")
            st.session_state.cam_running = False
            st.stop()

        prev_reps = detector.reps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, landmarks = pose_det.find_pose(frame)
            result = detector.process(landmarks)

            h, w, _ = frame.shape

            # Angle label on joint
            if landmarks:
                mp_pose = mp.solutions.pose.PoseLandmark
                knee_lm = landmarks[mp_pose.LEFT_KNEE.value]
                cx, cy  = int(knee_lm.x * w), int(knee_lm.y * h)
                cv2.putText(frame, f"{result['angle']}deg",
                            (cx - 25, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Form banner — small strip at bottom
            if result["form_ok"]:
                banner_color = (0, 180, 70)
                banner_text  = "GOOD FORM"
            else:
                banner_color = (40, 40, 200)
                banner_text  = "FIX FORM"

            cv2.rectangle(frame, (0, h - 36), (w, h), (15, 15, 15), -1)
            cv2.putText(frame, banner_text, (12, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

            if result["form_errors"]:
                err_text = result["form_errors"][0].replace("⚠️ ", "")
                cv2.putText(frame, err_text, (int(w * 0.28), h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

            # Reps overlay — smaller, top left
            cv2.rectangle(frame, (0, 0), (180, 52), (15, 15, 15), -1)
            cv2.putText(frame, f"REPS  {result['reps']}/{target}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            cv2.putText(frame, result["stage"].upper(), (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # ── Stat boxes ────────────────────────────────────────────────
            progress = min(result["reps"] / target, 1.0)
            pct      = int(progress * 100)

            reps_display.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Reps Done</div>
                <div class='stat-value'>{result['reps']}
                    <span style='font-size:1.1rem;color:#888'>/ {target}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            progress_display.progress(progress, text=f"{pct}% of target complete")

            stage_display.markdown(f"""
            <div class='stat-box' style='border-color:#a855f7'>
                <div class='stat-label'>Stage</div>
                <div class='stat-value' style='font-size:1.5rem'>
                    {result['stage'].upper()}
                </div>
            </div>""", unsafe_allow_html=True)

            angle_display.markdown(f"""
            <div class='stat-box' style='border-color:#f59e0b'>
                <div class='stat-label'>Joint Angle</div>
                <div class='stat-value' style='font-size:1.5rem'>
                    {result['angle']}°
                </div>
            </div>""", unsafe_allow_html=True)

            feedback_display.markdown(f"""
            <div class='feedback-box'>
                💬 {result['feedback']}
            </div>""", unsafe_allow_html=True)

            # ── Posture box ───────────────────────────────────────────────
            if result["form_ok"]:
                posture_display.markdown("""
                <div class='good-form'>✅ Great form — keep it up!</div>
                """, unsafe_allow_html=True)
            else:
                errors = "<br>".join(result["form_errors"])
                posture_display.markdown(f"""
                <div class='bad-form'>{errors}</div>
                """, unsafe_allow_html=True)

            # ── History ───────────────────────────────────────────────────
            if result["reps"] > prev_reps:
                st.session_state.history.append({
                    "Rep":   result["reps"],
                    "Angle": f"{result['angle']}°",
                    "Form":  "✅" if result["form_ok"] else "⚠️",
                })
                prev_reps = result["reps"]

            if st.session_state.history:
                history_display.dataframe(
                    st.session_state.history,
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Done ──────────────────────────────────────────────────────
            if result["reps"] >= target:
                cap.release()
                st.session_state.cam_running = False
                st.success(f"🎉 You completed {target} reps of {ex['label']}! Great work!")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔀 Try Another Exercise", use_container_width=True):
                        st.session_state.page = "select_exercise"
                        st.rerun()
                with c2:
                    if st.button("🔁 Repeat This Exercise", use_container_width=True):
                        detector.reset()
                        st.session_state.history = []
                        st.rerun()
                break

        cap.release()


# ── Router ────────────────────────────────────────────────────────────────────
pages = {
    "welcome":          show_welcome,
    "select_exercise":  show_exercise_selection,
    "setup":            show_setup,
    "tracking":         show_tracking,
}
pages[st.session_state.page]()