"""
dashboard/app.py
----------------
AI Physiotherapy Dashboard with:
- Login / Signup
- Exercise tracking with pose detection
- Diet tracker
- Progress page
- Notes
Run with: streamlit run dashboard/app.py
"""
 
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
import cv2
import streamlit as st
import mediapipe as mp
from datetime import datetime
from core.pose_detector import PoseDetector
from core.exercise_detector import ExerciseDetector
from core.database import (
    create_user, login_user, get_user, update_user_profile,
    save_session, get_user_sessions, get_sessions_summary, get_recent_sessions,
    add_diet_entry, get_diet_today, get_diet_all, delete_diet_entry,
    add_note, get_notes, delete_note,
    add_doctor, get_all_doctors, get_doctor, delete_doctor,
    book_appointment, get_user_appointments, get_all_appointments,
    update_appointment_status, cancel_appointment,
    send_message, get_conversation, get_all_conversations, get_unread_count,
    log_guardian_alert, get_guardian_alerts,
)
 
ADMIN_CODE = "admin@physio123"
 
 
st.set_page_config(page_title="PhysioAI", page_icon="🏥", layout="wide")
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
 
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1e1e2e !important;
        border-right: 1px solid #2d2d44 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] button {
        background: #2d2d44 !important;
        color: #a5b4fc !important;
        border: 1px solid #3d3d5c !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        margin-bottom: 6px !important;
        text-align: left !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #3d3d6c !important;
        border-color: #6366f1 !important;
        color: #c7d2fe !important;
    }
 
    /* ── Main area ── */
    .stApp {
        background: #0f0f1a !important;
    }
    .main .block-container {
        background: #0f0f1a !important;
        padding-top: 2rem !important;
    }
 
    /* ── Typography ── */
    .big-title {
        font-size: 2.1rem;
        font-weight: 700;
        text-align: center;
        color: #f1f5f9 !important;
        letter-spacing: -0.3px;
    }
    .sub-title {
        font-size: 1rem;
        color: #94a3b8 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    h3 { color: #f1f5f9 !important; font-weight: 700 !important; }
    h4 { color: #e2e8f0 !important; font-weight: 600 !important; }
    p  { color: #94a3b8 !important; }
 
    /* ── Cards ── */
    .card {
        background: #1e1e2e !important;
        border: 1px solid #2d2d44 !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        transition: box-shadow 0.2s, border-color 0.2s !important;
    }
    .card:hover {
        box-shadow: 0 4px 16px rgba(99,102,241,0.15) !important;
        border-color: #4f46e5 !important;
    }
 
    /* ── Clock ── */
    .clock-card {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        text-align: center !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(79,70,229,0.4) !important;
    }
 
    /* ── Stat boxes ── */
    .stat-box {
        background: #1e1e2e !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        margin-bottom: 10px !important;
        border-left: 4px solid #6366f1 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    .stat-label {
        font-size: 0.72rem !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 4px !important;
        font-weight: 600 !important;
    }
    .stat-value {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
 
    /* ── Form feedback ── */
    .good-form {
        background: #052e16 !important;
        border-left: 4px solid #16a34a !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #4ade80 !important;
        font-weight: 600 !important;
    }
    .bad-form {
        background: #2d0a0a !important;
        border-left: 4px solid #dc2626 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #f87171 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    .feedback-box {
        background: #0c1a3a !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: #93c5fd !important;
        border-left: 4px solid #3b82f6 !important;
        font-size: 0.92rem !important;
    }
 
    /* ── Pills ── */
    .pill {
        display: inline-block !important;
        background: #1e2d5a !important;
        color: #a5b4fc !important;
        border-radius: 20px !important;
        padding: 3px 12px !important;
        font-size: 0.78rem !important;
        margin: 2px !important;
        font-weight: 500 !important;
    }
 
    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        border: 1px solid #2d2d44 !important;
        background: #1e1e2e !important;
        color: #a5b4fc !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #2d2d4e !important;
        border-color: #6366f1 !important;
        color: #c7d2fe !important;
    }
 
    /* ── Inputs ── */
    .stTextInput > div > input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #2d2d44 !important;
        background: #1e1e2e !important;
        color: #f1f5f9 !important;
        font-size: 0.92rem !important;
    }
 
    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
 
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #a5b4fc !important;
    }
 
    /* ── Divider ── */
    hr { border-color: #2d2d44 !important; }
 
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        background: #1e1e2e !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Exercise metadata ─────────────────────────────────────────────────────────
EXERCISES = {
    # ── Physiotherapy ──────────────────────────────────────────────────────
    "squat":                 {"label":"Squat",                "icon":"🦵","target":"Knee & Hip Rehab",      "type":"physio","video_id":"YaXPRqUwItQ","form_checks":["Back angle","Knees over toes","Spine"],    "tip":"Stand shoulder-width apart, keep chest up!",         "limitations":["knee_pain","hip_pain"]},
    "shoulder_raises":       {"label":"Shoulder Raises",      "icon":"💪","target":"Shoulder Rehab",        "type":"physio","video_id":"FeGNSMVFBHg","form_checks":["Arms straight","Height control","Spine"],  "tip":"Move slowly and in control — no swinging!",          "limitations":["shoulder_injury"]},
    "crossover_arm_stretch": {"label":"Crossover Arm Stretch","icon":"🤸","target":"Shoulder Mobility",     "type":"physio","video_id":"5bMBCOgFHug","form_checks":["No shrug","Torso still"],                  "tip":"Hold each stretch for a full 2 seconds!",            "limitations":["shoulder_injury"]},
    "lateral_walks":         {"label":"Lateral Walks",        "icon":"🚶","target":"Hip & Knee Rehab",      "type":"physio","video_id":"swFjPnGXFxk","form_checks":["Knee bend","Torso upright"],               "tip":"Stay low throughout — don't stand between steps!",   "limitations":["knee_pain"]},
    "lunges":                {"label":"Lunges",               "icon":"🏃","target":"Leg Strength Rehab",    "type":"physio","video_id":"QOVaHwm-Q6U","form_checks":["Knee alignment","Torso","Depth"],           "tip":"Keep front knee directly above ankle!",              "limitations":["knee_pain","hip_pain"]},
    "calf_raises":           {"label":"Calf Raises",          "icon":"👟","target":"Ankle & Calf Rehab",    "type":"physio","video_id":"J0DnG1_S92I","form_checks":["Legs straight","No forward lean"],          "tip":"Lower heels ALL the way down each rep!",             "limitations":["ankle_injury"]},
    "knee_raises":           {"label":"Knee Raises",          "icon":"🦿","target":"Hip Flexor & Core",     "type":"physio","video_id":"RHrGLFDRRCY","form_checks":["Back straight","Height","Control"],         "tip":"Engage your core — don't lean backward!",            "limitations":["hip_pain"]},
    # ── Yoga ───────────────────────────────────────────────────────────────
    "tree_pose":             {"label":"Tree Pose",            "icon":"🌳","target":"Balance & Stability",   "type":"yoga",  "video_id":"wdln9qWYloU","form_checks":["Spine straight","No lean","Balance"],      "tip":"Fix your gaze on one spot to help balance!",         "limitations":["balance_issues"]},
    "warrior_pose":          {"label":"Warrior Pose",         "icon":"⚔️","target":"Leg & Core Strength",   "type":"yoga",  "video_id":"Mn6RSIRCV3w","form_checks":["Torso upright","Arms wide","Knee bend"],   "tip":"Front knee tracks over front foot — not inward!",    "limitations":["knee_pain","hip_pain"]},
    "cat_cow_stretch":       {"label":"Cat-Cow Stretch",      "icon":"🐄","target":"Spine Flexibility",     "type":"yoga",  "video_id":"kqnua4rHVVA","form_checks":["Head position","Full arch","Full round"],   "tip":"Breathe in for cow, breathe out for cat!",           "limitations":["back_pain"]},
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
 
def get_user_limitations(user):
    """Check user's medical conditions/limitations against exercise limitations."""
    limitations = []
    text = (user.get("medical_conditions","") + " " +
            user.get("exercise_limitations","")).lower()
    for key, keywords in LIMITATION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            limitations.append(key)
    return limitations
 
def is_exercise_safe(ex_key, user):
    """Returns (safe: bool, warning: str)"""
    user_lims  = get_user_limitations(user)
    ex_lims    = EXERCISES[ex_key].get("limitations", [])
    conflicts  = [l for l in ex_lims if l in user_lims]
    if conflicts:
        conflict_str = ", ".join(c.replace("_"," ") for c in conflicts)
        return False, f"⚠️ Based on your medical profile, this exercise may affect: **{conflict_str}**. Proceed only if your doctor approves."
    return True, ""
 
def generate_guardian_whatsapp_link(user, alert_msg):
    """Generate WhatsApp link to send alert to guardian."""
    number = user.get("guardian_whatsapp","").replace("+","").replace(" ","").replace("-","")
    if not number:
        return None, "No guardian WhatsApp number set."
    text = f"🚨 PhysioAI Alert for {user['name']}: {alert_msg}"
    import urllib.parse
    encoded = urllib.parse.quote(text)
    return f"https://wa.me/{number}?text={encoded}", text
 
# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "page": "login", "user": None,
    "selected_exercise": None, "detector": None,
    "pose_detector": None, "history": [],
    "target_reps": 10, "cam_running": False,
    "auth_mode": "login", "is_admin": False,
    "admin_unlocked": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
 
# ── Sidebar nav (only when logged in) ────────────────────────────────────────
def show_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:16px 0 8px 0'>
            <div style='font-size:3rem'>👤</div>
            <div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-top:6px'>{user['name']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🏠  Home",              use_container_width=True): st.session_state.page = "welcome";        st.rerun()
        if st.button("👤  My Profile",        use_container_width=True): st.session_state.page = "profile";        st.rerun()
        if st.button("🏋️  Exercise",          use_container_width=True): st.session_state.page = "select_exercise"; st.rerun()
        if st.button("🥗  Diet Tracker",      use_container_width=True): st.session_state.page = "diet";           st.rerun()
        if st.button("📈  My Progress",       use_container_width=True): st.session_state.page = "progress";       st.rerun()
        if st.button("👨‍⚕️  Find a Doctor",     use_container_width=True): st.session_state.page = "doctors";        st.rerun()
        if st.button("📅  My Appointments",   use_container_width=True): st.session_state.page = "appointments";   st.rerun()
        if st.button("💬  Messages",          use_container_width=True): st.session_state.page = "messages";       st.rerun()
        if st.button("📝  Notes",             use_container_width=True): st.session_state.page = "notes";          st.rerun()
        st.markdown("---")
        if st.button("🚪  Logout",            use_container_width=True):
            for k in DEFAULTS: st.session_state[k] = DEFAULTS[k]
            st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN / SIGNUP
# ════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown("<p class='big-title'>🏥 AI Physiotherapy Assistant</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Your personal rehabilitation coach powered by AI</p>", unsafe_allow_html=True)
 
    col = st.columns([1.5, 2, 1.5])
    with col[1]:
        mode = st.radio("", ["Login", "Sign Up"], horizontal=True,
                        index=0 if st.session_state.auth_mode == "login" else 1,
                        label_visibility="collapsed")
        st.session_state.auth_mode = "login" if mode == "Login" else "signup"
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
 
        if st.session_state.auth_mode == "login":
            st.markdown("#### Welcome back!")
            email    = st.text_input("Email",    placeholder="you@email.com")
            password = st.text_input("Password", type="password", placeholder="Your password")
            if st.button("Login →", use_container_width=True):
                if email and password:
                    ok, result = login_user(email, password)
                    if ok:
                        st.session_state.user = result
                        st.session_state.page = "welcome"
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please fill in all fields.")
 
        else:
            st.markdown("#### Create your account")
            name     = st.text_input("Full Name",  placeholder="John Doe")
            age      = st.number_input("Age", min_value=5, max_value=100, value=25)
            email    = st.text_input("Email",      placeholder="you@email.com")
            password = st.text_input("Password",   type="password", placeholder="Choose a password")
            confirm  = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
 
            if st.button("Create Account →", use_container_width=True):
                if name and email and password and confirm:
                    if password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        ok, msg = create_user(name, age, email, password)
                        if ok:
                            st.success(msg + " Please login.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.warning("Please fill in all fields.")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: WELCOME
# ════════════════════════════════════════════════════════════════════════════
def show_welcome():
    import random
    show_sidebar()
    user   = st.session_state.user
    now    = datetime.now()
    hour   = now.hour
    minute = now.minute
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
 
    # ── Welcome + greeting + clock ──────────────────────────────────────
    col_greet, col_clock = st.columns([3, 1])
    with col_greet:
        st.markdown(f"""
        <div style='padding: 10px 0'>
            <div style='font-size:1.05rem;color:#818cf8;font-weight:600;
                        letter-spacing:1px;margin-bottom:6px'>
                {greeting} 👋
            </div>
            <div style='font-size:3rem;font-weight:800;color:#f1f5f9;
                        line-height:1.1;letter-spacing:-1px'>
                Welcome,<br>{user['name']}
            </div>
            <div style='font-size:1rem;color:#64748b;margin-top:10px'>
                Ready for your session today?
            </div>
        </div>""", unsafe_allow_html=True)
    with col_clock:
        am_pm   = "AM" if hour < 12 else "PM"
        h12     = hour % 12 or 12
        st.markdown(f"""
        <div class='clock-card'>
            <div style='font-size:0.72rem;letter-spacing:2px;opacity:0.8'>🕐 CURRENT TIME</div>
            <div style='font-size:2.2rem;font-weight:700;margin:6px 0'>
                {h12:02d}:{minute:02d}
                <span style='font-size:1rem;opacity:0.7'>{am_pm}</span>
            </div>
            <div style='font-size:0.78rem;opacity:0.75'>{now.strftime("%A, %d %B %Y")}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── Exercise timing warning ───────────────────────────────────────────
    MORNING_START, MORNING_END = 5, 9
    EVENING_START, EVENING_END = 17, 20
 
    in_morning = MORNING_START <= hour < MORNING_END
    in_evening = EVENING_START <= hour < EVENING_END
 
    if in_morning:
        st.success("✅ **Perfect time to exercise!** Morning sessions (5 AM – 9 AM) boost metabolism and energy for the whole day.")
    elif in_evening:
        st.success("✅ **Perfect time to exercise!** Evening sessions (5 PM – 8 PM) help relieve stress and improve muscle recovery.")
    elif hour < MORNING_START or hour >= 21:
        st.error("🚨 **Not recommended!** It is late night / very early morning. Exercising now can disturb your sleep cycle and stress your body. Please rest and exercise tomorrow morning.")
    elif MORNING_END <= hour < 12:
        st.warning("⚠️ **Late morning** — not ideal for intense exercise. Light stretching is okay but save intense rehab for the evening window (5 PM – 8 PM).")
    elif 12 <= hour < 14:
        st.error("🚨 **Post-meal danger zone!** It is around lunch time. Always wait at least **4 hours after eating** before physiotherapy exercises to avoid cramps and poor performance.")
    elif 14 <= hour < EVENING_START:
        st.warning("⚠️ **Afternoon slump** — energy levels are typically low between 2 PM – 5 PM. Consider waiting for the evening window (5 PM – 8 PM) for better results.")
    elif hour >= EVENING_END:
        st.warning("⚠️ **Late evening** — light stretching is okay but avoid intense workouts after 8 PM as it may affect your sleep quality.")
 
    st.info("🍽️ **Always remember:** Wait at least **4 hours after a heavy meal** before exercising. Exercising on a full stomach reduces performance and may cause nausea or cramps.")
 
    st.markdown("---")
 
    # ── Exercise timing guidelines ────────────────────────────────────────
    st.markdown("### 📋 Exercise Timing Guidelines")
    g1, g2, g3, g4 = st.columns(4)
    for col, (icon, title, detail, color) in zip(
        [g1, g2, g3, g4],
        [
            ("🌅", "Morning Window",  "5:00 AM – 9:00 AM", "#22c55e"),
            ("🌆", "Evening Window",  "5:00 PM – 8:00 PM", "#22c55e"),
            ("🍽️", "After Meals",     "Wait 4+ hours",     "#f59e0b"),
            ("😴", "Before Bed",      "Avoid 2hrs before", "#ef4444"),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;border-left:4px solid {color};border-top:3px solid {color}'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-weight:700;margin:6px 0 2px;color:#f1f5f9'>{title}</div>
                <div style='color:{color};font-size:0.85rem;font-weight:600'>{detail}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── Did You Know facts ────────────────────────────────────────────────
    facts = [
        "💡 Regular physiotherapy can reduce chronic pain by up to 60% without medication.",
        "💡 Doing 10 minutes of stretching daily improves joint flexibility within 3 weeks.",
        "💡 Squats strengthen not just your legs but also your lower back and core muscles.",
        "💡 Physiotherapy after injury reduces the chance of re-injury by nearly 50%.",
        "💡 Shoulder raises improve posture and reduce neck tension from long sitting hours.",
        "💡 Calf raises improve blood circulation and reduce ankle swelling significantly.",
        "💡 Lateral walks activate your hip abductors — muscles most people never train.",
        "💡 Lunges improve balance and coordination as much as they build leg strength.",
        "💡 Consistent rehab exercises can help avoid surgery in many knee conditions.",
        "💡 Even 15 minutes of daily physiotherapy improves mobility within 2 weeks.",
        "💡 Proper form during exercise is more important than the number of reps.",
        "💡 Morning exercise on an empty stomach burns 20% more fat than post-meal workouts.",
        "💡 Stretching before bed improves sleep quality and reduces morning stiffness.",
        "💡 The human body has 360 joints — physiotherapy keeps them all healthy.",
        "💡 Drinking water before exercise improves muscle performance by up to 25%.",
    ]
    st.markdown("### 🧠 Did You Know?")
    f1, f2, f3 = st.columns(3)
    for col, fact in zip([f1, f2, f3], random.sample(facts, 3)):
        with col:
            st.markdown(f"""
            <div class='card' style='min-height:90px;padding:16px'>
                <p style='color:#ccc;font-size:0.9rem;line-height:1.6;margin:0'>{fact}</p>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── Navigation cards ──────────────────────────────────────────────────
    st.markdown("### 🚀 What would you like to do?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='card' style='text-align:center;border-top:3px solid #4f46e5'>
            <div style='font-size:2.5rem'>🏋️</div>
            <h3 style='color:#f1f5f9'>Exercise</h3>
            <p style='color:#94a3b8;font-size:0.85rem'>Start a rehab session with AI tracking</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Start Exercise", use_container_width=True, key="w_ex"):
            st.session_state.page = "select_exercise"; st.rerun()
    with c2:
        st.markdown("""<div class='card' style='text-align:center;border-top:3px solid #059669'>
            <div style='font-size:2.5rem'>🥗</div>
            <h3 style='color:#f1f5f9'>Diet Tracker</h3>
            <p style='color:#94a3b8;font-size:0.85rem'>Log your meals and track nutrition</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Track Diet", use_container_width=True, key="w_diet"):
            st.session_state.page = "diet"; st.rerun()
    with c3:
        st.markdown("""<div class='card' style='text-align:center;border-top:3px solid #d97706'>
            <div style='font-size:2.5rem'>📈</div>
            <h3 style='color:#f1f5f9'>My Progress</h3>
            <p style='color:#94a3b8;font-size:0.85rem'>View your exercise history and stats</p>
        </div>""", unsafe_allow_html=True)
        if st.button("View Progress", use_container_width=True, key="w_prog"):
            st.session_state.page = "progress"; st.rerun()
    with c4:
        st.markdown("""<div class='card' style='text-align:center;border-top:3px solid #dc2626'>
            <div style='font-size:2.5rem'>📝</div>
            <h3 style='color:#f1f5f9'>Notes</h3>
            <p style='color:#94a3b8;font-size:0.85rem'>Doctor notes and personal reminders</p>
        </div>""", unsafe_allow_html=True)
        if st.button("View Notes", use_container_width=True, key="w_notes"):
            st.session_state.page = "notes"; st.rerun()
 
    # ── Quick stats ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Your Quick Stats")
    summary        = get_sessions_summary(user["id"])
    diet_today     = get_diet_today(user["id"])
    total_cals     = sum(d["calories"] for d in diet_today)
    total_sessions = sum(s["total_sessions"] for s in summary)
    total_reps     = sum(s["total_reps"]     for s in summary)
 
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Sessions",  total_sessions)
    s2.metric("Total Reps Done", total_reps)
    s3.metric("Exercises Tried", len(summary))
    s4.metric("Calories Today",  f"{total_cals} kcal")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: EXERCISE SELECTION
# ════════════════════════════════════════════════════════════════════════════
def show_exercise_selection():
    show_sidebar()
    st.markdown("<p class='big-title' style='font-size:2rem'>Which exercise today?</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Choose one — your AI coach will guide you</p>", unsafe_allow_html=True)
    st.markdown("---")
 
    user = st.session_state.user
    user_lims = get_user_limitations(user)
 
    # Show physio and yoga separately
    for section, etype, color in [("🏥 Physiotherapy Exercises","physio","#6366f1"),("🧘 Yoga Exercises","yoga","#10b981")]:
        st.markdown(f"### {section}")
        keys_filtered = [k for k,v in EXERCISES.items() if v.get("type")==etype]
        chunks = [keys_filtered[i:i+3] for i in range(0, len(keys_filtered), 3)]
        for chunk in chunks:
            cols = st.columns(3)
            for col, key in zip(cols, chunk):
                ex = EXERCISES[key]
                safe, warn = is_exercise_safe(key, user)
                with col:
                    border = "#ef4444" if not safe else color
                    st.markdown(f"""
                    <div class='card' style='text-align:center;border-top:3px solid {border}'>
                        <div style='font-size:2.5rem'>{ex['icon']}</div>
                        <h3 style='margin:8px 0 2px;color:#f1f5f9'>{ex['label']}</h3>
                        <div style='color:{color};font-size:0.78rem;margin-bottom:4px'>🎯 {ex['target']}</div>
                        {"<div style='color:#f87171;font-size:0.72rem'>⚠️ Check medical profile</div>" if not safe else ""}
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"Choose {ex['label']}", key=f"sel_{key}", use_container_width=True):
                        st.session_state.selected_exercise = key
                        st.session_state.page = "setup"
                        st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ════════════════════════════════════════════════════════════════════════════
def show_setup():
    show_sidebar()
    key = st.session_state.selected_exercise
    ex  = EXERCISES[key]
 
    st.markdown(f"<p class='big-title' style='font-size:2rem'>{ex['icon']} Get Ready for {ex['label']}</p>", unsafe_allow_html=True)
    st.markdown("---")
 
    col_vid, col_right = st.columns([2, 1])
    with col_vid:
        st.markdown("#### 🎥 Watch how it's done")
        st.markdown(
            f'<iframe width="100%" height="300" src="https://www.youtube.com/embed/{ex["video_id"]}" '
            f'frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
        st.markdown(f'<div class="feedback-box" style="margin-top:10px">💡 <b>Tip:</b> {ex["tip"]}</div>',
                    unsafe_allow_html=True)
 
    with col_right:
        st.markdown("#### ⚙️ Session Settings")
        target = st.slider("Target reps", 5, 50, 10, 5)
        st.session_state.target_reps = target
        st.markdown(f"""<div class='stat-box'>
            <div class='stat-label'>Your target</div>
            <div class='stat-value'>{target} reps</div>
        </div>""", unsafe_allow_html=True)
 
        st.markdown("#### ✅ Form checks")
        for c in ex["form_checks"]:
            st.markdown(f"✅ {c}")
 
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        # Safety check based on user medical profile
        user = st.session_state.user
        safe, warning_msg = is_exercise_safe(key, user)
        if not safe:
            st.warning(warning_msg)
 
        # Guardian alert for wrong time
        hour_now = datetime.now().hour
        is_bad_time = hour_now < 5 or hour_now >= 21
        if is_bad_time and user.get("guardian_whatsapp"):
            alert_msg = f"{user['name']} is about to exercise at {datetime.now().strftime('%I:%M %p')} — outside recommended hours."
            wa_link, wa_text = generate_guardian_whatsapp_link(user, alert_msg)
            if wa_link:
                st.error("It is currently outside recommended exercise hours!")
                st.markdown(f"[📲 Notify Guardian on WhatsApp]({wa_link})", unsafe_allow_html=False)
                log_guardian_alert(user["id"], "wrong_time", wa_text, user.get("guardian_whatsapp",""))
 
        if st.button("▶️ Start Exercise", use_container_width=True):
            st.session_state.detector      = ExerciseDetector(exercise=key)
            st.session_state.pose_detector = PoseDetector()
            st.session_state.history       = []
            st.session_state.page          = "tracking"
            st.rerun()
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "select_exercise"; st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: TRACKING
# ════════════════════════════════════════════════════════════════════════════
def show_tracking():
    import time
    show_sidebar()
    key      = st.session_state.selected_exercise
    ex       = EXERCISES[key]
    detector = st.session_state.detector
    pose_det = st.session_state.pose_detector
    target   = st.session_state.target_reps
    user     = st.session_state.user
 
    # ── Session state for alarm system ───────────────────────────────────
    if "alarm_active"      not in st.session_state: st.session_state.alarm_active      = False
    if "alarm_start_time"  not in st.session_state: st.session_state.alarm_start_time  = None
    if "alarm_dismissed"   not in st.session_state: st.session_state.alarm_dismissed   = False
    if "wa_sent_this_alarm"not in st.session_state: st.session_state.wa_sent_this_alarm= False
    if "consecutive_errors"not in st.session_state: st.session_state.consecutive_errors= 0
 
    nav1, nav2, nav3 = st.columns([3, 1, 1])
    with nav1: st.markdown(f"### {ex['icon']} {ex['label']} — Live Session")
    with nav2:
        if st.button("🔄 Reset", use_container_width=True):
            detector.reset()
            st.session_state.history          = []
            st.session_state.alarm_active     = False
            st.session_state.alarm_start_time = None
            st.session_state.consecutive_errors = 0
            st.rerun()
    with nav3:
        if st.button("🔀 Switch", use_container_width=True):
            st.session_state.page = "select_exercise"; st.rerun()
 
    st.markdown("---")
    col_video, col_stats = st.columns([4, 2])
 
    with col_stats:
        st.markdown("#### 📊 Live Stats")
        reps_display     = st.empty()
        progress_display = st.empty()
        c1, c2           = st.columns(2)
        with c1: stage_display = st.empty()
        with c2: angle_display = st.empty()
        st.markdown("#### 💬 Feedback")
        feedback_display  = st.empty()
        st.markdown("#### 🩺 Posture")
        posture_display   = st.empty()
        alarm_display     = st.empty()   # alarm banner
        dismiss_display   = st.empty()   # dismiss button area
        wa_display        = st.empty()   # WhatsApp link area
        st.markdown("#### 📋 Rep History")
        history_display   = st.empty()
 
    with col_video:
        # Alarm sound + visual injected via HTML
        alarm_html_placeholder = st.empty()
        camera_on = st.button("📸 Click here to Start your Exercise Session", use_container_width=True)
        if camera_on:
            st.session_state.cam_running      = True
            st.session_state.alarm_active     = False
            st.session_state.alarm_start_time = None
            st.session_state.consecutive_errors = 0
        frame_placeholder = st.empty()
        if not st.session_state.cam_running:
            st.markdown("""
            <div style='background:#111827;border-radius:14px;padding:80px 20px;
                        text-align:center;border:2px dashed #333'>
                <div style='font-size:3rem'>📸</div>
                <div style='font-size:1.1rem;color:#888;margin-top:12px'>
                    Click the button above to start</div>
                <div style='font-size:0.82rem;color:#555;margin-top:6px'>
                    Step back so your full body is visible</div>
            </div>""", unsafe_allow_html=True)
 
    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera.")
            st.session_state.cam_running = False
            st.stop()
 
        prev_reps  = detector.reps
        good_reps  = 0
 
        while True:
            ret, frame = cap.read()
            if not ret: break
 
            frame = cv2.flip(frame, 1)
            frame, landmarks = pose_det.find_pose(frame)
            result = detector.process(landmarks)
            h, w, _ = frame.shape
 
            if landmarks:
                mp_pose = mp.solutions.pose.PoseLandmark
                knee_lm = landmarks[mp_pose.LEFT_KNEE.value]
                cx, cy  = int(knee_lm.x * w), int(knee_lm.y * h)
                cv2.putText(frame, f"{result['angle']}deg", (cx-25, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
 
            # ── Form status on video ──────────────────────────────────────
            if result["form_ok"]:
                bc, bt = (0,180,70), "GOOD FORM"
            else:
                bc, bt = (40,40,200), "FIX FORM"
 
            cv2.rectangle(frame, (0,h-36),(w,h),(15,15,15),-1)
            cv2.putText(frame, bt, (12,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bc, 2)
            if result["form_errors"]:
                cv2.putText(frame, result["form_errors"][0].replace("⚠️ ",""),
                            (int(w*0.28),h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1)
 
            cv2.rectangle(frame,(0,0),(185,52),(15,15,15),-1)
            cv2.putText(frame, f"REPS  {result['reps']}/{target}", (8,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)
            cv2.putText(frame, result["stage"].upper(), (8,44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
 
            # ── Alarm logic ───────────────────────────────────────────────
            if not result["form_ok"]:
                st.session_state.consecutive_errors += 1
            else:
                st.session_state.consecutive_errors = 0
                if st.session_state.alarm_active and not st.session_state.alarm_dismissed:
                    # Form corrected — auto dismiss alarm
                    st.session_state.alarm_active     = False
                    st.session_state.alarm_start_time = None
                    st.session_state.wa_sent_this_alarm = False
 
            # Trigger alarm after 3 consecutive bad-form frames
            ALARM_THRESHOLD = 3
            if (st.session_state.consecutive_errors >= ALARM_THRESHOLD
                    and not st.session_state.alarm_active):
                st.session_state.alarm_active      = True
                st.session_state.alarm_dismissed   = False
                st.session_state.alarm_start_time  = time.time()
                st.session_state.wa_sent_this_alarm= False
 
            # ── Alarm flash on video ──────────────────────────────────────
            if st.session_state.alarm_active and not st.session_state.alarm_dismissed:
                # Red flash border on frame
                flash_alpha = int(abs((time.time() * 3) % 2 - 1) * 200) + 55
                cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,flash_alpha), 8)
                cv2.putText(frame, "!! FIX POSTURE !!", (int(w*0.18), int(h*0.5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
 
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)
 
            # ── Alarm sound (browser beep via HTML audio) ─────────────────
            if st.session_state.alarm_active and not st.session_state.alarm_dismissed:
                elapsed = time.time() - st.session_state.alarm_start_time
                remaining = max(0, 60 - int(elapsed))
 
                # Inject alarm sound + red flash into page
                alarm_html_placeholder.markdown("""
                <audio id="alarmAudio" autoplay loop>
                  <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
                  <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAA..." type="audio/wav">
                </audio>
                <script>
                  var a = document.getElementById("alarmAudio");
                  if(a) { a.volume = 1.0; a.play().catch(()=>{}); }
                </script>
                <style>
                @keyframes pulse {
                  0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
                  70%  { box-shadow: 0 0 0 20px rgba(239,68,68,0); }
                  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
                }
                .alarm-banner {
                  background: #dc2626;
                  color: white;
                  font-size: 1.2rem;
                  font-weight: 800;
                  padding: 14px 20px;
                  border-radius: 12px;
                  text-align: center;
                  animation: pulse 1s infinite;
                  margin-bottom: 8px;
                  letter-spacing: 1px;
                }
                </style>
                """, unsafe_allow_html=True)
 
                alarm_display.markdown(f"""
                <div class="alarm-banner">
                    🚨 WRONG POSTURE DETECTED! FIX IT NOW!<br>
                    <span style="font-size:0.85rem;font-weight:400">
                    WhatsApp alert in {remaining}s if not dismissed
                    </span>
                </div>""", unsafe_allow_html=True)
 
                # Dismiss button — fixed key so it doesn't duplicate each frame
                if dismiss_display.button("✅ Dismiss Alarm — I Fixed My Posture",
                                           use_container_width=True, key="dismiss_alarm_btn"):
                    st.session_state.alarm_active     = False
                    st.session_state.alarm_dismissed  = True
                    st.session_state.consecutive_errors = 0
                    alarm_html_placeholder.empty()
                    alarm_display.empty()
                    wa_display.empty()
                    st.rerun()
 
                # WhatsApp after 60 seconds
                if elapsed >= 60 and not st.session_state.wa_sent_this_alarm:
                    st.session_state.wa_sent_this_alarm = True
                    err_text = result["form_errors"][0] if result["form_errors"] else "wrong posture"
                    alert_msg = (f"URGENT: {user['name']} has had wrong posture during "
                                 f"{ex['label']} exercise for over 1 minute. "
                                 f"Issue: {err_text.replace(chr(9888)+chr(65039)+' ','')}. "
                                 f"Please check on them immediately.")
                    wa_link, wa_text = generate_guardian_whatsapp_link(user, alert_msg)
                    if wa_link:
                        log_guardian_alert(user["id"], "posture_alarm_1min", wa_text,
                                           user.get("guardian_whatsapp",""))
                        wa_display.markdown(f"""
                        <div style='background:#7f1d1d;border-radius:10px;padding:12px 16px;
                                    border:2px solid #ef4444;margin-top:8px'>
                            <div style='color:#fca5a5;font-weight:700;margin-bottom:6px'>
                                Guardian not notified automatically — click below:
                            </div>
                            <a href='{wa_link}' target='_blank'
                               style='background:#25D366;color:white;padding:8px 16px;
                                      border-radius:8px;text-decoration:none;
                                      font-weight:600;font-size:0.9rem'>
                                📲 Send WhatsApp Alert to Guardian
                            </a>
                        </div>""", unsafe_allow_html=True)
                    else:
                        wa_display.warning("No guardian WhatsApp set. Add it in your profile!")
 
            else:
                # Clear alarm UI when not active
                alarm_html_placeholder.markdown("""
                <script>
                  var a = document.getElementById("alarmAudio");
                  if(a) { a.pause(); a.currentTime = 0; }
                </script>""", unsafe_allow_html=True)
                alarm_display.empty()
                dismiss_display.empty()
 
            # ── Stats panel ───────────────────────────────────────────────
            progress = min(result["reps"]/target, 1.0)
            pct      = int(progress * 100)
            reps_display.markdown(f"""<div class='stat-box'>
                <div class='stat-label'>Reps Done</div>
                <div class='stat-value'>{result['reps']}
                    <span style='font-size:1rem;color:#888'>/ {target}</span></div>
            </div>""", unsafe_allow_html=True)
            progress_display.progress(progress, text=f"{pct}% complete")
            stage_display.markdown(f"""<div class='stat-box' style='border-color:#a855f7'>
                <div class='stat-label'>Stage</div>
                <div class='stat-value' style='font-size:1.4rem'>{result['stage'].upper()}</div>
            </div>""", unsafe_allow_html=True)
            angle_display.markdown(f"""<div class='stat-box' style='border-color:#f59e0b'>
                <div class='stat-label'>Angle</div>
                <div class='stat-value' style='font-size:1.4rem'>{result['angle']}°</div>
            </div>""", unsafe_allow_html=True)
            feedback_display.markdown(f'<div class="feedback-box">💬 {result["feedback"]}</div>',
                                      unsafe_allow_html=True)
            if result["form_ok"]:
                posture_display.markdown("<div class='good-form'>✅ Great form — keep it up!</div>",
                                          unsafe_allow_html=True)
            else:
                posture_display.markdown(
                    f"<div class='bad-form'>{'<br>'.join(result['form_errors'])}</div>",
                    unsafe_allow_html=True)
 
            if result["reps"] > prev_reps:
                if result["form_ok"]: good_reps += 1
                st.session_state.history.append({
                    "Rep":   result["reps"],
                    "Angle": f"{result['angle']}°",
                    "Form":  "✅" if result["form_ok"] else "⚠️",
                })
                prev_reps = result["reps"]
 
            if st.session_state.history:
                history_display.dataframe(st.session_state.history,
                                          use_container_width=True, hide_index=True)
 
            if result["reps"] >= target:
                cap.release()
                st.session_state.cam_running = False
                st.session_state.alarm_active = False
                form_acc = int((good_reps / target) * 100) if target > 0 else 0
                save_session(user["id"], key, result["reps"], form_acc)
                st.success(f"Done! {target} reps of {ex['label']} — Form accuracy: {form_acc}%")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Try Another", use_container_width=True):
                        st.session_state.page = "select_exercise"; st.rerun()
                with c2:
                    if st.button("Repeat", use_container_width=True):
                        detector.reset(); st.session_state.history = []; st.rerun()
                break
 
        cap.release()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: DIET TRACKER
# ════════════════════════════════════════════════════════════════════════════
def show_diet():
    show_sidebar()
    user = st.session_state.user
    st.markdown("<p class='big-title' style='font-size:2rem'>🥗 Diet Tracker</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Log your meals and track daily nutrition</p>", unsafe_allow_html=True)
    st.markdown("---")
 
    col_form, col_today = st.columns([1, 1])
 
    with col_form:
        st.markdown("### ➕ Add a Meal")
        meal     = st.text_input("Meal name", placeholder="e.g. Oats with milk")
        cal_col, pro_col = st.columns(2)
        carb_col, fat_col = st.columns(2)
        with cal_col:  calories = st.number_input("Calories (kcal)", min_value=0, value=0)
        with pro_col:  protein  = st.number_input("Protein (g)",     min_value=0.0, value=0.0, step=0.5)
        with carb_col: carbs    = st.number_input("Carbs (g)",       min_value=0.0, value=0.0, step=0.5)
        with fat_col:  fats     = st.number_input("Fats (g)",        min_value=0.0, value=0.0, step=0.5)
 
        if st.button("✅ Add Meal", use_container_width=True):
            if meal:
                add_diet_entry(user["id"], meal, calories, protein, carbs, fats)
                st.success(f"Added: {meal}")
                st.rerun()
            else:
                st.warning("Please enter a meal name.")
 
    with col_today:
        st.markdown("### 📅 Today's Meals")
        today_meals = get_diet_today(user["id"])
        if today_meals:
            total_cal  = sum(m["calories"] for m in today_meals)
            total_pro  = sum(m["protein"]  for m in today_meals)
            total_carb = sum(m["carbs"]    for m in today_meals)
            total_fat  = sum(m["fats"]     for m in today_meals)
 
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Calories",  f"{total_cal} kcal")
            t2.metric("Protein",   f"{total_pro}g")
            t3.metric("Carbs",     f"{total_carb}g")
            t4.metric("Fats",      f"{total_fat}g")
 
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for meal in today_meals:
                mc1, mc2 = st.columns([4, 1])
                with mc1:
                    st.markdown(f"""<div class='card' style='padding:12px 16px'>
                        <b>{meal['meal']}</b> &nbsp;
                        <span class='pill'>{meal['calories']} kcal</span>
                        <span class='pill'>P: {meal['protein']}g</span>
                        <span class='pill'>C: {meal['carbs']}g</span>
                        <span class='pill'>F: {meal['fats']}g</span>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    if st.button("🗑️", key=f"del_diet_{meal['id']}"):
                        delete_diet_entry(meal["id"]); st.rerun()
        else:
            st.info("No meals logged today. Add your first meal!")
 
    st.markdown("---")
    st.markdown("### 📋 Full Diet History")
    all_meals = get_diet_all(user["id"])
    if all_meals:
        import pandas as pd
        df = pd.DataFrame(all_meals)[["date","meal","calories","protein","carbs","fats"]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d %b %Y %H:%M")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No diet history yet.")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: PROGRESS
# ════════════════════════════════════════════════════════════════════════════
def show_progress():
    show_sidebar()
    import pandas as pd
    user = st.session_state.user
    st.markdown("<p class='big-title' style='font-size:2rem'>📈 My Progress</p>", unsafe_allow_html=True)
    st.markdown("---")
 
    sessions = get_user_sessions(user["id"])
    summary  = get_sessions_summary(user["id"])
 
    if not sessions:
        st.info("No exercise sessions yet. Complete an exercise to see your progress!")
        return
 
    # Summary metrics
    total_sessions  = len(sessions)
    total_reps      = sum(s["reps"] for s in sessions)
    avg_form        = sum(s["form_accuracy"] for s in sessions) / total_sessions
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sessions",   total_sessions)
    m2.metric("Total Reps",       total_reps)
    m3.metric("Avg Form Accuracy",f"{avg_form:.0f}%")
    m4.metric("Exercises Done",   len(summary))
 
    st.markdown("---")
 
    # Per exercise breakdown
    st.markdown("### 🏋️ Exercise Breakdown")
    ex_cols = st.columns(min(len(summary), 3))
    for i, s in enumerate(summary):
        ex_info = EXERCISES.get(s["exercise"], {})
        with ex_cols[i % 3]:
            st.markdown(f"""<div class='card' style='text-align:center'>
                <div style='font-size:2rem'>{ex_info.get('icon','🏋️')}</div>
                <h4>{ex_info.get('label', s['exercise'])}</h4>
                <div><span class='pill'>{s['total_sessions']} sessions</span></div>
                <div><span class='pill'>{s['total_reps']} total reps</span></div>
                <div><span class='pill'>{s['avg_form']:.0f}% form</span></div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    # Session history table
    st.markdown("### 📋 Session History")
    df = pd.DataFrame(sessions)[["date","exercise","reps","form_accuracy"]]
    df["exercise"]      = df["exercise"].apply(lambda x: EXERCISES.get(x,{}).get("label", x))
    df["form_accuracy"] = df["form_accuracy"].apply(lambda x: f"{x}%")
    df["date"]          = pd.to_datetime(df["date"]).dt.strftime("%d %b %Y %H:%M")
    df.columns         = ["Date","Exercise","Reps","Form Accuracy"]
    st.dataframe(df, use_container_width=True, hide_index=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: NOTES
# ════════════════════════════════════════════════════════════════════════════
def show_notes():
    show_sidebar()
    user = st.session_state.user
    st.markdown("<p class='big-title' style='font-size:2rem'>📝 Notes</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Doctor notes, reminders, and personal observations</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    col_add, col_list = st.columns([1, 1])
 
    with col_add:
        st.markdown("### ➕ Add a Note")
        note_text = st.text_area("Write your note here", height=150,
                                 placeholder="e.g. Doctor advised to do 10 squats daily...")
        if st.button("💾 Save Note", use_container_width=True):
            if note_text.strip():
                add_note(user["id"], note_text.strip())
                st.success("Note saved!")
                st.rerun()
            else:
                st.warning("Please write something first.")
 
    with col_list:
        st.markdown("### 📋 Your Notes")
        notes = get_notes(user["id"])
        if notes:
            for note in notes:
                nc1, nc2 = st.columns([5, 1])
                with nc1:
                    date_str = note["date"][:16]
                    st.markdown(f"""<div class='card'>
                        <div style='font-size:0.75rem;color:#64748b;margin-bottom:6px'>🕐 {date_str}</div>
                        <div>{note['note_text']}</div>
                    </div>""", unsafe_allow_html=True)
                with nc2:
                    if st.button("🗑️", key=f"del_note_{note['id']}"):
                        delete_note(note["id"]); st.rerun()
        else:
            st.info("No notes yet. Add your first note!")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: FIND A DOCTOR
# ════════════════════════════════════════════════════════════════════════════
def show_doctors():
    show_sidebar()
    user    = st.session_state.user
    doctors = get_all_doctors()
 
    st.markdown("<p class='big-title' style='font-size:2rem'>👨‍⚕️ Find a Doctor</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Browse our physiotherapy specialists and book an appointment</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if not doctors:
        st.info("No doctors available yet. Please check back later.")
        return
 
    for doc in doctors:
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class='card'>
                    <div style='display:flex;align-items:center;gap:16px'>
                        <div style='font-size:3rem'>👨‍⚕️</div>
                        <div>
                            <h3 style='margin:0'>{doc['name']}</h3>
                            <span class='pill'>{doc['specialization']}</span>
                            <span class='pill'>{doc['experience']} yrs exp</span>
                            <span class='pill'>{doc['qualification']}</span>
                        </div>
                    </div>
                    <hr style='border-color:#333;margin:12px 0'>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                        <div><span style='color:#64748b;font-size:0.8rem'>📅 Available:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['available_days']}</span></div>
                        <div><span style='color:#64748b;font-size:0.8rem'>🕐 Timings:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['timings']}</span></div>
                        <div><span style='color:#64748b;font-size:0.8rem'>📞 Contact:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['contact']}</span></div>
                    </div>
                    <p style='color:#94a3b8;font-size:0.88rem;margin-top:10px'>{doc['about']}</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button(f"📅 Book Appointment", key=f"book_{doc['id']}",
                             use_container_width=True):
                    st.session_state["booking_doctor_id"] = doc["id"]
                    st.session_state.page = "book_appointment"
                    st.rerun()
                if st.button(f"💬 Send Message", key=f"msg_{doc['id']}",
                             use_container_width=True):
                    st.session_state["chat_doctor_id"] = doc["id"]
                    st.session_state.page = "chat"
                    st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: BOOK APPOINTMENT
# ════════════════════════════════════════════════════════════════════════════
def show_book_appointment():
    show_sidebar()
    user      = st.session_state.user
    doctor_id = st.session_state.get("booking_doctor_id")
    if not doctor_id:
        st.session_state.page = "doctors"; st.rerun()
    doc = get_doctor(doctor_id)
 
    st.markdown(f"<p class='big-title' style='font-size:2rem'>📅 Book Appointment</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    col_info, col_form = st.columns([1, 1])
    with col_info:
        st.markdown(f"""
        <div class='card'>
            <h3>👨‍⚕️ {doc['name']}</h3>
            <span class='pill'>{doc['specialization']}</span>
            <span class='pill'>{doc['experience']} yrs exp</span>
            <hr style='border-color:#333;margin:10px 0'>
            <p style='color:#64748b;font-size:0.85rem'>📅 {doc['available_days']}</p>
            <p style='color:#64748b;font-size:0.85rem'>🕐 {doc['timings']}</p>
        </div>""", unsafe_allow_html=True)
 
    with col_form:
        st.markdown("#### Fill in your appointment details")
        date   = st.date_input("Select Date")
        time   = st.selectbox("Select Time",
                    ["9:00 AM","10:00 AM","11:00 AM","12:00 PM",
                     "2:00 PM","3:00 PM","4:00 PM","5:00 PM","6:00 PM"])
        reason = st.text_area("Reason for appointment",
                    placeholder="Describe your issue or reason for visiting...")
 
        if st.button("✅ Confirm Booking", use_container_width=True):
            if reason.strip():
                book_appointment(user["id"], doctor_id, str(date), time, reason.strip())
                st.success(f"✅ Appointment booked with Dr. {doc['name']} on {date} at {time}!")
                st.session_state.page = "appointments"
                st.rerun()
            else:
                st.warning("Please enter a reason for your appointment.")
 
        if st.button("← Back to Doctors", use_container_width=True):
            st.session_state.page = "doctors"; st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: MY APPOINTMENTS
# ════════════════════════════════════════════════════════════════════════════
def show_appointments():
    show_sidebar()
    user  = st.session_state.user
    appts = get_user_appointments(user["id"])
 
    st.markdown("<p class='big-title' style='font-size:2rem'>📅 My Appointments</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if st.button("➕ Book New Appointment", use_container_width=False):
        st.session_state.page = "doctors"; st.rerun()
 
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
 
    if not appts:
        st.info("No appointments yet. Find a doctor and book your first appointment!")
        return
 
    STATUS_COLOR = {"pending": "#f59e0b", "approved": "#22c55e", "rejected": "#ef4444"}
 
    for a in appts:
        color = STATUS_COLOR.get(a["status"], "#888")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"""
            <div class='card' style='border-left:4px solid {color}'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <h4 style='margin:0'>Dr. {a['doctor_name']}</h4>
                    <span style='background:{color}33;color:{color};padding:4px 12px;
                           border-radius:20px;font-size:0.8rem;font-weight:600'>
                        {a['status'].upper()}
                    </span>
                </div>
                <span class='pill'>{a['specialization']}</span>
                <hr style='border-color:#333;margin:8px 0'>
                <p style='color:#64748b;font-size:0.85rem;margin:2px 0'>📅 {a['date']}  🕐 {a['time']}</p>
                <p style='color:#cbd5e1;font-size:0.88rem;margin:4px 0'><b>Reason:</b> {a['reason']}</p>
                {f"<p style='color:#22c55e;font-size:0.85rem'><b>Doctor note:</b> {a['admin_note']}</p>" if a['admin_note'] else ""}
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            if st.button("❌ Cancel", key=f"cancel_appt_{a['id']}",
                         use_container_width=True):
                cancel_appointment(a["id"]); st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: MESSAGES / CHAT
# ════════════════════════════════════════════════════════════════════════════
def show_messages():
    show_sidebar()
    user    = st.session_state.user
    doctors = get_all_doctors()
 
    st.markdown("<p class='big-title' style='font-size:2rem'>💬 Messages</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if not doctors:
        st.info("No doctors available to message yet.")
        return
 
    doc_options = {d["id"]: f"Dr. {d['name']} — {d['specialization']}" for d in doctors}
    selected_id = st.selectbox("Select a doctor to message",
                               options=list(doc_options.keys()),
                               format_func=lambda x: doc_options[x])
 
    st.session_state["chat_doctor_id"] = selected_id
    show_chat()
 
 
def show_chat():
    show_sidebar()
    user      = st.session_state.user
    doctor_id = st.session_state.get("chat_doctor_id")
    if not doctor_id:
        st.session_state.page = "doctors"; st.rerun()
 
    doc  = get_doctor(doctor_id)
    msgs = get_conversation(user["id"], doctor_id)
 
    st.markdown(f"#### 💬 Chat with Dr. {doc['name']} — {doc['specialization']}")
    st.markdown("---")
 
    # Chat history
    chat_container = st.container()
    with chat_container:
        if not msgs:
            st.markdown("""
            <div style='text-align:center;padding:30px;color:#555'>
                No messages yet. Send your first message below!
            </div>""", unsafe_allow_html=True)
        for msg in msgs:
            is_patient = msg["sender"] == "patient"
            align  = "flex-end"   if is_patient else "flex-start"
            bg     = "#1e3a5f"    if is_patient else "#1a2e1a"
            label  = "You"        if is_patient else f"Dr. {doc['name']}"
            color  = "#93c5fd"    if is_patient else "#86efac"
            st.markdown(f"""
            <div style='display:flex;justify-content:{align};margin:6px 0'>
                <div style='background:{bg};border-radius:12px;padding:10px 16px;
                            max-width:70%'>
                    <div style='font-size:0.72rem;color:{color};margin-bottom:4px'>{label}</div>
                    <div style='color:#fff;font-size:0.92rem'>{msg['message']}</div>
                    <div style='font-size:0.68rem;color:#555;margin-top:4px;text-align:right'>
                        {msg['timestamp'][11:16]}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        new_msg = st.text_input("Type your message...", label_visibility="collapsed",
                                placeholder="Type your message here...")
    with col_btn:
        if st.button("Send 📤", use_container_width=True):
            if new_msg.strip():
                send_message(user["id"], doctor_id, "patient", new_msg.strip())
                st.rerun()
 
    if st.button("← Back to Doctor List"):
        st.session_state.page = "doctors"; st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL
# ════════════════════════════════════════════════════════════════════════════
def show_admin():
    st.markdown("<p class='big-title' style='font-size:2rem'>🔐 Admin Panel</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    tab1, tab2, tab3 = st.tabs(["👨‍⚕️ Manage Doctors", "📅 Appointments", "💬 Messages"])
 
    # ── Tab 1: Manage Doctors ─────────────────────────────────────────────
    with tab1:
        st.markdown("### ➕ Add New Doctor")
        c1, c2 = st.columns(2)
        with c1:
            name           = st.text_input("Full Name",         placeholder="Dr. Arjun Mehta")
            specialization = st.text_input("Specialization",    placeholder="Orthopedic Physiotherapy")
            experience     = st.number_input("Experience (years)", min_value=0, value=5)
            qualification  = st.text_input("Qualification",     placeholder="BPT, MPT")
        with c2:
            available_days = st.text_input("Available Days",    placeholder="Mon, Wed, Fri")
            timings        = st.text_input("Timings",            placeholder="9:00 AM – 5:00 PM")
            contact        = st.text_input("Contact",            placeholder="+91 98765 43210")
            about          = st.text_area("About",               placeholder="Brief description...")
 
        if st.button("✅ Add Doctor", use_container_width=True):
            if name and specialization:
                add_doctor(name, specialization, experience, qualification,
                           available_days, timings, about, contact)
                st.success(f"Dr. {name} added successfully!")
                st.rerun()
            else:
                st.warning("Name and specialization are required.")
 
        st.markdown("---")
        st.markdown("### 📋 All Doctors")
        doctors = get_all_doctors()
        if doctors:
            for doc in doctors:
                dc1, dc2 = st.columns([4, 1])
                with dc1:
                    st.markdown(f"""
                    <div class='card' style='padding:12px 16px'>
                        <b>Dr. {doc['name']}</b> &nbsp;
                        <span class='pill'>{doc['specialization']}</span>
                        <span class='pill'>{doc['experience']} yrs</span>
                        <span class='pill'>{doc['timings']}</span>
                    </div>""", unsafe_allow_html=True)
                with dc2:
                    if st.button("🗑️ Remove", key=f"del_doc_{doc['id']}",
                                 use_container_width=True):
                        delete_doctor(doc["id"]); st.rerun()
        else:
            st.info("No doctors added yet.")
 
    # ── Tab 2: Appointments ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 📅 All Appointments")
        appts = get_all_appointments()
        if not appts:
            st.info("No appointments yet.")
        else:
            STATUS_COLOR = {"pending":"#f59e0b","approved":"#22c55e","rejected":"#ef4444"}
            for a in appts:
                color = STATUS_COLOR.get(a["status"], "#888")
                st.markdown(f"""
                <div class='card' style='border-left:4px solid {color}'>
                    <div style='display:flex;justify-content:space-between'>
                        <div>
                            <b>{a['patient_name']}</b>
                            <span style='color:#64748b;font-size:0.85rem'> ({a['email']})</span>
                            → Dr. {a['doctor_name']}
                        </div>
                        <span style='color:{color};font-weight:600'>{a['status'].upper()}</span>
                    </div>
                    <p style='color:#64748b;font-size:0.85rem;margin:4px 0'>
                        📅 {a['date']}  🕐 {a['time']}  |  <i>{a['reason']}</i>
                    </p>
                </div>""", unsafe_allow_html=True)
 
                ac1, ac2, ac3 = st.columns([2, 1, 1])
                with ac1:
                    note = st.text_input("Note to patient", key=f"note_{a['id']}",
                                         placeholder="e.g. Please bring your reports")
                with ac2:
                    if st.button("✅ Approve", key=f"appr_{a['id']}",
                                 use_container_width=True):
                        update_appointment_status(a["id"], "approved", note); st.rerun()
                with ac3:
                    if st.button("❌ Reject", key=f"rej_{a['id']}",
                                 use_container_width=True):
                        update_appointment_status(a["id"], "rejected", note); st.rerun()
                st.markdown("<hr style='border-color:#222'>", unsafe_allow_html=True)
 
    # ── Tab 3: Messages ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### 💬 Patient Conversations")
        convos = get_all_conversations()
        if not convos:
            st.info("No messages yet.")
        else:
            for conv in convos:
                with st.expander(f"💬 {conv['patient_name']} → Dr. {conv['doctor_name']}  |  Last: {conv['last_message'][11:16]}"):
                    msgs = get_conversation(conv["user_id"], conv["doctor_id"])
                    for msg in msgs:
                        who   = conv["patient_name"] if msg["sender"] == "patient" else f"Dr. {conv['doctor_name']}"
                        color = "#93c5fd" if msg["sender"] == "patient" else "#86efac"
                        st.markdown(f"<span style='color:{color};font-weight:600'>{who}:</span> {msg['message']}",
                                    unsafe_allow_html=True)
 
                    st.markdown("**Reply as doctor:**")
                    rc1, rc2 = st.columns([4, 1])
                    with rc1:
                        reply = st.text_input("", key=f"reply_{conv['user_id']}_{conv['doctor_id']}",
                                              placeholder="Type reply...",
                                              label_visibility="collapsed")
                    with rc2:
                        if st.button("Send", key=f"send_{conv['user_id']}_{conv['doctor_id']}",
                                     use_container_width=True):
                            if reply.strip():
                                send_message(conv["user_id"], conv["doctor_id"],
                                             "doctor", reply.strip())
                                st.rerun()
 
    st.markdown("---")
    if st.button("🚪 Exit Admin Panel"):
        st.session_state.is_admin      = False
        st.session_state.admin_unlocked = False
        st.session_state.page          = "login"
        st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# HIDDEN ADMIN — URL param based, no visible button on main page
# ════════════════════════════════════════════════════════════════════════════
def show_admin_login():
    """Hidden — triggered only via ?admin=1 in URL or secret input."""
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    col = st.columns([2, 1, 2])
    with col[1]:
        if not st.session_state.admin_unlocked:
            if st.button("🔐", use_container_width=True, help="Admin access"):
                st.session_state.admin_unlocked = True
                st.rerun()
        else:
            code = st.text_input("", type="password",
                                 label_visibility="collapsed",
                                 placeholder="Enter admin code")
            if st.button("→", use_container_width=True):
                if code == ADMIN_CODE:
                    st.session_state.is_admin = True
                    st.session_state.page     = "admin"
                    st.rerun()
                else:
                    st.error("Wrong code.")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
# PAGE: PROFILE
# ================================================================================
def get_recommended_exercises(user):
    """
    Returns list of recommended exercise keys based on user's functional problems,
    pain location, medical conditions and rehab goals.
    """
    recommendations = {}
    text = " ".join([
        user.get("current_problem",""),
        user.get("pain_location",""),
        user.get("functional_problems",""),
        user.get("movement_restrictions",""),
        user.get("rehab_goals",""),
        user.get("medical_conditions",""),
    ]).lower()
 
    goals = user.get("rehab_goals","").lower()
 
    # Rules engine
    rules = {
        "squat": ["knee","leg","hip","mobility","strength","lower body","walking difficulty","bending"],
        "lunges": ["knee","hip","leg","balance","sports","lower body","strength"],
        "lateral_walks": ["hip","knee","balance","stability","walking","lateral"],
        "calf_raises": ["ankle","calf","foot","achilles","walking","standing"],
        "knee_raises": ["hip","core","lower back","abdominal","posture","sitting"],
        "shoulder_raises": ["shoulder","arm","upper body","rotator","neck","posture"],
        "crossover_arm_stretch": ["shoulder","upper back","stiffness","flexibility","mobility"],
        "tree_pose": ["balance","stability","posture","fall prevention","coordination"],
        "warrior_pose": ["strength","flexibility","posture","sports","return to activity"],
        "cat_cow_stretch": ["back","spine","lower back","stiffness","posture","flexibility","sitting","desk"],
    }
 
    for ex_key, keywords in rules.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            recommendations[ex_key] = score
 
    # Goal-based additions
    if "pain relief" in goals or "relief" in goals:
        for k in ["cat_cow_stretch","crossover_arm_stretch","tree_pose"]:
            recommendations[k] = recommendations.get(k,0) + 2
    if "posture" in goals:
        for k in ["cat_cow_stretch","knee_raises","shoulder_raises","tree_pose"]:
            recommendations[k] = recommendations.get(k,0) + 2
    if "mobility" in goals or "flexibility" in goals:
        for k in ["cat_cow_stretch","crossover_arm_stretch","warrior_pose","lunges"]:
            recommendations[k] = recommendations.get(k,0) + 2
    if "sports" in goals or "return" in goals:
        for k in ["squat","lunges","warrior_pose","lateral_walks"]:
            recommendations[k] = recommendations.get(k,0) + 2
    if "strength" in goals:
        for k in ["squat","lunges","warrior_pose","calf_raises"]:
            recommendations[k] = recommendations.get(k,0) + 2
 
    # Sort by score, return top 5
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [k for k,_ in sorted_recs[:5]] if sorted_recs else ["cat_cow_stretch","shoulder_raises","tree_pose"]
 
 
def show_profile():
    show_sidebar()
    user = st.session_state.user
    uid  = user["id"]
 
    # Refresh user from DB to get latest data
    fresh_user = get_user(uid)
    if fresh_user:
        user = fresh_user
        st.session_state.user = user
 
    sessions = get_user_sessions(uid)
    summary  = get_sessions_summary(uid)
    diet_all = get_diet_all(uid)
    alerts   = get_guardian_alerts(uid)
 
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center;padding:20px 0 8px'>
        <div style='font-size:4rem'>👤</div>
        <div style='font-size:2.2rem;font-weight:800;color:#f1f5f9;margin-top:8px'>
            {user['name']}
        </div>
        <div style='margin-top:8px'>
            <span class='pill'>{user.get('gender','') or 'Gender not set'}</span>
            <span class='pill'>{user.get('blood_group','') or 'Blood group not set'}</span>
            <span class='pill'>{user.get('occupation','') or 'Occupation not set'}</span>
            <span class='pill'>Member since {user['created_at'][:10]}</span>
        </div>
    </div>""", unsafe_allow_html=True)
 
    # Quick stats
    total_sessions = len(sessions)
    total_reps     = sum(s["reps"] for s in sessions)
    avg_form       = (sum(s["form_accuracy"] for s in sessions)/total_sessions) if total_sessions else 0
    s1,s2,s3,s4   = st.columns(4)
    s1.metric("Sessions",     total_sessions)
    s2.metric("Total Reps",   total_reps)
    s3.metric("Avg Form",     f"{avg_form:.0f}%")
    s4.metric("Meals Logged", len(diet_all))
 
    st.markdown("---")
 
    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Basic Details",
        "Medical History",
        "Pain Details",
        "Lifestyle",
        "Medications & Goals",
        "Exercise Plan",
        "Alerts & History",
    ])
 
    # ════════════════════════════════════════
    # TAB 1 — BASIC DETAILS
    # ════════════════════════════════════════
    with tab1:
        st.markdown("### Basic Details")
        c1, c2 = st.columns(2)
        with c1:
            b_name    = st.text_input("Full Name",     value=user.get("name",""))
            b_dob     = st.text_input("Date of Birth", value=user.get("dob","") or "",
                                       placeholder="DD/MM/YYYY")
            b_age     = st.number_input("Age", min_value=1, max_value=120,
                                         value=int(user.get("age") or 18))
            gender_opts = ["","Male","Female","Other","Prefer not to say"]
            cur_g       = user.get("gender","") or ""
            b_gender  = st.selectbox("Gender", gender_opts,
                                      index=gender_opts.index(cur_g) if cur_g in gender_opts else 0)
        with c2:
            b_contact = st.text_input("Contact Number", value=user.get("contact_number","") or "",
                                       placeholder="+91 XXXXX XXXXX")
            blood_opts = ["","A+","A-","B+","B-","AB+","AB-","O+","O-"]
            cur_b      = user.get("blood_group","") or ""
            b_blood   = st.selectbox("Blood Group", blood_opts,
                                      index=blood_opts.index(cur_b) if cur_b in blood_opts else 0)
            b_height  = st.number_input("Height (cm)", min_value=0.0,
                                         value=float(user.get("height_cm") or 0))
            b_weight  = st.number_input("Weight (kg)", min_value=0.0,
                                         value=float(user.get("weight_kg") or 0))
 
        b_occupation = st.text_input("Occupation",
                                      value=user.get("occupation","") or "",
                                      placeholder="e.g. Student, Desk job, Athlete, Teacher...")
 
        st.markdown("#### Emergency Contact")
        ec1, ec2 = st.columns(2)
        with ec1:
            b_ec_name  = st.text_input("Emergency Contact Name",
                                        value=user.get("emergency_contact_name","") or "")
        with ec2:
            b_ec_phone = st.text_input("Emergency Contact Phone",
                                        value=user.get("emergency_contact_phone","") or "",
                                        placeholder="+91 XXXXX XXXXX")
        b_guardian = st.text_input("Guardian WhatsApp (for alerts)",
                                    value=user.get("guardian_whatsapp","") or "",
                                    placeholder="+91 XXXXX XXXXX")
        b_doctor   = st.text_input("Your Doctor's Name",
                                    value=user.get("doctor_name","") or "")
 
        if st.button("Save Basic Details", use_container_width=True, key="save_basic"):
            update_user_profile(uid, {
                "name": b_name, "age": b_age, "dob": b_dob,
                "gender": b_gender, "contact_number": b_contact,
                "blood_group": b_blood, "height_cm": b_height,
                "weight_kg": b_weight, "occupation": b_occupation,
                "emergency_contact_name": b_ec_name,
                "emergency_contact_phone": b_ec_phone,
                "guardian_whatsapp": b_guardian,
                "doctor_name": b_doctor,
            })
            st.session_state.user = get_user(uid)
            st.success("Basic details saved!")
            st.rerun()
 
        # BMI
        h = float(user.get("height_cm") or 0)
        w = float(user.get("weight_kg") or 0)
        if h > 0 and w > 0:
            bmi = w / ((h/100)**2)
            if bmi < 18.5:   cat, col = "Underweight", "#f59e0b"
            elif bmi < 25:   cat, col = "Normal", "#22c55e"
            elif bmi < 30:   cat, col = "Overweight", "#f59e0b"
            else:             cat, col = "Obese", "#ef4444"
            st.markdown(f"""
            <div class='stat-box' style='border-color:{col};margin-top:16px'>
                <div class='stat-label'>BMI</div>
                <div class='stat-value' style='color:{col}'>{bmi:.1f}
                    <span style='font-size:1rem;color:#64748b'> — {cat}</span>
                </div>
                <div style='color:#64748b;font-size:0.8rem'>Height: {h}cm | Weight: {w}kg</div>
            </div>""", unsafe_allow_html=True)
 
    # ════════════════════════════════════════
    # TAB 2 — MEDICAL HISTORY
    # ════════════════════════════════════════
    with tab2:
        st.markdown("### Medical History")
        m_problem  = st.text_area("Current Problem (pain, injury, stiffness, etc.)",
                                   value=user.get("current_problem","") or "",
                                   placeholder="e.g. Lower back pain, right knee swelling, neck stiffness...",
                                   height=80)
        mc1, mc2 = st.columns(2)
        with mc1:
            m_start = st.text_input("When did it start?",
                                     value=user.get("problem_start_date","") or "",
                                     placeholder="e.g. 2 weeks ago, Jan 2024...")
        with mc2:
            cause_opts = ["","Accident","Sports injury","Poor posture","Repetitive strain",
                          "Age-related","Post-surgery","Unknown","Other"]
            cur_cause  = user.get("problem_cause","") or ""
            m_cause   = st.selectbox("Cause", cause_opts,
                                      index=cause_opts.index(cur_cause) if cur_cause in cause_opts else 0)
 
        m_prev_inj  = st.text_area("Previous Injuries (if any)",
                                    value=user.get("previous_injuries","") or "",
                                    placeholder="e.g. Right ankle sprain 2021, left shoulder dislocation...",
                                    height=60)
        m_surgeries = st.text_area("Past Surgeries",
                                    value=user.get("past_surgeries","") or "",
                                    placeholder="e.g. ACL reconstruction 2020, herniated disc surgery...",
                                    height=60)
        m_conditions = st.text_area("Ongoing Medical Conditions",
                                     value=user.get("medical_conditions","") or "",
                                     placeholder="e.g. Diabetes Type 2, Hypertension, Arthritis...",
                                     height=60)
 
        if st.button("Save Medical History", use_container_width=True, key="save_medical"):
            update_user_profile(uid, {
                "current_problem": m_problem,
                "problem_start_date": m_start,
                "problem_cause": m_cause,
                "previous_injuries": m_prev_inj,
                "past_surgeries": m_surgeries,
                "medical_conditions": m_conditions,
            })
            st.session_state.user = get_user(uid)
            st.success("Medical history saved!")
            st.rerun()
 
    # ════════════════════════════════════════
    # TAB 3 — PAIN DETAILS
    # ════════════════════════════════════════
    with tab3:
        st.markdown("### Pain Details")
        p_location = st.text_input("Location of Pain",
                                    value=user.get("pain_location","") or "",
                                    placeholder="e.g. Lower back, right knee, left shoulder, neck...")
 
        p_intensity = st.slider("Pain Intensity (0 = no pain, 10 = worst pain)",
                                 min_value=0, max_value=10,
                                 value=int(user.get("pain_intensity") or 0))
        pain_colors = {0:"#22c55e",1:"#84cc16",2:"#a3e635",3:"#fbbf24",4:"#fb923c",
                       5:"#f97316",6:"#ef4444",7:"#dc2626",8:"#b91c1c",9:"#991b1b",10:"#7f1d1d"}
        pain_labels = {0:"No pain",1:"Minimal",2:"Mild",3:"Moderate",4:"Moderate",5:"Moderate",
                       6:"Severe",7:"Severe",8:"Very severe",9:"Very severe",10:"Worst possible"}
        pcol = pain_colors.get(p_intensity,"#ef4444")
        st.markdown(f"""
        <div style='background:{pcol}22;border-left:4px solid {pcol};
                    border-radius:10px;padding:10px 16px;margin-bottom:12px'>
            <span style='color:{pcol};font-weight:700;font-size:1.1rem'>
                Pain Level: {p_intensity}/10 — {pain_labels.get(p_intensity,"")}
            </span>
        </div>""", unsafe_allow_html=True)
 
        pc1, pc2 = st.columns(2)
        with pc1:
            type_opts  = ["","Sharp","Dull/Aching","Burning","Throbbing","Stabbing","Cramping","Stiffness"]
            cur_type   = user.get("pain_type","") or ""
            p_type    = st.selectbox("Type of Pain", type_opts,
                                      index=type_opts.index(cur_type) if cur_type in type_opts else 0)
        with pc2:
            dur_opts  = ["","Constant","Comes and goes","Only during activity","Only at rest","Morning only","Night only"]
            cur_dur   = user.get("pain_duration","") or ""
            p_duration = st.selectbox("Duration", dur_opts,
                                       index=dur_opts.index(cur_dur) if cur_dur in dur_opts else 0)
 
        p_triggers = st.text_area("When does it increase / decrease?",
                                   value=user.get("pain_triggers","") or "",
                                   placeholder="e.g. Increases when sitting for long, decreases with rest / heat...",
                                   height=70)
 
        if st.button("Save Pain Details", use_container_width=True, key="save_pain"):
            update_user_profile(uid, {
                "pain_location": p_location,
                "pain_intensity": p_intensity,
                "pain_type": p_type,
                "pain_duration": p_duration,
                "pain_triggers": p_triggers,
            })
            st.session_state.user = get_user(uid)
            st.success("Pain details saved!")
            st.rerun()
 
    # ════════════════════════════════════════
    # TAB 4 — LIFESTYLE
    # ════════════════════════════════════════
    with tab4:
        st.markdown("### Lifestyle & Activity")
        l_sitting = st.slider("Daily Sitting Hours", 0, 18,
                               value=int(user.get("daily_sitting_hours") or 0))
        lc1, lc2 = st.columns(2)
        with lc1:
            act_opts  = ["","Sedentary (mostly sitting)","Lightly active","Moderately active",
                          "Very active","Extremely active"]
            cur_act   = user.get("activity_level","") or ""
            l_activity = st.selectbox("Activity Level", act_opts,
                                       index=act_opts.index(cur_act) if cur_act in act_opts else 0)
        with lc2:
            l_sports = st.text_input("Sports Involvement",
                                      value=user.get("sports_involvement","") or "",
                                      placeholder="e.g. Cricket, Swimming, Gym, None...")
 
        l_exercise = st.text_area("Exercise Habits",
                                   value=user.get("exercise_habits","") or "",
                                   placeholder="e.g. Walk 30 min daily, gym 3x/week, no exercise currently...",
                                   height=70)
 
        if l_sitting >= 8:
            st.warning("You sit for long hours. Cat-Cow Stretch and Knee Raises are strongly recommended for you.")
 
        if st.button("Save Lifestyle Info", use_container_width=True, key="save_lifestyle"):
            update_user_profile(uid, {
                "daily_sitting_hours": l_sitting,
                "activity_level": l_activity,
                "sports_involvement": l_sports,
                "exercise_habits": l_exercise,
            })
            st.session_state.user = get_user(uid)
            st.success("Lifestyle info saved!")
            st.rerun()
 
    # ════════════════════════════════════════
    # TAB 5 — MEDICATIONS, FUNCTIONAL PROBLEMS & GOALS
    # ════════════════════════════════════════
    with tab5:
        st.markdown("### Medications & Reports")
        med_meds    = st.text_area("Current Medications",
                                    value=user.get("current_medications","") or "",
                                    placeholder="e.g. Ibuprofen 400mg, Calcium supplement...",
                                    height=70)
        med_reports = st.text_area("X-rays / MRI / Reports Available",
                                    value=user.get("reports_available","") or "",
                                    placeholder="e.g. MRI lower back done Jan 2024, X-ray knee normal...",
                                    height=60)
        med_allergy = st.text_area("Allergies or Precautions",
                                    value=user.get("allergies","") or "",
                                    placeholder="e.g. Allergic to Aspirin, avoid cold compress...",
                                    height=60)
 
        st.markdown("---")
        st.markdown("### Functional Problems")
        func_opts = ["Walking difficulty","Sitting difficulty","Bending difficulty",
                     "Lifting difficulty","Climbing stairs","Standing for long",
                     "Sleep disturbance due to pain","Limited arm movement","Limited neck movement"]
        cur_func  = user.get("functional_problems","").split(",") if user.get("functional_problems") else []
        cur_func  = [f.strip() for f in cur_func if f.strip()]
        func_sel  = st.multiselect("Select all that apply", func_opts,
                                    default=[f for f in cur_func if f in func_opts])
        func_other = st.text_input("Other functional problems",
                                    value=user.get("movement_restrictions","") or "",
                                    placeholder="Describe any other movement restriction...")
        func_sleep = st.text_input("Sleep disturbance details",
                                    value=user.get("sleep_disturbance","") or "",
                                    placeholder="e.g. Cannot sleep on right side, wakes up at night due to pain...")
 
        st.markdown("---")
        st.markdown("### Rehabilitation Goals")
        goal_opts = ["Pain relief","Return to sports","Better mobility","Posture correction",
                     "Muscle strengthening","Weight management","Post-surgery recovery","General fitness"]
        cur_goals = user.get("rehab_goals","").split(",") if user.get("rehab_goals") else []
        cur_goals = [g.strip() for g in cur_goals if g.strip()]
        goal_sel  = st.multiselect("What do you want to achieve?", goal_opts,
                                    default=[g for g in cur_goals if g in goal_opts])
        goal_other = st.text_area("Describe your goals in detail",
                                   value="" if user.get("rehab_goals","") in goal_opts else user.get("rehab_goals","") or "",
                                   placeholder="e.g. I want to be able to play cricket again without knee pain...",
                                   height=60)
 
        st.markdown("---")
        st.markdown("### Consent & Declaration")
        consent_given = st.checkbox(
            "I consent to use this AI physiotherapy system and confirm the information provided is accurate.",
            value=bool(user.get("consent_given") or False)
        )
 
        if st.button("Save All", use_container_width=True, key="save_med_goals"):
            combined_goals = ", ".join(goal_sel)
            if goal_other.strip():
                combined_goals += (", " + goal_other.strip()) if combined_goals else goal_other.strip()
            update_user_profile(uid, {
                "current_medications": med_meds,
                "reports_available": med_reports,
                "allergies": med_allergy,
                "functional_problems": ", ".join(func_sel),
                "movement_restrictions": func_other,
                "sleep_disturbance": func_sleep,
                "rehab_goals": combined_goals,
                "consent_given": 1 if consent_given else 0,
            })
            st.session_state.user = get_user(uid)
            st.success("Saved successfully!")
            st.rerun()
 
    # ════════════════════════════════════════
    # TAB 6 — PERSONALIZED EXERCISE PLAN
    # ════════════════════════════════════════
    with tab6:
        st.markdown("### Your Personalized Exercise Plan")
 
        fresh = get_user(uid)
        has_profile = any([
            fresh.get("current_problem"),
            fresh.get("pain_location"),
            fresh.get("rehab_goals"),
            fresh.get("functional_problems"),
        ])
 
        if not has_profile:
            st.info("Fill in your Medical History, Pain Details, and Goals tabs first to get a personalized exercise plan!")
        else:
            recommended = get_recommended_exercises(fresh)
            user_lims   = get_user_limitations(fresh)
 
            # Profile summary
            st.markdown(f"""
            <div class='card' style='border-left:4px solid #6366f1;margin-bottom:16px'>
                <div style='font-weight:700;color:#a5b4fc;margin-bottom:8px'>Your Profile Summary</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                    <div><span style='color:#64748b;font-size:0.8rem'>Current Problem:</span>
                         <div style='color:#e2e8f0;font-size:0.88rem'>{fresh.get('current_problem','Not specified') or 'Not specified'}</div></div>
                    <div><span style='color:#64748b;font-size:0.8rem'>Pain Location:</span>
                         <div style='color:#e2e8f0;font-size:0.88rem'>{fresh.get('pain_location','Not specified') or 'Not specified'}</div></div>
                    <div><span style='color:#64748b;font-size:0.8rem'>Goals:</span>
                         <div style='color:#e2e8f0;font-size:0.88rem'>{fresh.get('rehab_goals','Not specified') or 'Not specified'}</div></div>
                    <div><span style='color:#64748b;font-size:0.8rem'>Pain Intensity:</span>
                         <div style='color:#e2e8f0;font-size:0.88rem'>{fresh.get('pain_intensity',0)}/10</div></div>
                </div>
            </div>""", unsafe_allow_html=True)
 
            st.markdown("#### Recommended Exercises For You")
 
            # Pain level warning
            pain_level = int(fresh.get("pain_intensity") or 0)
            if pain_level >= 8:
                st.error("Your pain intensity is very high (8+/10). Please consult your doctor before starting any exercise.")
            elif pain_level >= 6:
                st.warning("Your pain intensity is high. Start with gentle exercises only and stop if pain increases.")
            elif pain_level >= 4:
                st.info("Moderate pain detected. Perform exercises gently. Stop if discomfort increases.")
            else:
                st.success("Pain level is manageable. You can perform exercises as guided.")
 
            # Show recommended exercise cards
            chunks = [recommended[i:i+3] for i in range(0, len(recommended), 3)]
            for chunk in chunks:
                cols = st.columns(3)
                for col, ex_key in zip(cols, chunk):
                    ex   = EXERCISES.get(ex_key, {})
                    safe, warn_msg = is_exercise_safe(ex_key, fresh)
                    border = "#ef4444" if not safe else "#22c55e"
                    with col:
                        st.markdown(f"""
                        <div class='card' style='text-align:center;border-top:3px solid {border}'>
                            <div style='font-size:2.2rem'>{ex.get('icon','🏋')}</div>
                            <div style='font-weight:700;color:#f1f5f9;margin:6px 0'>{ex.get('label',ex_key)}</div>
                            <div style='color:#818cf8;font-size:0.78rem;margin-bottom:4px'>{ex.get('target','')}</div>
                            <span class='pill'>{ex.get('type','').upper()}</span>
                            {"<div style='color:#f87171;font-size:0.72rem;margin-top:4px'>Medical caution</div>" if not safe else "<div style='color:#4ade80;font-size:0.72rem;margin-top:4px'>Recommended for you</div>"}
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"Start {ex.get('label',ex_key)}", key=f"rec_{ex_key}",
                                     use_container_width=True):
                            st.session_state.selected_exercise = ex_key
                            st.session_state.detector          = ExerciseDetector(exercise=ex_key)
                            st.session_state.pose_detector     = PoseDetector()
                            st.session_state.history           = []
                            st.session_state.page              = "tracking"
                            st.rerun()
 
            # Non-recommended exercises
            not_recommended = [k for k in EXERCISES.keys() if k not in recommended]
            if not_recommended:
                with st.expander("Other available exercises (not prioritized for your condition)"):
                    cols = st.columns(3)
                    for i, ex_key in enumerate(not_recommended):
                        ex = EXERCISES.get(ex_key,{})
                        safe, _ = is_exercise_safe(ex_key, fresh)
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div class='card' style='text-align:center;opacity:0.7'>
                                <div style='font-size:1.8rem'>{ex.get('icon','🏋')}</div>
                                <div style='font-size:0.88rem;font-weight:600;color:#94a3b8'>{ex.get('label',ex_key)}</div>
                            </div>""", unsafe_allow_html=True)
                            if st.button(f"Start", key=f"other_{ex_key}", use_container_width=True):
                                st.session_state.selected_exercise = ex_key
                                st.session_state.detector          = ExerciseDetector(exercise=ex_key)
                                st.session_state.pose_detector     = PoseDetector()
                                st.session_state.history           = []
                                st.session_state.page              = "tracking"
                                st.rerun()
 
    # ════════════════════════════════════════
    # TAB 7 — ALERTS & HISTORY
    # ════════════════════════════════════════
    with tab7:
        st.markdown("### Guardian Alert History")
        guardian_wa = user.get("guardian_whatsapp","")
        if not guardian_wa:
            st.warning("No guardian WhatsApp set. Go to Basic Details to add one.")
        else:
            st.success(f"Guardian WhatsApp: {guardian_wa}")
 
        if not alerts:
            st.info("No alerts sent yet.")
        else:
            for a in alerts:
                st.markdown(f"""
                <div class='card' style='border-left:4px solid #ef4444'>
                    <div style='color:#f87171;font-size:0.78rem'>{a['timestamp'][:16]} — {a['alert_type']}</div>
                    <div style='color:#e2e8f0;margin-top:4px'>{a['message']}</div>
                    <div style='color:#64748b;font-size:0.78rem;margin-top:4px'>Sent to: {a['sent_to']}</div>
                </div>""", unsafe_allow_html=True)
 
        st.markdown("---")
        st.markdown("### Exercise History")
        recent = get_recent_sessions(uid, 15)
        if recent:
            import pandas as pd
            df = pd.DataFrame(recent)[["date","exercise","reps","form_accuracy"]]
            df["exercise"]      = df["exercise"].apply(lambda x: EXERCISES.get(x,{}).get("label",x))
            df["form_accuracy"] = df["form_accuracy"].apply(lambda x: f"{x}%")
            df["date"]          = pd.to_datetime(df["date"]).dt.strftime("%d %b %Y %H:%M")
            df.columns          = ["Date","Exercise","Reps","Form"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No sessions yet.")
 
 
def show_doctors():
    show_sidebar()
    user    = st.session_state.user
    doctors = get_all_doctors()
 
    st.markdown("<p class='big-title' style='font-size:2rem'>👨‍⚕️ Find a Doctor</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Browse our physiotherapy specialists and book an appointment</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if not doctors:
        st.info("No doctors available yet. Please check back later.")
        return
 
    for doc in doctors:
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"""
                <div class='card'>
                    <div style='display:flex;align-items:center;gap:16px'>
                        <div style='font-size:3rem'>👨‍⚕️</div>
                        <div>
                            <h3 style='margin:0'>{doc['name']}</h3>
                            <span class='pill'>{doc['specialization']}</span>
                            <span class='pill'>{doc['experience']} yrs exp</span>
                            <span class='pill'>{doc['qualification']}</span>
                        </div>
                    </div>
                    <hr style='border-color:#333;margin:12px 0'>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                        <div><span style='color:#64748b;font-size:0.8rem'>📅 Available:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['available_days']}</span></div>
                        <div><span style='color:#64748b;font-size:0.8rem'>🕐 Timings:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['timings']}</span></div>
                        <div><span style='color:#64748b;font-size:0.8rem'>📞 Contact:</span>
                             <span style='color:#cbd5e1;font-size:0.88rem'> {doc['contact']}</span></div>
                    </div>
                    <p style='color:#94a3b8;font-size:0.88rem;margin-top:10px'>{doc['about']}</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button(f"📅 Book Appointment", key=f"book_{doc['id']}",
                             use_container_width=True):
                    st.session_state["booking_doctor_id"] = doc["id"]
                    st.session_state.page = "book_appointment"
                    st.rerun()
                if st.button(f"💬 Send Message", key=f"msg_{doc['id']}",
                             use_container_width=True):
                    st.session_state["chat_doctor_id"] = doc["id"]
                    st.session_state.page = "chat"
                    st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: BOOK APPOINTMENT
# ════════════════════════════════════════════════════════════════════════════
def show_book_appointment():
    show_sidebar()
    user      = st.session_state.user
    doctor_id = st.session_state.get("booking_doctor_id")
    if not doctor_id:
        st.session_state.page = "doctors"; st.rerun()
    doc = get_doctor(doctor_id)
 
    st.markdown(f"<p class='big-title' style='font-size:2rem'>📅 Book Appointment</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    col_info, col_form = st.columns([1, 1])
    with col_info:
        st.markdown(f"""
        <div class='card'>
            <h3>👨‍⚕️ {doc['name']}</h3>
            <span class='pill'>{doc['specialization']}</span>
            <span class='pill'>{doc['experience']} yrs exp</span>
            <hr style='border-color:#333;margin:10px 0'>
            <p style='color:#64748b;font-size:0.85rem'>📅 {doc['available_days']}</p>
            <p style='color:#64748b;font-size:0.85rem'>🕐 {doc['timings']}</p>
        </div>""", unsafe_allow_html=True)
 
    with col_form:
        st.markdown("#### Fill in your appointment details")
        date   = st.date_input("Select Date")
        time   = st.selectbox("Select Time",
                    ["9:00 AM","10:00 AM","11:00 AM","12:00 PM",
                     "2:00 PM","3:00 PM","4:00 PM","5:00 PM","6:00 PM"])
        reason = st.text_area("Reason for appointment",
                    placeholder="Describe your issue or reason for visiting...")
 
        if st.button("✅ Confirm Booking", use_container_width=True):
            if reason.strip():
                book_appointment(user["id"], doctor_id, str(date), time, reason.strip())
                st.success(f"✅ Appointment booked with Dr. {doc['name']} on {date} at {time}!")
                st.session_state.page = "appointments"
                st.rerun()
            else:
                st.warning("Please enter a reason for your appointment.")
 
        if st.button("← Back to Doctors", use_container_width=True):
            st.session_state.page = "doctors"; st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: MY APPOINTMENTS
# ════════════════════════════════════════════════════════════════════════════
def show_appointments():
    show_sidebar()
    user  = st.session_state.user
    appts = get_user_appointments(user["id"])
 
    st.markdown("<p class='big-title' style='font-size:2rem'>📅 My Appointments</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if st.button("➕ Book New Appointment", use_container_width=False):
        st.session_state.page = "doctors"; st.rerun()
 
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
 
    if not appts:
        st.info("No appointments yet. Find a doctor and book your first appointment!")
        return
 
    STATUS_COLOR = {"pending": "#f59e0b", "approved": "#22c55e", "rejected": "#ef4444"}
 
    for a in appts:
        color = STATUS_COLOR.get(a["status"], "#888")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"""
            <div class='card' style='border-left:4px solid {color}'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <h4 style='margin:0'>Dr. {a['doctor_name']}</h4>
                    <span style='background:{color}33;color:{color};padding:4px 12px;
                           border-radius:20px;font-size:0.8rem;font-weight:600'>
                        {a['status'].upper()}
                    </span>
                </div>
                <span class='pill'>{a['specialization']}</span>
                <hr style='border-color:#333;margin:8px 0'>
                <p style='color:#64748b;font-size:0.85rem;margin:2px 0'>📅 {a['date']}  🕐 {a['time']}</p>
                <p style='color:#cbd5e1;font-size:0.88rem;margin:4px 0'><b>Reason:</b> {a['reason']}</p>
                {f"<p style='color:#22c55e;font-size:0.85rem'><b>Doctor note:</b> {a['admin_note']}</p>" if a['admin_note'] else ""}
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            if st.button("❌ Cancel", key=f"cancel_appt_{a['id']}",
                         use_container_width=True):
                cancel_appointment(a["id"]); st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: MESSAGES / CHAT
# ════════════════════════════════════════════════════════════════════════════
def show_messages():
    show_sidebar()
    user    = st.session_state.user
    doctors = get_all_doctors()
 
    st.markdown("<p class='big-title' style='font-size:2rem'>💬 Messages</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    if not doctors:
        st.info("No doctors available to message yet.")
        return
 
    doc_options = {d["id"]: f"Dr. {d['name']} — {d['specialization']}" for d in doctors}
    selected_id = st.selectbox("Select a doctor to message",
                               options=list(doc_options.keys()),
                               format_func=lambda x: doc_options[x])
 
    st.session_state["chat_doctor_id"] = selected_id
    show_chat()
 
 
def show_chat():
    show_sidebar()
    user      = st.session_state.user
    doctor_id = st.session_state.get("chat_doctor_id")
    if not doctor_id:
        st.session_state.page = "doctors"; st.rerun()
 
    doc  = get_doctor(doctor_id)
    msgs = get_conversation(user["id"], doctor_id)
 
    st.markdown(f"#### 💬 Chat with Dr. {doc['name']} — {doc['specialization']}")
    st.markdown("---")
 
    # Chat history
    chat_container = st.container()
    with chat_container:
        if not msgs:
            st.markdown("""
            <div style='text-align:center;padding:30px;color:#555'>
                No messages yet. Send your first message below!
            </div>""", unsafe_allow_html=True)
        for msg in msgs:
            is_patient = msg["sender"] == "patient"
            align  = "flex-end"   if is_patient else "flex-start"
            bg     = "#1e3a5f"    if is_patient else "#1a2e1a"
            label  = "You"        if is_patient else f"Dr. {doc['name']}"
            color  = "#93c5fd"    if is_patient else "#86efac"
            st.markdown(f"""
            <div style='display:flex;justify-content:{align};margin:6px 0'>
                <div style='background:{bg};border-radius:12px;padding:10px 16px;
                            max-width:70%'>
                    <div style='font-size:0.72rem;color:{color};margin-bottom:4px'>{label}</div>
                    <div style='color:#fff;font-size:0.92rem'>{msg['message']}</div>
                    <div style='font-size:0.68rem;color:#555;margin-top:4px;text-align:right'>
                        {msg['timestamp'][11:16]}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        new_msg = st.text_input("Type your message...", label_visibility="collapsed",
                                placeholder="Type your message here...")
    with col_btn:
        if st.button("Send 📤", use_container_width=True):
            if new_msg.strip():
                send_message(user["id"], doctor_id, "patient", new_msg.strip())
                st.rerun()
 
    if st.button("← Back to Doctor List"):
        st.session_state.page = "doctors"; st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL
# ════════════════════════════════════════════════════════════════════════════
def show_admin():
    st.markdown("<p class='big-title' style='font-size:2rem'>🔐 Admin Panel</p>",
                unsafe_allow_html=True)
    st.markdown("---")
 
    tab1, tab2, tab3 = st.tabs(["👨‍⚕️ Manage Doctors", "📅 Appointments", "💬 Messages"])
 
    # ── Tab 1: Manage Doctors ─────────────────────────────────────────────
    with tab1:
        st.markdown("### ➕ Add New Doctor")
        c1, c2 = st.columns(2)
        with c1:
            name           = st.text_input("Full Name",         placeholder="Dr. Arjun Mehta")
            specialization = st.text_input("Specialization",    placeholder="Orthopedic Physiotherapy")
            experience     = st.number_input("Experience (years)", min_value=0, value=5)
            qualification  = st.text_input("Qualification",     placeholder="BPT, MPT")
        with c2:
            available_days = st.text_input("Available Days",    placeholder="Mon, Wed, Fri")
            timings        = st.text_input("Timings",            placeholder="9:00 AM – 5:00 PM")
            contact        = st.text_input("Contact",            placeholder="+91 98765 43210")
            about          = st.text_area("About",               placeholder="Brief description...")
 
        if st.button("✅ Add Doctor", use_container_width=True):
            if name and specialization:
                add_doctor(name, specialization, experience, qualification,
                           available_days, timings, about, contact)
                st.success(f"Dr. {name} added successfully!")
                st.rerun()
            else:
                st.warning("Name and specialization are required.")
 
        st.markdown("---")
        st.markdown("### 📋 All Doctors")
        doctors = get_all_doctors()
        if doctors:
            for doc in doctors:
                dc1, dc2 = st.columns([4, 1])
                with dc1:
                    st.markdown(f"""
                    <div class='card' style='padding:12px 16px'>
                        <b>Dr. {doc['name']}</b> &nbsp;
                        <span class='pill'>{doc['specialization']}</span>
                        <span class='pill'>{doc['experience']} yrs</span>
                        <span class='pill'>{doc['timings']}</span>
                    </div>""", unsafe_allow_html=True)
                with dc2:
                    if st.button("🗑️ Remove", key=f"del_doc_{doc['id']}",
                                 use_container_width=True):
                        delete_doctor(doc["id"]); st.rerun()
        else:
            st.info("No doctors added yet.")
 
    # ── Tab 2: Appointments ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 📅 All Appointments")
        appts = get_all_appointments()
        if not appts:
            st.info("No appointments yet.")
        else:
            STATUS_COLOR = {"pending":"#f59e0b","approved":"#22c55e","rejected":"#ef4444"}
            for a in appts:
                color = STATUS_COLOR.get(a["status"], "#888")
                st.markdown(f"""
                <div class='card' style='border-left:4px solid {color}'>
                    <div style='display:flex;justify-content:space-between'>
                        <div>
                            <b>{a['patient_name']}</b>
                            <span style='color:#64748b;font-size:0.85rem'> ({a['email']})</span>
                            → Dr. {a['doctor_name']}
                        </div>
                        <span style='color:{color};font-weight:600'>{a['status'].upper()}</span>
                    </div>
                    <p style='color:#64748b;font-size:0.85rem;margin:4px 0'>
                        📅 {a['date']}  🕐 {a['time']}  |  <i>{a['reason']}</i>
                    </p>
                </div>""", unsafe_allow_html=True)
 
                ac1, ac2, ac3 = st.columns([2, 1, 1])
                with ac1:
                    note = st.text_input("Note to patient", key=f"note_{a['id']}",
                                         placeholder="e.g. Please bring your reports")
                with ac2:
                    if st.button("✅ Approve", key=f"appr_{a['id']}",
                                 use_container_width=True):
                        update_appointment_status(a["id"], "approved", note); st.rerun()
                with ac3:
                    if st.button("❌ Reject", key=f"rej_{a['id']}",
                                 use_container_width=True):
                        update_appointment_status(a["id"], "rejected", note); st.rerun()
                st.markdown("<hr style='border-color:#222'>", unsafe_allow_html=True)
 
    # ── Tab 3: Messages ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### 💬 Patient Conversations")
        convos = get_all_conversations()
        if not convos:
            st.info("No messages yet.")
        else:
            for conv in convos:
                with st.expander(f"💬 {conv['patient_name']} → Dr. {conv['doctor_name']}  |  Last: {conv['last_message'][11:16]}"):
                    msgs = get_conversation(conv["user_id"], conv["doctor_id"])
                    for msg in msgs:
                        who   = conv["patient_name"] if msg["sender"] == "patient" else f"Dr. {conv['doctor_name']}"
                        color = "#93c5fd" if msg["sender"] == "patient" else "#86efac"
                        st.markdown(f"<span style='color:{color};font-weight:600'>{who}:</span> {msg['message']}",
                                    unsafe_allow_html=True)
 
                    st.markdown("**Reply as doctor:**")
                    rc1, rc2 = st.columns([4, 1])
                    with rc1:
                        reply = st.text_input("", key=f"reply_{conv['user_id']}_{conv['doctor_id']}",
                                              placeholder="Type reply...",
                                              label_visibility="collapsed")
                    with rc2:
                        if st.button("Send", key=f"send_{conv['user_id']}_{conv['doctor_id']}",
                                     use_container_width=True):
                            if reply.strip():
                                send_message(conv["user_id"], conv["doctor_id"],
                                             "doctor", reply.strip())
                                st.rerun()
 
    st.markdown("---")
    if st.button("🚪 Exit Admin Panel"):
        st.session_state.is_admin      = False
        st.session_state.admin_unlocked = False
        st.session_state.page          = "login"
        st.rerun()
 
 
# ════════════════════════════════════════════════════════════════════════════
# HIDDEN ADMIN — URL param based, no visible button on main page
# ════════════════════════════════════════════════════════════════════════════
def show_admin_login():
    """Hidden — triggered only via ?admin=1 in URL or secret input."""
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    col = st.columns([2, 1, 2])
    with col[1]:
        if not st.session_state.admin_unlocked:
            if st.button("🔐", use_container_width=True, help="Admin access"):
                st.session_state.admin_unlocked = True
                st.rerun()
        else:
            code = st.text_input("", type="password",
                                 label_visibility="collapsed",
                                 placeholder="Enter admin code")
            if st.button("→", use_container_width=True):
                if code == ADMIN_CODE:
                    st.session_state.is_admin = True
                    st.session_state.page     = "admin"
                    st.rerun()
                else:
                    st.error("Wrong code.")
 
 
# ════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
def show_footer():
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='border-top:1px solid #2d2d44;padding:30px 0;margin-top:20px;text-align:center'>
        <div style='font-size:1.4rem;font-weight:700;color:#a5b4fc;margin-bottom:6px'>
            🏥 PhysioAI
        </div>
        <div style='color:#64748b;font-size:0.85rem;margin-bottom:12px'>
            AI-powered physiotherapy assistant for rehabilitation and fitness tracking
        </div>
        <div style='color:#475569;font-size:0.82rem;margin-bottom:10px'>
            Built with ❤️ by the PhysioAI Team
        </div>
        <div style='margin-bottom:8px'>
            <span style='color:#64748b;font-size:0.82rem'>📧 Contact us: &nbsp;</span>
            <a href='mailto:physioteam@gmail.com'
               style='color:#818cf8;font-size:0.85rem;text-decoration:none'>
                physioteam@gmail.com
            </a>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <a href='mailto:support@physioai.in'
               style='color:#818cf8;font-size:0.85rem;text-decoration:none'>
                support@physioai.in
            </a>
        </div>
        <div style='color:#334155;font-size:0.78rem;margin-top:10px'>
            © 2026 PhysioAI. All rights reserved.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.is_admin and st.session_state.page == "admin":
    show_admin()
elif st.session_state.user is None:
    show_login()
    show_admin_login()
else:
    pages = {
        "welcome":          show_welcome,
        "profile":          show_profile,
        "select_exercise":  show_exercise_selection,
        "setup":            show_setup,
        "tracking":         show_tracking,
        "diet":             show_diet,
        "progress":         show_progress,
        "notes":            show_notes,
        "doctors":          show_doctors,
        "book_appointment": show_book_appointment,
        "appointments":     show_appointments,
        "messages":         show_messages,
        "chat":             show_chat,
    }
    pages.get(st.session_state.page, show_welcome)()
    show_footer()
 