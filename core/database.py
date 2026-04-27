"""
core/database.py
----------------
SQLite database — all tables and CRUD functions.
Tables: users, sessions, diet_log, notes, doctors,
        appointments, messages, alerts
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiphysio.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c    = conn.cursor()

    # ── Users (extended with medical + emergency fields) ──────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        name                TEXT NOT NULL,
        age                 INTEGER,
        gender              TEXT DEFAULT '',
        email               TEXT UNIQUE NOT NULL,
        password            TEXT NOT NULL,
        blood_group         TEXT DEFAULT '',
        height_cm           REAL DEFAULT 0,
        weight_kg           REAL DEFAULT 0,
        medical_conditions  TEXT DEFAULT '',
        current_medications TEXT DEFAULT '',
        exercise_limitations TEXT DEFAULT '',
        emergency_contact_name  TEXT DEFAULT '',
        emergency_contact_phone TEXT DEFAULT '',
        guardian_whatsapp   TEXT DEFAULT '',
        doctor_name         TEXT DEFAULT '',
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── Migrate old users table if missing new columns ────────────────────
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()]
    new_cols = [
        # Basic
        ("gender",                  "TEXT DEFAULT ''"),
        ("dob",                     "TEXT DEFAULT ''"),
        ("contact_number",          "TEXT DEFAULT ''"),
        ("blood_group",             "TEXT DEFAULT ''"),
        ("height_cm",               "REAL DEFAULT 0"),
        ("weight_kg",               "REAL DEFAULT 0"),
        ("occupation",              "TEXT DEFAULT ''"),
        # Medical history
        ("current_problem",         "TEXT DEFAULT ''"),
        ("problem_start_date",      "TEXT DEFAULT ''"),
        ("problem_cause",           "TEXT DEFAULT ''"),
        ("previous_injuries",       "TEXT DEFAULT ''"),
        ("past_surgeries",          "TEXT DEFAULT ''"),
        ("medical_conditions",      "TEXT DEFAULT ''"),
        # Pain details
        ("pain_location",           "TEXT DEFAULT ''"),
        ("pain_intensity",          "INTEGER DEFAULT 0"),
        ("pain_type",               "TEXT DEFAULT ''"),
        ("pain_triggers",           "TEXT DEFAULT ''"),
        ("pain_duration",           "TEXT DEFAULT ''"),
        # Lifestyle
        ("daily_sitting_hours",     "INTEGER DEFAULT 0"),
        ("activity_level",          "TEXT DEFAULT ''"),
        ("exercise_habits",         "TEXT DEFAULT ''"),
        ("sports_involvement",      "TEXT DEFAULT ''"),
        # Medications & reports
        ("current_medications",     "TEXT DEFAULT ''"),
        ("reports_available",       "TEXT DEFAULT ''"),
        # Functional problems
        ("functional_problems",     "TEXT DEFAULT ''"),
        ("sleep_disturbance",       "TEXT DEFAULT ''"),
        ("movement_restrictions",   "TEXT DEFAULT ''"),
        # Goals
        ("rehab_goals",             "TEXT DEFAULT ''"),
        # Consent
        ("consent_given",           "INTEGER DEFAULT 0"),
        ("allergies",               "TEXT DEFAULT ''"),
        ("precautions",             "TEXT DEFAULT ''"),
        # Limitations & emergency
        ("exercise_limitations",    "TEXT DEFAULT ''"),
        ("emergency_contact_name",  "TEXT DEFAULT ''"),
        ("emergency_contact_phone", "TEXT DEFAULT ''"),
        ("guardian_whatsapp",       "TEXT DEFAULT ''"),
        ("doctor_name",             "TEXT DEFAULT ''"),
    ]
    for col_name, col_def in new_cols:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")

    # ── Sessions ──────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL,
        exercise      TEXT NOT NULL,
        reps          INTEGER DEFAULT 0,
        form_accuracy INTEGER DEFAULT 0,
        duration_sec  INTEGER DEFAULT 0,
        notes         TEXT DEFAULT '',
        date          TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")

    # ── Diet log ──────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS diet_log (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id  INTEGER NOT NULL,
        meal     TEXT NOT NULL,
        calories INTEGER DEFAULT 0,
        protein  REAL DEFAULT 0,
        carbs    REAL DEFAULT 0,
        fats     REAL DEFAULT 0,
        date     TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")

    # ── Notes ─────────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL,
        note_text TEXT NOT NULL,
        date      TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")

    # ── Doctors ───────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS doctors (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        name           TEXT NOT NULL,
        specialization TEXT NOT NULL,
        experience     INTEGER DEFAULT 0,
        qualification  TEXT,
        available_days TEXT,
        timings        TEXT,
        about          TEXT,
        contact        TEXT,
        whatsapp       TEXT DEFAULT '',
        email          TEXT DEFAULT '',
        created_at     TEXT DEFAULT CURRENT_TIMESTAMP)""")

    # ── Migrate old doctors table if missing new columns ──────────────────
    existing_doctor_cols = [row[1] for row in c.execute("PRAGMA table_info(doctors)").fetchall()]
    doctor_new_cols = [
        ("whatsapp", "TEXT DEFAULT ''"),
        ("email",    "TEXT DEFAULT ''"),
    ]
    for col_name, col_def in doctor_new_cols:
        if col_name not in existing_doctor_cols:
            c.execute(f"ALTER TABLE doctors ADD COLUMN {col_name} {col_def}")

    # ── Appointments ──────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS appointments (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        doctor_id  INTEGER NOT NULL,
        date       TEXT NOT NULL,
        time       TEXT NOT NULL,
        reason     TEXT,
        status     TEXT DEFAULT 'pending',
        admin_note TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (doctor_id) REFERENCES doctors(id))""")

    # ── Messages ──────────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        sender    TEXT NOT NULL,
        message   TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (doctor_id) REFERENCES doctors(id))""")

    # ── Guardian alerts ───────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS guardian_alerts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        alert_type TEXT NOT NULL,
        message    TEXT NOT NULL,
        sent_to    TEXT NOT NULL,
        timestamp  TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")

    conn.commit()
    conn.close()


# ── Password ──────────────────────────────────────────────────────────────────
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════════════════════════
def create_user(name, age, email, password):
    try:
        conn = get_connection()
        conn.execute("INSERT INTO users (name,age,email,password) VALUES (?,?,?,?)",
                     (name, age, email, hash_password(password)))
        conn.commit(); conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already registered. Please login."

def login_user(email, password):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                        (email, hash_password(password))).fetchone()
    conn.close()
    return (True, dict(user)) if user else (False, "Invalid email or password.")

def get_user(user_id):
    conn = get_connection()
    u = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(u) if u else None

def update_user_profile(user_id, data: dict):
    """Update any user fields. data = {column: value}"""
    if not data: return
    cols = ", ".join(f"{k}=?" for k in data.keys())
    vals = list(data.values()) + [user_id]
    conn = get_connection()
    conn.execute(f"UPDATE users SET {cols} WHERE id=?", vals)
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
def save_session(user_id, exercise, reps, form_accuracy, duration_sec=0, notes=""):
    conn = get_connection()
    conn.execute("""INSERT INTO sessions
        (user_id,exercise,reps,form_accuracy,duration_sec,notes)
        VALUES (?,?,?,?,?,?)""",
        (user_id, exercise, reps, form_accuracy, duration_sec, notes))
    conn.commit(); conn.close()

def get_user_sessions(user_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM sessions WHERE user_id=? ORDER BY date DESC",
                        (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_sessions_summary(user_id):
    conn = get_connection()
    rows = conn.execute("""SELECT exercise, COUNT(*) AS total_sessions,
        SUM(reps) AS total_reps, AVG(form_accuracy) AS avg_form
        FROM sessions WHERE user_id=? GROUP BY exercise""", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_sessions(user_id, limit=5):
    conn = get_connection()
    rows = conn.execute("""SELECT * FROM sessions WHERE user_id=?
        ORDER BY date DESC LIMIT ?""", (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# DIET
# ══════════════════════════════════════════════════════════════════════════════
def add_diet_entry(user_id, meal, calories, protein, carbs, fats):
    conn = get_connection()
    conn.execute("INSERT INTO diet_log (user_id,meal,calories,protein,carbs,fats) VALUES (?,?,?,?,?,?)",
                 (user_id, meal, calories, protein, carbs, fats))
    conn.commit(); conn.close()

def get_diet_today(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = get_connection()
    rows  = conn.execute("SELECT * FROM diet_log WHERE user_id=? AND date LIKE ?",
                         (user_id, f"{today}%")).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_diet_all(user_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM diet_log WHERE user_id=? ORDER BY date DESC",
                        (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_diet_entry(entry_id):
    conn = get_connection()
    conn.execute("DELETE FROM diet_log WHERE id=?", (entry_id,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# NOTES
# ══════════════════════════════════════════════════════════════════════════════
def add_note(user_id, note_text):
    conn = get_connection()
    conn.execute("INSERT INTO notes (user_id,note_text) VALUES (?,?)", (user_id, note_text))
    conn.commit(); conn.close()

def get_notes(user_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM notes WHERE user_id=? ORDER BY date DESC",
                        (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_note(note_id):
    conn = get_connection()
    conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# DOCTORS
# ══════════════════════════════════════════════════════════════════════════════
def add_doctor(name, specialization, experience, qualification,
               available_days, timings, about, contact, whatsapp="", email=""):
    conn = get_connection()
    conn.execute("""INSERT INTO doctors
        (name,specialization,experience,qualification,
         available_days,timings,about,contact,whatsapp,email)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (name, specialization, experience, qualification,
         available_days, timings, about, contact, whatsapp, email))
    conn.commit(); conn.close()

def get_all_doctors():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM doctors ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_doctor(doctor_id):
    conn = get_connection()
    d = conn.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,)).fetchone()
    conn.close()
    return dict(d) if d else None

def delete_doctor(doctor_id):
    conn = get_connection()
    conn.execute("DELETE FROM doctors WHERE id=?", (doctor_id,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# APPOINTMENTS
# ══════════════════════════════════════════════════════════════════════════════
def book_appointment(user_id, doctor_id, date, time, reason):
    conn = get_connection()
    conn.execute("INSERT INTO appointments (user_id,doctor_id,date,time,reason) VALUES (?,?,?,?,?)",
                 (user_id, doctor_id, date, time, reason))
    conn.commit(); conn.close()

def get_user_appointments(user_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.*, d.name AS doctor_name, d.specialization
        FROM appointments a JOIN doctors d ON a.doctor_id=d.id
        WHERE a.user_id=? ORDER BY a.created_at DESC""", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_appointments():
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.*, d.name AS doctor_name, u.name AS patient_name, u.email
        FROM appointments a
        JOIN doctors d ON a.doctor_id=d.id
        JOIN users u ON a.user_id=u.id
        ORDER BY a.created_at DESC""").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_appointment_status(appt_id, status, admin_note=""):
    conn = get_connection()
    conn.execute("UPDATE appointments SET status=?, admin_note=? WHERE id=?",
                 (status, admin_note, appt_id))
    conn.commit(); conn.close()

def cancel_appointment(appt_id):
    conn = get_connection()
    conn.execute("DELETE FROM appointments WHERE id=?", (appt_id,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGES
# ══════════════════════════════════════════════════════════════════════════════
def send_message(user_id, doctor_id, sender, message):
    conn = get_connection()
    conn.execute("INSERT INTO messages (user_id,doctor_id,sender,message) VALUES (?,?,?,?)",
                 (user_id, doctor_id, sender, message))
    conn.commit(); conn.close()

def get_conversation(user_id, doctor_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM messages WHERE user_id=? AND doctor_id=? ORDER BY timestamp ASC",
                        (user_id, doctor_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_conversations():
    conn = get_connection()
    rows = conn.execute("""
        SELECT DISTINCT m.user_id, m.doctor_id,
               u.name AS patient_name, d.name AS doctor_name,
               MAX(m.timestamp) AS last_message
        FROM messages m
        JOIN users u ON m.user_id=u.id
        JOIN doctors d ON m.doctor_id=d.id
        GROUP BY m.user_id, m.doctor_id
        ORDER BY last_message DESC""").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_unread_count(user_id):
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id=? AND sender='doctor'",
                         (user_id,)).fetchone()[0]
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# GUARDIAN ALERTS
# ══════════════════════════════════════════════════════════════════════════════
def log_guardian_alert(user_id, alert_type, message, sent_to):
    conn = get_connection()
    conn.execute("INSERT INTO guardian_alerts (user_id,alert_type,message,sent_to) VALUES (?,?,?,?)",
                 (user_id, alert_type, message, sent_to))
    conn.commit(); conn.close()

def get_guardian_alerts(user_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM guardian_alerts WHERE user_id=? ORDER BY timestamp DESC",
                        (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_db()