"""
Tests for core/database.py
"""

import sys
import os
import gc
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a temp database for testing
import core.database as db

original_db_path = db.DB_PATH
_temp_db_path = None


def setup_function():
    """Set up a fresh temp database before each test."""
    global _temp_db_path
    fd, _temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db.DB_PATH = _temp_db_path
    # Create schema in the temp database
    db.init_db()


def teardown_function():
    """Clean up temp database after each test."""
    db.DB_PATH = original_db_path
    gc.collect()
    if _temp_db_path and os.path.exists(_temp_db_path):
        try:
            os.unlink(_temp_db_path)
        except PermissionError:
            pass  # Windows may hold the file briefly


def test_create_user():
    ok, msg = db.create_user("Test User", 25, "test@example.com", "password123")
    assert ok is True
    assert "created" in msg.lower()


def test_duplicate_email():
    db.create_user("User 1", 25, "dup@example.com", "password123")
    ok, msg = db.create_user("User 2", 30, "dup@example.com", "password456")
    assert ok is False
    assert "already" in msg.lower()


def test_login_user():
    db.create_user("Login Test", 25, "login@example.com", "mypass123")
    ok, user = db.login_user("login@example.com", "mypass123")
    assert ok is True
    assert user["name"] == "Login Test"
    assert user["email"] == "login@example.com"


def test_login_wrong_password():
    db.create_user("Login Test", 25, "login2@example.com", "correct")
    ok, msg = db.login_user("login2@example.com", "wrong")
    assert ok is False


def test_get_user():
    db.create_user("Get Test", 25, "get@example.com", "pass")
    user = db.login_user("get@example.com", "pass")[1]
    fetched = db.get_user(user["id"])
    assert fetched is not None
    assert fetched["name"] == "Get Test"


def test_update_user_profile():
    db.create_user("Update Test", 25, "update@example.com", "pass")
    user = db.login_user("update@example.com", "pass")[1]
    db.update_user_profile(user["id"], {"name": "Updated Name", "age": 30})
    updated = db.get_user(user["id"])
    assert updated["name"] == "Updated Name"
    assert updated["age"] == 30


def test_save_and_get_sessions():
    db.create_user("Session Test", 25, "session@example.com", "pass")
    user = db.login_user("session@example.com", "pass")[1]
    db.save_session(user["id"], "squat", 10, 85)
    db.save_session(user["id"], "lunges", 8, 90)
    sessions = db.get_user_sessions(user["id"])
    assert len(sessions) == 2
    assert sessions[0]["exercise"] == "lunges"  # newest first (id DESC tiebreaker)


def test_diet():
    db.create_user("Diet Test", 25, "diet@example.com", "pass")
    user = db.login_user("diet@example.com", "pass")[1]
    db.add_diet_entry(user["id"], "Oats", 300, 10, 50, 5)
    today = db.get_diet_today(user["id"])
    assert len(today) == 1
    assert today[0]["meal"] == "Oats"
    all_meals = db.get_diet_all(user["id"])
    assert len(all_meals) == 1


def test_notes():
    db.create_user("Notes Test", 25, "notes@example.com", "pass")
    user = db.login_user("notes@example.com", "pass")[1]
    db.add_note(user["id"], "Test note")
    notes = db.get_notes(user["id"])
    assert len(notes) == 1
    assert notes[0]["note_text"] == "Test note"
    db.delete_note(notes[0]["id"])
    assert len(db.get_notes(user["id"])) == 0


def test_hash_password():
    """Password hashing should be deterministic."""
    h1 = db.hash_password("test")
    h2 = db.hash_password("test")
    assert h1 == h2
    assert h1 != "test"  # Should be hashed, not plaintext
