"""
Tests for backend services.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.exercise_service import (
    get_all_exercises,
    get_exercise,
    get_user_limitations,
    is_exercise_safe,
)


def test_get_all_exercises():
    exercises = get_all_exercises()
    assert len(exercises) == 10
    ids = [e["id"] for e in exercises]
    assert "squat" in ids
    assert "tree_pose" in ids


def test_get_exercise():
    ex = get_exercise("squat")
    assert ex is not None
    assert ex["label"] == "Squat"
    assert ex["type"] == "physio"


def test_get_exercise_not_found():
    ex = get_exercise("nonexistent")
    assert ex is None


def test_user_limitations():
    user = {
        "medical_conditions": "knee pain, previous ACL surgery",
        "exercise_limitations": "",
    }
    lims = get_user_limitations(user)
    assert "knee_pain" in lims


def test_user_no_limitations():
    user = {
        "medical_conditions": "",
        "exercise_limitations": "",
    }
    lims = get_user_limitations(user)
    assert len(lims) == 0


def test_exercise_safe_no_limitations():
    user = {"medical_conditions": "", "exercise_limitations": ""}
    safe, msg = is_exercise_safe("squat", user)
    assert safe is True


def test_exercise_unsafe_with_knee_pain():
    user = {"medical_conditions": "knee pain", "exercise_limitations": ""}
    safe, msg = is_exercise_safe("squat", user)
    assert safe is False
    assert "knee" in msg.lower()
