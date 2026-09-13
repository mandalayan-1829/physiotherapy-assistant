"""
Session service — wraps database session CRUD operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.database import (
    save_session,
    get_user_sessions,
    get_sessions_summary,
    get_recent_sessions,
)


def create_session(
    user_id: int,
    exercise: str,
    reps: int,
    form_accuracy: int,
    duration_sec: int = 0,
    notes: str = "",
) -> None:
    """Save a completed exercise session."""
    save_session(user_id, exercise, reps, form_accuracy, duration_sec, notes)


def get_sessions(user_id: int) -> list[dict]:
    """Get all sessions for a user, newest first."""
    return get_user_sessions(user_id)


def get_summary(user_id: int) -> list[dict]:
    """Get aggregated session summary grouped by exercise."""
    return get_sessions_summary(user_id)


def get_recent(user_id: int, limit: int = 5) -> list[dict]:
    """Get the most recent sessions."""
    return get_recent_sessions(user_id, limit)
