"""
Progress service — computes user progress analytics.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.database import (
    get_user_sessions,
    get_sessions_summary,
)


def get_progress(user_id: int) -> dict:
    """Compute full progress data for a user."""
    sessions = get_user_sessions(user_id)
    summary = get_sessions_summary(user_id)

    total_sessions = len(sessions)
    total_reps = sum(s.get("reps", 0) for s in sessions)
    avg_form = (
        sum(s.get("form_accuracy", 0) for s in sessions) / total_sessions
        if total_sessions > 0
        else 0
    )

    return {
        "total_sessions": total_sessions,
        "total_reps": total_reps,
        "avg_form_accuracy": round(avg_form, 1),
        "exercises_done": len(summary),
        "summary": summary,
        "recent_sessions": sessions[:10],
    }


def get_streak(user_id: int) -> int:
    """Compute the current exercise streak (consecutive days with sessions)."""
    from datetime import datetime, timedelta

    sessions = get_user_sessions(user_id)
    if not sessions:
        return 0

    # Get unique dates (just the date part)
    dates = set()
    for s in sessions:
        d = s.get("date", "")
        if d:
            dates.add(d[:10])

    # Sort dates descending
    sorted_dates = sorted(dates, reverse=True)
    today = datetime.now().strftime("%Y-%m-%d")

    streak = 0
    current = datetime.strptime(today, "%Y-%m-%d")

    for d_str in sorted_dates:
        d = datetime.strptime(d_str, "%Y-%m-%d")
        if d.date() == current.date() or d.date() == (current - timedelta(days=1)).date():
            streak += 1
            current = d
        else:
            break

    return streak
