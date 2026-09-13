"""
Progress routes — analytics and statistics.
"""

from fastapi import APIRouter, Depends

from backend.schemas.session import ProgressResponse, SessionSummary, SessionResponse
from backend.api.dependencies import get_current_user
from backend.services.progress_service import get_progress, get_streak

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("", response_model=ProgressResponse)
def user_progress(user: dict = Depends(get_current_user)):
    """Get full progress data for the current user."""
    data = get_progress(user["id"])
    return ProgressResponse(
        total_sessions=data["total_sessions"],
        total_reps=data["total_reps"],
        avg_form_accuracy=data["avg_form_accuracy"],
        exercises_done=data["exercises_done"],
        summary=[SessionSummary(**s) for s in data["summary"]],
        recent_sessions=[SessionResponse(**s) for s in data["recent_sessions"]],
    )


@router.get("/summary")
def progress_summary(user: dict = Depends(get_current_user)):
    """Get a quick progress summary."""
    data = get_progress(user["id"])
    streak = get_streak(user["id"])
    return {
        "total_sessions": data["total_sessions"],
        "total_reps": data["total_reps"],
        "avg_form_accuracy": data["avg_form_accuracy"],
        "exercises_done": data["exercises_done"],
        "current_streak": streak,
    }
