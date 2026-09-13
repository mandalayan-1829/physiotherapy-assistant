"""
Session routes — start, end, and list exercise sessions.
"""

from fastapi import APIRouter, HTTPException, Depends

from backend.schemas.exercise import SessionStartRequest, SessionEndRequest
from backend.schemas.session import SessionResponse
from backend.api.dependencies import get_current_user
from backend.services.session_service import (
    create_session,
    get_sessions,
    get_recent,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def start_session(data: SessionStartRequest, user: dict = Depends(get_current_user)):
    """Create a new session record (called when user starts an exercise).
    The actual session state lives on the client / WebSocket.
    """
    from backend.services.exercise_service import get_exercise

    if not get_exercise(data.exercise):
        raise HTTPException(status_code=400, detail=f"Unknown exercise: {data.exercise}")

    # Create a session record with 0 reps initially
    create_session(
        user_id=user["id"],
        exercise=data.exercise,
        reps=0,
        form_accuracy=0,
        notes="",
    )
    # Get the latest session (the one we just created)
    sessions = get_sessions(user["id"])
    if sessions:
        return SessionResponse(**sessions[0])
    raise HTTPException(status_code=500, detail="Failed to create session")


@router.put("/{session_id}/end")
def end_session(
    session_id: int,
    data: SessionEndRequest,
    user: dict = Depends(get_current_user),
):
    """End a session and record final results."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from core.database import get_connection

    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET reps=?, form_accuracy=?, duration_sec=?, notes=? WHERE id=? AND user_id=?",
        (data.reps, data.form_accuracy, data.duration_sec, data.notes, session_id, user["id"]),
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": "Session saved"}


@router.get("", response_model=list[SessionResponse])
def list_sessions(user: dict = Depends(get_current_user)):
    """Get all sessions for the current user."""
    sessions = get_sessions(user["id"])
    return [SessionResponse(**s) for s in sessions]


@router.get("/recent", response_model=list[SessionResponse])
def list_recent_sessions(user: dict = Depends(get_current_user)):
    """Get the 5 most recent sessions."""
    sessions = get_recent(user["id"], limit=5)
    return [SessionResponse(**s) for s in sessions]
