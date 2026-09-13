"""
Pydantic schemas for session data.
"""

from pydantic import BaseModel
from typing import Optional


class SessionResponse(BaseModel):
    id: int
    user_id: int
    exercise: str
    reps: int
    form_accuracy: int
    duration_sec: int = 0
    notes: str = ""
    date: Optional[str] = None


class SessionSummary(BaseModel):
    exercise: str
    total_sessions: int
    total_reps: int
    avg_form: float


class ProgressResponse(BaseModel):
    total_sessions: int
    total_reps: int
    avg_form_accuracy: float
    exercises_done: int
    summary: list[SessionSummary]
    recent_sessions: list[SessionResponse]
