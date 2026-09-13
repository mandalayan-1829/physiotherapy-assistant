"""
Pydantic schemas for exercise data.
"""

from pydantic import BaseModel
from typing import Optional


class ExerciseInfo(BaseModel):
    id: str
    label: str
    icon: str
    target: str
    type: str  # "physio" or "yoga"
    video_id: str
    form_checks: list[str]
    tip: str
    limitations: list[str]
    description: Optional[str] = None
    difficulty: Optional[str] = "moderate"
    recommended_reps: int = 10


class ExerciseListResponse(BaseModel):
    exercises: list[ExerciseInfo]
    total: int


class PoseState(BaseModel):
    """Real-time pose analysis result sent over WebSocket."""
    rep_count: int
    angle: float
    stage: str
    feedback: str
    form_status: str  # "good", "warning", "incorrect", "no_pose"
    form_errors: list[str]
    form_ok: bool
    exercise: str
    hold_count: int = 0


class SessionStartRequest(BaseModel):
    exercise: str
    target_reps: int = 10


class SessionEndRequest(BaseModel):
    session_id: int
    reps: int
    form_accuracy: int
    duration_sec: int = 0
    notes: str = ""
