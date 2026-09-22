"""Exercise session and progress schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: How the metrics in the row were produced. Only ``pose_inference`` represents a
#: real measurement; see ``app.models.session``.
MetricsSource = Literal["pose_inference", "simulated", "manual"]


class WorkoutSessionCreate(BaseModel):
    exercise: str = Field(min_length=1, max_length=120)
    exercise_label: str = Field(min_length=1, max_length=255)
    reps: int = Field(default=0, ge=0, le=10_000)
    target_reps: int = Field(default=0, ge=0, le=10_000)
    form_accuracy: int = Field(default=0, ge=0, le=100)
    duration_sec: int = Field(default=0, ge=0, le=86_400)
    notes: str = Field(default="", max_length=2000)
    performed_at: datetime | None = None
    # Defaults to the least-trusted value: a client that does not explicitly
    # claim real pose inference is not recorded as having measured anything.
    metrics_source: MetricsSource = "manual"


class WorkoutSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    exercise: str
    exercise_label: str
    reps: int
    target_reps: int
    form_accuracy: int
    duration_sec: int
    notes: str
    metrics_source: str
    performed_at: datetime


class DailyProgressOut(BaseModel):
    date: str
    sessions: int
    total_reps: int
    #: Averaged over measured (``pose_inference``) sessions only.
    avg_form_accuracy: float
    total_duration_sec: int
    #: Number of sessions that actually contained a real measurement.
    measured_sessions: int = 0
