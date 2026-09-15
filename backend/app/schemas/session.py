"""Exercise session and progress schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WorkoutSessionCreate(BaseModel):
    exercise: str = Field(min_length=1, max_length=120)
    exercise_label: str = Field(min_length=1, max_length=255)
    reps: int = Field(default=0, ge=0, le=10_000)
    target_reps: int = Field(default=0, ge=0, le=10_000)
    form_accuracy: int = Field(default=0, ge=0, le=100)
    duration_sec: int = Field(default=0, ge=0, le=86_400)
    notes: str = ""
    performed_at: datetime | None = None


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
    performed_at: datetime


class DailyProgressOut(BaseModel):
    date: str
    sessions: int
    total_reps: int
    avg_form_accuracy: float
    total_duration_sec: int
