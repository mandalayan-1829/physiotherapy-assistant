"""Appointment schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AppointmentStatusLiteral = Literal[
    "pending",
    "scheduled",
    "confirmed",
    "approved",
    "ready",
    "completed",
    "cancelled",
]


class AppointmentCreate(BaseModel):
    doctor_profile_id: int
    date: str = Field(min_length=4, max_length=30)
    time: str = Field(min_length=2, max_length=30)
    reason: str = ""


class AppointmentUpdate(BaseModel):
    status: AppointmentStatusLiteral | None = None
    clinician_note: str | None = None
    date: str | None = None
    time: str | None = None
    reason: str | None = None


class AppointmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    doctor_profile_id: int
    patient_name: str
    doctor_name: str
    specialization: str
    email: str
    date: str
    time: str
    reason: str
    status: str
    clinician_note: str
    created_at: datetime
