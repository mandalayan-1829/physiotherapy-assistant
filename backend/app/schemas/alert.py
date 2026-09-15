"""Safety alert and telehealth message schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AlertCreate(BaseModel):
    alert_type: Literal["pain_spike", "workout_milestone", "emergency_help", "missed_routine"]
    message: str = Field(min_length=1)
    sent_to: str = ""


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    alert_type: str
    message: str
    sent_to: str
    created_at: datetime


class MessageCreate(BaseModel):
    doctor_profile_id: int
    patient_account_id: int | None = None
    message: str = Field(min_length=1)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    doctor_profile_id: int
    sender: str
    message: str
    created_at: datetime
