"""Diet record schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DietRecordCreate(BaseModel):
    meal: str = Field(min_length=1, max_length=255)
    calories: int = Field(default=0, ge=0, le=20_000)
    protein: float = Field(default=0.0, ge=0)
    carbs: float = Field(default=0.0, ge=0)
    fats: float = Field(default=0.0, ge=0)
    recorded_at: datetime | None = None


class DietRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    meal: str
    calories: int
    protein: float
    carbs: float
    fats: float
    recorded_at: datetime
