"""Monthly report schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReportGenerateRequest(BaseModel):
    month_key: str = Field(pattern=r"^\d{4}-\d{2}$", examples=["2026-09"])
    patient_account_id: int | None = None


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    month_key: str
    month_name: str
    payload: dict
    generated_at: datetime
