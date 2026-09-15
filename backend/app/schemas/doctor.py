"""Doctor / physiotherapist schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DoctorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    specialization: str
    experience: int
    qualification: str
    available_days: str
    timings: str
    about: str
    contact: str
    whatsapp: str
    email: str
    hospital: str
    rating: float


class PatientSummary(BaseModel):
    """Minimal patient record exposed to an authorized doctor."""

    account_id: int
    name: str
    email: str
    current_problem: str = ""
    pain_intensity: int = 0
