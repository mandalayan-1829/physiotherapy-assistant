"""Patient medical profile schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientProfileUpdate(BaseModel):
    """Every field is optional so partial updates are possible."""

    age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = None
    dob: str | None = None
    contact_number: str | None = None
    blood_group: str | None = None
    height_cm: float | None = Field(default=None, ge=0)
    weight_kg: float | None = Field(default=None, ge=0)
    occupation: str | None = None

    current_problem: str | None = None
    problem_start_date: str | None = None
    problem_cause: str | None = None
    previous_injuries: str | None = None
    past_surgeries: str | None = None

    medical_conditions: str | None = None
    current_medications: str | None = None
    allergies: str | None = None
    precautions: str | None = None
    exercise_limitations: str | None = None

    pain_location: str | None = None
    pain_intensity: int | None = Field(default=None, ge=0, le=10)
    pain_type: str | None = None
    pain_triggers: str | None = None
    pain_duration: str | None = None

    daily_sitting_hours: int | None = Field(default=None, ge=0, le=24)
    activity_level: str | None = None
    exercise_habits: str | None = None
    movement_restrictions: str | None = None
    rehab_goals: str | None = None

    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None
    guardian_whatsapp: str | None = None
    doctor_name: str | None = None


class PatientProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    account_id: int
    age: int | None
    gender: str
    dob: str
    contact_number: str
    blood_group: str
    height_cm: float
    weight_kg: float
    occupation: str

    current_problem: str
    problem_start_date: str
    problem_cause: str
    previous_injuries: str
    past_surgeries: str

    medical_conditions: str
    current_medications: str
    allergies: str
    precautions: str
    exercise_limitations: str

    pain_location: str
    pain_intensity: int
    pain_type: str
    pain_triggers: str
    pain_duration: str

    daily_sitting_hours: int
    activity_level: str
    exercise_habits: str
    movement_restrictions: str
    rehab_goals: str

    emergency_contact_name: str
    emergency_contact_phone: str
    guardian_whatsapp: str
    doctor_name: str


class MeResponse(BaseModel):
    """Combined identity + profile payload returned to the frontend."""

    account: "AccountOut"
    profile: PatientProfileOut | None = None
    doctor_profile: "DoctorOut | None" = None


from app.schemas.auth import AccountOut  # noqa: E402
from app.schemas.doctor import DoctorOut  # noqa: E402

MeResponse.model_rebuild()
