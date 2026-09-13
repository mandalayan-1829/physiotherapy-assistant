"""
Pydantic schemas for user data.
"""

from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    name: str
    age: int
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    gender: Optional[str] = ""
    email: str
    blood_group: Optional[str] = ""
    height_cm: Optional[float] = 0
    weight_kg: Optional[float] = 0
    medical_conditions: Optional[str] = ""
    exercise_limitations: Optional[str] = ""
    rehab_goals: Optional[str] = None
    created_at: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    blood_group: Optional[str] = None
    medical_conditions: Optional[str] = None
    current_medications: Optional[str] = None
    exercise_limitations: Optional[str] = None
    current_problem: Optional[str] = None
    problem_start_date: Optional[str] = None
    problem_cause: Optional[str] = None
    previous_injuries: Optional[str] = None
    past_surgeries: Optional[str] = None
    pain_location: Optional[str] = None
    pain_intensity: Optional[int] = None
    pain_type: Optional[str] = None
    pain_triggers: Optional[str] = None
    pain_duration: Optional[str] = None
    daily_sitting_hours: Optional[int] = None
    activity_level: Optional[str] = None
    exercise_habits: Optional[str] = None
    sports_involvement: Optional[str] = None
    rehab_goals: Optional[str] = None
    allergies: Optional[str] = None
    precautions: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    guardian_whatsapp: Optional[str] = None
    doctor_name: Optional[str] = None
    occupation: Optional[str] = None
    functional_problems: Optional[str] = None
    sleep_disturbance: Optional[str] = None
    movement_restrictions: Optional[str] = None
    consent_given: Optional[int] = None


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None
