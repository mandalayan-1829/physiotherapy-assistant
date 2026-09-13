"""
Pydantic schemas for doctor and appointment data.
"""

from pydantic import BaseModel
from typing import Optional


class DoctorResponse(BaseModel):
    id: int
    name: str
    specialization: str
    experience: int
    qualification: Optional[str] = ""
    available_days: Optional[str] = ""
    timings: Optional[str] = ""
    about: Optional[str] = ""
    contact: Optional[str] = ""
    whatsapp: Optional[str] = ""
    email: Optional[str] = ""


class AppointmentCreate(BaseModel):
    doctor_id: int
    date: str
    time: str
    reason: str


class AppointmentResponse(BaseModel):
    id: int
    user_id: int
    doctor_id: int
    doctor_name: Optional[str] = None
    specialization: Optional[str] = None
    patient_name: Optional[str] = None
    date: str
    time: str
    reason: Optional[str] = ""
    status: str = "pending"
    admin_note: Optional[str] = ""
    created_at: Optional[str] = None


class NoteCreate(BaseModel):
    note_text: str


class NoteResponse(BaseModel):
    id: int
    user_id: int
    note_text: str
    date: Optional[str] = None


class MessageCreate(BaseModel):
    doctor_id: int
    message: str


class MessageResponse(BaseModel):
    id: int
    user_id: int
    doctor_id: int
    sender: str
    message: str
    timestamp: Optional[str] = None
