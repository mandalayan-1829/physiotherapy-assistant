"""
Doctor routes — browse doctors, book appointments, messages.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends

from backend.schemas.doctor import (
    DoctorResponse,
    AppointmentCreate,
    AppointmentResponse,
    MessageCreate,
    MessageResponse,
)
from backend.api.dependencies import get_current_user
from core.database import (
    get_all_doctors,
    get_doctor,
    book_appointment,
    get_user_appointments,
    get_all_appointments,
    update_appointment_status,
    cancel_appointment,
    send_message,
    get_conversation,
    get_all_conversations,
)

router = APIRouter(prefix="/api", tags=["doctors"])


# ── Doctors ────────────────────────────────────────────────────────────
@router.get("/doctors", response_model=list[DoctorResponse])
def list_doctors():
    """Get all available doctors."""
    docs = get_all_doctors()
    return [DoctorResponse(**d) for d in docs]


@router.get("/doctors/{doctor_id}", response_model=DoctorResponse)
def get_doctor_detail(doctor_id: int):
    """Get a specific doctor's details."""
    doc = get_doctor(doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return DoctorResponse(**doc)


# ── Appointments ───────────────────────────────────────────────────────
@router.post("/appointments", response_model=AppointmentResponse)
def create_appointment(data: AppointmentCreate, user: dict = Depends(get_current_user)):
    """Book an appointment with a doctor."""
    doc = get_doctor(data.doctor_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    book_appointment(user["id"], data.doctor_id, data.date, data.time, data.reason)
    # Return latest appointment
    appts = get_user_appointments(user["id"])
    if appts:
        return AppointmentResponse(**appts[0])
    raise HTTPException(status_code=500, detail="Failed to book appointment")


@router.get("/appointments", response_model=list[AppointmentResponse])
def list_appointments(user: dict = Depends(get_current_user)):
    """Get all appointments for the current user."""
    appts = get_user_appointments(user["id"])
    return [AppointmentResponse(**a) for a in appts]


@router.get("/admin/appointments", response_model=list[AppointmentResponse])
def list_all_appointments():
    """Get all appointments (admin view)."""
    appts = get_all_appointments()
    return [AppointmentResponse(**a) for a in appts]


@router.put("/appointments/{appt_id}/status")
def update_status(appt_id: int, status: str, admin_note: str = ""):
    """Update appointment status (admin)."""
    update_appointment_status(appt_id, status, admin_note)
    return {"success": True}


@router.delete("/appointments/{appt_id}")
def remove_appointment(appt_id: int):
    """Cancel an appointment."""
    cancel_appointment(appt_id)
    return {"success": True}


# ── Messages ───────────────────────────────────────────────────────────
@router.post("/messages", response_model=MessageResponse)
def send_msg(data: MessageCreate, user: dict = Depends(get_current_user)):
    """Send a message to a doctor."""
    send_message(user["id"], data.doctor_id, "patient", data.message)
    return MessageResponse(
        id=0,
        user_id=user["id"],
        doctor_id=data.doctor_id,
        sender="patient",
        message=data.message,
    )


@router.get("/messages/{doctor_id}", response_model=list[MessageResponse])
def get_msgs(doctor_id: int, user: dict = Depends(get_current_user)):
    """Get conversation with a doctor."""
    msgs = get_conversation(user["id"], doctor_id)
    return [MessageResponse(**m) for m in msgs]


@router.get("/messages")
def list_conversations(user: dict = Depends(get_current_user)):
    """List all conversations for the current user."""
    convos = get_all_conversations()
    # Filter to this user's conversations
    return [c for c in convos if c["user_id"] == user["id"]]
