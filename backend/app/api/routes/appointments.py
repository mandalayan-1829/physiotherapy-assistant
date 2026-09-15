"""Appointment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentAccount, CurrentPatient, DbSession
from app.schemas.appointment import AppointmentCreate, AppointmentOut, AppointmentUpdate
from app.services import appointment_service

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.get("", response_model=list[AppointmentOut])
def list_appointments(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[AppointmentOut]:
    return appointment_service.list_appointments(db, account, patient_account_id)


@router.post("", response_model=AppointmentOut, status_code=status.HTTP_201_CREATED)
def book_appointment(
    payload: AppointmentCreate, patient: CurrentPatient, db: DbSession
) -> AppointmentOut:
    return appointment_service.create_appointment(db, patient, payload)


@router.patch("/{appointment_id}", response_model=AppointmentOut)
def update_appointment(
    appointment_id: int, payload: AppointmentUpdate, account: CurrentAccount, db: DbSession
) -> AppointmentOut:
    """Status/notes updates. Patients may only cancel; doctors may do more."""
    return appointment_service.update_appointment(db, account, appointment_id, payload)
