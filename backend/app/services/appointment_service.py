"""Appointment booking and status management."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, Appointment, DoctorProfile
from app.schemas.appointment import AppointmentCreate, AppointmentUpdate
from app.services.access import ensure_doctor_verified, ensure_link, linked_patient_ids
from app.services.errors import ForbiddenError, NotFoundError


def list_appointments(db: Session, requester: Account, patient_account_id: int | None) -> list[Appointment]:
    if requester.role == "patient":
        stmt = select(Appointment).where(Appointment.patient_account_id == requester.id)
    else:
        # Doctor: only appointments that involve them.
        profile = db.execute(
            select(DoctorProfile).where(DoctorProfile.account_id == requester.id)
        ).scalar_one_or_none()
        if profile is None:
            return []
        stmt = select(Appointment).where(Appointment.doctor_profile_id == profile.id)
        if patient_account_id is not None:
            if patient_account_id not in linked_patient_ids(db, profile.id):
                raise ForbiddenError("You are not assigned to this patient.")
            stmt = stmt.where(Appointment.patient_account_id == patient_account_id)
    return list(db.execute(stmt.order_by(Appointment.date.desc())).scalars().all())


def create_appointment(
    db: Session, patient: Account, payload: AppointmentCreate
) -> Appointment:
    doctor = db.get(DoctorProfile, payload.doctor_profile_id)
    if doctor is None:
        raise NotFoundError("Doctor not found.")

    # An unverified clinician is not bookable. This is also what stops the
    # booking from silently creating the doctor -> patient access link below.
    ensure_doctor_verified(doctor)

    appointment = Appointment(
        patient_account_id=patient.id,
        doctor_profile_id=doctor.id,
        patient_name=patient.full_name,
        doctor_name=doctor.name,
        specialization=doctor.specialization,
        email=patient.email,
        date=payload.date,
        time=payload.time,
        reason=payload.reason,
        status="pending",
    )
    db.add(appointment)
    db.commit()
    db.refresh(appointment)

    # Booking a checkup establishes the clinical relationship.
    ensure_link(db, doctor.id, patient.id)
    return appointment


def update_appointment(
    db: Session, requester: Account, appointment_id: int, payload: AppointmentUpdate
) -> Appointment:
    appointment = db.get(Appointment, appointment_id)
    if appointment is None:
        raise NotFoundError("Appointment not found.")

    if requester.role == "patient":
        if appointment.patient_account_id != requester.id:
            raise ForbiddenError("This appointment does not belong to you.")
        # Patients may only cancel their own appointment.
        if payload.status not in (None, "cancelled"):
            raise ForbiddenError("Patients may only cancel an appointment.")
    else:
        profile = db.execute(
            select(DoctorProfile).where(DoctorProfile.account_id == requester.id)
        ).scalar_one_or_none()
        if profile is None or appointment.doctor_profile_id != profile.id:
            raise ForbiddenError("This appointment is not assigned to you.")

    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(appointment, field, value)
    db.commit()
    db.refresh(appointment)
    return appointment
