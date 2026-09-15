"""Telehealth messages and safety alerts."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, DoctorProfile, GuardianAlert, Message
from app.schemas.alert import AlertCreate, MessageCreate
from app.services.access import ensure_link, ensure_patient_access, get_doctor_profile
from app.services.errors import ForbiddenError, NotFoundError


# --- Messages ---------------------------------------------------------------


def list_messages(db: Session, requester: Account, patient_account_id: int) -> list[Message]:
    ensure_patient_access(db, requester, patient_account_id)
    stmt = select(Message).where(Message.patient_account_id == patient_account_id)
    if requester.role == "doctor":
        profile = get_doctor_profile(db, requester)
        stmt = stmt.where(Message.doctor_profile_id == profile.id)
    return list(db.execute(stmt.order_by(Message.created_at.asc())).scalars().all())


def send_message(db: Session, requester: Account, payload: MessageCreate) -> Message:
    doctor = db.get(DoctorProfile, payload.doctor_profile_id)
    if doctor is None:
        raise NotFoundError("Doctor not found.")

    if requester.role == "patient":
        ensure_link(db, doctor.id, requester.id)
        patient_account_id = requester.id
        sender = "patient"
    else:
        profile = get_doctor_profile(db, requester)
        if profile.id != doctor.id:
            raise ForbiddenError("You can only reply as yourself.")
        if payload.patient_account_id is None:
            raise ForbiddenError("patient_account_id is required for clinician replies.")
        ensure_patient_access(db, requester, payload.patient_account_id)
        patient_account_id = payload.patient_account_id
        sender = "doctor"

    message = Message(
        patient_account_id=patient_account_id,
        doctor_profile_id=doctor.id,
        sender=sender,
        message=payload.message,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


# --- Alerts -----------------------------------------------------------------


def list_alerts(db: Session, requester: Account, patient_account_id: int) -> list[GuardianAlert]:
    ensure_patient_access(db, requester, patient_account_id)
    return list(
        db.execute(
            select(GuardianAlert)
            .where(GuardianAlert.patient_account_id == patient_account_id)
            .order_by(GuardianAlert.created_at.desc())
        ).scalars().all()
    )


def create_alert(db: Session, patient: Account, payload: AlertCreate) -> GuardianAlert:
    alert = GuardianAlert(
        patient_account_id=patient.id,
        alert_type=payload.alert_type,
        message=payload.message,
        sent_to=payload.sent_to,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert
