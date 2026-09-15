"""Clinical note service."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, DoctorNote, DoctorProfile
from app.schemas.note import NoteCreate
from app.services.access import ensure_patient_access, get_doctor_profile
from app.services.errors import ForbiddenError, NotFoundError


def list_notes(db: Session, requester: Account, patient_account_id: int) -> list[DoctorNote]:
    ensure_patient_access(db, requester, patient_account_id)
    return list(
        db.execute(
            select(DoctorNote)
            .where(DoctorNote.patient_account_id == patient_account_id)
            .order_by(DoctorNote.created_at.desc())
        ).scalars().all()
    )


def create_note(db: Session, requester: Account, payload: NoteCreate) -> DoctorNote:
    """Create a note.

    A patient may only record notes on their own record (``doctor_profile_id``
    stays null). A clinician must be linked to the patient.
    """
    if requester.role == "patient":
        if payload.patient_account_id is not None and payload.patient_account_id != requester.id:
            raise ForbiddenError("You are not allowed to write notes for another patient.")
        patient_account_id = requester.id
        doctor_profile_id = None
    else:
        if payload.patient_account_id is None:
            raise ForbiddenError("patient_account_id is required for clinician notes.")
        ensure_patient_access(db, requester, payload.patient_account_id)
        profile = get_doctor_profile(db, requester)
        patient_account_id = payload.patient_account_id
        doctor_profile_id = profile.id

    note = DoctorNote(
        patient_account_id=patient_account_id,
        doctor_profile_id=doctor_profile_id,
        note_text=payload.note_text,
        category=payload.category,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def delete_note(db: Session, requester: Account, note_id: int) -> None:
    note = db.get(DoctorNote, note_id)
    if note is None:
        raise NotFoundError("Note not found.")

    if requester.role == "patient":
        # Patients may only delete the journal notes they authored themselves.
        if note.patient_account_id != requester.id or note.doctor_profile_id is not None:
            raise ForbiddenError("You can only delete notes you authored.")
    else:
        profile = db.execute(
            select(DoctorProfile).where(DoctorProfile.account_id == requester.id)
        ).scalar_one_or_none()
        if profile is None or note.doctor_profile_id != profile.id:
            raise ForbiddenError("You can only delete notes you authored.")

    db.delete(note)
    db.commit()
