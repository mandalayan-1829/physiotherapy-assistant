"""Authorization helpers.

Authorization is always enforced on the server. A patient may only reach their
own records; a doctor may only reach patients they hold an active link with.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, DoctorProfile, PatientDoctorLink, PatientProfile
from app.services.errors import ForbiddenError, NotFoundError


def get_doctor_profile(db: Session, account: Account) -> DoctorProfile:
    profile = db.execute(
        select(DoctorProfile).where(DoctorProfile.account_id == account.id)
    ).scalar_one_or_none()
    if profile is None:
        raise NotFoundError("No clinical profile is associated with this account.")
    return profile


def get_patient_profile(db: Session, account_id: int) -> PatientProfile | None:
    return db.execute(
        select(PatientProfile).where(PatientProfile.account_id == account_id)
    ).scalar_one_or_none()


def link_exists(db: Session, doctor_profile_id: int, patient_account_id: int) -> bool:
    return (
        db.execute(
            select(PatientDoctorLink.id).where(
                PatientDoctorLink.doctor_profile_id == doctor_profile_id,
                PatientDoctorLink.patient_account_id == patient_account_id,
                PatientDoctorLink.status == "active",
            )
        ).scalar_one_or_none()
        is not None
    )


def ensure_link(db: Session, doctor_profile_id: int, patient_account_id: int) -> None:
    """Idempotently create an active patient ↔ doctor link."""
    if link_exists(db, doctor_profile_id, patient_account_id):
        return
    db.add(
        PatientDoctorLink(
            patient_account_id=patient_account_id, doctor_profile_id=doctor_profile_id
        )
    )
    db.commit()


def linked_patient_ids(db: Session, doctor_profile_id: int) -> list[int]:
    rows = db.execute(
        select(PatientDoctorLink.patient_account_id).where(
            PatientDoctorLink.doctor_profile_id == doctor_profile_id,
            PatientDoctorLink.status == "active",
        )
    ).scalars().all()
    return list(rows)


def ensure_patient_access(db: Session, requester: Account, patient_account_id: int) -> None:
    """Raise ``ForbiddenError`` unless ``requester`` may read this patient's data."""
    if requester.role == "patient":
        if requester.id != patient_account_id:
            raise ForbiddenError("You are not allowed to access another patient's records.")
        return

    profile = get_doctor_profile(db, requester)
    if not link_exists(db, profile.id, patient_account_id):
        raise ForbiddenError(
            "You are not assigned to this patient and cannot access their records."
        )


def resolve_patient_scope(db: Session, requester: Account, requested_patient_id: int | None) -> int:
    """Determine which patient's data a request applies to, enforcing access."""
    if requester.role == "patient":
        if requested_patient_id is not None and requested_patient_id != requester.id:
            raise ForbiddenError("You are not allowed to access another patient's records.")
        return requester.id

    # Doctor: an explicit patient id is required and must be linked.
    if requested_patient_id is None:
        raise ForbiddenError("A patient_account_id is required for doctor requests.")
    ensure_patient_access(db, requester, requested_patient_id)
    return requested_patient_id
