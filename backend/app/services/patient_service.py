"""Patient profile reads and updates."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, PatientProfile
from app.schemas.patient import PatientProfileUpdate
from app.services.access import get_patient_profile, linked_patient_ids
from app.services.errors import NotFoundError


def profile_for(db: Session, patient_account_id: int) -> PatientProfile:
    profile = get_patient_profile(db, patient_account_id)
    if profile is None:
        # Profiles are created on registration; this covers legacy accounts.
        profile = PatientProfile(account_id=patient_account_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def update_profile(db: Session, patient_account_id: int, payload: PatientProfileUpdate) -> PatientProfile:
    profile = profile_for(db, patient_account_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(profile, field, value)
    db.commit()
    db.refresh(profile)
    return profile


def summarise_patients(db: Session, patients: list[Account]) -> list[dict]:
    """Build the clinician-facing patient summary list."""
    if not patients:
        return []
    ids = [p.id for p in patients]
    profiles = {
        p.account_id: p
        for p in db.execute(
            select(PatientProfile).where(PatientProfile.account_id.in_(ids))
        ).scalars().all()
    }
    summaries = []
    for account in patients:
        profile = profiles.get(account.id)
        summaries.append(
            {
                "account_id": account.id,
                "name": account.full_name,
                "email": account.email,
                "current_problem": profile.current_problem if profile else "",
                "pain_intensity": profile.pain_intensity if profile else 0,
            }
        )
    return summaries


def patients_for_doctor(db: Session, doctor_profile_id: int) -> list[Account]:
    ids = linked_patient_ids(db, doctor_profile_id)
    if not ids:
        return []
    return list(
        db.execute(select(Account).where(Account.id.in_(ids))).scalars().all()
    )


__all__ = [
    "profile_for",
    "update_profile",
    "summarise_patients",
    "patients_for_doctor",
    "NotFoundError",
]
