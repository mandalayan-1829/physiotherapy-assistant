"""Doctor directory service."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, DoctorProfile
from app.services.access import link_exists
from app.services.errors import NotFoundError


def get_profile(db: Session, doctor_profile_id: int) -> DoctorProfile:
    profile = db.get(DoctorProfile, doctor_profile_id)
    if profile is None:
        raise NotFoundError("Doctor not found.")
    return profile


def list_doctor_profiles(db: Session, patient_account: Account) -> list[DoctorProfile]:
    """Return clinicians visible to this patient.

    Clinicians the patient already has a relationship with are shown first,
    followed by the rest of the directory.
    """
    profiles = list(db.execute(select(DoctorProfile)).scalars().all())
    linked_ids = [
        p.id
        for p in profiles
        if link_exists(db, p.id, patient_account.id)
    ]
    profiles.sort(key=lambda p: (p.id not in linked_ids, p.name.lower()))
    return profiles
