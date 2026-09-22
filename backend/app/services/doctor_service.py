"""Doctor directory service.

Authorization posture
---------------------
Self-registration as a clinician does **not** confer clinical authority. Only
verified profiles are discoverable or bookable, so a random internet user cannot
appear in a patient's directory by choosing ``role: "doctor"`` at signup. A
clinician whose profile is still pending can always read their *own* profile, so
the portal can explain the pending state.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, DoctorProfile, ROLE_DOCTOR
from app.services.access import link_exists
from app.services.errors import NotFoundError


def get_profile(db: Session, doctor_profile_id: int, requester: Account | None = None) -> DoctorProfile:
    """Return a clinician profile, or raise ``NotFoundError``.

    Unverified profiles are indistinguishable from non-existent ones, except to
    the account that owns them.
    """
    profile = db.get(DoctorProfile, doctor_profile_id)
    if profile is None:
        raise NotFoundError("Doctor not found.")
    if not profile.is_verified:
        owns_profile = (
            requester is not None
            and requester.role == ROLE_DOCTOR
            and profile.account_id == requester.id
        )
        if not owns_profile:
            # 404 rather than 403: an unverified profile should not be
            # enumerable, and confirming its existence would leak onboarding
            # state.
            raise NotFoundError("Doctor not found.")
    return profile


def list_doctor_profiles(db: Session, requester: Account) -> list[DoctorProfile]:
    """Return clinicians visible to ``requester``.

    Verified clinicians only, ordered with the requester's existing contacts
    first. A doctor's own profile is always included, even while it is pending
    verification, so the clinical console can render itself.
    """
    profiles = list(
        db.execute(select(DoctorProfile).where(DoctorProfile.is_verified.is_(True)))
        .scalars()
        .all()
    )

    if requester.role == ROLE_DOCTOR:
        own = db.execute(
            select(DoctorProfile).where(DoctorProfile.account_id == requester.id)
        ).scalar_one_or_none()
        if own is not None and all(profile.id != own.id for profile in profiles):
            profiles.append(own)

    linked_ids = [p.id for p in profiles if link_exists(db, p.id, requester.id)]
    profiles.sort(key=lambda p: (p.id not in linked_ids, p.name.lower()))
    return profiles
