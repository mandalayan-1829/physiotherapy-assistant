"""Doctor / physiotherapist profile."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DoctorProfile(Base):
    """Clinical profile for a doctor / physiotherapist.

    ``account_id`` is nullable so that directory-only clinicians carried over
    from the legacy database are preserved even before they register a login
    account to claim the profile.

    ``is_verified`` is the gate that stops self-registration from conferring
    clinical authority (OWASP API6). An unverified clinician is invisible in the
    patient-facing directory, cannot be booked, and cannot acquire access to a
    patient's records. Verification is an out-of-band operations action
    (``backend/scripts/verify_doctor.py``) because this application deliberately
    has no admin role.
    """

    __tablename__ = "doctor_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int | None] = mapped_column(
        ForeignKey("accounts.id", ondelete="SET NULL"), unique=True, nullable=True
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialization: Mapped[str] = mapped_column(String(255), nullable=False)
    experience: Mapped[int] = mapped_column(Integer, default=0)
    qualification: Mapped[str] = mapped_column(String(255), default="")
    available_days: Mapped[str] = mapped_column(String(120), default="")
    timings: Mapped[str] = mapped_column(String(120), default="")
    about: Mapped[str] = mapped_column(Text, default="")
    contact: Mapped[str] = mapped_column(String(50), default="")
    whatsapp: Mapped[str] = mapped_column(String(50), default="")
    email: Mapped[str] = mapped_column(String(255), default="")
    hospital: Mapped[str] = mapped_column(String(255), default="")
    rating: Mapped[float] = mapped_column(Float, default=0.0)

    # Deny-by-default: a clinician is not bookable or discoverable until an
    # operator verifies them.
    is_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    account: Mapped["Account | None"] = relationship(back_populates="doctor_profile")  # noqa: F821
