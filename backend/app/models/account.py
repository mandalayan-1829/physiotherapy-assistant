"""User account model.

A single ``accounts`` table represents every authenticated identity. The
``role`` column is constrained to exactly two values: ``patient`` and
``doctor`` (doctor / physiotherapist).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

ROLE_PATIENT = "patient"
ROLE_DOCTOR = "doctor"


class Account(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        CheckConstraint("role IN ('patient', 'doctor')", name="role_valid"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Incremented whenever the password changes. Tokens embed this value, so a
    # password reset invalidates every previously issued session token.
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Retained only while migrating unsalted SHA-256 digests from aiphysio.db.
    # Cleared as soon as the owner logs in and the password is re-hashed.
    legacy_password_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    patient_profile: Mapped["PatientProfile | None"] = relationship(  # noqa: F821
        back_populates="account", cascade="all, delete-orphan", uselist=False
    )
    doctor_profile: Mapped["DoctorProfile | None"] = relationship(  # noqa: F821
        back_populates="account", cascade="all, delete-orphan", uselist=False
    )

    @property
    def is_patient(self) -> bool:
        return self.role == ROLE_PATIENT

    @property
    def is_doctor(self) -> bool:
        return self.role == ROLE_DOCTOR
