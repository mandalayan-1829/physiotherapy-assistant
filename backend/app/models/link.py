"""Patient ↔ doctor relationship.

This table is the source of truth for authorization: a doctor may only read a
patient's records when an active link exists between them.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PatientDoctorLink(Base):
    __tablename__ = "patient_doctor_links"
    __table_args__ = (
        UniqueConstraint("patient_account_id", "doctor_profile_id", name="patient_doctor"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doctor_profile_id: Mapped[int] = mapped_column(
        ForeignKey("doctor_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(30), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
