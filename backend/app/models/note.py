"""Clinical note written by a doctor about a patient."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

NOTE_CATEGORIES = ("clinical", "exercise", "symptom", "general")


class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Null for notes a patient writes in their own journal; set when a
    # clinician authors the note.
    doctor_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("doctor_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )

    note_text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(30), default="clinical")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
