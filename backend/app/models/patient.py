"""Patient medical profile."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # Demographics
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str] = mapped_column(String(50), default="")
    dob: Mapped[str] = mapped_column(String(20), default="")
    contact_number: Mapped[str] = mapped_column(String(50), default="")
    blood_group: Mapped[str] = mapped_column(String(10), default="")
    height_cm: Mapped[float] = mapped_column(Float, default=0.0)
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    occupation: Mapped[str] = mapped_column(String(255), default="")

    # Presenting problem
    current_problem: Mapped[str] = mapped_column(Text, default="")
    problem_start_date: Mapped[str] = mapped_column(String(30), default="")
    problem_cause: Mapped[str] = mapped_column(Text, default="")
    previous_injuries: Mapped[str] = mapped_column(Text, default="")
    past_surgeries: Mapped[str] = mapped_column(Text, default="")

    # Medical history
    medical_conditions: Mapped[str] = mapped_column(Text, default="")
    current_medications: Mapped[str] = mapped_column(Text, default="")
    allergies: Mapped[str] = mapped_column(Text, default="")
    precautions: Mapped[str] = mapped_column(Text, default="")
    exercise_limitations: Mapped[str] = mapped_column(Text, default="")

    # Pain assessment
    pain_location: Mapped[str] = mapped_column(String(255), default="")
    pain_intensity: Mapped[int] = mapped_column(Integer, default=0)
    pain_type: Mapped[str] = mapped_column(String(120), default="")
    pain_triggers: Mapped[str] = mapped_column(Text, default="")
    pain_duration: Mapped[str] = mapped_column(String(120), default="")

    # Lifestyle
    daily_sitting_hours: Mapped[int] = mapped_column(Integer, default=0)
    activity_level: Mapped[str] = mapped_column(String(120), default="")
    exercise_habits: Mapped[str] = mapped_column(Text, default="")
    movement_restrictions: Mapped[str] = mapped_column(Text, default="")
    rehab_goals: Mapped[str] = mapped_column(Text, default="")

    # Emergency contact
    emergency_contact_name: Mapped[str] = mapped_column(String(255), default="")
    emergency_contact_phone: Mapped[str] = mapped_column(String(50), default="")
    guardian_whatsapp: Mapped[str] = mapped_column(String(50), default="")

    # Free-text link to an assigned clinician (kept for UI parity)
    doctor_name: Mapped[str] = mapped_column(String(255), default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    account: Mapped["Account"] = relationship(back_populates="patient_profile")  # noqa: F821
