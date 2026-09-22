"""Completed exercise session record.

``metrics_source`` records how the numbers in the row were produced. This
matters clinically: only ``pose_inference`` rows are measurements of a real
movement, so only those may be averaged into a form-accuracy score. Rows from the
former simulated tracker (``simulated``) and manually logged rows (``manual``)
are retained for history but are excluded from clinical aggregates.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: ``pose_inference`` - landmarks came from real on-device pose estimation.
#: ``simulated``      - landmarks were fabricated by the retired demo tracker.
#: ``manual``         - the patient logged the set themselves; no measurement.
METRICS_SOURCES = ("pose_inference", "simulated", "manual")

#: The only source whose ``form_accuracy`` is a real measurement.
MEASURED_METRICS_SOURCE = "pose_inference"


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    exercise: Mapped[str] = mapped_column(String(120), nullable=False)
    exercise_label: Mapped[str] = mapped_column(String(255), nullable=False)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    target_reps: Mapped[int] = mapped_column(Integer, default=0)
    form_accuracy: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")

    # Deny-by-default: a caller that does not explicitly state that real pose
    # inference produced these numbers is not treated as having measured them.
    metrics_source: Mapped[str] = mapped_column(
        String(20), default="manual", nullable=False, index=True
    )

    performed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
