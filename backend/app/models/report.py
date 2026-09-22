"""Persisted monthly rehabilitation report."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Report(Base):
    __tablename__ = "reports"
    # A patient has at most one report per month. Without this, two concurrent
    # "generate" requests can both miss the SELECT and insert, leaving duplicate
    # rows for the same month that then double-count in any aggregate.
    __table_args__ = (
        UniqueConstraint(
            "patient_account_id", "month_key", name="uq_reports_patient_month"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    month_key: Mapped[str] = mapped_column(String(7), nullable=False)  # e.g. "2026-09"
    month_name: Mapped[str] = mapped_column(String(40), default="")
    # Aggregated figures are stored as JSON so the schema stays stable as the
    # report contents evolve.
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
