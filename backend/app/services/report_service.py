"""Monthly report generation, computed from persisted exercise sessions."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Account, Report, WorkoutSession
from app.services.errors import ValidationError

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _month_name(month_key: str) -> str:
    year, month = month_key.split("-")
    index = int(month) - 1
    if not 0 <= index < 12:
        raise ValidationError("month_key must look like '2026-09'.")
    return f"{MONTH_NAMES[index]} {year}"


def list_reports(db: Session, patient_account_id: int) -> list[Report]:
    return list(
        db.execute(
            select(Report)
            .where(Report.patient_account_id == patient_account_id)
            .order_by(Report.month_key.desc())
        ).scalars().all()
    )


def generate_report(db: Session, patient: Account, month_key: str) -> Report:
    """Build (or refresh) the monthly report for ``patient`` from real sessions."""
    sessions = list(
        db.execute(
            select(WorkoutSession).where(WorkoutSession.patient_account_id == patient.id)
        ).scalars().all()
    )
    month_sessions = [s for s in sessions if s.performed_at.date().isoformat().startswith(month_key)]

    total_sessions = len(month_sessions)
    total_reps = sum(s.reps for s in month_sessions)
    total_duration = sum(s.duration_sec for s in month_sessions)
    avg_accuracy = (
        round(sum(s.form_accuracy for s in month_sessions) / total_sessions, 1)
        if total_sessions
        else 0.0
    )

    by_exercise: dict[str, dict] = {}
    for session in month_sessions:
        entry = by_exercise.setdefault(
            session.exercise_label,
            {"exercise_label": session.exercise_label, "sessions": 0, "reps": 0, "_acc": 0},
        )
        entry["sessions"] += 1
        entry["reps"] += session.reps
        entry["_acc"] += session.form_accuracy

    breakdown = []
    for entry in by_exercise.values():
        entry["avg_accuracy"] = round(entry.pop("_acc") / entry["sessions"], 1)
        breakdown.append(entry)

    payload = {
        "patient_name": patient.full_name,
        "patient_email": patient.email,
        "total_sessions": total_sessions,
        "total_reps": total_reps,
        "total_duration_sec": total_duration,
        "avg_form_accuracy": avg_accuracy,
        "exercise_breakdown": breakdown,
        "insufficient_data": total_sessions == 0,
    }

    existing = db.execute(
        select(Report).where(
            Report.patient_account_id == patient.id, Report.month_key == month_key
        )
    ).scalar_one_or_none()

    if existing:
        existing.payload = payload
        existing.month_name = _month_name(month_key)
        db.commit()
        db.refresh(existing)
        return existing

    report = Report(
        patient_account_id=patient.id,
        month_key=month_key,
        month_name=_month_name(month_key),
        payload=payload,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report
