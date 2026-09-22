"""Monthly report generation, computed from persisted exercise sessions.

Only sessions whose metrics came from real on-device pose inference contribute
to ``avg_form_accuracy``. Rows logged manually, or produced by the retired
simulated tracker, are reported separately as counts - they are never averaged
in as if they were measurements.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Account, Report, WorkoutSession
from app.models.session import MEASURED_METRICS_SOURCE
from app.services.errors import NotFoundError, ValidationError

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


def _active_days_percent(month_key: str, month_sessions: list[WorkoutSession]) -> int:
    """Share of the month's *elapsed* days that contain at least one session.

    This is a defined, computable proportion - unlike an "adherence score"
    produced by a lookup table, it describes something that actually happened.
    For the month in progress only the days that have already occurred are
    counted, so a patient is not penalised for the future.

    Computed server-side so the figure does not depend on the client's clock or
    timezone, and so two clients cannot disagree about the same month.
    """
    year, month = (int(part) for part in month_key.split("-"))
    days_in_month = calendar.monthrange(year, month)[1]

    today = datetime.now(timezone.utc).date()
    elapsed_days = (
        today.day if (today.year, today.month) == (year, month) else days_in_month
    )
    if elapsed_days <= 0:
        return 0

    active_days = len({s.performed_at.date() for s in month_sessions})
    return min(100, round(active_days / elapsed_days * 100))


def list_reports(db: Session, patient_account_id: int) -> list[Report]:
    return list(
        db.execute(
            select(Report)
            .where(Report.patient_account_id == patient_account_id)
            .order_by(Report.month_key.desc())
        ).scalars().all()
    )


def get_report(db: Session, report_id: int) -> Report:
    """Fetch one report by id.

    This does not authorise the caller: object-level access is enforced in the
    route, which checks the owning patient against the requester's scope before
    this is called.
    """
    report = db.get(Report, report_id)
    if report is None:
        raise NotFoundError("Report not found.")
    return report


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
    measured_sessions = [s for s in month_sessions if s.metrics_source == MEASURED_METRICS_SOURCE]
    unmeasured_sessions = total_sessions - len(measured_sessions)
    avg_accuracy = (
        round(sum(s.form_accuracy for s in measured_sessions) / len(measured_sessions), 1)
        if measured_sessions
        else 0.0
    )

    by_exercise: dict[str, dict] = {}
    for session in month_sessions:
        entry = by_exercise.setdefault(
            session.exercise_label,
            {
                "exercise_label": session.exercise_label,
                "sessions": 0,
                "reps": 0,
                "measured_sessions": 0,
                "_acc": 0,
            },
        )
        entry["sessions"] += 1
        entry["reps"] += session.reps
        if session.metrics_source == MEASURED_METRICS_SOURCE:
            entry["measured_sessions"] += 1
            entry["_acc"] += session.form_accuracy

    breakdown = []
    for entry in by_exercise.values():
        measured = entry["measured_sessions"]
        entry["avg_accuracy"] = round(entry.pop("_acc") / measured, 1) if measured else 0.0
        breakdown.append(entry)

    # Validated first: this raises for a malformed or out-of-range month before
    # any other parsing happens.
    month_name = _month_name(month_key)

    payload = {
        "patient_name": patient.full_name,
        "patient_email": patient.email,
        "total_sessions": total_sessions,
        "total_reps": total_reps,
        "total_duration_sec": total_duration,
        # Averaged over measured sessions only. ``measured_sessions`` tells the
        # reader how much of the month actually contains a measurement, so a
        # report can never imply more clinical evidence than exists.
        "avg_form_accuracy": avg_accuracy,
        "measured_sessions": len(measured_sessions),
        "unmeasured_sessions": unmeasured_sessions,
        "exercise_breakdown": breakdown,
        "insufficient_data": total_sessions == 0,
        "no_measured_sessions": len(measured_sessions) == 0,
        # Computed here rather than in the browser so the same month reports the
        # same figure regardless of the client's clock or timezone.
        "active_days_percent": _active_days_percent(month_key, month_sessions),
    }

    existing = db.execute(
        select(Report).where(
            Report.patient_account_id == patient.id, Report.month_key == month_key
        )
    ).scalar_one_or_none()

    if existing:
        existing.payload = payload
        existing.month_name = month_name
        db.commit()
        db.refresh(existing)
        return existing

    report = Report(
        patient_account_id=patient.id,
        month_key=month_key,
        month_name=month_name,
        payload=payload,
    )
    db.add(report)
    try:
        db.commit()
    except IntegrityError:
        # A concurrent request inserted the same (patient, month) between our
        # SELECT and this INSERT. The unique constraint rejected it, so refresh
        # the row that won rather than returning a 500 for a harmless race.
        db.rollback()
        concurrent = db.execute(
            select(Report).where(
                Report.patient_account_id == patient.id, Report.month_key == month_key
            )
        ).scalar_one()
        concurrent.payload = payload
        concurrent.month_name = month_name
        db.commit()
        db.refresh(concurrent)
        return concurrent

    db.refresh(report)
    return report
