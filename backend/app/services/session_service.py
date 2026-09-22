"""Exercise session persistence and daily progress aggregation.

Aggregation is deliberately split by provenance: only sessions whose metrics
came from real pose inference contribute to ``avg_form_accuracy``. Simulated and
manually logged sessions are still counted and stored (a patient's history is
not discarded) but are never presented as measured clinical performance.
"""

from __future__ import annotations

from collections import OrderedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import WorkoutSession
from app.models.session import MEASURED_METRICS_SOURCE
from app.schemas.session import WorkoutSessionCreate


def list_sessions(db: Session, patient_account_id: int, limit: int = 200) -> list[WorkoutSession]:
    return list(
        db.execute(
            select(WorkoutSession)
            .where(WorkoutSession.patient_account_id == patient_account_id)
            .order_by(WorkoutSession.performed_at.desc())
            .limit(limit)
        ).scalars().all()
    )


def create_session(
    db: Session, patient_account_id: int, payload: WorkoutSessionCreate
) -> WorkoutSession:
    data = payload.model_dump(exclude={"performed_at"})
    session = WorkoutSession(patient_account_id=patient_account_id, **data)
    if payload.performed_at is not None:
        session.performed_at = payload.performed_at
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def daily_progress(db: Session, patient_account_id: int) -> list[dict]:
    """Aggregate sessions into per-day totals (newest day first)."""
    sessions = list_sessions(db, patient_account_id, limit=5000)

    buckets: "OrderedDict[str, dict]" = OrderedDict()
    for session in sessions:
        day = session.performed_at.date().isoformat()
        bucket = buckets.setdefault(
            day,
            {
                "date": day,
                "sessions": 0,
                "total_reps": 0,
                "total_duration_sec": 0,
                "measured_sessions": 0,
                "_accuracy_sum": 0,
            },
        )
        bucket["sessions"] += 1
        bucket["total_reps"] += session.reps
        bucket["total_duration_sec"] += session.duration_sec
        if session.metrics_source == MEASURED_METRICS_SOURCE:
            bucket["measured_sessions"] += 1
            bucket["_accuracy_sum"] += session.form_accuracy

    results = []
    for day in sorted(buckets.keys(), reverse=True):
        bucket = buckets[day]
        measured = bucket["measured_sessions"]
        # Only real measurements feed the average; with none, the day has no
        # measurable form score rather than a fabricated 0%.
        bucket["avg_form_accuracy"] = (
            round(bucket["_accuracy_sum"] / measured, 1) if measured else 0.0
        )
        bucket.pop("_accuracy_sum")
        results.append(bucket)
    return results
