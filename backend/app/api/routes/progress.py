"""Progress and history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import CurrentAccount, DbSession
from app.schemas.session import DailyProgressOut
from app.services import session_service
from app.services.access import resolve_patient_scope

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("/daily", response_model=list[DailyProgressOut])
def daily_progress(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[DailyProgressOut]:
    """Per-day aggregates for the scoped patient (newest first)."""
    scope = resolve_patient_scope(db, account, patient_account_id)
    return session_service.daily_progress(db, scope)
