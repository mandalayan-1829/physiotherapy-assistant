"""Exercise session endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentAccount, CurrentPatient, DbSession
from app.schemas.session import WorkoutSessionCreate, WorkoutSessionOut
from app.services import session_service
from app.services.access import resolve_patient_scope

router = APIRouter(tags=["sessions"])


@router.get("/sessions", response_model=list[WorkoutSessionOut])
def list_sessions(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[WorkoutSessionOut]:
    """Patients see their own sessions; doctors must pass a linked patient id."""
    scope = resolve_patient_scope(db, account, patient_account_id)
    return session_service.list_sessions(db, scope)


@router.post("/sessions", response_model=WorkoutSessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: WorkoutSessionCreate, patient: CurrentPatient, db: DbSession
) -> WorkoutSessionOut:
    return session_service.create_session(db, patient.id, payload)
