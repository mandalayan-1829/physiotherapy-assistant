"""Telehealth message and safety alert endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentAccount, CurrentPatient, DbSession
from app.schemas.alert import AlertCreate, AlertOut, MessageCreate, MessageOut
from app.services import communication_service
from app.services.access import resolve_patient_scope

router = APIRouter(tags=["communication"])


@router.get("/messages", response_model=list[MessageOut])
def list_messages(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[MessageOut]:
    scope = resolve_patient_scope(db, account, patient_account_id)
    return communication_service.list_messages(db, account, scope)


@router.post("/messages", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def send_message(
    payload: MessageCreate, account: CurrentAccount, db: DbSession
) -> MessageOut:
    return communication_service.send_message(db, account, payload)


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[AlertOut]:
    scope = resolve_patient_scope(db, account, patient_account_id)
    return communication_service.list_alerts(db, account, scope)


@router.post("/alerts", response_model=AlertOut, status_code=status.HTTP_201_CREATED)
def create_alert(
    payload: AlertCreate, patient: CurrentPatient, db: DbSession
) -> AlertOut:
    return communication_service.create_alert(db, patient, payload)
