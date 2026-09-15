"""Diet record endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Response, status

from app.api.deps import CurrentAccount, CurrentPatient, DbSession
from app.schemas.diet import DietRecordCreate, DietRecordOut
from app.services import diet_service
from app.services.access import resolve_patient_scope

router = APIRouter(prefix="/diet", tags=["diet"])


@router.get("", response_model=list[DietRecordOut])
def list_diet(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[DietRecordOut]:
    scope = resolve_patient_scope(db, account, patient_account_id)
    return diet_service.list_records(db, scope)


@router.post("", response_model=DietRecordOut, status_code=status.HTTP_201_CREATED)
def add_diet(
    payload: DietRecordCreate, patient: CurrentPatient, db: DbSession
) -> DietRecordOut:
    return diet_service.create_record(db, patient.id, payload)


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_diet(
    record_id: int, patient: CurrentPatient, db: DbSession
) -> Response:
    diet_service.delete_record(db, patient.id, record_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
