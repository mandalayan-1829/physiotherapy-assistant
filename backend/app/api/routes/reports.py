"""Monthly report endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentAccount, DbSession
from app.models import Account
from app.schemas.report import ReportGenerateRequest, ReportOut
from app.services import report_service
from app.services.access import resolve_patient_scope
from app.services.errors import NotFoundError

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[ReportOut])
def list_reports(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[ReportOut]:
    scope = resolve_patient_scope(db, account, patient_account_id)
    return report_service.list_reports(db, scope)


@router.post("/generate", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def generate_report(
    payload: ReportGenerateRequest, account: CurrentAccount, db: DbSession
) -> ReportOut:
    """Generate (or refresh) a monthly report from persisted sessions."""
    if account.role == "patient":
        patient = account
    else:
        if payload.patient_account_id is None:
            from app.services.errors import ForbiddenError

            raise ForbiddenError("patient_account_id is required for doctor requests.")
        # Access is enforced inside resolve_patient_scope.
        resolve_patient_scope(db, account, payload.patient_account_id)
        patient = db.get(Account, payload.patient_account_id)
        if patient is None:
            raise NotFoundError("Patient not found.")

    return report_service.generate_report(db, patient, payload.month_key)
