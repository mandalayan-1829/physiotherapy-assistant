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


@router.get("/{report_id}", response_model=ReportOut)
def retrieve_report(report_id: int, account: CurrentAccount, db: DbSession) -> ReportOut:
    """Retrieve one monthly report.

    Object-level authorization (OWASP API1): the report's *owning* patient is
    resolved from the row itself and checked against the requester's scope. The
    path never carries a patient id, so a caller cannot reach another patient's
    report by editing a query parameter - a doctor with no link to the owner, and
    any patient but the owner, are both refused.
    """
    report = report_service.get_report(db, report_id)
    resolve_patient_scope(db, account, report.patient_account_id)
    return report


@router.post("/generate", response_model=ReportOut, status_code=status.HTTP_201_CREATED)
def generate_report(
    payload: ReportGenerateRequest, account: CurrentAccount, db: DbSession
) -> ReportOut:
    """Generate (or refresh) a monthly report from persisted sessions.

    The target patient is resolved through :func:`resolve_patient_scope`, the
    same helper every other patient-scoped endpoint uses. That helper refuses a
    patient who names another patient, and requires a doctor to name a patient
    they are linked to.

    Resolving the scope first is deliberate. The previous version branched on the
    role and used ``patient = account`` for patients, which *silently ignored* a
    client-supplied ``patient_account_id`` pointing at someone else: the request
    succeeded and returned a 201, but for the wrong patient. Failing closed is
    the only safe behaviour when a caller and the payload disagree about whose
    record is being written.
    """
    patient_account_id = resolve_patient_scope(db, account, payload.patient_account_id)

    patient = db.get(Account, patient_account_id)
    if patient is None:
        raise NotFoundError("Patient not found.")

    return report_service.generate_report(db, patient, payload.month_key)
