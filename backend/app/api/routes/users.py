"""Current-user profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentAccount, DbSession, CurrentPatient
from app.schemas.doctor import DoctorOut
from app.schemas.patient import PatientProfileOut, PatientProfileUpdate
from app.services import patient_service
from app.services.access import get_doctor_profile

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me/profile", response_model=PatientProfileOut)
def get_my_profile(account: CurrentPatient, db: DbSession) -> PatientProfileOut:
    return patient_service.profile_for(db, account.id)


@router.put("/me/profile", response_model=PatientProfileOut)
def update_my_profile(
    payload: PatientProfileUpdate, account: CurrentPatient, db: DbSession
) -> PatientProfileOut:
    """Save the authenticated patient's medical profile."""
    return patient_service.update_profile(db, account.id, payload)


@router.get("/me/doctor-profile", response_model=DoctorOut)
def get_my_doctor_profile(account: CurrentAccount, db: DbSession) -> DoctorOut:
    return get_doctor_profile(db, account)
