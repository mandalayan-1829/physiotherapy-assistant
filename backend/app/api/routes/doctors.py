"""Doctor directory endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentAccount, DbSession, CurrentDoctor, CurrentPatient
from app.schemas.doctor import DoctorOut, PatientSummary
from app.services import doctor_service, patient_service
from app.services.access import get_doctor_profile

router = APIRouter(tags=["doctors"])


@router.get("/doctors", response_model=list[DoctorOut])
def list_doctors(account: CurrentAccount, db: DbSession) -> list[DoctorOut]:
    """Directory of clinicians visible to the caller."""
    return doctor_service.list_doctor_profiles(db, account)


@router.get("/doctors/{doctor_profile_id}", response_model=DoctorOut)
def get_doctor(doctor_profile_id: int, account: CurrentAccount, db: DbSession) -> DoctorOut:
    """One clinician profile.

    Unverified (pending) profiles are not enumerable: they 404 for everyone
    except the account that owns them.
    """
    return doctor_service.get_profile(db, doctor_profile_id, account)


@router.get("/patients", response_model=list[PatientSummary])
def list_my_patients(account: CurrentDoctor, db: DbSession) -> list[PatientSummary]:
    """Patients linked to the authenticated doctor."""
    profile = get_doctor_profile(db, account)
    patients = patient_service.patients_for_doctor(db, profile.id)
    return patient_service.summarise_patients(db, patients)
