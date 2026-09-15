"""Diet record service."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DietRecord
from app.schemas.diet import DietRecordCreate
from app.services.errors import NotFoundError


def list_records(db: Session, patient_account_id: int) -> list[DietRecord]:
    return list(
        db.execute(
            select(DietRecord)
            .where(DietRecord.patient_account_id == patient_account_id)
            .order_by(DietRecord.recorded_at.desc())
        ).scalars().all()
    )


def create_record(
    db: Session, patient_account_id: int, payload: DietRecordCreate
) -> DietRecord:
    record = DietRecord(
        patient_account_id=patient_account_id,
        **payload.model_dump(exclude={"recorded_at"}),
    )
    if payload.recorded_at is not None:
        record.recorded_at = payload.recorded_at
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def delete_record(db: Session, patient_account_id: int, record_id: int) -> None:
    record = db.get(DietRecord, record_id)
    if record is None or record.patient_account_id != patient_account_id:
        raise NotFoundError("Diet record not found.")
    db.delete(record)
    db.commit()
