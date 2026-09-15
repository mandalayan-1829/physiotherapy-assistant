"""Import data from the legacy ``aiphysio.db`` into the new schema.

The legacy database is only ever *read* — it is never modified or deleted.

What is migrated
----------------
* ``users``         -> ``accounts`` (role=patient) + ``patient_profiles``
                      The unsalted SHA-256 password digest is carried over in
                      ``legacy_password_hash`` so the owner can still sign in
                      once; the hash is upgraded to bcrypt on that login.
* ``doctors``       -> ``doctor_profiles`` (directory entries, no login yet)
* ``appointments``  -> ``appointments`` (only rows whose patient/doctor exist)
* ``diet_log``      -> ``diet_records``
* ``sessions``      -> ``workout_sessions``
* ``notes``         -> ``doctor_notes`` when a doctor can be resolved

Usage (from the backend/ directory):

    python scripts/migrate_legacy.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.init_db import init_db  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    Account,
    Appointment,
    DietRecord,
    DoctorProfile,
    PatientDoctorLink,
    PatientProfile,
    WorkoutSession,
)

LEGACY_PATH = Path(
    os.getenv("LEGACY_DATABASE_PATH", settings.legacy_database_path)
).resolve()


def _legacy_rows(table: str) -> list[sqlite3.Row]:
    if not LEGACY_PATH.exists():
        print(f"[skip] legacy database not found at {LEGACY_PATH}")
        return []
    conn = sqlite3.connect(f"file:{LEGACY_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(f"SELECT * FROM {table}"))
    except sqlite3.OperationalError:
        print(f"[skip] legacy table '{table}' not present")
        return []
    finally:
        conn.close()


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def migrate() -> None:
    init_db()
    db = SessionLocal()
    try:
        # --- Users -------------------------------------------------------
        for row in _legacy_rows("users"):
            email = (row["email"] or "").strip().lower()
            if not email:
                continue
            if db.execute(select(Account).where(Account.email == email)).scalar_one_or_none():
                print(f"[skip] account already exists: {email}")
                continue

            account = Account(
                email=email,
                full_name=(row["name"] or "Patient").title(),
                # Placeholder that can never match a real password; the legacy
                # digest below is what actually authenticates until first login.
                password_hash="!legacy-account-must-reset!",
                role="patient",
                legacy_password_hash=(row["password"] or None),
            )
            db.add(account)
            db.flush()

            db.add(
                PatientProfile(
                    account_id=account.id,
                    age=row["age"],
                    gender=row["gender"] or "",
                    dob=row["dob"] or "",
                    contact_number=row["contact_number"] or "",
                    blood_group=row["blood_group"] or "",
                    height_cm=row["height_cm"] or 0.0,
                    weight_kg=row["weight_kg"] or 0.0,
                    occupation=row["occupation"] or "",
                    current_problem=row["current_problem"] or "",
                    problem_start_date=row["problem_start_date"] or "",
                    problem_cause=row["problem_cause"] or "",
                    previous_injuries=row["previous_injuries"] or "",
                    past_surgeries=row["past_surgeries"] or "",
                    medical_conditions=row["medical_conditions"] or "",
                    current_medications=row["current_medications"] or "",
                    allergies=row["allergies"] or "",
                    precautions=row["precautions"] or "",
                    exercise_limitations=row["exercise_limitations"] or "",
                    pain_location=row["pain_location"] or "",
                    pain_intensity=row["pain_intensity"] or 0,
                    pain_type=row["pain_type"] or "",
                    pain_triggers=row["pain_triggers"] or "",
                    pain_duration=row["pain_duration"] or "",
                    daily_sitting_hours=row["daily_sitting_hours"] or 0,
                    activity_level=row["activity_level"] or "",
                    exercise_habits=row["exercise_habits"] or "",
                    movement_restrictions=row["movement_restrictions"] or "",
                    rehab_goals=row["rehab_goals"] or "",
                    emergency_contact_name=row["emergency_contact_name"] or "",
                    emergency_contact_phone=row["emergency_contact_phone"] or "",
                    guardian_whatsapp=row["guardian_whatsapp"] or "",
                    doctor_name=row["doctor_name"] or "",
                )
            )
            print(f"[ok]   imported patient: {email}")

        # --- Doctors (directory entries) ---------------------------------
        legacy_doctor_map: dict[int, int] = {}
        for row in _legacy_rows("doctors"):
            name = (row["name"] or "").strip()
            if not name:
                continue
            existing = db.execute(
                select(DoctorProfile).where(DoctorProfile.name == name.title())
            ).scalar_one_or_none()
            if existing:
                legacy_doctor_map[row["id"]] = existing.id
                continue
            profile = DoctorProfile(
                name=name.title(),
                specialization=row["specialization"] or "",
                experience=row["experience"] or 0,
                qualification=row["qualification"] or "",
                available_days=row["available_days"] or "",
                timings=row["timings"] or "",
                about=row["about"] or "",
                contact=row["contact"] or "",
                whatsapp=row["whatsapp"] or "",
                email=row["email"] or "",
            )
            db.add(profile)
            db.flush()
            legacy_doctor_map[row["id"]] = profile.id
            print(f"[ok]   imported doctor profile: {profile.name}")

        db.commit()

        # --- Exercise sessions -------------------------------------------
        legacy_user_ids = {
            row["id"]: row["email"].strip().lower() for row in _legacy_rows("users")
        }
        for row in _legacy_rows("sessions"):
            email = legacy_user_ids.get(row["user_id"])
            if not email:
                continue
            account = db.execute(select(Account).where(Account.email == email)).scalar_one_or_none()
            if not account:
                continue
            db.add(
                WorkoutSession(
                    patient_account_id=account.id,
                    exercise=row["exercise"] or "unknown",
                    exercise_label=(row["exercise"] or "unknown").replace("_", " ").title(),
                    reps=row["reps"] or 0,
                    form_accuracy=row["form_accuracy"] or 0,
                    performed_at=_parse_date(row["date"]),
                )
            )

        # --- Diet records -------------------------------------------------
        for row in _legacy_rows("diet_log"):
            email = legacy_user_ids.get(row["user_id"])
            if not email:
                continue
            account = db.execute(select(Account).where(Account.email == email)).scalar_one_or_none()
            if not account:
                continue
            db.add(
                DietRecord(
                    patient_account_id=account.id,
                    meal=row["meal"] or "meal",
                    calories=row["calories"] or 0,
                    protein=row["protein"] or 0.0,
                    carbs=row["carbs"] or 0.0,
                    fats=row["fats"] or 0.0,
                    recorded_at=_parse_date(row["date"]),
                )
            )

        # --- Appointments -------------------------------------------------
        for row in _legacy_rows("appointments"):
            email = legacy_user_ids.get(row["user_id"])
            doctor_profile_id = legacy_doctor_map.get(row["doctor_id"])
            if not email or doctor_profile_id is None:
                continue
            account = db.execute(select(Account).where(Account.email == email)).scalar_one_or_none()
            profile = db.get(DoctorProfile, doctor_profile_id)
            if not account or not profile:
                continue
            db.add(
                Appointment(
                    patient_account_id=account.id,
                    doctor_profile_id=profile.id,
                    patient_name=account.full_name,
                    doctor_name=profile.name,
                    specialization=profile.specialization,
                    email=account.email,
                    date=row["date"] or "",
                    time=row["time"] or "",
                    reason=row["reason"] or "",
                    status=row["status"] or "pending",
                    clinician_note=row["admin_note"] or "",
                )
            )
            # Existing bookings imply an existing clinical relationship.
            if not db.execute(
                select(PatientDoctorLink).where(
                    PatientDoctorLink.patient_account_id == account.id,
                    PatientDoctorLink.doctor_profile_id == profile.id,
                )
            ).scalar_one_or_none():
                db.add(
                    PatientDoctorLink(
                        patient_account_id=account.id, doctor_profile_id=profile.id
                    )
                )

        db.commit()
        print("\nMigration complete. Legacy aiphysio.db was read-only and is unchanged.")
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
