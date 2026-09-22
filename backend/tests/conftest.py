"""Pytest fixtures.

The environment is configured *before* the application is imported so the API
under test uses a throwaway SQLite file instead of the development database.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

_TEST_DB = Path(tempfile.mkdtemp(prefix="physioai-tests-")) / "test.db"

os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
# A >=32 character value: the application now refuses to start with a weak or
# placeholder signing key, which is asserted by tests/test_security_config.py.
os.environ["SECRET_KEY"] = "test-only-signing-key-0123456789abcdefghijklmnop"
os.environ["ENVIRONMENT"] = "test"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"
os.environ["LEGACY_DATABASE_PATH"] = str(_TEST_DB)  # never touch the real legacy db
# The insecure legacy SHA-256 path is off by default; the single test that covers
# it switches it on explicitly.
os.environ["ALLOW_LEGACY_PASSWORD_LOGIN"] = "false"

# Abuse limits are raised for the suite because every request in a test run
# appears to come from one client address, so production values would throttle
# the suite itself. The limiters are exercised explicitly, with tightened
# settings, in tests/test_rate_limiting.py.
os.environ["REGISTER_MAX_REQUESTS_PER_HOUR"] = "100000"
os.environ["LOGIN_MAX_REQUESTS_PER_IP_PER_WINDOW"] = "100000"
os.environ["PASSWORD_RESET_MAX_REQUESTS_PER_IP_PER_HOUR"] = "100000"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import DoctorProfile  # noqa: E402
from app.services.email_service import clear_console_outbox  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    # The context manager runs the lifespan hook, which creates the tables.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_email_outbox():
    """Keep recorded development emails isolated between tests."""
    clear_console_outbox()
    yield
    clear_console_outbox()


# --- Helpers ----------------------------------------------------------------


def register(client: TestClient, email: str, password: str, role: str, **extra):
    payload = {
        "full_name": extra.pop("full_name", "Test User"),
        "email": email,
        "password": password,
        "role": role,
    }
    payload.update(extra)
    return client.post("/auth/register", json=payload)


def login(client: TestClient, email: str, password: str, role: str):
    return client.post("/auth/login", json={"email": email, "password": password, "role": role})


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def verify_doctor(doctor_profile_id: int) -> None:
    """Simulate the out-of-band verification an operator performs.

    Clinician profiles are unverified (and therefore unbookable and invisible in
    the directory) until verified. There is no admin API by design, so the tests
    flip the flag the same way ``backend/scripts/verify_doctor.py`` does.
    """
    db = SessionLocal()
    try:
        profile = db.get(DoctorProfile, doctor_profile_id)
        assert profile is not None, f"no doctor profile with id={doctor_profile_id}"
        profile.is_verified = True
        db.commit()
    finally:
        db.close()


def verify_doctor_by_email(email: str) -> int:
    """Verify a clinician by e-mail and return the profile id."""
    db = SessionLocal()
    try:
        profile = db.execute(
            select(DoctorProfile).where(DoctorProfile.email == email.lower())
        ).scalar_one()
        profile.is_verified = True
        db.commit()
        return profile.id
    finally:
        db.close()


@pytest.fixture
def new_patient(client: TestClient):
    """Factory fixture returning (account, token) for a fresh patient."""

    counter = {"n": 0}

    def _make(prefix: str = "patient") -> tuple[dict, str]:
        counter["n"] += 1
        email = f"{prefix}{counter['n']}.{os.urandom(3).hex()}@example.com"
        response = register(client, email, "PatientPass123", "patient", full_name="Pat Patient")
        assert response.status_code == 201, response.text
        body = response.json()
        return body["account"], body["access_token"]

    return _make


@pytest.fixture
def new_doctor(client: TestClient):
    counter = {"n": 0}

    def _make(prefix: str = "doctor") -> tuple[dict, str]:
        counter["n"] += 1
        email = f"{prefix}{counter['n']}.{os.urandom(3).hex()}@example.com"
        response = register(
            client,
            email,
            "DoctorPass123",
            "doctor",
            full_name="Dr Test Clinician",
            specialization="Orthopedic Rehabilitation",
            qualification="MPT",
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return body["account"], body["access_token"]

    return _make
