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
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"
os.environ["LEGACY_DATABASE_PATH"] = str(_TEST_DB)  # never touch the real legacy db

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
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
