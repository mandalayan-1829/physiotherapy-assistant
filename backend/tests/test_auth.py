"""Registration, login and password handling."""

from __future__ import annotations

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Account
from tests.conftest import auth_headers, login, register

LEGACY_PASSWORD = "LegacyPass123"


def _create_legacy_account(email: str) -> None:
    """Insert an account exactly as the aiphysio.db importer would."""
    from app.core.security import hash_legacy_sha256
    from app.models import PatientProfile

    db = SessionLocal()
    try:
        account = Account(
            email=email,
            full_name="Legacy User",
            password_hash="!legacy-account-must-reset!",
            role="patient",
            legacy_password_hash=hash_legacy_sha256(LEGACY_PASSWORD),
        )
        db.add(account)
        db.flush()
        db.add(PatientProfile(account_id=account.id))
        db.commit()
    finally:
        db.close()


def test_registration_works(client, new_patient):
    account, token = new_patient()
    assert account["role"] == "patient"
    assert account["email"].endswith("@example.com")

    me = client.get("/auth/me", headers=auth_headers(token))
    assert me.status_code == 200
    assert me.json()["account"]["id"] == account["id"]


def test_duplicate_email_is_rejected(client):
    email = "duplicate@example.com"
    first = register(client, email, "Password123", "patient")
    assert first.status_code == 201

    second = register(client, email, "Password123", "patient")
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"].lower()


def test_password_is_hashed_not_plaintext(client):
    email = "hashcheck@example.com"
    plaintext = "SuperSecret123"
    assert register(client, email, plaintext, "patient").status_code == 201

    db = SessionLocal()
    try:
        account = db.execute(select(Account).where(Account.email == email)).scalar_one()
    finally:
        db.close()

    assert account.password_hash != plaintext
    assert plaintext not in account.password_hash
    # bcrypt hashes are prefixed with the algorithm identifier.
    assert account.password_hash.startswith("$2")
    assert account.password_hash.startswith("$2") and len(account.password_hash) >= 55


def test_valid_login_returns_token(client):
    email = "login.ok@example.com"
    register(client, email, "Password123", "patient")

    response = login(client, email, "Password123", "patient")
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["account"]["role"] == "patient"


def test_invalid_password_fails(client):
    email = "login.bad@example.com"
    register(client, email, "Password123", "patient")

    response = login(client, email, "WrongPassword123", "patient")
    assert response.status_code == 401


def test_unknown_email_fails(client):
    response = login(client, "nobody@example.com", "Password123", "patient")
    assert response.status_code == 401
    # Same message as a wrong password: no account enumeration.
    assert response.json()["detail"] == "Invalid email or password."


def test_legacy_sha256_digest_cannot_authenticate_by_default(client):
    """An unsalted SHA-256 digest is not an acceptable authentication factor.

    ``ALLOW_LEGACY_PASSWORD_LOGIN`` is off by default, so the migration-only
    verifier is never consulted and the account cannot sign in with the legacy
    password. The owner resets the password through the normal flow instead.
    """
    email = "legacy.default.off@example.com"
    _create_legacy_account(email)

    response = login(client, email, LEGACY_PASSWORD, "patient")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_legacy_sha256_password_is_upgraded_to_bcrypt_when_explicitly_enabled(
    client, monkeypatch
):
    """With the migration flag switched on, a legacy digest upgrades to bcrypt once.

    This is an operator-controlled, one-time path used while importing
    aiphysio.db; it is never active in a normal deployment.
    """
    email = "legacy.upgrade@example.com"
    monkeypatch.setattr(settings, "allow_legacy_password_login", True)
    _create_legacy_account(email)

    assert login(client, email, LEGACY_PASSWORD, "patient").status_code == 200

    db = SessionLocal()
    try:
        refreshed = db.execute(select(Account).where(Account.email == email)).scalar_one()
    finally:
        db.close()

    # The insecure digest is discarded after the first successful sign-in.
    assert refreshed.legacy_password_hash is None
    assert refreshed.password_hash.startswith("$2")

    # The account now authenticates with bcrypt even with the flag turned off again.
    monkeypatch.setattr(settings, "allow_legacy_password_login", False)
    assert login(client, email, LEGACY_PASSWORD, "patient").status_code == 200


def test_legacy_digest_cannot_be_used_after_upgrade(client, monkeypatch):
    """A wrong legacy password is still refused while the migration flag is on."""
    email = "legacy.wrongpass@example.com"
    monkeypatch.setattr(settings, "allow_legacy_password_login", True)
    _create_legacy_account(email)

    assert login(client, email, "NotTheLegacyPass123", "patient").status_code == 401


def test_login_wrong_portal_role_is_rejected(client, new_doctor):
    account, _ = new_doctor()
    response = login(client, account["email"], "DoctorPass123", "patient")
    assert response.status_code == 401


def test_patient_role_works(client, new_patient):
    _, token = new_patient()
    response = client.get("/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    assert response.json()["account"]["role"] == "patient"
    assert response.json()["profile"] is not None


def test_doctor_role_works(client, new_doctor):
    _, token = new_doctor()
    response = client.get("/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    assert response.json()["account"]["role"] == "doctor"
    assert response.json()["doctor_profile"] is not None


def test_registration_rejects_admin_role(client):
    response = register(client, "admin@example.com", "Password123", "admin")
    assert response.status_code == 422


def test_weak_password_is_rejected(client):
    response = register(client, "weak@example.com", "short", "patient")
    assert response.status_code == 422


def test_logout_requires_authentication(client, new_patient):
    _, token = new_patient()
    assert client.post("/auth/logout", headers=auth_headers(token)).status_code == 204
    assert client.post("/auth/logout").status_code == 401
