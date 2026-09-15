"""Failed sign-in tracking and the secure password reset flow."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import SCOPE_PASSWORD_RESET_REQUEST, Account, AuthThrottle, PasswordResetToken
from app.services.email_service import get_console_outbox
from tests.conftest import auth_headers, login, register

CODE_PATTERN = re.compile(r"code is:\s*(\d{6})")
GENERIC_MESSAGE = "If an account exists for this email, a verification code has been sent."


# --- helpers ----------------------------------------------------------------


def latest_code() -> str:
    outbox = get_console_outbox()
    assert outbox, "no email was recorded in the development outbox"
    match = CODE_PATTERN.search(outbox[-1].text_body)
    assert match, f"no verification code found in email body: {outbox[-1].text_body!r}"
    return match.group(1)


def emails_sent() -> int:
    return len(get_console_outbox())


def db_session():
    return SessionLocal()


def expire_code(email: str) -> None:
    """Force the newest reset code for ``email`` into the past."""
    db = db_session()
    try:
        account = db.execute(select(Account).where(Account.email == email)).scalar_one()
        token = (
            db.execute(
                select(PasswordResetToken)
                .where(PasswordResetToken.account_id == account.id)
                .order_by(PasswordResetToken.id.desc())
            )
            .scalars()
            .first()
        )
        assert token is not None
        token.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
    finally:
        db.close()


def clear_reset_cooldown(email: str) -> None:
    """Simulate the resend cooldown having elapsed."""
    db = db_session()
    try:
        row = db.execute(
            select(AuthThrottle).where(
                AuthThrottle.scope == SCOPE_PASSWORD_RESET_REQUEST,
                AuthThrottle.key == email.lower(),
            )
        ).scalar_one_or_none()
        if row is not None:
            row.last_attempt_at = datetime.now(timezone.utc) - timedelta(minutes=5)
            row.window_started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
            db.commit()
    finally:
        db.close()


def make_patient(client, prefix: str, password: str = "OriginalPass123") -> str:
    email = f"{prefix}.{__import__('os').urandom(3).hex()}@example.com"
    response = register(client, email, password, "patient")
    assert response.status_code == 201, response.text
    return email


def start_reset(client, email: str, role: str = "patient"):
    return client.post("/auth/forgot-password", json={"email": email, "role": role})


def verify_code(client, email: str, code: str, role: str = "patient"):
    return client.post(
        "/auth/verify-reset-code",
        json={"email": email, "verification_code": code, "role": role},
    )


def complete_reset(client, reset_token: str, password: str, confirm: str | None = None):
    return client.post(
        "/auth/reset-password",
        json={
            "reset_token": reset_token,
            "new_password": password,
            "confirm_password": confirm if confirm is not None else password,
        },
    )


# --- 1-4: sign-in failure tracking ------------------------------------------


def test_normal_login_still_works(client):
    email = make_patient(client, "normal")
    response = login(client, email, "OriginalPass123", "patient")
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_wrong_password_attempts_are_counted_and_offer_forgot_password(client):
    email = make_patient(client, "attempts")

    first = login(client, email, "WrongPass123", "patient")
    assert first.status_code == 401
    assert first.json()["detail"] == "Invalid email or password."
    assert first.json()["failed_attempts"] == 1
    assert first.json()["show_forgot_password"] is False

    second = login(client, email, "WrongPass123", "patient")
    assert second.status_code == 401
    assert second.json()["failed_attempts"] == 2
    assert second.json()["show_forgot_password"] is False, "must not prompt before 3 failures"

    third = login(client, email, "WrongPass123", "patient")
    assert third.status_code == 401
    assert third.json()["failed_attempts"] == 3
    assert third.json()["show_forgot_password"] is True


def test_unknown_email_gets_the_same_failure_response(client):
    known = make_patient(client, "known")

    unknown = login(client, "nobody-at-all@example.com", "WrongPass123", "patient")
    assert unknown.status_code == 401
    assert unknown.json()["detail"] == "Invalid email or password."
    assert unknown.json()["show_forgot_password"] is False  # attempt 1 for this address

    for _ in range(2):
        response = login(client, "nobody-at-all@example.com", "WrongPass123", "patient")
    assert response.json()["show_forgot_password"] is True

    # ...and the known address behaves identically on its own counter.
    assert login(client, known, "WrongPass123", "patient").status_code == 401


def test_too_many_failed_attempts_are_rate_limited(client):
    email = make_patient(client, "ratelimit")
    statuses = [login(client, email, "WrongPass123", "patient").status_code for _ in range(6)]
    assert statuses[:5] == [401] * 5
    assert statuses[5] == 429

    blocked = login(client, email, "OriginalPass123", "patient")
    assert blocked.status_code == 429
    assert blocked.json()["show_forgot_password"] is True
    assert blocked.json()["retry_after_seconds"] > 0


def test_successful_login_clears_the_failure_counter(client):
    email = make_patient(client, "cleared")
    login(client, email, "WrongPass123", "patient")
    login(client, email, "WrongPass123", "patient")

    assert login(client, email, "OriginalPass123", "patient").status_code == 200

    after = login(client, email, "WrongPass123", "patient")
    assert after.json()["failed_attempts"] == 1


# --- 5-8: requesting a code -------------------------------------------------


def test_forgot_password_returns_generic_message_and_sends_a_code(client):
    email = make_patient(client, "forgot")
    response = start_reset(client, email)

    assert response.status_code == 200
    assert response.json()["message"] == GENERIC_MESSAGE
    assert emails_sent() == 1
    assert latest_code().isdigit()


def test_nonexistent_email_gets_identical_response_and_no_email(client):
    known = make_patient(client, "privacy")

    unknown = start_reset(client, "does.not.exist@example.com")
    assert unknown.status_code == 200
    assert unknown.json()["message"] == GENERIC_MESSAGE
    assert emails_sent() == 0, "no email may be sent for an unknown address"

    # The known address produces the same body.
    assert start_reset(client, known).json()["message"] == GENERIC_MESSAGE


def test_role_mismatch_is_indistinguishable_from_unknown_email(client):
    email = make_patient(client, "rolemismatch")
    response = start_reset(client, email, role="doctor")
    assert response.status_code == 200
    assert response.json()["message"] == GENERIC_MESSAGE
    assert emails_sent() == 0


def test_codes_are_random_secure_and_not_stored_in_plaintext(client):
    email = make_patient(client, "random")
    start_reset(client, email)
    first = latest_code()

    assert len(first) == 6 and first.isdigit()

    clear_reset_cooldown(email)
    start_reset(client, email)
    second = latest_code()

    assert first != second, "each request must produce a fresh random code"

    db = db_session()
    try:
        account = db.execute(select(Account).where(Account.email == email)).scalar_one()
        token = (
            db.execute(
                select(PasswordResetToken)
                .where(PasswordResetToken.account_id == account.id)
                .order_by(PasswordResetToken.id.desc())
            )
            .scalars()
            .first()
        )
    finally:
        db.close()

    assert token is not None
    assert token.code_hash != second
    assert second not in token.code_hash
    assert token.code_hash.startswith("$2"), "codes must be bcrypt-hashed at rest"


def test_reset_requests_are_rate_limited(client):
    email = make_patient(client, "cooldown")
    assert start_reset(client, email).status_code == 200

    blocked = start_reset(client, email)
    assert blocked.status_code == 429
    assert blocked.json()["retry_after_seconds"] > 0


# --- 9-13: verifying the code ----------------------------------------------


def test_expired_code_is_rejected(client):
    email = make_patient(client, "expired")
    start_reset(client, email)
    code = latest_code()

    expire_code(email)

    response = verify_code(client, email, code)
    assert response.status_code == 422
    assert "expired" in response.json()["detail"].lower()

    # An expired code must not lead to a password change.
    assert complete_reset(client, "not-a-real-token", "BrandNew123").status_code == 401


def test_wrong_code_is_rejected_without_leaking_information(client):
    email = make_patient(client, "wrongcode")
    start_reset(client, email)
    code = latest_code()
    wrong = "000000" if code != "000000" else "111111"

    response = verify_code(client, email, wrong)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid verification code."

    # An unknown address returns the identical error.
    other = verify_code(client, "unknown.person@example.com", wrong)
    assert other.status_code == 401
    assert other.json()["detail"] == "Invalid verification code."


def test_too_many_wrong_attempts_invalidate_the_request(client):
    email = make_patient(client, "bruteforce")
    start_reset(client, email)
    code = latest_code()

    for _ in range(5):
        assert verify_code(client, email, "000000").status_code == 401

    exhausted = verify_code(client, email, "000000")
    assert exhausted.status_code == 429

    # Even the correct code no longer works.
    assert verify_code(client, email, code).status_code == 401


def test_resend_creates_new_code_and_invalidates_the_previous_one(client):
    email = make_patient(client, "resend")
    start_reset(client, email)
    old_code = latest_code()

    clear_reset_cooldown(email)
    start_reset(client, email)
    new_code = latest_code()

    assert old_code != new_code
    assert verify_code(client, email, old_code).status_code == 401

    verified = verify_code(client, email, new_code)
    assert verified.status_code == 200
    assert verified.json()["reset_token"]


# --- 14-18 + session invalidation ------------------------------------------


@pytest.fixture
def reset_ready_patient(client):
    """A patient with a verified reset token, plus a pre-reset session token."""
    email = make_patient(client, "ready")
    session = login(client, email, "OriginalPass123", "patient").json()
    pre_reset_access_token = session["access_token"]

    start_reset(client, email)
    code = latest_code()
    verified = verify_code(client, email, code)
    assert verified.status_code == 200

    return {
        "email": email,
        "old_password": "OriginalPass123",
        "new_password": "BrandNewSecret456",
        "reset_token": verified.json()["reset_token"],
        "access_token": pre_reset_access_token,
    }


def test_valid_code_permits_password_reset(client, reset_ready_patient):
    response = complete_reset(client, reset_ready_patient["reset_token"], reset_ready_patient["new_password"])
    assert response.status_code == 200
    assert "updated" in response.json()["message"].lower()


def test_new_password_is_securely_hashed(client, reset_ready_patient):
    complete_reset(client, reset_ready_patient["reset_token"], reset_ready_patient["new_password"])

    db = db_session()
    try:
        account = db.execute(
            select(Account).where(Account.email == reset_ready_patient["email"])
        ).scalar_one()
    finally:
        db.close()

    assert account.password_hash != reset_ready_patient["new_password"]
    assert account.password_hash.startswith("$2")
    assert account.legacy_password_hash is None


def test_old_password_stops_working_and_new_password_works(client, reset_ready_patient):
    complete_reset(client, reset_ready_patient["reset_token"], reset_ready_patient["new_password"])

    assert login(client, reset_ready_patient["email"], reset_ready_patient["old_password"], "patient").status_code == 401
    assert login(client, reset_ready_patient["email"], reset_ready_patient["new_password"], "patient").status_code == 200


def test_reset_token_cannot_be_reused(client, reset_ready_patient):
    first = complete_reset(client, reset_ready_patient["reset_token"], reset_ready_patient["new_password"])
    assert first.status_code == 200

    second = complete_reset(client, reset_ready_patient["reset_token"], "AnotherPass789")
    assert second.status_code == 401


def test_password_change_invalidates_existing_sessions(client, reset_ready_patient):
    headers = auth_headers(reset_ready_patient["access_token"])
    assert client.get("/auth/me", headers=headers).status_code == 200

    complete_reset(client, reset_ready_patient["reset_token"], reset_ready_patient["new_password"])

    # The token issued before the change is no longer accepted.
    assert client.get("/auth/me", headers=headers).status_code == 401

    fresh = login(client, reset_ready_patient["email"], reset_ready_patient["new_password"], "patient")
    assert fresh.status_code == 200
    assert client.get("/auth/me", headers=auth_headers(fresh.json()["access_token"])).status_code == 200


def test_verified_code_cannot_be_verified_twice(client):
    email = make_patient(client, "singleuse")
    start_reset(client, email)
    code = latest_code()

    assert verify_code(client, email, code).status_code == 200
    assert verify_code(client, email, code).status_code == 401


# --- 8: password policy is enforced by the backend --------------------------


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("short1", "at least"),
        ("alllettersonly", "number"),
        ("12345678", "letter"),
        (" PaddedPass123 ", "whitespace"),
    ],
)
def test_new_password_policy_is_enforced_server_side(client, candidate, expected):
    email = make_patient(client, "policy")
    start_reset(client, email)
    verified = verify_code(client, email, latest_code())
    assert verified.status_code == 200

    response = complete_reset(client, verified.json()["reset_token"], candidate)
    assert response.status_code == 422
    assert expected in response.json()["detail"].lower()


def test_mismatched_confirmation_is_rejected(client):
    email = make_patient(client, "mismatch")
    start_reset(client, email)
    verified = verify_code(client, email, latest_code())

    response = complete_reset(
        client, verified.json()["reset_token"], "BrandNewSecret456", "DifferentSecret789"
    )
    assert response.status_code == 422


def test_reset_token_is_not_accepted_as_a_session_token(client, reset_ready_patient):
    headers = {"Authorization": f"Bearer {reset_ready_patient['reset_token']}"}
    assert client.get("/auth/me", headers=headers).status_code == 401


# --- 19-21: existing roles and authorization remain intact -------------------


def test_patient_authentication_still_works(client):
    email = make_patient(client, "stillpatient")
    response = login(client, email, "OriginalPass123", "patient")
    assert response.status_code == 200
    assert response.json()["account"]["role"] == "patient"


def test_doctor_authentication_still_works(client):
    from tests.conftest import register as register_account

    email = f"doctor.still.{__import__('os').urandom(3).hex()}@example.com"
    created = register_account(
        client,
        email,
        "DoctorPass123",
        "doctor",
        full_name="Dr Still Working",
        specialization="Orthopedic Rehabilitation",
    )
    assert created.status_code == 201
    assert created.json()["account"]["role"] == "doctor"

    assert login(client, email, "DoctorPass123", "doctor").status_code == 200
    # The doctor credentials are not accepted in the patient portal.
    assert login(client, email, "DoctorPass123", "patient").status_code == 401


def test_role_based_authorization_still_enforced(client):
    email = make_patient(client, "authz")
    token = login(client, email, "OriginalPass123", "patient").json()["access_token"]
    assert client.get("/patients", headers=auth_headers(token)).status_code == 403
    assert client.get("/sessions").status_code == 401
