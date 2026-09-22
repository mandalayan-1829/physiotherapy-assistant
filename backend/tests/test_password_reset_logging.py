"""Password reset codes must never be observable outside the e-mail itself.

A reset code that reaches an application log has effectively leaked: logs are
shipped to third-party aggregators, rotated to disk, and read by operators who
have no business being able to reset a patient's password. The same applies to
returning the code in an API response.

The tests therefore assert three things:

* the code does not appear in any log record, at any level,
* the code does not appear in any API response, and
* the suite can still *complete* the flow, because the console transport keeps the
  message in memory for tests instead of logging it.
"""

from __future__ import annotations

import logging
import re

from app.services.email_service import get_console_outbox
from tests.conftest import register

CODE_PATTERN = re.compile(r"code is:\s*(\d{6})")


def _latest_code() -> str:
    outbox = get_console_outbox()
    assert outbox, "the console transport should have recorded the message"
    match = CODE_PATTERN.search(outbox[-1].text_body)
    assert match, "the recorded message should contain a verification code"
    return match.group(1)


def _patient(client, prefix: str) -> str:
    email = f"{prefix}.{__import__('os').urandom(3).hex()}@example.com"
    assert register(client, email, "OriginalPass123", "patient").status_code == 201
    return email


def test_reset_code_is_never_written_to_logs(client, caplog):
    caplog.set_level(logging.DEBUG)
    email = _patient(client, "logfree")

    response = client.post("/auth/forgot-password", json={"email": email, "role": "patient"})
    assert response.status_code == 200

    code = _latest_code()
    assert len(code) == 6 and code.isdigit()

    # Nothing that was logged may contain the code.
    assert code not in caplog.text, "the password reset code leaked into the logs"
    # Nor the message body it was taken from.
    for record in caplog.records:
        message = record.getMessage()
        assert "Your PhysioAI password reset code" not in message
        assert code not in message


def test_console_transport_warns_without_revealing_the_body(client, caplog):
    """The development transport must be loud about not sending, and silent about the code."""
    caplog.set_level(logging.DEBUG)
    email = _patient(client, "loudwarning")

    client.post("/auth/forgot-password", json={"email": email, "role": "patient"})
    code = _latest_code()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("EMAIL NOT SENT" in r.getMessage() for r in warnings), (
        "an undelivered reset e-mail must produce a visible warning"
    )
    assert all(code not in r.getMessage() for r in caplog.records)


def test_reset_code_is_not_returned_in_the_api_response(client):
    email = _patient(client, "apiresponse")

    response = client.post("/auth/forgot-password", json={"email": email, "role": "patient"})
    assert response.status_code == 200

    code = _latest_code()
    body = response.text
    payload = response.json()

    assert code not in body
    assert set(payload.keys()) == {"message"}, "the response must not carry extra fields"
    assert payload["message"] == (
        "If an account exists for this email, a verification code has been sent."
    )


def test_failed_verification_attempts_are_not_logged_with_the_code(client, caplog):
    """Neither the submitted nor the expected code may be logged on a failure."""
    caplog.set_level(logging.DEBUG)
    email = _patient(client, "failedverify")

    client.post("/auth/forgot-password", json={"email": email, "role": "patient"})
    real_code = _latest_code()
    wrong_code = "000000" if real_code != "000000" else "111111"

    response = client.post(
        "/auth/verify-reset-code",
        json={"email": email, "verification_code": wrong_code, "role": "patient"},
    )
    assert response.status_code == 401
    assert real_code not in caplog.text
    assert wrong_code not in caplog.text
    assert real_code not in response.text


def test_verification_code_is_only_stored_hashed(client):
    """The code must not be recoverable from the database either."""
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models import Account, PasswordResetToken

    email = _patient(client, "hashedatrest")
    client.post("/auth/forgot-password", json={"email": email, "role": "patient"})
    code = _latest_code()

    db = SessionLocal()
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
    assert token.code_hash.startswith("$2"), "codes must be bcrypt-hashed at rest"
    assert code not in token.code_hash


def test_forgot_password_never_discloses_whether_the_account_exists(client, caplog):
    caplog.set_level(logging.DEBUG)
    known = _patient(client, "existscheck")

    known_response = client.post(
        "/auth/forgot-password", json={"email": known, "role": "patient"}
    )
    unknown_response = client.post(
        "/auth/forgot-password", json={"email": "ghost@example.com", "role": "patient"}
    )

    assert known_response.status_code == unknown_response.status_code == 200
    assert known_response.json() == unknown_response.json()
