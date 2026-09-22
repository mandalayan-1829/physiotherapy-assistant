"""Password reset flow.

Design notes
------------
* The verification code is generated server-side with a CSPRNG and stored only
  as a bcrypt hash.
* Every request creates a new row and invalidates any earlier unused row, so a
  code is single-use and a resend always supersedes the previous code.
* Responses never reveal whether an email address belongs to an account, in
  either the patient or the doctor portal.
* Changing the password increments ``Account.token_version``, which invalidates
  every access token that was issued before the change.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    PURPOSE_PASSWORD_RESET,
    create_password_reset_token,
    decode_access_token,
    generate_verification_code,
    hash_password,
    hash_verification_code,
    validate_password_policy,
    verify_verification_code,
)
from app.models import (
    SCOPE_AUTH_IP,
    SCOPE_LOGIN,
    SCOPE_PASSWORD_RESET_REQUEST,
    Account,
    PasswordResetToken,
)
from app.schemas.password_reset import GENERIC_FORGOT_PASSWORD_MESSAGE
from app.services import rate_limit
from app.services.email_service import OutboundEmail, get_email_sender
from app.services.errors import AuthError, TooManyRequestsError, ValidationError

logger = logging.getLogger("physioai.password_reset")

INVALID_CODE_MESSAGE = "Invalid verification code."
EXPIRED_CODE_MESSAGE = "This verification code has expired. Please request a new code."
TOO_MANY_ATTEMPTS_MESSAGE = (
    "Too many incorrect attempts. Please request a new verification code."
)
INVALID_RESET_TOKEN_MESSAGE = (
    "This password reset request is no longer valid. Please start again."
)

RESET_REQUEST_WINDOW_SECONDS = 3600


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _find_account(db: Session, email: str) -> Account | None:
    return db.execute(
        select(Account).where(Account.email == email.strip().lower())
    ).scalar_one_or_none()


def _active_tokens(db: Session, account_id: int) -> list[PasswordResetToken]:
    return list(
        db.execute(
            select(PasswordResetToken)
            .where(
                PasswordResetToken.account_id == account_id,
                PasswordResetToken.used.is_(False),
                PasswordResetToken.invalidated.is_(False),
            )
            .order_by(PasswordResetToken.created_at.desc(), PasswordResetToken.id.desc())
        ).scalars().all()
    )


def _invalidate_account_tokens(db: Session, account_id: int, *, except_id: int | None = None) -> None:
    for token in _active_tokens(db, account_id):
        if except_id is not None and token.id == except_id:
            continue
        token.invalidated = True


def _enforce_source_limit(db: Session, action: str, client_ip: str) -> None:
    """Spend one unit of the per-source budget for a recovery-flow action.

    The whole flow (request, verify, reset) is covered, because an attacker who
    can enumerate codes does not care which step is throttled as long as one of
    them is not. Keys are namespaced per action inside one shared scope.
    """
    rate_limit.enforce_limit(
        db,
        SCOPE_AUTH_IP,
        f"{action}:{client_ip}",
        settings.password_reset_max_requests_per_ip_per_hour,
        RESET_REQUEST_WINDOW_SECONDS,
        "Too many password reset attempts from this network. Please try again later.",
    )


# --- 1. Request a code ------------------------------------------------------


def request_password_reset(db: Session, email: str, role: str | None, client_ip: str) -> str:
    """Create a reset request and email a verification code.

    Always returns the same generic message, whether or not the address exists.
    Throttled both per address (cooldown + hourly cap) and per source address.
    """
    normalized = email.strip().lower()

    _enforce_source_limit(db, "forgot", client_ip)

    # Throttle every request (including unknown addresses) so this endpoint
    # cannot be used to probe for registered users.
    cooldown = rate_limit.is_cooldown_active(
        db,
        SCOPE_PASSWORD_RESET_REQUEST,
        normalized,
        settings.password_reset_resend_cooldown_seconds,
    )
    if cooldown > 0:
        raise TooManyRequestsError(
            f"Please wait {cooldown} seconds before requesting another code.",
            extra={"retry_after_seconds": cooldown},
        )

    state = rate_limit.register_attempt(
        db, SCOPE_PASSWORD_RESET_REQUEST, normalized, RESET_REQUEST_WINDOW_SECONDS
    )
    if state.attempt_count > settings.password_reset_max_requests_per_hour:
        raise TooManyRequestsError(
            "Too many reset requests. Please try again later.",
            extra={"retry_after_seconds": state.seconds_until_window_end(RESET_REQUEST_WINDOW_SECONDS)},
        )

    account = _find_account(db, normalized)

    # A role mismatch behaves exactly like an unknown address: nothing is sent
    # and the response is identical, so portal membership stays private.
    if account is None or not account.is_active or (role is not None and account.role != role):
        logger.info("Password reset requested for an address with no matching active account.")
        return GENERIC_FORGOT_PASSWORD_MESSAGE

    # Supersede any earlier unused code.
    _invalidate_account_tokens(db, account.id)

    code = generate_verification_code()
    token = PasswordResetToken(
        account_id=account.id,
        portal_role=account.role,
        code_hash=hash_verification_code(code),
        expires_at=_now() + timedelta(minutes=settings.password_reset_code_ttl_minutes),
    )
    db.add(token)
    db.commit()
    db.refresh(token)

    _send_code_email(account, code, token)
    return GENERIC_FORGOT_PASSWORD_MESSAGE


def _send_code_email(account: Account, code: str, token: PasswordResetToken) -> None:
    minutes = settings.password_reset_code_ttl_minutes
    text_body = (
        f"Hello {account.full_name},\n\n"
        f"Your PhysioAI password reset code is: {code}\n\n"
        f"This code expires in {minutes} minutes and can be used once.\n"
        "If you did not request a password reset you can ignore this email.\n\n"
        "— PhysioAI Clinical Suite"
    )
    html_body = (
        f"<p>Hello {account.full_name},</p>"
        f"<p>Your PhysioAI password reset code is:</p>"
        f"<p style=\"font-size:24px;font-weight:700;letter-spacing:4px\">{code}</p>"
        f"<p>This code expires in {minutes} minutes and can be used once.</p>"
        "<p>If you did not request a password reset you can ignore this email.</p>"
        "<p>— PhysioAI Clinical Suite</p>"
    )

    result = get_email_sender().send(
        OutboundEmail(
            to=account.email,
            subject="Your PhysioAI password reset code",
            text_body=text_body,
            html_body=html_body,
        )
    )

    if not result.delivered:
        # Deliberately explicit: the caller must not treat this as a send.
        logger.warning(
            "Password reset code was NOT delivered (transport=%s): %s",
            result.transport,
            result.detail,
        )
    else:
        logger.info("Password reset code delivered to %s via %s", account.email, result.transport)


# --- 2. Verify the code -----------------------------------------------------


def verify_reset_code(
    db: Session, email: str, code: str, role: str | None, client_ip: str
) -> tuple[str, int]:
    """Return ``(reset_token, expires_in_seconds)`` when the code is valid."""
    normalized = email.strip().lower()

    # Guards against brute-forcing a 6-digit code across many accounts.
    _enforce_source_limit(db, "verify", client_ip)

    account = _find_account(db, normalized)

    if account is None or not account.is_active:
        raise AuthError(INVALID_CODE_MESSAGE)

    tokens = _active_tokens(db, account.id)
    live = [t for t in tokens if t.verified_at is None]
    if not live:
        raise AuthError(INVALID_CODE_MESSAGE)

    token = live[0]

    if role is not None and token.portal_role != role:
        raise AuthError(INVALID_CODE_MESSAGE)

    if _as_aware(token.expires_at) <= _now():
        token.invalidated = True
        db.commit()
        raise ValidationError(EXPIRED_CODE_MESSAGE)

    if token.attempt_count >= settings.password_reset_max_attempts:
        token.invalidated = True
        db.commit()
        raise TooManyRequestsError(
            TOO_MANY_ATTEMPTS_MESSAGE, extra={"restart_required": True}
        )

    # Count the attempt before comparing so a crash cannot give free guesses.
    token.attempt_count += 1

    if not verify_verification_code(code.strip(), token.code_hash):
        db.commit()
        raise AuthError(INVALID_CODE_MESSAGE)

    if token.attempt_count > settings.password_reset_max_attempts:
        token.invalidated = True
        db.commit()
        raise TooManyRequestsError(
            TOO_MANY_ATTEMPTS_MESSAGE, extra={"restart_required": True}
        )

    token.verified_at = _now()
    db.commit()

    reset_token, expires_in = create_password_reset_token(account.id, token.id)
    return reset_token, expires_in


# --- 3. Set the new password ------------------------------------------------


def reset_password(
    db: Session,
    reset_token: str,
    new_password: str,
    confirm_password: str | None = None,
    client_ip: str = "unknown",
) -> None:
    """Validate the temporary token, set the new password and end old sessions."""
    _enforce_source_limit(db, "reset", client_ip)

    payload = decode_access_token(reset_token)
    if payload is None or payload.get("purpose") != PURPOSE_PASSWORD_RESET:
        raise AuthError(INVALID_RESET_TOKEN_MESSAGE)

    try:
        account_id = int(payload["sub"])
        token_id = int(payload["jti"])
    except (KeyError, TypeError, ValueError):
        raise AuthError(INVALID_RESET_TOKEN_MESSAGE) from None

    token = db.get(PasswordResetToken, token_id)
    if token is None or token.account_id != account_id:
        raise AuthError(INVALID_RESET_TOKEN_MESSAGE)

    if token.used or token.invalidated or token.verified_at is None:
        raise AuthError(INVALID_RESET_TOKEN_MESSAGE)

    if _as_aware(token.expires_at) <= _now():
        token.invalidated = True
        db.commit()
        raise ValidationError(EXPIRED_CODE_MESSAGE)

    # Final authority on the password policy is the backend.
    if confirm_password is not None and new_password != confirm_password:
        raise ValidationError("Passwords do not match.")
    policy_error = validate_password_policy(new_password)
    if policy_error:
        raise ValidationError(policy_error)

    account = db.get(Account, account_id)
    if account is None or not account.is_active:
        raise AuthError(INVALID_RESET_TOKEN_MESSAGE)

    account.password_hash = hash_password(new_password)
    account.legacy_password_hash = None
    # Invalidate every access token issued before this change.
    account.token_version = (account.token_version or 0) + 1

    token.used = True
    _invalidate_account_tokens(db, account.id, except_id=token.id)
    db.commit()

    # A successful reset also clears the sign-in failure counter.
    rate_limit.reset(db, SCOPE_LOGIN, account.email)
    logger.info("Password reset completed for account id=%s", account.id)
