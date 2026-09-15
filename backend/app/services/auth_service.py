"""Authentication service: registration, credential verification and login."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    create_access_token,
    hash_password,
    validate_password_policy,
    verify_legacy_password,
    verify_password,
)
from app.models import SCOPE_LOGIN, Account, PatientProfile, ROLE_DOCTOR, ROLE_PATIENT
from app.models.doctor import DoctorProfile
from app.schemas.auth import RegisterRequest
from app.services import rate_limit
from app.services.errors import (
    AuthError,
    ConflictError,
    TooManyRequestsError,
    ValidationError,
)

LOGIN_WINDOW_SECONDS = settings.login_attempt_window_minutes * 60


def get_account_by_email(db: Session, email: str) -> Account | None:
    return db.execute(
        select(Account).where(Account.email == email.lower())
    ).scalar_one_or_none()


def get_account(db: Session, account_id: int) -> Account | None:
    return db.get(Account, account_id)


def register_account(db: Session, payload: RegisterRequest) -> Account:
    """Create a new account with exactly one of the two allowed roles."""
    email = payload.email.lower()

    if get_account_by_email(db, email):
        raise ConflictError("An account with this email already exists.")

    if payload.role == ROLE_DOCTOR and not payload.specialization:
        raise ValidationError("A specialization is required to register as a doctor.")

    # Registration is a password-setting path, so it is subject to the same
    # backend policy as a password reset.
    policy_error = validate_password_policy(payload.password)
    if policy_error:
        raise ValidationError(policy_error)

    account = Account(
        email=email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(account)
    db.flush()  # assign account.id

    if payload.role == ROLE_PATIENT:
        db.add(PatientProfile(account_id=account.id))
    else:
        db.add(
            DoctorProfile(
                account_id=account.id,
                name=account.full_name,
                specialization=payload.specialization or "",
                qualification=payload.qualification or "",
                email=email,
            )
        )

    db.commit()
    db.refresh(account)
    return account


def _too_many_attempts_error(seconds_remaining: int) -> TooManyRequestsError:
    return TooManyRequestsError(
        "Too many unsuccessful sign-in attempts. Please wait and try again, "
        "or reset your password.",
        extra={
            "retry_after_seconds": seconds_remaining,
            "show_forgot_password": True,
        },
    )


def authenticate(db: Session, email: str, password: str, role: str) -> Account:
    """Validate credentials and return the account.

    Failed attempts are counted server-side. After
    ``login_forgot_password_threshold`` failures the response tells the client
    to offer "Forgot password?"; past ``login_max_failed_attempts`` the address
    is temporarily rate-limited.

    The same messages and counters are used for unknown addresses, so the
    endpoint cannot be used to discover registered accounts.
    """
    normalized = email.strip().lower()

    state = rate_limit.peek(db, SCOPE_LOGIN, normalized, LOGIN_WINDOW_SECONDS)
    if state.attempt_count >= settings.login_max_failed_attempts:
        raise _too_many_attempts_error(state.seconds_until_window_end(LOGIN_WINDOW_SECONDS))

    account = get_account_by_email(db, normalized)

    def fail() -> AuthError:
        new_state = rate_limit.register_attempt(
            db, SCOPE_LOGIN, normalized, LOGIN_WINDOW_SECONDS
        )
        # ``login_max_failed_attempts`` failures are answered with the normal
        # error; the next request is rate-limited.
        if new_state.attempt_count > settings.login_max_failed_attempts:
            raise _too_many_attempts_error(
                new_state.seconds_until_window_end(LOGIN_WINDOW_SECONDS)
            )
        return AuthError(
            "Invalid email or password.",
            extra={
                "failed_attempts": new_state.attempt_count,
                "show_forgot_password": new_state.attempt_count
                >= settings.login_forgot_password_threshold,
            },
        )

    if account is None or not account.is_active or account.role != role:
        raise fail()

    if not verify_password(password, account.password_hash):
        # Legacy digest fallback (migration path only, from aiphysio.db).
        if account.legacy_password_hash and verify_legacy_password(
            password, account.legacy_password_hash
        ):
            account.password_hash = hash_password(password)
            account.legacy_password_hash = None
            db.commit()
            db.refresh(account)
        else:
            raise fail()

    # Successful sign-in clears the failure counter.
    rate_limit.reset(db, SCOPE_LOGIN, normalized)
    return account


def issue_token(account: Account) -> tuple[str, int]:
    """Return an access token and its lifetime in seconds."""
    token = create_access_token(
        subject=account.id,
        role=account.role,
        token_version=account.token_version or 0,
    )
    expires_in = settings.access_token_expire_minutes * 60
    return token, expires_in
