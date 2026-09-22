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
from app.models import (
    SCOPE_AUTH_IP,
    SCOPE_LOGIN,
    Account,
    PatientProfile,
    ROLE_DOCTOR,
    ROLE_PATIENT,
)
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


def _enforce_source_limit(
    db: Session,
    action: str,
    client_ip: str,
    max_attempts: int,
    window_seconds: int,
    message: str,
) -> None:
    """Throttle an unauthenticated action per source address.

    Keyed by ``"<action>:<ip>"`` inside the shared ``auth_ip`` scope so every
    auth action has its own budget while sharing one table.
    """
    rate_limit.enforce_limit(
        db, SCOPE_AUTH_IP, f"{action}:{client_ip}", max_attempts, window_seconds, message
    )


def register_account(db: Session, payload: RegisterRequest, client_ip: str) -> Account:
    """Create a new account with exactly one of the two allowed roles.

    ``client_ip`` is required: registration is a fully unauthenticated write
    path, so it must be rate-limited per source address rather than left open
    (OWASP API4).
    """
    _enforce_source_limit(
        db,
        "register",
        client_ip,
        settings.register_max_requests_per_hour,
        3600,
        "Too many accounts have been created from this address. Please try again later.",
    )

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
        # Registering as a doctor creates a *pending* profile. It confers no
        # clinical authority until an operator verifies it: the profile is not
        # listed in the directory, cannot be booked, and cannot acquire access
        # to a patient's records. See app.services.access.ensure_doctor_verified.
        db.add(
            DoctorProfile(
                account_id=account.id,
                name=account.full_name,
                specialization=payload.specialization or "",
                qualification=payload.qualification or "",
                email=email,
                is_verified=settings.doctor_default_verified,
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


def authenticate(
    db: Session, email: str, password: str, role: str, client_ip: str
) -> Account:
    """Validate credentials and return the account.

    Failed attempts are counted server-side. After
    ``login_forgot_password_threshold`` failures the response tells the client
    to offer "Forgot password?"; past ``login_max_failed_attempts`` the address
    is temporarily rate-limited.

    The same messages and counters are used for unknown addresses, so the
    endpoint cannot be used to discover registered accounts.

    In addition to the per-address counter, one budget is spent per source
    address. Per-address counters alone do nothing against password spraying -
    an attacker simply rotates the address - so the two limits are complementary
    (OWASP API2).
    """
    normalized = email.strip().lower()

    _enforce_source_limit(
        db,
        "login",
        client_ip,
        settings.login_max_requests_per_ip_per_window,
        LOGIN_WINDOW_SECONDS,
        "Too many sign-in attempts from this network. Please wait and try again.",
    )

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
        # Legacy digest migration path (unsalted SHA-256, inherited from
        # aiphysio.db). It is *disabled by default*: an unsalted digest must not
        # be usable as an authentication factor unless an operator has
        # deliberately switched it on while importing legacy data.
        legacy_allowed = settings.allow_legacy_password_login
        if (
            legacy_allowed
            and account.legacy_password_hash
            and verify_legacy_password(password, account.legacy_password_hash)
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
