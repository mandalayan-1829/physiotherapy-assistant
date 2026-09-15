"""Password hashing, secure code generation and JWT token helpers.

Passwords are never stored in plaintext. New passwords use bcrypt. The legacy
``aiphysio.db`` stored unsalted SHA-256 digests, so a verification helper for
that format is kept purely to allow a one-time upgrade on successful login.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import settings

# --- Password hashing -------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash for ``password``."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def hash_legacy_sha256(password: str) -> str:
    """Reproduce the unsalted SHA-256 digest used by the legacy database."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_legacy_password(password: str, legacy_hash: str) -> bool:
    """Check ``password`` against a legacy unsalted SHA-256 digest."""
    if not legacy_hash:
        return False
    return hmac.compare_digest(hash_legacy_sha256(password), legacy_hash.lower())


# --- Password policy --------------------------------------------------------

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128


def validate_password_policy(password: str) -> str | None:
    """Return an error message when ``password`` fails the backend policy.

    The backend is the final authority: registration and password reset both
    call this, regardless of any client-side validation.
    """
    if password is None or password == "":
        return "Password must not be empty."
    if password != password.strip():
        return "Password must not start or end with whitespace."
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters long."
    if len(password) > PASSWORD_MAX_LENGTH:
        return f"Password must be at most {PASSWORD_MAX_LENGTH} characters long."
    if not any(character.isalpha() for character in password):
        return "Password must contain at least one letter."
    if not any(character.isdigit() for character in password):
        return "Password must contain at least one number."
    return None


# --- Verification codes -----------------------------------------------------


def generate_verification_code(length: int | None = None) -> str:
    """Return a cryptographically random numeric verification code.

    Uses ``secrets`` (CSPRNG). Never generated on the client and never
    hardcoded.
    """
    digits = length or settings.password_reset_code_length
    lower = 10 ** (digits - 1)
    upper = 10**digits
    return str(secrets.randbelow(upper - lower) + lower)


def hash_verification_code(code: str) -> str:
    """Hash a verification code for storage (bcrypt, same as passwords)."""
    return hash_password(code)


def verify_verification_code(code: str, code_hash: str) -> bool:
    """Constant-time-ish comparison of a submitted code against its hash."""
    return verify_password(code, code_hash)


# --- JWT --------------------------------------------------------------------

PURPOSE_ACCESS = "access"
PURPOSE_PASSWORD_RESET = "password_reset"


def create_access_token(
    subject: str | int,
    role: str,
    token_version: int = 0,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a session access token.

    ``token_version`` mirrors the account's counter so that changing a password
    can invalidate every previously issued token.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "tv": token_version,
        "purpose": PURPOSE_ACCESS,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_password_reset_token(subject: str | int, reset_token_id: int) -> tuple[str, int]:
    """Create the short-lived authorisation issued after code verification.

    Returns ``(token, expires_in_seconds)``.
    """
    now = datetime.now(timezone.utc)
    ttl_minutes = settings.password_reset_token_ttl_minutes
    payload: dict[str, Any] = {
        "sub": str(subject),
        "jti": str(reset_token_id),
        "purpose": PURPOSE_PASSWORD_RESET,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return (
        jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm),
        ttl_minutes * 60,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode a JWT, returning ``None`` when it is invalid or expired."""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None
