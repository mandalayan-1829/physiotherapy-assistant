"""Application settings.

All configuration is read from environment variables (or a local ``.env`` file)
so that the same code can run against SQLite in development and a managed
PostgreSQL instance in production without code changes.

Security posture
----------------
There is deliberately **no default value for ``SECRET_KEY``**. The JWT signing
key is the single credential that protects every session in the system: if it is
predictable, anyone can mint a token for any account and any role. The
configuration therefore *fails fast* — importing this module raises
``ConfigurationError`` when the key is missing, too short, or one of the values
that have historically been shipped in templates.

A key is never generated automatically: doing so would silently invalidate every
issued token on each restart. The operator supplies it:

    python -c "import secrets; print(secrets.token_urlsafe(48))"

The secret is never logged and never committed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("physioai.config")

ENVIRONMENT_DEVELOPMENT = "development"
ENVIRONMENT_TEST = "test"
ENVIRONMENT_PRODUCTION = "production"

Environment = Literal["development", "test", "production"]

#: Minimum acceptable length of the JWT signing key, in characters. 32 bytes of
#: entropy is the floor recommended for HS256.
MIN_SECRET_KEY_LENGTH = 32

#: Values that must never be used to sign tokens. Every one of these has
#: appeared in a template, a tutorial, or an earlier revision of this file, so
#: they are exactly what an attacker would try first.
INSECURE_SECRET_VALUES = frozenset(
    {
        "change-me-to-a-long-random-value",
        "changeme",
        "change-me",
        "changethis",
        "change_this",
        "dev",
        "dev-secret",
        "dev-insecure-change-me",
        "development",
        "example",
        "insecure",
        "jwt-secret",
        "jwt_secret",
        "physioai",
        "placeholder",
        "please-change-me",
        "replace-me",
        "secret",
        "secret-key",
        "test",
        "testing",
        "todo",
        "your-secret-key",
        "your-secret-key-here",
    }
)


class ConfigurationError(RuntimeError):
    """Raised when the application is configured in an unsafe way.

    This is intentionally raised while settings are constructed, so an
    insecurely configured deployment fails *before* it can accept traffic.
    """


def is_insecure_secret_key(value: str | None) -> bool:
    """Return ``True`` when ``value`` is empty, a known placeholder or too short."""
    if value is None:
        return True
    candidate = value.strip()
    if not candidate:
        return True
    if len(candidate) < MIN_SECRET_KEY_LENGTH:
        return True
    if candidate.lower() in INSECURE_SECRET_VALUES:
        return True
    # A secret made of a single repeated character is no secret at all.
    return len(set(candidate)) < 4


def validate_secret_key(value: str | None) -> str:
    """Validate the JWT signing key, or raise :class:`ConfigurationError`.

    Returns the validated value so it can be assigned inside a model validator.
    The offending value is never included in the error message or in logs.
    """
    if value is None or not value.strip():
        raise ConfigurationError(
            "SECRET_KEY is not set. The API refuses to start without a JWT signing "
            'key. Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))" '
            "and provide it as the SECRET_KEY environment variable."
        )

    candidate = value.strip()
    lowered = candidate.lower()

    if lowered in INSECURE_SECRET_VALUES:
        raise ConfigurationError(
            "SECRET_KEY is set to a well-known placeholder value and is therefore "
            "not secret. Generate a real key with: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )

    if len(candidate) < MIN_SECRET_KEY_LENGTH:
        raise ConfigurationError(
            f"SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters long "
            "(a shorter key can be brute-forced, which would allow forged tokens)."
        )

    if len(set(candidate)) < 4:
        raise ConfigurationError(
            "SECRET_KEY has too little entropy (it is built from fewer than four "
            "distinct characters). Generate a random key instead."
        )

    return candidate


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_name: str = "PhysioAI API"
    api_version: str = "0.2.0"

    # Deployment context. Drives the additional production-only checks below.
    environment: Environment = ENVIRONMENT_DEVELOPMENT

    # Database -----------------------------------------------------------
    # Default targets a clean, SQLAlchemy-managed database file.
    database_url: str = "sqlite:///./physioai.db"
    # The previous architecture's database. Never deleted, only read from.
    legacy_database_path: str = "../aiphysio.db"

    # Authentication -----------------------------------------------------
    # No default. Must be supplied through the environment. See module docstring.
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # Issuer claim embedded in, and required from, every token this API issues.
    jwt_issuer: str = "physioai-api"

    # Legacy credential migration ----------------------------------------
    # The pre-existing ``aiphysio.db`` stored unsalted SHA-256 digests. Verifying
    # against that format is a one-time migration aid, so it is *off by default*
    # and must be switched on deliberately (and only while importing legacy
    # data). When disabled, no SHA-256 digest can ever authenticate.
    allow_legacy_password_login: bool = False

    # CORS ---------------------------------------------------------------
    # Comma separated origins. Never use "*" in production.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Interactive API docs. Disabled automatically in production: publishing the
    # full route inventory is free reconnaissance for an attacker.
    enable_api_docs: bool = True

    # When the API sits behind a reverse proxy (Render, Railway, Fly, an ingress),
    # the client address arrives in X-Forwarded-For. Only trust that header when
    # a proxy you control actually sets it - otherwise the header is attacker
    # controlled and per-IP throttling becomes trivially bypassable.
    trust_proxy_headers: bool = False

    # Login throttling ---------------------------------------------------
    # Number of failed sign-ins allowed inside the window before the account
    # e-mail (or unknown address) is temporarily locked out.
    login_max_failed_attempts: int = 5
    login_attempt_window_minutes: int = 15
    # After this many failures the UI is told to offer "Forgot password?".
    login_forgot_password_threshold: int = 3

    # Per-source (IP) throttling. These exist because per-email counters alone
    # are bypassed by rotating addresses, and because registration is a fully
    # unauthenticated write path.
    register_max_requests_per_hour: int = 5
    login_max_requests_per_ip_per_window: int = 30
    password_reset_max_requests_per_ip_per_hour: int = 20

    # Password reset -----------------------------------------------------
    password_reset_code_ttl_minutes: int = 10
    password_reset_max_attempts: int = 5
    password_reset_resend_cooldown_seconds: int = 60
    password_reset_max_requests_per_hour: int = 5
    # Lifetime of the short-lived token returned after a code is verified.
    password_reset_token_ttl_minutes: int = 10
    password_reset_code_length: int = 6

    # Email --------------------------------------------------------------
    # "smtp" sends real mail, "console" records the message server-side without
    # logging its body (used for local development and tests), "auto" picks smtp
    # when SMTP_HOST is set.
    email_backend: str = "auto"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "no-reply@physioai.local"
    smtp_from_name: str = "PhysioAI"
    # STARTTLS on port 587 is the common default. Port 465 is treated as implicit
    # TLS automatically, so no provider-specific code is needed.
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    email_timeout_seconds: int = 20

    # Production normally refuses to start without a real mail transport, because
    # a password-reset code that cannot be delivered has to be either shown to
    # the user or written to a log to be useful - and both would leak it.
    #
    # This is a deliberate escape hatch for a deployment window in which password
    # reset is knowingly unavailable (e.g. the SMTP provider has not been chosen
    # yet). It is off by default, it does *not* weaken SECRET_KEY, CORS or the
    # JWT algorithm checks, and while it is on the API logs a warning at startup.
    # Codes are still never logged or returned; reset simply cannot complete.
    allow_production_without_email: bool = False

    # Doctor onboarding --------------------------------------------------
    # Newly registered clinicians start unverified: they are hidden from the
    # patient-facing directory and cannot be booked until an operator verifies
    # them (see backend/scripts/verify_doctor.py). There is no admin role by
    # design, so verification is an out-of-band operations action.
    doctor_default_verified: bool = False

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == ENVIRONMENT_PRODUCTION

    @property
    def docs_url(self) -> str | None:
        return "/docs" if (self.enable_api_docs and not self.is_production) else None

    @property
    def redoc_url(self) -> str | None:
        return "/redoc" if (self.enable_api_docs and not self.is_production) else None

    @property
    def openapi_url(self) -> str | None:
        return "/openapi.json" if (self.enable_api_docs and not self.is_production) else None

    def resolved_email_backend(self) -> str:
        """Return the effective mail backend ('smtp' or 'console')."""
        backend = (self.email_backend or "auto").strip().lower()
        if backend == "auto":
            return "smtp" if self.smtp_host else "console"
        return backend

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_security_configuration(self) -> "Settings":
        """Fail fast on an unsafe configuration.

        Runs for every environment so that a misconfigured deployment cannot
        start and serve traffic. The production-only checks are the ones that
        would otherwise leak credentials or health data.
        """
        self.secret_key = validate_secret_key(self.secret_key)

        if self.jwt_algorithm.upper() not in {"HS256", "HS384", "HS512"}:
            raise ConfigurationError(
                "JWT_ALGORITHM must be one of HS256, HS384 or HS512 (HMAC). "
                f"Received {self.jwt_algorithm!r}. Only a secret-based algorithm may "
                "be used because the API signs tokens with SECRET_KEY."
            )

        if self.is_production:
            effective_email_backend = self.resolved_email_backend()
            if effective_email_backend != "smtp":
                if not self.allow_production_without_email:
                    raise ConfigurationError(
                        "ENVIRONMENT=production requires a real mail transport "
                        "(EMAIL_BACKEND=smtp with SMTP_HOST configured). Without it, "
                        "password reset codes cannot be delivered and must never be "
                        "written to logs or returned to the client. If you accept "
                        "that password reset is unavailable, set "
                        "ALLOW_PRODUCTION_WITHOUT_EMAIL=true deliberately."
                    )
                logger.warning(
                    "ALLOW_PRODUCTION_WITHOUT_EMAIL is enabled and the mail transport "
                    "is %r: /auth/forgot-password cannot deliver a code, so password "
                    "reset is unavailable to every user. Reset codes are still never "
                    "logged or returned to the client. Configure SMTP_* and remove "
                    "this flag to restore the flow.",
                    effective_email_backend,
                )
            local_origins = [
                origin
                for origin in self.cors_origin_list
                if "localhost" in origin or "127.0.0.1" in origin
            ]
            if local_origins:
                raise ConfigurationError(
                    "ENVIRONMENT=production must not list localhost origins in "
                    "CORS_ORIGINS. Set CORS_ORIGINS to the exact deployed frontend "
                    "origin(s)."
                )
            if not self.cors_origin_list:
                raise ConfigurationError(
                    "ENVIRONMENT=production requires CORS_ORIGINS to be set to the "
                    "exact deployed frontend origin(s)."
                )

        if self.allow_legacy_password_login:
            logger.warning(
                "ALLOW_LEGACY_PASSWORD_LOGIN is enabled: unsalted legacy SHA-256 "
                "digests will be accepted once per account and upgraded to bcrypt. "
                "Use this only while migrating aiphysio.db, then turn it off."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Constructing the settings at import time is deliberate: a missing or insecure
# SECRET_KEY must stop the process immediately rather than surface as an obscure
# 500 on the first login attempt.
settings = get_settings()
