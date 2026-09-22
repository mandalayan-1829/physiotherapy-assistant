"""Fail-fast configuration: the API must never start insecurely.

The JWT signing key is the single credential that protects every session, so a
missing, guessable or placeholder value has to stop the process rather than
degrade quietly. These tests pin that behaviour down.

Every ``Settings`` instance is built with ``_env_file=None`` and explicit keyword
arguments so the result does not depend on the ambient environment.
"""

from __future__ import annotations

import logging

import pytest

from app.core.config import (
    MIN_SECRET_KEY_LENGTH,
    ConfigurationError,
    Settings,
    is_insecure_secret_key,
    validate_secret_key,
)

STRONG_SECRET = "uR2mQ9-zA7Xw4bT1cLp0sVf6yHj3kNd8eG5iOqZx"  # 44 chars, high entropy


def make_settings(**overrides) -> Settings:
    """Build isolated settings, ignoring any ``.env`` and ambient overrides."""
    base: dict[str, object] = {
        "secret_key": STRONG_SECRET,
        "cors_origins": "http://localhost:3000",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


# --- 1. No insecure default exists ------------------------------------------


def test_secret_key_has_no_default_value():
    """The field must be empty by default, never a working placeholder."""
    default = Settings.model_fields["secret_key"].default
    assert default == "", "SECRET_KEY must not ship with a usable default"


def test_the_old_hardcoded_placeholder_is_rejected():
    """The value that used to be the default must now be refused outright."""
    assert is_insecure_secret_key("dev-insecure-change-me")
    with pytest.raises(ConfigurationError):
        make_settings(secret_key="dev-insecure-change-me")


def test_env_example_placeholder_is_rejected():
    assert is_insecure_secret_key("change-me-to-a-long-random-value")
    with pytest.raises(ConfigurationError):
        make_settings(secret_key="change-me-to-a-long-random-value")


# --- 2. Missing SECRET_KEY fails fast ---------------------------------------


@pytest.mark.parametrize("value", ["", "   "])
def test_missing_secret_key_fails_fast(value):
    with pytest.raises(ConfigurationError) as excinfo:
        make_settings(secret_key=value)
    assert "SECRET_KEY is not set" in str(excinfo.value)


def test_missing_secret_key_fails_in_production_too():
    with pytest.raises(ConfigurationError):
        make_settings(
            secret_key="",
            environment="production",
            email_backend="smtp",
            smtp_host="smtp.example.com",
            cors_origins="https://app.example.com",
        )


# --- 3. Weak values are rejected --------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "short",
        "secret",
        "changeme",
        "development",
        "a" * 64,  # long but zero entropy
        "abababababababababababababababababab",  # long but two characters
    ],
)
def test_weak_or_placeholder_secret_keys_are_rejected(value):
    with pytest.raises(ConfigurationError):
        make_settings(secret_key=value)


def test_short_secret_key_reports_the_minimum_length():
    # 31 distinct characters: long enough to have entropy, but below the minimum.
    too_short = "abcdefghijklmnopqrstuvwxyz01234"
    assert len(too_short) == MIN_SECRET_KEY_LENGTH - 1
    with pytest.raises(ConfigurationError) as excinfo:
        make_settings(secret_key=too_short)
    assert str(MIN_SECRET_KEY_LENGTH) in str(excinfo.value)


def test_error_message_never_contains_the_offending_value():
    """A rejected key must not be echoed back into logs or tracebacks."""
    offending = "LooksSensitiveButIsTooShort1234"
    with pytest.raises(ConfigurationError) as excinfo:
        make_settings(secret_key=offending)
    assert offending not in str(excinfo.value)


# --- 4. A strong key is accepted --------------------------------------------


def test_strong_secret_key_is_accepted():
    settings = make_settings()
    assert settings.secret_key == STRONG_SECRET
    assert settings.is_production is False


def test_development_behaves_intentionally():
    settings = make_settings(environment="development")
    # Local development may use the console mail transport and localhost origins.
    assert settings.resolved_email_backend() == "console"
    assert settings.secret_key  # still validated, never bypassed
    assert settings.docs_url == "/docs"
    assert settings.openapi_url == "/openapi.json"
    # The insecure legacy password path is off unless explicitly enabled.
    assert settings.allow_legacy_password_login is False


# --- 5. Production-only guarantees ------------------------------------------


def test_production_requires_a_real_mail_transport():
    with pytest.raises(ConfigurationError) as excinfo:
        make_settings(
            environment="production",
            email_backend="auto",  # no SMTP_HOST -> resolves to console
            cors_origins="https://app.example.com",
        )
    assert "production" in str(excinfo.value).lower()


def test_production_rejects_the_console_mail_transport():
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="console",
            cors_origins="https://app.example.com",
        )


def test_production_rejects_localhost_cors_origins():
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="smtp",
            smtp_host="smtp.example.com",
            cors_origins="http://localhost:3000",
        )


def test_production_without_email_requires_the_explicit_opt_out():
    """The escape hatch is off by default and must be asked for by name."""
    assert Settings.model_fields["allow_production_without_email"].default is False
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="console",
            cors_origins="https://physio.example.com",
        )


def test_production_without_email_is_accepted_when_deliberately_enabled(caplog):
    """Opting out starts the API, but only after logging what it costs."""
    with caplog.at_level(logging.WARNING, logger="physioai.config"):
        settings = make_settings(
            environment="production",
            email_backend="console",
            allow_production_without_email=True,
            cors_origins="https://physio.example.com",
        )
    assert settings.is_production is True
    assert settings.resolved_email_backend() == "console"
    # The operator must be told, in the log, that the flow is unavailable.
    assert any("password reset is unavailable" in record.message for record in caplog.records)


def test_production_without_email_does_not_weaken_other_guards():
    """The opt-out is narrow: it never relaxes CORS, SECRET_KEY or the algorithm."""
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="console",
            allow_production_without_email=True,
            cors_origins="http://localhost:3000",
        )
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="console",
            allow_production_without_email=True,
            cors_origins="",
        )
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="console",
            allow_production_without_email=True,
            cors_origins="https://physio.example.com",
            secret_key="short",
        )


def test_production_requires_cors_origins():
    with pytest.raises(ConfigurationError):
        make_settings(
            environment="production",
            email_backend="smtp",
            smtp_host="smtp.example.com",
            cors_origins="",
        )


def test_production_disables_interactive_api_docs():
    settings = make_settings(
        environment="production",
        email_backend="smtp",
        smtp_host="smtp.example.com",
        cors_origins="https://app.example.com",
    )
    assert settings.docs_url is None
    assert settings.redoc_url is None
    assert settings.openapi_url is None
    assert settings.is_production is True


def test_production_with_a_complete_configuration_is_accepted():
    settings = make_settings(
        environment="production",
        email_backend="smtp",
        smtp_host="smtp.example.com",
        cors_origins="https://app.example.com,https://physio.example.com",
    )
    assert settings.cors_origin_list == [
        "https://app.example.com",
        "https://physio.example.com",
    ]
    assert settings.secret_key == STRONG_SECRET


# --- 6. Algorithm allow-list ------------------------------------------------


@pytest.mark.parametrize("algorithm", ["none", "RS256", "ES256", "PS256"])
def test_non_hmac_jwt_algorithms_are_rejected(algorithm):
    """Tokens are signed with SECRET_KEY, so only HMAC algorithms are valid.

    Allowing 'none' here would mean unsigned tokens are accepted.
    """
    with pytest.raises(ConfigurationError):
        make_settings(jwt_algorithm=algorithm)


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_hmac_jwt_algorithms_are_accepted(algorithm):
    assert make_settings(jwt_algorithm=algorithm).jwt_algorithm == algorithm


# --- 7. The validator itself -------------------------------------------------


def test_validate_secret_key_returns_the_stripped_value():
    assert validate_secret_key(f"  {STRONG_SECRET}  ") == STRONG_SECRET


def test_is_insecure_secret_key_accepts_a_generated_key():
    import secrets

    assert not is_insecure_secret_key(secrets.token_urlsafe(48))
