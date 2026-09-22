"""Abuse protection on the unauthenticated authentication surface.

OWASP API2 and API4. Two properties are checked here:

1. every anonymous entry point (registration, sign-in, and the whole password
   recovery flow) is throttled per source address, so rotating the e-mail address
   does not buy an attacker unlimited attempts; and
2. the counters live in the database rather than in process memory, which is what
   makes them hold across restarted or load-balanced backend instances.

The suite runs with generous limits (see ``conftest.py``) because every request
originates from one client address. Each test below tightens the relevant limit
and clears the matching counters first, so the assertions are about the limiter
and nothing else.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import SCOPE_AUTH_IP, AuthThrottle
from tests.conftest import auth_headers, register

TEST_CLIENT_HOST = "testclient"


def _clear_source_counters() -> None:
    """Drop every per-source counter so a test starts from a clean budget."""
    db = SessionLocal()
    try:
        db.execute(delete(AuthThrottle).where(AuthThrottle.scope == SCOPE_AUTH_IP))
        db.commit()
    finally:
        db.close()


def _counter_for(action: str, source: str = TEST_CLIENT_HOST) -> AuthThrottle | None:
    db = SessionLocal()
    try:
        return db.execute(
            select(AuthThrottle).where(
                AuthThrottle.scope == SCOPE_AUTH_IP,
                AuthThrottle.key == f"{action}:{source}".lower(),
            )
        ).scalar_one_or_none()
    finally:
        db.close()


@pytest.fixture
def clean_source_counters():
    """Start and finish each test with an empty per-source budget."""
    _clear_source_counters()
    yield
    _clear_source_counters()


# --- Registration ------------------------------------------------------------


def test_registration_is_rate_limited_per_source(client, monkeypatch, clean_source_counters):
    monkeypatch.setattr(settings, "register_max_requests_per_hour", 2)

    for index in range(2):
        response = register(client, f"limit{index}@example.com", "Password123", "patient")
        assert response.status_code == 201, response.text

    blocked = register(client, "limit3@example.com", "Password123", "patient")
    assert blocked.status_code == 429
    assert blocked.json()["retry_after_seconds"] > 0


def test_registration_limit_still_returns_a_generic_message(client, monkeypatch, clean_source_counters):
    """The throttle must not disclose whether an address already exists."""
    monkeypatch.setattr(settings, "register_max_requests_per_hour", 1)

    assert register(client, "first@example.com", "Password123", "patient").status_code == 201

    known = register(client, "first@example.com", "Password123", "patient")
    unknown = register(client, "second@example.com", "Password123", "patient")
    assert known.status_code == unknown.status_code == 429
    assert "already exists" not in unknown.json()["detail"].lower()


def test_source_counters_live_in_the_database(client, monkeypatch, clean_source_counters):
    """Proves the limiter is shared state, not a per-process counter.

    A process-local counter would be reset by a restart and would not be seen by a
    second instance behind a load balancer; a row in the shared database is.
    """
    monkeypatch.setattr(settings, "register_max_requests_per_hour", 1)
    assert register(client, "dbcounter@example.com", "Password123", "patient").status_code == 201

    row = _counter_for("register")
    assert row is not None
    assert row.attempt_count == 1
    assert row.window_started_at is not None


def test_source_cannot_be_spoofed_by_default(client, monkeypatch, clean_source_counters):
    """With TRUST_PROXY_HEADERS off, X-Forwarded-For must not reset the budget."""
    monkeypatch.setattr(settings, "register_max_requests_per_hour", 1)
    monkeypatch.setattr(settings, "trust_proxy_headers", False)

    assert register(client, "spoof1@example.com", "Password123", "patient").status_code == 201

    spoofed = client.post(
        "/auth/register",
        json={
            "full_name": "Spoof One",
            "email": "spoof2@example.com",
            "password": "Password123",
            "role": "patient",
        },
        headers={"X-Forwarded-For": "203.0.113.99"},
    )
    assert spoofed.status_code == 429, "an attacker-controlled header must not bypass the limit"


def test_forwarded_header_is_honoured_when_explicitly_trusted(
    client, monkeypatch, clean_source_counters
):
    """Behind a proxy you control, the forwarded address *is* the real source."""
    monkeypatch.setattr(settings, "register_max_requests_per_hour", 1)
    monkeypatch.setattr(settings, "trust_proxy_headers", True)

    first = client.post(
        "/auth/register",
        json={
            "full_name": "Proxied One",
            "email": "proxied1@example.com",
            "password": "Password123",
            "role": "patient",
        },
        headers={"X-Forwarded-For": "203.0.113.1"},
    )
    assert first.status_code == 201

    same_source = client.post(
        "/auth/register",
        json={
            "full_name": "Proxied Two",
            "email": "proxied2@example.com",
            "password": "Password123",
            "role": "patient",
        },
        headers={"X-Forwarded-For": "203.0.113.1"},
    )
    assert same_source.status_code == 429

    other_source = client.post(
        "/auth/register",
        json={
            "full_name": "Proxied Three",
            "email": "proxied3@example.com",
            "password": "Password123",
            "role": "patient",
        },
        headers={"X-Forwarded-For": "198.51.100.7"},
    )
    assert other_source.status_code == 201, "a different source has its own budget"


# --- Sign-in -----------------------------------------------------------------


def test_login_is_rate_limited_per_source(client, monkeypatch, clean_source_counters):
    monkeypatch.setattr(settings, "login_max_requests_per_ip_per_window", 2)

    email = "loginlimit@example.com"
    assert register(client, email, "Password123", "patient").status_code == 201

    payload = {"email": email, "password": "Password123", "role": "patient"}
    assert client.post("/auth/login", json=payload).status_code == 200
    assert client.post("/auth/login", json=payload).status_code == 200

    blocked = client.post("/auth/login", json=payload)
    assert blocked.status_code == 429
    assert blocked.json()["retry_after_seconds"] > 0


def test_login_source_limit_applies_to_unknown_accounts_too(
    client, monkeypatch, clean_source_counters
):
    """Spraying many unknown addresses from one host is throttled as well."""
    monkeypatch.setattr(settings, "login_max_requests_per_ip_per_window", 1)

    first = client.post(
        "/auth/login",
        json={"email": "nobody1@example.com", "password": "Password123", "role": "patient"},
    )
    assert first.status_code == 401

    second = client.post(
        "/auth/login",
        json={"email": "nobody2@example.com", "password": "Password123", "role": "patient"},
    )
    assert second.status_code == 429


# --- Password recovery flow --------------------------------------------------


def test_forgot_password_is_rate_limited_per_source(client, monkeypatch, clean_source_counters):
    """A per-address cooldown alone is bypassed by rotating the address."""
    monkeypatch.setattr(settings, "password_reset_max_requests_per_ip_per_hour", 1)

    assert register(client, "reset1@example.com", "Password123", "patient").status_code == 201
    assert register(client, "reset2@example.com", "Password123", "patient").status_code == 201

    first = client.post(
        "/auth/forgot-password", json={"email": "reset1@example.com", "role": "patient"}
    )
    assert first.status_code == 200

    # A *different* address from the same source: the per-address cooldown does
    # not apply, but the per-source budget does.
    second = client.post(
        "/auth/forgot-password", json={"email": "reset2@example.com", "role": "patient"}
    )
    assert second.status_code == 429


def test_verify_reset_code_is_rate_limited_per_source(client, monkeypatch, clean_source_counters):
    monkeypatch.setattr(settings, "password_reset_max_requests_per_ip_per_hour", 1)

    first = client.post(
        "/auth/verify-reset-code",
        json={"email": "someone@example.com", "verification_code": "000000", "role": "patient"},
    )
    assert first.status_code == 401

    second = client.post(
        "/auth/verify-reset-code",
        json={"email": "someone@example.com", "verification_code": "000001", "role": "patient"},
    )
    assert second.status_code == 429


def test_reset_password_is_rate_limited_per_source(client, monkeypatch, clean_source_counters):
    monkeypatch.setattr(settings, "password_reset_max_requests_per_ip_per_hour", 1)

    body = {
        "reset_token": "not-a-real-token-but-long-enough",
        "new_password": "BrandNewSecret456",
        "confirm_password": "BrandNewSecret456",
    }
    assert client.post("/auth/reset-password", json=body).status_code == 401
    assert client.post("/auth/reset-password", json=body).status_code == 429


# --- Limits do not break the happy path --------------------------------------


def test_authenticated_endpoints_are_not_affected_by_source_limits(client, new_patient):
    """Throttling applies to the anonymous surface only."""
    _, token = new_patient("unthrottled")
    for _ in range(5):
        assert client.get("/sessions", headers=auth_headers(token)).status_code == 200
