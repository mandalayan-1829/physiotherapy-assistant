"""JWT hardening: signatures, algorithms, claims and role trust.

A token is a bearer credential, so every one of these properties matters:

* an unsigned token (``alg: none``) must be rejected,
* a token signed with a different key must be rejected,
* an expired token must be rejected,
* a token missing required claims, or issued for a different audience, must be
  rejected,
* and - most importantly - the ``role`` claim must never be trusted, because
  authorization has to come from the database rather than from a value the client
  happens to be holding.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core.config import settings
from app.core.security import (
    PURPOSE_PASSWORD_RESET,
    create_access_token,
    decode_access_token,
)
from tests.conftest import auth_headers

ALGORITHM = settings.jwt_algorithm


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def unsigned_token(payload: dict) -> str:
    """Build an ``alg: none`` token by hand (no library will do this for us).

    Numeric date claims are used because the payload is serialised with plain
    ``json``, not through PyJWT's datetime conversion.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    claims = {
        "sub": "1",
        "role": "patient",
        "purpose": "access",
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + 300,
    }
    claims.update(payload)
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(claims).encode())
    return f"{header}.{body}."


def base_payload(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "1",
        "purpose": "access",
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    payload.update(overrides)
    return payload


# --- Signature / algorithm ---------------------------------------------------


def test_unsigned_token_is_rejected(client, new_patient):
    """``alg: none`` must never authenticate, even with valid-looking claims."""
    account, _ = new_patient("nonealg")
    # Every claim except the signature is valid: only ``alg: none`` is wrong.
    token = unsigned_token({"sub": str(account["id"]), "role": "patient"})
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401

    # ...and the same payload signed with the real key *is* accepted, which
    # proves the rejection above was caused by the missing signature.
    genuinely_signed = jwt.encode(
        base_payload(sub=str(account["id"]), role="patient"),
        settings.secret_key,
        algorithm=ALGORITHM,
    )
    assert client.get("/auth/me", headers=auth_headers(genuinely_signed)).status_code == 200


def test_token_signed_with_a_different_key_is_rejected(client, new_patient):
    account, _ = new_patient("otherkey")
    forged = jwt.encode(
        base_payload(sub=str(account["id"])),
        "an-attacker-controlled-signing-key-0123456789abcdef",
        algorithm=ALGORITHM,
    )
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_tampered_payload_fails_signature_check(client, new_patient):
    """Editing the payload of a real token invalidates the signature."""
    account, token = new_patient("tampered")

    header, payload, signature = token.split(".")
    decoded = json.loads(base64.urlsafe_b64decode(payload + "=="))
    decoded["role"] = "doctor"
    decoded["sub"] = str(account["id"])
    forged = ".".join([header, _b64url(json.dumps(decoded).encode()), signature])

    assert client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


@pytest.mark.parametrize(
    "token",
    ["", "not-a-jwt", "a.b", "a.b.c", "...", "Bearer", "eyJhbGciOiJIUzI1NiJ9"],
)
def test_malformed_tokens_are_rejected(client, token):
    assert (
        client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    )


# --- Expiry and required claims ---------------------------------------------


def test_expired_token_is_rejected(client, new_patient):
    """A correctly signed token whose ``exp`` has passed must not authenticate.

    The token is minted here with the real signing key, so the *only* thing wrong
    with it is the expiry - which is exactly the property under test.
    """
    account, _ = new_patient("expired")
    past = jwt.encode(
        base_payload(
            sub=str(account["id"]),
            role="patient",
            exp=datetime.now(timezone.utc) - timedelta(seconds=5),
        ),
        settings.secret_key,
        algorithm=ALGORITHM,
    )
    assert decode_access_token(past) is None
    assert client.get("/auth/me", headers=auth_headers(past)).status_code == 401


def test_token_without_expiry_is_rejected(client, new_patient):
    account, _ = new_patient("noexp")
    payload = base_payload(sub=str(account["id"]))
    payload.pop("exp")
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_without_issuer_is_rejected(client, new_patient):
    account, _ = new_patient("noiss")
    payload = base_payload(sub=str(account["id"]))
    payload.pop("iss")
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_from_a_foreign_issuer_is_rejected(client, new_patient):
    account, _ = new_patient("foreigniss")
    token = jwt.encode(
        base_payload(sub=str(account["id"]), iss="some-other-service"),
        settings.secret_key,
        algorithm=ALGORITHM,
    )
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_token_without_purpose_claim_is_rejected(client, new_patient):
    account, _ = new_patient("nopurpose")
    payload = base_payload(sub=str(account["id"]))
    payload.pop("purpose")
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_password_reset_token_cannot_authenticate(client, new_patient):
    account, _ = new_patient("resetpurpose")
    token = jwt.encode(
        base_payload(
            sub=str(account["id"]), purpose=PURPOSE_PASSWORD_RESET, jti="1"
        ),
        settings.secret_key,
        algorithm=ALGORITHM,
    )
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


# --- The role claim is not trusted -------------------------------------------


def test_role_claim_cannot_grant_doctor_access(client, new_patient):
    """Authorization reads the role from the database, never from the token.

    Even a correctly signed token that *claims* ``role: doctor`` must not unlock
    doctor-only operations, because the claim is attacker-controlled data the
    moment a signing key leaks or a token is re-issued by mistake.
    """
    account, patient_token = new_patient("roleescalation")
    assert account["role"] == "patient"

    escalated = create_access_token(subject=account["id"], role="doctor")
    payload = decode_access_token(escalated)
    assert payload is not None and payload["role"] == "doctor"

    headers = auth_headers(escalated)
    # Doctor-only endpoint -> still forbidden.
    assert client.get("/patients", headers=headers).status_code == 403
    # Identity is reported from the database, not the claim.
    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["account"]["role"] == "patient"

    # The genuine patient token behaves identically.
    assert client.get("/patients", headers=auth_headers(patient_token)).status_code == 403


def test_stale_token_version_is_rejected(client, new_patient):
    """A token minted before a password change must stop working."""
    account, token = new_patient("tokenversion")
    assert client.get("/auth/me", headers=auth_headers(token)).status_code == 200

    stale = create_access_token(subject=account["id"], role="patient", token_version=99)
    assert client.get("/auth/me", headers=auth_headers(stale)).status_code == 401


def test_token_for_a_non_existent_account_is_rejected(client):
    token = create_access_token(subject=999_999, role="patient")
    assert client.get("/auth/me", headers=auth_headers(token)).status_code == 401


# --- Dependency hygiene ------------------------------------------------------


def test_pyjwt_is_at_least_the_patched_version():
    """Regression guard for the audited advisories in PyJWT 2.10.1."""
    from packaging.version import Version

    assert Version(jwt.__version__) >= Version("2.12.0"), (
        "PyJWT must be upgraded past the advisories reported against 2.10.1"
    )
