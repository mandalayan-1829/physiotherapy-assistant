"""Shared FastAPI dependencies: authentication, role guards and client identity."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import PURPOSE_ACCESS, decode_access_token
from app.db.session import get_db
from app.models import Account, ROLE_DOCTOR, ROLE_PATIENT

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


def get_current_account(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> Account:
    """Resolve the authenticated account from the Bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    # Only session tokens may authenticate API calls; a password-reset token is
    # never accepted here.
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("purpose", PURPOSE_ACCESS) != PURPOSE_ACCESS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    account = db.get(Account, int(payload["sub"]))
    if account is None or not account.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account no longer exists or is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # A password change bumps this counter, which invalidates every token that
    # was issued before it.
    if int(payload.get("tv", 0)) != int(account.token_version or 0):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return account


CurrentAccount = Annotated[Account, Depends(get_current_account)]


def require_patient(account: CurrentAccount) -> Account:
    if account.role != ROLE_PATIENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action is only available to patient accounts.",
        )
    return account


def require_doctor(account: CurrentAccount) -> Account:
    if account.role != ROLE_DOCTOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action is only available to doctor accounts.",
        )
    return account


CurrentPatient = Annotated[Account, Depends(require_patient)]
CurrentDoctor = Annotated[Account, Depends(require_doctor)]
DbSession = Annotated[Session, Depends(get_db)]


def get_client_ip(request: Request) -> str:
    """Best-effort source address of the caller, used for abuse throttling.

    ``X-Forwarded-For`` is only consulted when ``TRUST_PROXY_HEADERS`` is
    enabled, because otherwise a client could spoof the header and reset its own
    rate-limit bucket on every request.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    client = request.client
    return client.host if client and client.host else "unknown"


ClientIp = Annotated[str, Depends(get_client_ip)]
