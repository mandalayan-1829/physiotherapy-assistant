"""Domain-level errors.

Services raise these instead of importing FastAPI, keeping business logic
framework-independent. ``app.main`` maps them to JSON responses.

An error may carry ``extra`` fields which are merged into the JSON body. This
is how the sign-in endpoint reports how many attempts have failed without
leaking whether an account exists.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    status_code = 400

    def __init__(self, detail: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra: dict[str, Any] = extra or {}


class ValidationError(ServiceError):
    status_code = 422


class AuthError(ServiceError):
    """Invalid credentials or token."""

    status_code = 401


class ForbiddenError(ServiceError):
    """Authenticated, but not allowed to access this resource."""

    status_code = 403


class NotFoundError(ServiceError):
    status_code = 404


class ConflictError(ServiceError):
    """The request conflicts with existing state (e.g. duplicate email)."""

    status_code = 409


class TooManyRequestsError(ServiceError):
    """Rate limit exceeded."""

    status_code = 429
