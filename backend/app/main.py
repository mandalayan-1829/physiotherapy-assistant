"""PhysioAI FastAPI application.

Run locally with:

    cd backend
    uvicorn app.main:app --reload --port 8000

The application refuses to start unless it is configured safely (see
``app.core.config``); an insecure or missing ``SECRET_KEY`` raises
``ConfigurationError`` at import time, before any request is served.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    appointments,
    auth,
    communication,
    diet,
    doctors,
    notes,
    progress,
    reports,
    sessions,
    users,
)
from app.core.config import settings
from app.db.init_db import init_db
from app.services.errors import ServiceError


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Create any missing tables so a clean checkout can start immediately.
    init_db()
    yield


app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    lifespan=lifespan,
    # Interactive documentation is disabled in production: the OpenAPI document
    # is a complete map of the attack surface.
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Attach conservative security headers to every response.

    A strict Content-Security-Policy is intentionally *not* set here: this
    single-page app loads Google Fonts and YouTube iframes, so a CSP must be
    written and verified against the real bundle in a browser before it can be
    enabled. That is tracked as a follow-up rather than guessed at.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # The exercise tracker needs the camera; nothing else is required.
    response.headers.setdefault(
        "Permissions-Policy", "camera=(self), microphone=(), geolocation=()"
    )
    if settings.is_production:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    """Translate domain errors into consistent JSON responses.

    Any ``extra`` fields are merged in, which is how the sign-in endpoint
    reports the number of failed attempts without disclosing whether an email
    address is registered.
    """
    content: dict[str, object] = {"detail": exc.detail}
    if exc.extra:
        content.update(exc.extra)
    return JSONResponse(status_code=exc.status_code, content=content)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe used by the frontend and deployment platforms."""
    return {"status": "ok", "service": settings.project_name, "version": settings.api_version}


for router in (
    auth.router,
    users.router,
    doctors.router,
    sessions.router,
    progress.router,
    diet.router,
    appointments.router,
    notes.router,
    reports.router,
    communication.router,
):
    app.include_router(router)
