"""PhysioAI FastAPI application.

Run locally with:

    cd backend
    uvicorn app.main:app --reload --port 8000
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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
