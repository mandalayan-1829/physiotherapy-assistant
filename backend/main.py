"""
AI Physio — FastAPI Backend
Run with: uvicorn backend.main:app --reload
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.api.routes import auth, users, exercises, sessions, progress, diet, notes, doctors, websocket

# Initialize database
from core.database import init_db  # noqa: E402

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Physiotherapy Assistant — Backend API",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(exercises.router)
app.include_router(sessions.router)
app.include_router(progress.router)
app.include_router(diet.router)
app.include_router(notes.router)
app.include_router(doctors.router)
app.include_router(websocket.router)


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
