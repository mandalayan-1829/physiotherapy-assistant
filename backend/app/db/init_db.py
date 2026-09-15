"""Database bootstrap.

``create_all`` is safe to call repeatedly and is used for local development and
tests. Schema evolution in deployed environments is handled by Alembic.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

import app.models  # noqa: F401  (ensures every model is registered)
from app.db.base import Base
from app.db.session import engine


def init_db() -> None:
    """Create any missing tables."""
    Base.metadata.create_all(bind=engine)


def init_db_with_session(db: Session) -> None:  # pragma: no cover - convenience helper
    init_db()
