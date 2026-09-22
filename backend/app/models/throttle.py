"""Generic authentication throttle counters.

Used to rate-limit sign-in attempts and password-reset requests server-side so
abuse protection never depends on a browser counter.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

SCOPE_LOGIN = "login"
SCOPE_PASSWORD_RESET_REQUEST = "password_reset_request"
#: Per-source (IP) counters for the unauthenticated auth surface. Keys are
#: namespaced per action, e.g. ``register:203.0.113.7``, so one scope holds every
#: action while each keeps its own limit.
SCOPE_AUTH_IP = "auth_ip"


class AuthThrottle(Base):
    __tablename__ = "auth_throttles"
    __table_args__ = (UniqueConstraint("scope", "key", name="scope_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    # e.g. "login" or "password_reset_request"
    scope: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    # Lower-cased e-mail address, or an action-prefixed source address, for the
    # action being throttled (e.g. ``register:203.0.113.7``).
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    window_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
