"""Server-side rate limiting for authentication actions.

Counters live in the database (``auth_throttles``) rather than in browser
storage, so they survive restarts and apply across workers.

Because the counters are rows in the shared database, the limits hold across
multiple backend instances without Redis. That is the property that makes this
implementation safe to run behind a load balancer; a process-local in-memory
counter would not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuthThrottle
from app.services.errors import TooManyRequestsError


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime) -> datetime:
    """SQLite returns naive datetimes; normalise to UTC-aware."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass
class ThrottleState:
    attempt_count: int
    window_started_at: datetime
    last_attempt_at: datetime

    def seconds_until_window_end(self, window_seconds: int) -> int:
        elapsed = (_now() - _as_aware(self.window_started_at)).total_seconds()
        return max(0, int(window_seconds - elapsed))

    def seconds_since_last_attempt(self) -> int:
        return int((_now() - _as_aware(self.last_attempt_at)).total_seconds())


def _get_row(db: Session, scope: str, key: str) -> AuthThrottle | None:
    return db.execute(
        select(AuthThrottle).where(AuthThrottle.scope == scope, AuthThrottle.key == key.lower())
    ).scalar_one_or_none()


def peek(db: Session, scope: str, key: str, window_seconds: int) -> ThrottleState:
    """Read the current state without modifying it."""
    row = _get_row(db, scope, key)
    if row is None or (_now() - _as_aware(row.window_started_at)) > timedelta(seconds=window_seconds):
        return ThrottleState(0, _now(), _now())
    return ThrottleState(row.attempt_count, row.window_started_at, row.last_attempt_at)


def register_attempt(
    db: Session, scope: str, key: str, window_seconds: int
) -> ThrottleState:
    """Record one attempt inside the rolling window, resetting it if elapsed."""
    now = _now()
    row = _get_row(db, scope, key)

    if row is None:
        row = AuthThrottle(
            scope=scope,
            key=key.lower(),
            attempt_count=1,
            window_started_at=now,
            last_attempt_at=now,
        )
        db.add(row)
    else:
        if (now - _as_aware(row.window_started_at)) > timedelta(seconds=window_seconds):
            row.attempt_count = 1
            row.window_started_at = now
        else:
            row.attempt_count += 1
        row.last_attempt_at = now

    db.commit()
    db.refresh(row)
    return ThrottleState(row.attempt_count, row.window_started_at, row.last_attempt_at)


def reset(db: Session, scope: str, key: str) -> None:
    """Clear a counter (used after a successful sign-in or reset)."""
    row = _get_row(db, scope, key)
    if row is not None:
        db.delete(row)
        db.commit()


def is_cooldown_active(
    db: Session, scope: str, key: str, cooldown_seconds: int
) -> int:
    """Return remaining cooldown seconds, or 0 when a new attempt is allowed."""
    row = _get_row(db, scope, key)
    if row is None:
        return 0
    elapsed = (_now() - _as_aware(row.last_attempt_at)).total_seconds()
    remaining = int(cooldown_seconds - elapsed)
    return max(0, remaining)


def enforce_limit(
    db: Session,
    scope: str,
    key: str,
    max_attempts: int,
    window_seconds: int,
    message: str,
) -> ThrottleState:
    """Record one attempt, raising ``TooManyRequestsError`` once the cap is hit.

    The counter is incremented *before* the limit is evaluated on the next call,
    so a burst that races past the check still ends up throttled. The caller is
    responsible for supplying a stable, non-spoofable key.
    """
    state = peek(db, scope, key, window_seconds)
    if state.attempt_count >= max_attempts:
        raise TooManyRequestsError(
            message,
            extra={
                "retry_after_seconds": state.seconds_until_window_end(window_seconds),
            },
        )
    return register_attempt(db, scope, key, window_seconds)
