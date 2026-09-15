"""Password reset request.

A row is created for every "forgot password" request. The verification code is
stored only as a salted bcrypt hash — never in plaintext — and the row carries
its own expiry, attempt counter and invalidation state.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which portal the request came from ("patient" or "doctor").
    portal_role: Mapped[str] = mapped_column(String(20), nullable=False)

    # bcrypt hash of the 6-digit verification code.
    code_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    invalidated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Set once the code has been verified, so the row can only be used for a
    # single password change.
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def is_usable(self, at: datetime) -> bool:
        """True when the row can still be used to verify a code."""
        return (
            not self.used
            and not self.invalidated
            and self.verified_at is None
            and self.expires_at > at
        )
