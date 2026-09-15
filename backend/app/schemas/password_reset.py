"""Password reset request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field

Role = Literal["patient", "doctor"]

# Generic message returned for every forgot-password request so that the
# existence of an account (or its role) can never be inferred.
GENERIC_FORGOT_PASSWORD_MESSAGE = (
    "If an account exists for this email, a verification code has been sent."
)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    # The portal the user selected. Used only to route the message; a mismatch
    # never changes the response.
    role: Role | None = None


class ForgotPasswordResponse(BaseModel):
    message: str = GENERIC_FORGOT_PASSWORD_MESSAGE


class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    verification_code: str = Field(min_length=4, max_length=12)
    role: Role | None = None


class VerifyResetCodeResponse(BaseModel):
    reset_token: str
    token_type: str = "bearer"
    expires_in: int


class ResetPasswordRequest(BaseModel):
    reset_token: str = Field(min_length=10)
    # Length and content rules are enforced by the backend password policy
    # (``core.security.validate_password_policy``), which is the single source
    # of truth. The client is never trusted for these checks.
    new_password: str = Field(min_length=1, max_length=128)
    confirm_password: str = Field(min_length=1, max_length=128)


class ResetPasswordResponse(BaseModel):
    message: str = "Your password has been updated. Please sign in with your new password."
