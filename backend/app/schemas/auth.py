"""Authentication request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Role = Literal["patient", "doctor"]


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    # Content rules come from the backend password policy, not the schema.
    password: str = Field(min_length=1, max_length=128)
    role: Role
    # Optional clinical details supplied when a doctor registers.
    specialization: str | None = Field(default=None, max_length=255)
    qualification: str | None = Field(default=None, max_length=255)
    license_number: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    role: Role


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    account: "AccountOut"


class AccountOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: Role
    is_active: bool


TokenResponse.model_rebuild()
