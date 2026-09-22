"""Authentication routes: register, login, logout, current identity."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import ClientIp, CurrentAccount, DbSession
from app.schemas.auth import AccountOut, LoginRequest, RegisterRequest, TokenResponse
from app.schemas.password_reset import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    VerifyResetCodeRequest,
    VerifyResetCodeResponse,
)
from app.schemas.patient import MeResponse
from app.services import auth_service, password_reset_service, patient_service
from app.services.access import get_doctor_profile

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: DbSession, client_ip: ClientIp) -> TokenResponse:
    """Create an account and return an access token.

    Registration is patient-facing in effect: a ``doctor`` signup creates a
    *pending* clinician profile that confers no clinical authority until it is
    verified out of band. The per-source rate limit is enforced server-side.
    """
    account = auth_service.register_account(db, payload, client_ip)
    token, expires_in = auth_service.issue_token(account)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        account=AccountOut.model_validate(account),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: DbSession, client_ip: ClientIp) -> TokenResponse:
    """Validate credentials for the selected portal and return a token."""
    account = auth_service.authenticate(
        db, payload.email, payload.password, payload.role, client_ip
    )
    token, expires_in = auth_service.issue_token(account)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        account=AccountOut.model_validate(account),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(_: CurrentAccount) -> Response:
    """Tokens are stateless; the client discards its token.

    Exposed so the frontend performs a real, authenticated round-trip instead
    of clearing local state silently.
    """
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=MeResponse)
def me(account: CurrentAccount, db: DbSession) -> MeResponse:
    """Return the authenticated identity plus role-specific profile."""
    profile = None
    doctor_profile = None

    if account.role == "patient":
        profile = patient_service.profile_for(db, account.id)
    else:
        doctor_profile = get_doctor_profile(db, account)

    return MeResponse(
        account=AccountOut.model_validate(account),
        profile=profile,
        doctor_profile=doctor_profile,
    )


# --- Password reset ---------------------------------------------------------


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(
    payload: ForgotPasswordRequest, db: DbSession, client_ip: ClientIp
) -> ForgotPasswordResponse:
    """Start a password reset.

    The response is identical whether or not the address belongs to an
    account, in either portal. The verification code is emailed and is never
    returned in the response or written to logs.
    """
    message = password_reset_service.request_password_reset(
        db, payload.email, payload.role, client_ip
    )
    return ForgotPasswordResponse(message=message)


@router.post("/verify-reset-code", response_model=VerifyResetCodeResponse)
def verify_reset_code(
    payload: VerifyResetCodeRequest, db: DbSession, client_ip: ClientIp
) -> VerifyResetCodeResponse:
    """Exchange a valid verification code for a short-lived reset token."""
    reset_token, expires_in = password_reset_service.verify_reset_code(
        db, payload.email, payload.verification_code, payload.role, client_ip
    )
    return VerifyResetCodeResponse(reset_token=reset_token, expires_in=expires_in)


@router.post("/reset-password", response_model=ResetPasswordResponse)
def reset_password(
    payload: ResetPasswordRequest, db: DbSession, client_ip: ClientIp
) -> ResetPasswordResponse:
    """Set a new password using the reset token and end all existing sessions."""
    password_reset_service.reset_password(
        db, payload.reset_token, payload.new_password, payload.confirm_password, client_ip
    )
    return ResetPasswordResponse()
