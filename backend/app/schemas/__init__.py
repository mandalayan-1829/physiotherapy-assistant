"""Pydantic request/response schemas."""

from app.schemas.alert import AlertCreate, AlertOut, MessageCreate, MessageOut
from app.schemas.appointment import (
    AppointmentCreate,
    AppointmentOut,
    AppointmentStatusLiteral,
    AppointmentUpdate,
)
from app.schemas.auth import AccountOut, LoginRequest, RegisterRequest, Role, TokenResponse
from app.schemas.diet import DietRecordCreate, DietRecordOut
from app.schemas.doctor import DoctorOut, PatientSummary
from app.schemas.note import NoteCreate, NoteOut
from app.schemas.password_reset import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    VerifyResetCodeRequest,
    VerifyResetCodeResponse,
)
from app.schemas.patient import MeResponse, PatientProfileOut, PatientProfileUpdate
from app.schemas.report import ReportGenerateRequest, ReportOut
from app.schemas.session import DailyProgressOut, WorkoutSessionCreate, WorkoutSessionOut

__all__ = [
    "AccountOut",
    "LoginRequest",
    "RegisterRequest",
    "Role",
    "TokenResponse",
    "PatientProfileOut",
    "PatientProfileUpdate",
    "MeResponse",
    "ForgotPasswordRequest",
    "ForgotPasswordResponse",
    "VerifyResetCodeRequest",
    "VerifyResetCodeResponse",
    "ResetPasswordRequest",
    "ResetPasswordResponse",
    "DoctorOut",
    "PatientSummary",
    "WorkoutSessionCreate",
    "WorkoutSessionOut",
    "DailyProgressOut",
    "DietRecordCreate",
    "DietRecordOut",
    "AppointmentCreate",
    "AppointmentUpdate",
    "AppointmentOut",
    "AppointmentStatusLiteral",
    "NoteCreate",
    "NoteOut",
    "ReportGenerateRequest",
    "ReportOut",
    "AlertCreate",
    "AlertOut",
    "MessageCreate",
    "MessageOut",
]
