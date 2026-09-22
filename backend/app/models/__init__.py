"""SQLAlchemy models.

Importing this package registers every model on ``Base.metadata`` and lets
string-based relationships resolve.
"""

from app.models.account import ROLE_DOCTOR, ROLE_PATIENT, Account
from app.models.alert import ALERT_TYPES, GuardianAlert
from app.models.appointment import APPOINTMENT_STATUSES, Appointment
from app.models.diet import DietRecord
from app.models.doctor import DoctorProfile
from app.models.link import PatientDoctorLink
from app.models.message import Message
from app.models.note import NOTE_CATEGORIES, DoctorNote
from app.models.patient import PatientProfile
from app.models.report import Report
from app.models.reset import PasswordResetToken
from app.models.session import WorkoutSession
from app.models.throttle import (
    SCOPE_AUTH_IP,
    SCOPE_LOGIN,
    SCOPE_PASSWORD_RESET_REQUEST,
    AuthThrottle,
)

__all__ = [
    "Account",
    "ROLE_PATIENT",
    "ROLE_DOCTOR",
    "PatientProfile",
    "DoctorProfile",
    "PatientDoctorLink",
    "WorkoutSession",
    "DietRecord",
    "Appointment",
    "APPOINTMENT_STATUSES",
    "DoctorNote",
    "NOTE_CATEGORIES",
    "Report",
    "Message",
    "GuardianAlert",
    "ALERT_TYPES",
    "PasswordResetToken",
    "AuthThrottle",
    "SCOPE_LOGIN",
    "SCOPE_PASSWORD_RESET_REQUEST",
    "SCOPE_AUTH_IP",
]
