"""
User service — wraps database user CRUD operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.database import (
    create_user,
    login_user,
    get_user,
    update_user_profile,
)


def register(name: str, age: int, email: str, password: str) -> tuple[bool, str]:
    """Register a new user."""
    return create_user(name, age, email, password)


def authenticate(email: str, password: str) -> tuple[bool, str]:
    """Authenticate a user by email and password."""
    return login_user(email, password)


def get_profile(user_id: int) -> dict | None:
    """Get user profile by ID."""
    return get_user(user_id)


def update_profile(user_id: int, data: dict) -> None:
    """Update user profile fields."""
    update_user_profile(user_id, data)
