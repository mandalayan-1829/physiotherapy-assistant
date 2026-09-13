"""
FastAPI dependencies — authentication, DB connections, etc.
"""

from fastapi import Header, HTTPException
from typing import Optional

from backend.services.user_service import get_profile


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Extract user from Authorization header.

    For this initial build we use a simple token = user_id.
    A production system would use JWT/OAuth.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        user_id = int(authorization.replace("Bearer ", "").strip())
    except (ValueError, AttributeError):
        raise HTTPException(status_code=401, detail="Invalid token format")

    user = get_profile(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
