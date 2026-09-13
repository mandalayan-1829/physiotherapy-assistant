"""
User routes — profile management.
"""

from fastapi import APIRouter, Depends

from backend.schemas.user import UserResponse, UserUpdate
from backend.api.dependencies import get_current_user
from backend.services.user_service import update_profile, get_profile

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
def get_me(user: dict = Depends(get_current_user)):
    """Get the current user's profile."""
    return UserResponse(**user)


@router.put("/me", response_model=UserResponse)
def update_me(data: UserUpdate, user: dict = Depends(get_current_user)):
    """Update the current user's profile."""
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    if update_data:
        update_profile(user["id"], update_data)
    updated = get_profile(user["id"])
    return UserResponse(**updated)
