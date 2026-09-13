"""
Auth routes — login / signup.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas.user import UserCreate, UserLogin, AuthResponse, UserResponse
from backend.services.user_service import register, authenticate

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
def signup(data: UserCreate):
    """Register a new user account."""
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    ok, msg = register(data.name, data.age, data.email, data.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return AuthResponse(success=True, message=msg)


@router.post("/login", response_model=AuthResponse)
def login(data: UserLogin):
    """Authenticate and return user profile."""
    ok, result = authenticate(data.email, data.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return AuthResponse(
        success=True,
        message="Login successful",
        user=UserResponse(**result),
    )
