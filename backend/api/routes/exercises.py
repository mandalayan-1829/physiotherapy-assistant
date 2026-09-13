"""
Exercise routes — library of supported exercises.
"""

from fastapi import APIRouter, HTTPException, Depends

from backend.schemas.exercise import ExerciseInfo, ExerciseListResponse
from backend.api.dependencies import get_current_user
from backend.services.exercise_service import (
    get_all_exercises,
    get_exercise,
    is_exercise_safe,
)

router = APIRouter(prefix="/api/exercises", tags=["exercises"])


@router.get("", response_model=ExerciseListResponse)
def list_exercises():
    """Get all available exercises."""
    exercises = get_all_exercises()
    return ExerciseListResponse(exercises=exercises, total=len(exercises))


@router.get("/{exercise_id}", response_model=ExerciseInfo)
def get_exercise_detail(exercise_id: str):
    """Get details for a specific exercise."""
    exercise = get_exercise(exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return ExerciseInfo(**exercise)


@router.get("/{exercise_id}/safety")
def check_exercise_safety(exercise_id: str, user: dict = Depends(get_current_user)):
    """Check if an exercise is safe for the current user."""
    safe, msg = is_exercise_safe(exercise_id, user)
    return {"safe": safe, "warning": msg}
