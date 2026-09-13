"""
Diet routes — meal tracking.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends

from backend.schemas.diet import DietEntryCreate, DietEntryResponse, DietSummary
from backend.api.dependencies import get_current_user
from core.database import (
    add_diet_entry,
    get_diet_today,
    get_diet_all,
    delete_diet_entry,
)

router = APIRouter(prefix="/api/diet", tags=["diet"])


@router.post("", response_model=DietEntryResponse)
def add_meal(data: DietEntryCreate, user: dict = Depends(get_current_user)):
    """Add a diet entry for the current user."""
    add_diet_entry(user["id"], data.meal, data.calories, data.protein, data.carbs, data.fats)
    # Return the latest entry
    entries = get_diet_today(user["id"])
    if entries:
        return DietEntryResponse(**entries[-1])
    raise HTTPException(status_code=500, detail="Failed to save entry")


@router.get("/today", response_model=DietSummary)
def today_meals(user: dict = Depends(get_current_user)):
    """Get today's diet entries and summary."""
    entries = get_diet_today(user["id"])
    return DietSummary(
        total_calories=sum(e["calories"] for e in entries),
        total_protein=sum(e["protein"] for e in entries),
        total_carbs=sum(e["carbs"] for e in entries),
        total_fats=sum(e["fats"] for e in entries),
        entries=[DietEntryResponse(**e) for e in entries],
    )


@router.get("", response_model=list[DietEntryResponse])
def all_meals(user: dict = Depends(get_current_user)):
    """Get all diet entries for the current user."""
    entries = get_diet_all(user["id"])
    return [DietEntryResponse(**e) for e in entries]


@router.delete("/{entry_id}")
def remove_meal(entry_id: int, user: dict = Depends(get_current_user)):
    """Delete a diet entry."""
    delete_diet_entry(entry_id)
    return {"success": True}
