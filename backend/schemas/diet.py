"""
Pydantic schemas for diet data.
"""

from pydantic import BaseModel
from typing import Optional


class DietEntryCreate(BaseModel):
    meal: str
    calories: int = 0
    protein: float = 0
    carbs: float = 0
    fats: float = 0


class DietEntryResponse(BaseModel):
    id: int
    user_id: int
    meal: str
    calories: int
    protein: float
    carbs: float
    fats: float
    date: Optional[str] = None


class DietSummary(BaseModel):
    total_calories: int
    total_protein: float
    total_carbs: float
    total_fats: float
    entries: list[DietEntryResponse]
