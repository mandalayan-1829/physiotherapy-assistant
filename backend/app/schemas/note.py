"""Clinical note schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NoteCreate(BaseModel):
    """A patient may write a note about themselves; a clinician must supply the patient id."""

    patient_account_id: int | None = None
    note_text: str = Field(min_length=1)
    category: Literal["clinical", "exercise", "symptom", "general"] = "clinical"


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_account_id: int
    doctor_profile_id: int | None
    note_text: str
    category: str
    created_at: datetime
