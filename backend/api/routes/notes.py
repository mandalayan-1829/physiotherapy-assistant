"""
Notes routes — user notes management.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from fastapi import APIRouter, Depends

from backend.schemas.doctor import NoteCreate, NoteResponse
from backend.api.dependencies import get_current_user
from core.database import add_note, get_notes, delete_note

router = APIRouter(prefix="/api/notes", tags=["notes"])


@router.post("", response_model=NoteResponse)
def create_note(data: NoteCreate, user: dict = Depends(get_current_user)):
    """Add a new note."""
    add_note(user["id"], data.note_text.strip())
    notes = get_notes(user["id"])
    if notes:
        return NoteResponse(**notes[0])
    return NoteResponse(id=0, user_id=user["id"], note_text=data.note_text)


@router.get("", response_model=list[NoteResponse])
def list_notes(user: dict = Depends(get_current_user)):
    """Get all notes for the current user."""
    notes = get_notes(user["id"])
    return [NoteResponse(**n) for n in notes]


@router.delete("/{note_id}")
def remove_note(note_id: int, user: dict = Depends(get_current_user)):
    """Delete a note."""
    delete_note(note_id)
    return {"success": True}
