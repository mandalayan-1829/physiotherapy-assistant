"""Clinical note endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Response, status

from app.api.deps import CurrentAccount, DbSession
from app.schemas.note import NoteCreate, NoteOut
from app.services import note_service
from app.services.access import resolve_patient_scope

router = APIRouter(prefix="/notes", tags=["notes"])


@router.get("", response_model=list[NoteOut])
def list_notes(
    account: CurrentAccount,
    db: DbSession,
    patient_account_id: int | None = Query(default=None),
) -> list[NoteOut]:
    scope = resolve_patient_scope(db, account, patient_account_id)
    return note_service.list_notes(db, account, scope)


@router.post("", response_model=NoteOut, status_code=status.HTTP_201_CREATED)
def create_note(payload: NoteCreate, account: CurrentAccount, db: DbSession) -> NoteOut:
    """Patients journal about themselves; clinicians author notes for linked patients."""
    return note_service.create_note(db, account, payload)


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: int, account: CurrentAccount, db: DbSession) -> Response:
    note_service.delete_note(db, account, note_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
