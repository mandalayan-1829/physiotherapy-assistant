"""clinician verification and session metrics provenance

Two security/correctness changes:

* ``doctor_profiles.is_verified`` - stops self-registration from conferring
  clinical authority (OWASP API6). Freshly registered clinicians are unverified
  and therefore hidden from the directory and unbookable. Rows that existed
  before this revision are grandfathered in as verified, because they were
  created under the previous policy and removing them would break every existing
  patient/clinician relationship.
* ``workout_sessions.metrics_source`` - records how the numbers in a row were
  produced. Every pre-existing row came from the retired simulated tracker, so it
  is labelled ``simulated`` rather than being presented as a measurement.

Both columns are added with a server-side default so the statements succeed on a
non-empty database, then the rows are backfilled explicitly.

Revision ID: c8d1e4a72b90
Revises: b0411f7bc699
Create Date: 2026-09-17
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c8d1e4a72b90'
down_revision: Union[str, None] = 'b0411f7bc699'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Clinician verification ------------------------------------------------
    op.add_column(
        'doctor_profiles',
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        'doctor_profiles',
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f('ix_doctor_profiles_is_verified'), 'doctor_profiles', ['is_verified'], unique=False
    )
    # Grandfather clinicians that already exist: they were created before the
    # verification gate, and de-verifying them would revoke access from patients
    # who are already under their care.
    op.execute('UPDATE doctor_profiles SET is_verified = true, verified_at = created_at')

    # --- Session metrics provenance -------------------------------------------
    # Existing rows were produced by the simulated tracker, so they are marked as
    # such and excluded from clinical aggregates.
    op.add_column(
        'workout_sessions',
        sa.Column(
            'metrics_source',
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'simulated'"),
        ),
    )
    op.create_index(
        op.f('ix_workout_sessions_metrics_source'),
        'workout_sessions',
        ['metrics_source'],
        unique=False,
    )
    # New rows must state their provenance explicitly; the model defaults to the
    # least-trusted value ('manual'), so the database no longer implies that a
    # row without a declared source was measured.
    if op.get_bind().dialect.name == 'postgresql':
        op.alter_column('workout_sessions', 'metrics_source', server_default=None)
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_index(op.f('ix_workout_sessions_metrics_source'), table_name='workout_sessions')
    op.drop_column('workout_sessions', 'metrics_source')
    op.drop_index(op.f('ix_doctor_profiles_is_verified'), table_name='doctor_profiles')
    op.drop_column('doctor_profiles', 'verified_at')
    op.drop_column('doctor_profiles', 'is_verified')
