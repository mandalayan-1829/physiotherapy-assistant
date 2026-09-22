"""one report per patient per month

``report_service.generate_report`` decided whether to insert or update with a
SELECT followed by an INSERT. Two concurrent requests for the same month could
both miss the SELECT and both insert, leaving duplicate rows that then
double-count wherever reports are aggregated.

This revision makes that impossible at the database level.

Two things this migration has to get right:

* **A populated database may already contain duplicates.** The de-duplication
  runs *before* the constraint is created, rather than the constraint failing on
  existing data. The surviving row is the most recently generated one - it holds
  the newest computation of the same month, so nothing is lost that regenerating
  the report would not reproduce.
* **SQLite cannot ALTER a constraint.** ``op.create_unique_constraint`` raises
  ``NotImplementedError`` on SQLite, which is the local development and test
  database. Batch mode performs the equivalent copy-and-move there and emits a
  plain ``ALTER TABLE`` on PostgreSQL, so one migration serves both.

Revision ID: d2f5a3c1e847
Revises: c8d1e4a72b90
Create Date: 2026-09-19
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = 'd2f5a3c1e847'
down_revision: Union[str, None] = 'c8d1e4a72b90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT_NAME = 'uq_reports_patient_month'
TABLE = 'reports'


def upgrade() -> None:
    # 1. Collapse pre-existing duplicates, keeping the newest row per
    #    (patient, month). A deleted row is a superseded computation of the same
    #    month, not an independent record.
    op.execute(
        """
        DELETE FROM reports
        WHERE id NOT IN (
            SELECT keep_id FROM (
                SELECT MAX(id) AS keep_id
                FROM reports
                GROUP BY patient_account_id, month_key
            ) AS newest_per_month
        )
        """
    )

    # 2. Now the constraint cannot fail on existing data. Batch mode is required
    #    for SQLite, which has no ALTER for constraints.
    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.create_unique_constraint(
            CONSTRAINT_NAME, ['patient_account_id', 'month_key']
        )


def downgrade() -> None:
    with op.batch_alter_table(TABLE) as batch_op:
        batch_op.drop_constraint(CONSTRAINT_NAME, type_='unique')
