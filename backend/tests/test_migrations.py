"""Alembic migrations must work on a database that already contains data.

The revision that adds ``accounts.token_version`` originally created a NOT NULL
column with no server-side default, so it raised "column contains null values"
the moment it met a real install - i.e. it only ever worked on an empty database.

These tests build a throwaway SQLite database, populate it at the *older*
revision, and then upgrade, which is the sequence a live deployment actually
performs. They also check that the migration chain still matches the models, so
the two cannot drift apart silently.
"""

from __future__ import annotations

import logging.config
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext

import app.models  # noqa: F401  (registers every model on Base.metadata)
from app.core.config import settings
from app.db.base import Base

BACKEND_ROOT = Path(__file__).resolve().parents[1]

INITIAL_REVISION = "c4c7db25443c"  # accounts/profiles/sessions, no token_version
TOKEN_VERSION_REVISION = "b0411f7bc699"  # the revision that used to fail
HEAD_REVISION = "c8d1e4a72b90"  # clinician verification + metrics provenance


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    """An isolated database plus Alembic config pointing at it."""
    db_path = tmp_path / "migration-test.db"
    url = f"sqlite:///{db_path.as_posix()}"

    # alembic/env.py reads DATABASE_URL through the cached settings object.
    monkeypatch.setattr(settings, "database_url", url)
    # env.py calls logging.config.fileConfig, which would disable the loggers the
    # rest of the suite relies on. The migration itself does not need it.
    monkeypatch.setattr(logging.config, "fileConfig", lambda *args, **kwargs: None)

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config, url


def _seed_rows_at_initial_revision(url: str) -> None:
    """Insert one account, one clinician and one session, as an old install would."""
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO accounts "
                "(id, email, full_name, password_hash, role, is_active, legacy_password_hash,"
                " created_at, updated_at) "
                "VALUES (1, 'legacy@example.com', 'Legacy Patient', '$2b$12$abcdefghijklmnop',"
                " 'patient', 1, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO doctor_profiles "
                "(id, account_id, name, specialization, experience, qualification, available_days,"
                " timings, about, contact, whatsapp, email, hospital, rating, created_at) "
                "VALUES (1, NULL, 'Dr Legacy', 'Orthopedic Rehabilitation', 10, 'MPT', 'Mon-Fri',"
                " '9-5', '', '', '', 'legacy.doctor@example.com', '', 4.5, CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO workout_sessions "
                "(id, patient_account_id, exercise, exercise_label, reps, target_reps,"
                " form_accuracy, duration_sec, notes, performed_at, created_at) "
                "VALUES (1, 1, 'squat', 'Squat', 10, 10, 95, 60, '', CURRENT_TIMESTAMP,"
                " CURRENT_TIMESTAMP)"
            )
        )
    engine.dispose()


def _column_info(url: str, table: str) -> dict[str, dict]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()
    finally:
        engine.dispose()
    return {row[1]: {"notnull": row[3], "default": row[4]} for row in rows}


def _scalar(url: str, statement: str):
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(sa.text(statement)).scalar()
    finally:
        engine.dispose()


# --- The regression that mattered -------------------------------------------


def test_token_version_migration_upgrades_a_non_empty_database(migrated_db):
    """Upgrading to the token_version revision must not fail on existing rows."""
    config, url = migrated_db
    command.upgrade(config, INITIAL_REVISION)
    _seed_rows_at_initial_revision(url)
    assert _scalar(url, "SELECT COUNT(*) FROM accounts") == 1

    # This is the statement that previously raised on a populated table.
    command.upgrade(config, TOKEN_VERSION_REVISION)

    assert _scalar(url, "SELECT token_version FROM accounts WHERE id = 1") == 0
    info = _column_info(url, "accounts")["token_version"]
    assert info["notnull"] == 1, "token_version must remain NOT NULL"
    # A default is retained on SQLite so the column stays insertable without a
    # table rebuild; the model supplies the value from Python.
    assert info["default"] is not None


def test_full_upgrade_of_a_non_empty_database(migrated_db):
    """The whole chain applies, and the pre-existing rows are labelled honestly."""
    config, url = migrated_db
    command.upgrade(config, INITIAL_REVISION)
    _seed_rows_at_initial_revision(url)

    command.upgrade(config, "head")

    # Accounts keep working and gained a token version.
    assert _scalar(url, "SELECT token_version FROM accounts WHERE id = 1") == 0

    # Clinicians that already existed are grandfathered as verified, so no
    # existing patient/clinician relationship is revoked by the new gate.
    assert _scalar(url, "SELECT is_verified FROM doctor_profiles WHERE id = 1") in (1, True)
    assert _column_info(url, "doctor_profiles")["is_verified"]["notnull"] == 1

    # Historic sessions came from the simulated tracker, so they are labelled as
    # such rather than being presented as measurements.
    assert _scalar(url, "SELECT metrics_source FROM workout_sessions WHERE id = 1") == "simulated"
    assert _column_info(url, "workout_sessions")["metrics_source"]["notnull"] == 1


def test_upgrade_on_a_fresh_database_then_downgrade(migrated_db):
    """A clean install works, and the downgrade path is functional."""
    config, _ = migrated_db
    command.upgrade(config, "head")
    # Downgrading returns to an empty schema (the project supports downgrades for
    # development databases).
    command.downgrade(config, INITIAL_REVISION)


# --- Drift between migrations and models ------------------------------------


def test_migrations_match_the_models(migrated_db):
    """``upgrade head`` must produce exactly the schema the models describe.

    Without this, a column added to a model but not to a migration would work
    locally (where tables come from ``create_all``) and fail in production (where
    they come from migrations).
    """
    config, url = migrated_db
    command.upgrade(config, "head")

    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            differences = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert differences == [], f"migrations and models disagree: {differences}"
