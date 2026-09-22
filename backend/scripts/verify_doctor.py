"""Verify (or revoke) a clinician account out of band.

This application has exactly two roles and deliberately no administrator role, so
clinician verification is an operations action rather than an API call. Until a
profile is verified it is absent from the patient-facing directory and cannot be
booked or messaged, which is what stops public signup from conferring clinical
authority (OWASP API6).

Usage (from the ``backend/`` directory):

    python scripts/verify_doctor.py --list
    python scripts/verify_doctor.py --email dr.smith@example.com
    python scripts/verify_doctor.py --email dr.smith@example.com --revoke

Revoking a profile does not delete anything and does not touch existing
records; it only removes the clinician's ability to be newly booked or listed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.db.init_db import init_db  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models import Account, DoctorProfile  # noqa: E402


def _rows(db) -> list[tuple[DoctorProfile, str]]:
    profiles = list(db.execute(select(DoctorProfile).order_by(DoctorProfile.id)).scalars().all())
    emails = {
        account.id: account.email
        for account in db.execute(select(Account)).scalars().all()
    }
    return [
        (profile, emails.get(profile.account_id) if profile.account_id else "(no login account)")
        for profile in profiles
    ]


def list_profiles() -> None:
    db = SessionLocal()
    try:
        rows = _rows(db)
        if not rows:
            print("No clinician profiles found.")
            return
        print(f"{'id':<6}{'verified':<10}{'email':<40}name")
        for profile, email in rows:
            print(f"{profile.id:<6}{str(profile.is_verified):<10}{str(email):<40}{profile.name}")
    finally:
        db.close()


def set_verified(email: str, verified: bool) -> int:
    db = SessionLocal()
    try:
        account = db.execute(
            select(Account).where(Account.email == email.strip().lower())
        ).scalar_one_or_none()
        if account is None:
            print(f"No account found for {email!r}.")
            return 1

        profile = db.execute(
            select(DoctorProfile).where(DoctorProfile.account_id == account.id)
        ).scalar_one_or_none()
        if profile is None:
            print(f"Account {email!r} is not a clinician (no doctor profile).")
            return 1

        profile.is_verified = verified
        profile.verified_at = datetime.now(timezone.utc) if verified else None
        db.commit()

        action = "verified" if verified else "revoked"
        print(f"{action}: id={profile.id} {profile.name} <{email}>")
        if verified:
            print("The clinician is now listed in the directory and can be booked.")
        else:
            print("The clinician is hidden from the directory and cannot be booked.")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", help="e-mail address of the clinician account")
    parser.add_argument("--revoke", action="store_true", help="un-verify instead of verify")
    parser.add_argument("--list", action="store_true", help="list every clinician profile")
    args = parser.parse_args(argv)

    if args.list:
        list_profiles()
        return 0

    if not args.email:
        parser.error("provide --email <address> or --list")

    # Ensure the schema exists before writing, mirroring application startup.
    init_db()
    return set_verified(args.email, verified=not args.revoke)


if __name__ == "__main__":
    raise SystemExit(main())
