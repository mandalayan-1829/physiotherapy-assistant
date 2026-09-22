"""Monthly report generation, retrieval, uniqueness and authorization.

Reports are served by the API and stored in the database. They are not a browser
artefact: the same month must produce the same figures for every client, a
patient must only ever see their own reports, and a clinician must hold an active
relationship with the patient before a report is disclosed.

The test month is derived from the current UTC date rather than hardcoded, so the
suite does not rot as time passes.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone

from tests.conftest import auth_headers, verify_doctor

NOW = datetime.now(timezone.utc)
CURRENT_MONTH = NOW.strftime("%Y-%m")
CURRENT_MONTH_NAME = f"{calendar.month_name[NOW.month]} {NOW.year}"

# A month that has definitely already passed, used to check that reports for
# different months are independent.
OTHER_MONTH = "2020-01"


def _add_session(client, headers, **overrides):
    payload = {
        "exercise": "squat",
        "exercise_label": "Squat",
        "reps": 10,
        "target_reps": 10,
        "form_accuracy": 90,
        "duration_sec": 60,
        "metrics_source": "pose_inference",
    }
    payload.update(overrides)
    return client.post("/sessions", headers=headers, json=payload)


def _generate(client, headers, month_key: str = CURRENT_MONTH):
    return client.post("/reports/generate", headers=headers, json={"month_key": month_key})


def _book(client, patient_token, doctor_profile_id: int):
    return client.post(
        "/appointments",
        headers=auth_headers(patient_token),
        json={
            "doctor_profile_id": doctor_profile_id,
            "date": "2026-10-01",
            "time": "10:00 AM",
            "reason": "Report review",
        },
    )


def _doctor_profile_id(client, doctor_token) -> int:
    return client.get("/auth/me", headers=auth_headers(doctor_token)).json()["doctor_profile"]["id"]


# --- Generation --------------------------------------------------------------


def test_generate_creates_a_report_for_the_requested_month(client, new_patient):
    account, token = new_patient("genreport")
    headers = auth_headers(token)

    response = _generate(client, headers)
    assert response.status_code == 201

    body = response.json()
    assert body["month_key"] == CURRENT_MONTH
    assert body["month_name"] == CURRENT_MONTH_NAME
    assert body["patient_account_id"] == account["id"]
    assert body["generated_at"]
    assert isinstance(body["payload"], dict)


def test_generated_payload_reports_only_measured_sessions(client, new_patient):
    _, token = new_patient("payloadmix")
    headers = auth_headers(token)

    _add_session(client, headers, reps=10, form_accuracy=100, metrics_source="pose_inference")
    _add_session(client, headers, reps=10, form_accuracy=0, metrics_source="manual")

    payload = _generate(client, headers).json()["payload"]

    assert payload["total_sessions"] == 2
    assert payload["measured_sessions"] == 1
    assert payload["unmeasured_sessions"] == 1
    assert payload["total_reps"] == 20, "reps from both sessions still count"
    assert payload["avg_form_accuracy"] == 100.0, "only the measured session may be averaged"
    assert payload["no_measured_sessions"] is False
    assert payload["insufficient_data"] is False


def test_active_days_percent_is_computed_server_side(client, new_patient):
    """Present in the payload so two clients cannot disagree about one month."""
    _, token = new_patient("activedays")
    headers = auth_headers(token)

    _add_session(client, headers)

    payload = _generate(client, headers).json()["payload"]

    assert "active_days_percent" in payload
    assert 0 < payload["active_days_percent"] <= 100


def test_a_month_with_no_sessions_is_flagged_not_invented(client, new_patient):
    _, token = new_patient("emptymonth")

    payload = _generate(client, auth_headers(token)).json()["payload"]

    assert payload["total_sessions"] == 0
    assert payload["insufficient_data"] is True
    assert payload["no_measured_sessions"] is True
    assert payload["avg_form_accuracy"] == 0.0
    assert payload["exercise_breakdown"] == []
    assert payload["active_days_percent"] == 0


# --- Uniqueness / regeneration ----------------------------------------------


def test_generating_the_same_month_twice_returns_the_same_report(client, new_patient):
    """Regeneration refreshes the row; it does not accumulate duplicates."""
    _, token = new_patient("idempotent")
    headers = auth_headers(token)

    first = _generate(client, headers)
    assert first.status_code == 201

    # A new session lands, then the report is refreshed.
    _add_session(client, headers, reps=5)
    second = _generate(client, headers)
    assert second.status_code == 201

    assert second.json()["id"] == first.json()["id"], "same month must be the same row"

    listed = client.get("/reports", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1, "exactly one report may exist per month"


def test_regenerating_picks_up_new_sessions(client, new_patient):
    _, token = new_patient("refreshed")
    headers = auth_headers(token)

    assert _generate(client, headers).json()["payload"]["total_sessions"] == 0

    _add_session(client, headers, reps=7)
    refreshed = _generate(client, headers).json()["payload"]

    assert refreshed["total_sessions"] == 1
    assert refreshed["total_reps"] == 7


def test_different_months_are_independent_reports(client, new_patient):
    _, token = new_patient("twomonths")
    headers = auth_headers(token)

    current = _generate(client, headers, CURRENT_MONTH)
    other = _generate(client, headers, OTHER_MONTH)
    assert current.status_code == 201 and other.status_code == 201
    assert current.json()["id"] != other.json()["id"]

    listed = client.get("/reports", headers=headers).json()
    assert len(listed) == 2
    # Newest month first.
    assert listed[0]["month_key"] == CURRENT_MONTH


# --- Retrieval ---------------------------------------------------------------


def test_retrieve_a_report_by_id(client, new_patient):
    _, token = new_patient("retrieve")
    headers = auth_headers(token)

    created = _generate(client, headers).json()

    response = client.get(f"/reports/{created['id']}", headers=headers)
    assert response.status_code == 200

    fetched = response.json()
    assert fetched["id"] == created["id"]
    assert fetched["month_key"] == created["month_key"]
    assert fetched["month_name"] == created["month_name"]
    assert fetched["payload"] == created["payload"], "retrieval must round-trip the payload"


def test_retrieving_an_unknown_report_is_404(client, new_patient):
    _, token = new_patient("unknownreport")

    assert client.get("/reports/999999", headers=auth_headers(token)).status_code == 404


def test_report_endpoints_require_authentication(client):
    assert client.get("/reports").status_code == 401
    assert client.get("/reports/1").status_code == 401
    assert client.post("/reports/generate", json={"month_key": "2026-09"}).status_code == 401


# --- Validation --------------------------------------------------------------


def test_a_malformed_month_key_is_rejected(client, new_patient):
    _, token = new_patient("badmonth")
    headers = auth_headers(token)

    for month_key in ("2026-9", "September", "26-09", ""):
        assert _generate(client, headers, month_key).status_code == 422, month_key


def test_an_out_of_range_month_is_rejected(client, new_patient):
    """'2026-13' matches the \\d{4}-\\d{2} shape but is not a month."""
    _, token = new_patient("rangemonth")

    response = _generate(client, auth_headers(token), "2026-13")
    assert response.status_code in (400, 422)


# --- Authorization -----------------------------------------------------------


def test_a_patient_cannot_retrieve_another_patients_report(client, new_patient):
    _, alice_token = new_patient("alice")
    alice_report = _generate(client, auth_headers(alice_token)).json()

    _, bob_token = new_patient("bob")

    # Bob may not read Alice's report even though he knows its id.
    assert client.get(f"/reports/{alice_report['id']}", headers=auth_headers(bob_token)).status_code == 403

    # And his own list must not contain it.
    listed = client.get("/reports", headers=auth_headers(bob_token)).json()
    assert listed == []


def test_a_patient_cannot_scope_reports_to_another_patient(client, new_patient):
    alice, alice_token = new_patient("scopealice")
    _, bob_token = new_patient("scopebob")

    response = client.get(
        f"/reports?patient_account_id={alice['id']}", headers=auth_headers(bob_token)
    )
    assert response.status_code == 403


def test_a_patient_cannot_generate_a_report_for_another_patient(client, new_patient):
    alice, _ = new_patient("genalice")
    _, bob_token = new_patient("genbob")

    response = client.post(
        "/reports/generate",
        headers=auth_headers(bob_token),
        json={"month_key": CURRENT_MONTH, "patient_account_id": alice["id"]},
    )
    assert response.status_code == 403


def test_an_unlinked_doctor_cannot_retrieve_a_report(client, new_patient, new_doctor):
    _, patient_token = new_patient("unlinkedrep")
    report = _generate(client, auth_headers(patient_token)).json()

    _, doctor_token = new_doctor("unlinkedrepdoc")
    doctor_profile_id = _doctor_profile_id(client, doctor_token)
    verify_doctor(doctor_profile_id)

    assert (
        client.get(f"/reports/{report['id']}", headers=auth_headers(doctor_token)).status_code
        == 403
    )


def test_a_linked_doctor_can_retrieve_a_report(client, new_patient, new_doctor):
    account, patient_token = new_patient("linkedrep")
    patient_headers = auth_headers(patient_token)
    _add_session(client, patient_headers, reps=12)
    report = _generate(client, patient_headers).json()

    _, doctor_token = new_doctor("linkedrepdoc")
    doctor_headers = auth_headers(doctor_token)
    doctor_profile_id = _doctor_profile_id(client, doctor_token)
    verify_doctor(doctor_profile_id)

    # Booking the appointment is what establishes the relationship.
    assert _book(client, patient_token, doctor_profile_id).status_code == 201

    response = client.get(f"/reports/{report['id']}", headers=doctor_headers)
    assert response.status_code == 200
    assert response.json()["patient_account_id"] == account["id"]


def test_a_doctor_must_name_the_patient_to_list_reports(client, new_patient, new_doctor):
    _, doctor_token = new_doctor("listingdoc")
    doctor_profile_id = _doctor_profile_id(client, doctor_token)
    verify_doctor(doctor_profile_id)

    assert client.get("/reports", headers=auth_headers(doctor_token)).status_code == 403
