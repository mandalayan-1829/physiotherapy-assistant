"""Session metrics must carry their provenance, and fabricated data must not be
presented as a clinical measurement.

The retired demo tracker generated landmarks with a trigonometric model and stored
the resulting "form accuracy" as though a real movement had been measured. Storing
provenance on every session is what makes that impossible to repeat: only rows
explicitly declared as ``pose_inference`` - i.e. produced by real on-device pose
estimation - are averaged into a form score. Everything else is history, not
evidence.
"""

from __future__ import annotations

from tests.conftest import auth_headers

MONTH_KEY = "2026-09"


def _add_session(client, headers, **overrides):
    payload = {
        "exercise": "squat",
        "exercise_label": "Squat",
        "reps": 10,
        "target_reps": 10,
        "form_accuracy": 90,
        "duration_sec": 60,
    }
    payload.update(overrides)
    return client.post("/sessions", headers=headers, json=payload)


# --- Recording the source ----------------------------------------------------


def test_a_session_without_a_declared_source_is_not_treated_as_measured(client, new_patient):
    """Deny-by-default: silence means "not a measurement", never "measured"."""
    _, token = new_patient("nodourcesource")
    headers = auth_headers(token)

    created = _add_session(client, headers)
    assert created.status_code == 201
    assert created.json()["metrics_source"] == "manual"


def test_real_pose_inference_is_recorded(client, new_patient):
    _, token = new_patient("realinference")
    headers = auth_headers(token)

    created = _add_session(client, headers, metrics_source="pose_inference")
    assert created.status_code == 201
    assert created.json()["metrics_source"] == "pose_inference"


def test_simulated_and_manual_sources_are_accepted_and_labelled(client, new_patient):
    _, token = new_patient("allssources")
    headers = auth_headers(token)

    for source in ("simulated", "manual", "pose_inference"):
        response = _add_session(client, headers, metrics_source=source)
        assert response.status_code == 201
        assert response.json()["metrics_source"] == source


def test_an_unknown_source_is_rejected(client, new_patient):
    """The value is a closed set, so it cannot be used to smuggle in a claim."""
    _, token = new_patient("badsource")
    response = _add_session(client, auth_headers(token), metrics_source="definitely_real_ai")
    assert response.status_code == 422


# --- Reporting excludes what was not measured --------------------------------


def test_report_averages_only_measured_sessions(client, new_patient):
    _, token = new_patient("reportmix")
    headers = auth_headers(token)

    _add_session(client, headers, form_accuracy=100, metrics_source="pose_inference")
    # These must not drag the average around, in either direction.
    _add_session(client, headers, form_accuracy=0, metrics_source="simulated")
    _add_session(client, headers, form_accuracy=0, metrics_source="manual")

    report = client.post("/reports/generate", headers=headers, json={"month_key": MONTH_KEY})
    assert report.status_code == 201
    payload = report.json()["payload"]

    assert payload["total_sessions"] == 3
    assert payload["measured_sessions"] == 1
    assert payload["unmeasured_sessions"] == 2
    assert payload["avg_form_accuracy"] == 100.0, "only the measured session may count"


def test_report_flags_a_month_with_no_measurements(client, new_patient):
    """A month of fabricated numbers must not read as a clinical result."""
    _, token = new_patient("nomeasurements")
    headers = auth_headers(token)

    _add_session(client, headers, form_accuracy=98, metrics_source="simulated")

    report = client.post("/reports/generate", headers=headers, json={"month_key": MONTH_KEY})
    payload = report.json()["payload"]

    assert payload["total_sessions"] == 1
    assert payload["measured_sessions"] == 0
    assert payload["no_measured_sessions"] is True
    assert payload["avg_form_accuracy"] == 0.0


def test_exercise_breakdown_marks_measured_sessions(client, new_patient):
    _, token = new_patient("breakdown")
    headers = auth_headers(token)

    _add_session(client, headers, exercise_label="Squat", metrics_source="pose_inference")
    _add_session(client, headers, exercise_label="Squat", metrics_source="manual")

    report = client.post("/reports/generate", headers=headers, json={"month_key": MONTH_KEY})
    entry = report.json()["payload"]["exercise_breakdown"][0]

    assert entry["exercise_label"] == "Squat"
    assert entry["sessions"] == 2
    assert entry["measured_sessions"] == 1


def test_daily_progress_separates_measured_from_logged_sessions(client, new_patient):
    _, token = new_patient("dailyprogress")
    headers = auth_headers(token)

    _add_session(client, headers, reps=12, form_accuracy=80, metrics_source="pose_inference")
    _add_session(client, headers, reps=8, form_accuracy=10, metrics_source="manual")

    daily = client.get("/progress/daily", headers=headers)
    assert daily.status_code == 200
    entry = daily.json()[0]

    assert entry["sessions"] == 2
    assert entry["total_reps"] == 20
    assert entry["measured_sessions"] == 1
    assert entry["avg_form_accuracy"] == 80.0
