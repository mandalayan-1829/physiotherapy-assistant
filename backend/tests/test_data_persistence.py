"""Data written by the API is persisted and read back from the database."""

from __future__ import annotations

from tests.conftest import auth_headers


def test_medical_profile_round_trip(client, new_patient):
    _, token = new_patient("profile")

    updated = client.put(
        "/users/me/profile",
        headers=auth_headers(token),
        json={
            "age": 34,
            "blood_group": "O+",
            "height_cm": 178,
            "weight_kg": 76,
            "current_problem": "Left knee stiffness",
            "pain_intensity": 4,
            "rehab_goals": "Full pain-free squat depth",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["current_problem"] == "Left knee stiffness"

    # A fresh request reads the persisted values back (not in-memory state).
    fetched = client.get("/users/me/profile", headers=auth_headers(token))
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["age"] == 34
    assert body["pain_intensity"] == 4
    assert body["rehab_goals"] == "Full pain-free squat depth"
    assert body["blood_group"] == "O+"


def test_exercise_sessions_persist_and_aggregate(client, new_patient):
    _, token = new_patient("sessions")
    headers = auth_headers(token)

    for reps in (10, 12):
        created = client.post(
            "/sessions",
            headers=headers,
            json={
                "exercise": "squat",
                "exercise_label": "Squat",
                "reps": reps,
                "target_reps": 10,
                "form_accuracy": 90,
                "duration_sec": 60,
                "notes": "felt good",
            },
        )
        assert created.status_code == 201

    listed = client.get("/sessions", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    daily = client.get("/progress/daily", headers=headers)
    assert daily.status_code == 200
    assert len(daily.json()) == 1
    assert daily.json()[0]["sessions"] == 2
    assert daily.json()[0]["total_reps"] == 22


def test_diet_records_persist_and_delete(client, new_patient):
    _, token = new_patient("diet")
    headers = auth_headers(token)

    created = client.post(
        "/diet",
        headers=headers,
        json={"meal": "Oats and berries", "calories": 420, "protein": 32, "carbs": 52, "fats": 9},
    )
    assert created.status_code == 201
    record_id = created.json()["id"]

    listed = client.get("/diet", headers=headers)
    assert [r["meal"] for r in listed.json()] == ["Oats and berries"]

    assert client.delete(f"/diet/{record_id}", headers=headers).status_code == 204
    assert client.get("/diet", headers=headers).json() == []


def test_report_generation_uses_persisted_sessions(client, new_patient):
    _, token = new_patient("report")
    headers = auth_headers(token)

    client.post(
        "/sessions",
        headers=headers,
        json={
            "exercise": "squat",
            "exercise_label": "Squat",
            "reps": 40,
            "target_reps": 40,
            "form_accuracy": 91,
            "duration_sec": 300,
        },
    )

    report = client.post("/reports/generate", headers=headers, json={"month_key": "2026-09"})
    assert report.status_code == 201
    payload = report.json()["payload"]
    assert payload["total_sessions"] == 1
    assert payload["total_reps"] == 40
    assert payload["avg_form_accuracy"] == 91.0

    listed = client.get("/reports", headers=headers)
    assert len(listed.json()) == 1


def test_alerts_are_recorded(client, new_patient):
    _, token = new_patient("alert")
    headers = auth_headers(token)

    created = client.post(
        "/alerts",
        headers=headers,
        json={"alert_type": "pain_spike", "message": "Pain rose to 7/10", "sent_to": "+9112345"},
    )
    assert created.status_code == 201

    listed = client.get("/alerts", headers=headers)
    assert listed.status_code == 200
    assert listed.json()[0]["alert_type"] == "pain_spike"
