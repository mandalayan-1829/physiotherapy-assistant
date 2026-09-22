"""Server-side authorization is enforced, not just hidden in the UI.

This suite is the guard rail on the authorization system described in
``app/services/access.py``. The system is deliberately not redesigned here; the
only adjustment is that a clinician must be verified before a patient can create
a relationship with them (see ``tests/test_doctor_verification.py``), so the
helpers below verify the clinician first, exactly as an operator would.
"""

from __future__ import annotations

from tests.conftest import auth_headers, verify_doctor


def _book(client, patient_token, doctor_profile_id: int):
    return client.post(
        "/appointments",
        headers=auth_headers(patient_token),
        json={
            "doctor_profile_id": doctor_profile_id,
            "date": "2026-10-01",
            "time": "10:00 AM",
            "reason": "Knee review",
        },
    )


def test_unauthenticated_requests_are_rejected(client):
    for path in ("/sessions", "/diet", "/appointments", "/progress/daily", "/auth/me"):
        assert client.get(path).status_code == 401, path


def test_invalid_token_is_rejected(client):
    response = client.get("/sessions", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_patient_cannot_access_another_patients_data(client, new_patient):
    _, token_a = new_patient("alice")
    account_b, token_b = new_patient("bob")

    # Alice records a session.
    created = client.post(
        "/sessions",
        headers=auth_headers(token_a),
        json={
            "exercise": "squat",
            "exercise_label": "Squat",
            "reps": 10,
            "target_reps": 10,
            "form_accuracy": 94,
            "duration_sec": 60,
        },
    )
    assert created.status_code == 201

    # Bob explicitly asks for Alice's data by id -> forbidden.
    response = client.get(
        "/sessions", headers=auth_headers(token_b), params={"patient_account_id": account_b["id"]}
    )
    assert response.status_code == 200  # his own id is fine

    response = client.get(
        f"/sessions?patient_account_id={created.json()['patient_account_id']}",
        headers=auth_headers(token_b),
    )
    assert response.status_code == 403

    # Bob never sees Alice's session in his own list.
    mine = client.get("/sessions", headers=auth_headers(token_b))
    assert mine.status_code == 200
    assert mine.json() == []


def test_patient_cannot_use_doctor_only_endpoints(client, new_patient):
    _, token = new_patient("notadoctor")
    response = client.get("/patients", headers=auth_headers(token))
    assert response.status_code == 403


def test_patient_can_only_note_themselves(client, new_patient):
    account, token = new_patient("notedoc")
    other, _ = new_patient("otherpatient")

    # A patient may keep a journal note about themselves.
    own = client.post(
        "/notes",
        headers=auth_headers(token),
        json={"patient_account_id": account["id"], "note_text": "Knee felt better today"},
    )
    assert own.status_code == 201
    assert own.json()["doctor_profile_id"] is None

    # ...but not about anybody else.
    forbidden = client.post(
        "/notes",
        headers=auth_headers(token),
        json={"patient_account_id": other["id"], "note_text": "not allowed"},
    )
    assert forbidden.status_code == 403


def test_doctor_cannot_access_unlinked_patient(client, new_patient, new_doctor):
    patient_account, patient_token = new_patient("unlinked")
    # Patient records something.
    client.post(
        "/sessions",
        headers=auth_headers(patient_token),
        json={"exercise": "lunge", "exercise_label": "Lunge", "reps": 8, "form_accuracy": 90},
    )

    _, doctor_token = new_doctor("unlinked")
    response = client.get(
        f"/sessions?patient_account_id={patient_account['id']}",
        headers=auth_headers(doctor_token),
    )
    assert response.status_code == 403

    # The doctor's patient list is empty too.
    patients = client.get("/patients", headers=auth_headers(doctor_token))
    assert patients.status_code == 200
    assert patients.json() == []


def test_doctor_gains_access_after_relationship_is_established(client, new_patient, new_doctor):
    patient_account, patient_token = new_patient("linked")
    client.post(
        "/sessions",
        headers=auth_headers(patient_token),
        json={"exercise": "squat", "exercise_label": "Squat", "reps": 12, "form_accuracy": 96},
    )

    _, doctor_token = new_doctor("linked")
    doctor_profile_id = client.get("/auth/me", headers=auth_headers(doctor_token)).json()[
        "doctor_profile"
    ]["id"]

    # The clinician is only discoverable/bookable once verified.
    verify_doctor(doctor_profile_id)

    # The patient can see the clinician in the directory.
    directory_ids = [d["id"] for d in client.get("/doctors", headers=auth_headers(patient_token)).json()]
    assert doctor_profile_id in directory_ids

    booking = _book(client, patient_token, doctor_profile_id)
    assert booking.status_code == 201

    # Now the doctor is linked and can read the patient's sessions.
    response = client.get(
        f"/sessions?patient_account_id={patient_account['id']}",
        headers=auth_headers(doctor_token),
    )
    assert response.status_code == 200
    assert len(response.json()) == 1

    patients = client.get("/patients", headers=auth_headers(doctor_token)).json()
    assert any(p["account_id"] == patient_account["id"] for p in patients)


def test_doctor_cannot_modify_another_doctors_appointment(client, new_patient, new_doctor):
    patient_account, patient_token = new_patient("appt")
    _, doctor_a_token = new_doctor("alpha")
    _, doctor_b_token = new_doctor("beta")

    doctor_a_profile = client.get("/auth/me", headers=auth_headers(doctor_a_token)).json()[
        "doctor_profile"
    ]["id"]
    verify_doctor(doctor_a_profile)
    booking = _book(client, patient_token, doctor_a_profile)
    appointment_id = booking.json()["id"]

    response = client.patch(
        f"/appointments/{appointment_id}",
        headers=auth_headers(doctor_b_token),
        json={"status": "confirmed"},
    )
    assert response.status_code == 403

    # The owning doctor can update it.
    ok = client.patch(
        f"/appointments/{appointment_id}",
        headers=auth_headers(doctor_a_token),
        json={"status": "confirmed", "clinician_note": "Reviewed."},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "confirmed"


def test_patient_cannot_approve_own_appointment(client, new_patient, new_doctor):
    _, patient_token = new_patient("approver")
    _, doctor_token = new_doctor("approver")
    doctor_profile = client.get("/auth/me", headers=auth_headers(doctor_token)).json()[
        "doctor_profile"
    ]["id"]
    verify_doctor(doctor_profile)
    appointment_id = _book(client, patient_token, doctor_profile).json()["id"]

    response = client.patch(
        f"/appointments/{appointment_id}",
        headers=auth_headers(patient_token),
        json={"status": "approved"},
    )
    assert response.status_code == 403
