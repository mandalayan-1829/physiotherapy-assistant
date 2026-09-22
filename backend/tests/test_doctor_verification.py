"""Clinician verification: self-registration must not confer clinical authority.

OWASP API6 (Unrestricted Access to Sensitive Business Flows). Anyone can reach
``POST /auth/register``, so ``role: "doctor"`` cannot be enough to become a
bookable clinician - otherwise any internet user could appear in a patient's
directory and, once booked, read that patient's entire medical record.

The rules pinned down here:

* a newly registered clinician is unverified,
* an unverified clinician is absent from the directory and cannot be booked or
  messaged (both of which are what create the access link),
* an unverified clinician can still see their own pending profile,
* the client cannot set the flag, and
* verification is what grants access - nothing else.
"""

from __future__ import annotations

from tests.conftest import auth_headers, register, verify_doctor


def _profile_id(client, token: str) -> int:
    response = client.get("/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    profile = response.json()["doctor_profile"]
    assert profile is not None, "a doctor account must have a clinical profile"
    return profile["id"]


def _book(client, patient_token: str, doctor_profile_id: int):
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


# --- Registration produces a pending, powerless profile ----------------------


def test_registered_doctor_is_unverified(client, new_doctor):
    account, token = new_doctor("pending")
    profile = client.get("/auth/me", headers=auth_headers(token)).json()["doctor_profile"]

    assert profile["is_verified"] is False
    assert account["role"] == "doctor", "the account itself is still a doctor account"


def test_client_cannot_self_verify_at_registration(client):
    """Extra body fields must not influence the verification flag."""
    response = register(
        client,
        "sneaky.clinician@example.com",
        "DoctorPass123",
        "doctor",
        full_name="Dr Sneaky",
        specialization="Orthopedic Rehabilitation",
        is_verified=True,
        verified=True,
        verified_at="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 201, response.text

    profile = client.get(
        "/auth/me", headers=auth_headers(response.json()["access_token"])
    ).json()["doctor_profile"]
    assert profile["is_verified"] is False, "the client must not be able to self-verify"


# --- An unverified clinician is invisible and unbookable --------------------


def test_unverified_doctor_is_absent_from_the_patient_directory(client, new_patient, new_doctor):
    patient, patient_token = new_patient("dirpatient")
    _, doctor_token = new_doctor("dirdoctor")
    doctor_profile_id = _profile_id(client, doctor_token)

    directory = client.get("/doctors", headers=auth_headers(patient_token)).json()
    assert doctor_profile_id not in [d["id"] for d in directory]


def test_unverified_doctor_can_still_see_their_own_profile(client, new_doctor):
    _, doctor_token = new_doctor("ownprofile")
    doctor_profile_id = _profile_id(client, doctor_token)

    directory = client.get("/doctors", headers=auth_headers(doctor_token)).json()
    mine = [d for d in directory if d["id"] == doctor_profile_id]
    assert len(mine) == 1, "clinicians must be able to see their own pending profile"
    assert mine[0]["is_verified"] is False


def test_unverified_doctor_profile_is_not_enumerable(client, new_patient, new_doctor):
    _, patient_token = new_patient("enumerate")
    _, doctor_token = new_doctor("hidden")
    doctor_profile_id = _profile_id(client, doctor_token)

    # A third party sees a 404, not a 403: confirming existence would leak
    # onboarding state.
    assert (
        client.get(f"/doctors/{doctor_profile_id}", headers=auth_headers(patient_token)).status_code
        == 404
    )
    # The owner still sees it.
    assert (
        client.get(f"/doctors/{doctor_profile_id}", headers=auth_headers(doctor_token)).status_code
        == 200
    )


def test_patient_cannot_book_an_unverified_doctor(client, new_patient, new_doctor):
    _, patient_token = new_patient("booker")
    _, doctor_token = new_doctor("unbookable")
    doctor_profile_id = _profile_id(client, doctor_token)

    response = _book(client, patient_token, doctor_profile_id)
    assert response.status_code == 403
    assert "verified" in response.json()["detail"].lower()


def test_booking_an_unverified_doctor_creates_no_access_link(client, new_patient, new_doctor):
    """The link is what grants record access, so it must not be created."""
    patient_account, patient_token = new_patient("nolink")
    _, doctor_token = new_doctor("nolinkdoc")
    doctor_profile_id = _profile_id(client, doctor_token)

    assert _book(client, patient_token, doctor_profile_id).status_code == 403

    # The clinician has no roster and cannot reach the patient's records.
    assert client.get("/patients", headers=auth_headers(doctor_token)).json() == []
    assert (
        client.get(
            f"/sessions?patient_account_id={patient_account['id']}",
            headers=auth_headers(doctor_token),
        ).status_code
        == 403
    )


def test_patient_cannot_message_an_unverified_doctor(client, new_patient, new_doctor):
    """Messaging also creates the link, so it is gated the same way."""
    _, patient_token = new_patient("messenger")
    _, doctor_token = new_doctor("unmessagable")
    doctor_profile_id = _profile_id(client, doctor_token)

    response = client.post(
        "/messages",
        headers=auth_headers(patient_token),
        json={"doctor_profile_id": doctor_profile_id, "message": "Hello doctor"},
    )
    assert response.status_code == 403

    # No relationship was established.
    assert client.get("/patients", headers=auth_headers(doctor_token)).json() == []


def test_unverified_doctor_cannot_read_another_patients_records(client, new_patient, new_doctor):
    patient_account, patient_token = new_patient("victim")
    client.post(
        "/sessions",
        headers=auth_headers(patient_token),
        json={"exercise": "squat", "exercise_label": "Squat", "reps": 10, "form_accuracy": 90},
    )

    _, doctor_token = new_doctor("nosy")
    assert (
        client.get(
            f"/sessions?patient_account_id={patient_account['id']}",
            headers=auth_headers(doctor_token),
        ).status_code
        == 403
    )


# --- Verification is what grants access ------------------------------------


def test_verified_doctor_becomes_discoverable_and_bookable(client, new_patient, new_doctor):
    patient_account, patient_token = new_patient("verifiedpatient")
    _, doctor_token = new_doctor("verifieddoctor")
    doctor_profile_id = _profile_id(client, doctor_token)

    # Before: hidden and unbookable.
    assert doctor_profile_id not in [
        d["id"] for d in client.get("/doctors", headers=auth_headers(patient_token)).json()
    ]
    assert _book(client, patient_token, doctor_profile_id).status_code == 403

    verify_doctor(doctor_profile_id)

    # After: listed, bookable, and the relationship grants record access.
    directory = client.get("/doctors", headers=auth_headers(patient_token)).json()
    listed = [d for d in directory if d["id"] == doctor_profile_id]
    assert len(listed) == 1 and listed[0]["is_verified"] is True

    booking = _book(client, patient_token, doctor_profile_id)
    assert booking.status_code == 201

    assert (
        client.get(
            f"/sessions?patient_account_id={patient_account['id']}",
            headers=auth_headers(doctor_token),
        ).status_code
        == 200
    )
    roster = client.get("/patients", headers=auth_headers(doctor_token)).json()
    assert any(p["account_id"] == patient_account["id"] for p in roster)


def test_patient_registration_remains_open_and_unaffected(client, new_patient):
    """The patient journey must not regress while clinicians are gated."""
    account, token = new_patient()
    assert account["role"] == "patient"
    assert client.get("/auth/me", headers=auth_headers(token)).status_code == 200
    assert client.get("/sessions", headers=auth_headers(token)).status_code == 200


def test_there_are_still_exactly_two_roles(client):
    assert register(client, "admin.attempt@example.com", "Password123", "admin").status_code == 422
    assert register(client, "admin2@example.com", "Password123", "staff").status_code == 422
