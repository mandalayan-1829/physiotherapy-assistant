# Threat Model

> **Scope and method.** This document describes the security posture of the
> repository as it exists at the time of writing (branch `security-documentation`,
> HEAD `a132eaa`, 2026-10-06). Every statement below is derived from reading the
> current source, configuration and migrations. Nothing here was fixed, and no
> code was changed. Where a claim could not be verified from the repository it is
> explicitly labelled **Potential risk requiring verification**.

## System Overview

The project is a two-tier web application for physiotherapy / rehabilitation:

```
Browser (React 19 + TypeScript + Vite SPA)
   │  HTTPS, fetch() from src/services/api.ts
   ▼
FastAPI application (backend/app)
   │  SQLAlchemy 2.0 ORM
   ▼
SQLite (local development)  /  PostgreSQL (production, via DATABASE_URL)
```

* **Frontend.** A single-page React 19 + TypeScript application built with Vite 6
  (`package.json`, `vite.config.ts`, `index.html` → `src/main.tsx`). Deployed as a
  static site on Vercel (`vercel.json`). All backend calls go through one module,
  `src/services/api.ts`, which attaches the `Authorization: Bearer` header.
  Authentication is in `src/services/auth.ts`; domain data is in
  `src/services/physio.ts`; identity is **always** re-resolved from
  `GET /auth/me`, never trusted from browser storage.
* **Backend.** A FastAPI application (`backend/app`) layered as
  `api/routes → services → models`. `backend/app/main.py` mounts ten routers
  (`auth`, `users`, `doctors`, `sessions`, `progress`, `diet`, `appointments`,
  `notes`, `reports`, `communication`). Cross-cutting concerns live in
  `app/core/config.py` (settings), `app/core/security.py` (hashing + JWT),
  `app/api/deps.py` (authentication, role guards, client identity) and
  `app/services/access.py` (object-level authorization).
* **Database.** SQLAlchemy 2.0 models (`backend/app/models`) over SQLite in
  development and PostgreSQL in production (`backend/app/db/session.py`,
  `DATABASE_URL`). Schema evolution is handled by four Alembic revisions
  (`backend/alembic/versions`). `create_all` runs at startup for convenience.
* **Pose estimation.** Real on-device inference with MediaPipe Tasks Vision
  (`src/utils/poseEstimator.ts`, `src/components/PoseTracker.tsx`). Camera frames
  are read from a `<video>` element and never uploaded; only aggregate metrics
  (reps, form score, duration, notes) are sent to `POST /sessions`.
* **Email.** Password-reset codes are delivered through an SMTP abstraction
  (`backend/app/services/email_service.py`). A `console` transport exists for
  development only and never logs the message body.
* **Deployment.** The repository contains Infrastructure-as-Code for the backend
  on Railway (`.railway/railway.ts`), a Vercel configuration for the frontend
  (`vercel.json`), and a deployment runbook (`docs/DEPLOYMENT.md`). The backend
  currently runs with `ENVIRONMENT=production`, `ENABLE_API_DOCS=false`,
  `TRUST_PROXY_HEADERS=true`, `EMAIL_BACKEND=console` and
  `ALLOW_PRODUCTION_WITHOUT_EMAIL=true` (i.e. password reset is unavailable until
  an SMTP provider is configured).

### Major data flows

| Flow | Path | Notes |
| --- | --- | --- |
| Registration / login | Browser → `POST /auth/register` / `/auth/login` → `accounts` | Returns a short‑lived JWT |
| Session restore | Browser → `GET /auth/me` | Identity/role are authoritative from the DB |
| Profile | Browser → `GET/PUT /users/me/profile` → `patient_profiles` | Full medical profile |
| Camera → metrics | Camera → on‑device MediaPipe → aggregate metrics | Frames never leave the device |
| Session capture | Browser → `POST /sessions` → `workout_sessions` | Aggregates only; `metrics_source` records provenance |
| Reports | Browser → `POST /reports/generate` / `GET /reports` → `reports` | Computed server-side from persisted sessions |
| Password reset | Browser → `/auth/forgot-password` → backend → SMTP → user | Code stored only as a bcrypt hash |
| Doctor discovery | Browser → `GET /doctors` → `doctor_profiles` | Verified clinicians only |
| Telehealth | Browser → `POST /messages` / `POST /appointments` → `messages` / `appointments` (+ `patient_doctor_links`) | Booking creates an access link |

## Assets

| Asset | Where it lives | Notes |
| --- | --- | --- |
| User accounts | `accounts` | email, full name, bcrypt `password_hash`, role, `is_active`, `token_version` |
| Authentication credentials | `accounts.password_hash`; `password_reset_tokens.code_hash` | bcrypt in both cases |
| JWT signing secret | `SECRET_KEY` (environment only) | Single key for all sessions |
| Patient information | `patient_profiles` | Demographics, medical history, pain profile, emergency contacts |
| Doctor information | `doctor_profiles` | Name, specialization, contact, email, WhatsApp, verification state |
| Physiotherapy session data | `workout_sessions` | Exercise, reps, form accuracy, duration, notes, provenance |
| Research / measurement provenance | `workout_sessions.metrics_source`, `reports.payload` | Distinguishes measured vs. simulated vs. manual |
| Database records | All tables | Managed PostgreSQL in production (per config) |
| Reports | `reports.payload` (JSON) | Monthly aggregates tied to a patient |
| Password-reset information | `password_reset_tokens`; SMTP credentials | Code hash, expiry, attempts, transport secrets |
| Secrets | Environment / Railway variables | `SECRET_KEY`, SMTP credentials, `DATABASE_URL` |
| Source code & deployment configuration | Repository, `.railway/railway.ts`, `vercel.json` | Public-facing repository |
| Legacy health data | `aiphysio.db` (git-ignored, on disk); **also present in git history** | See finding **TM‑1** |

## Trust Boundaries

* **Browser ↔ Backend (public internet).** Every browser request crosses this
  boundary over HTTPS; the backend is the sole authority for authentication and
  authorization.
* **Frontend ↔ Backend.** The SPA is served from Vercel and calls the backend
  origin configured in `VITE_API_URL`. CORS is restricted to an explicit origin
  list (`app/main.py`, `app/core/config.py`).
* **Backend ↔ Database.** Authenticated via `DATABASE_URL`; not publicly exposed
  by the application. Whether the database is reachable only from the private
  network is **Potential risk requiring verification**.
* **Backend ↔ Email provider (SMTP).** The reset code crosses to the configured
  SMTP host. No provider is currently configured.
* **Browser ↔ Camera.** `getUserMedia` runs entirely on the user's device; frames
  do not cross the network.
* **Browser ↔ Third-party CDNs.** Google Fonts (`index.html`), the MediaPipe model
  CDN (`src/utils/poseEstimator.ts`, overridable) and a YouTube-nocookie iframe
  (`src/views/ExerciseSelectionView.tsx`).
* **Developer environment ↔ Repository.** Deployment secrets are kept in the
  hosting platform; `.gitignore` excludes `.env*`, `*.db` and research artefacts.
* **Deployment platform ↔ Application.** Railway injects `DATABASE_URL`, `$PORT`
  and (per `.railway/railway.ts`) terminates TLS and sets `X-Forwarded-For`.

## Actors

* **Patient** – authenticated account with `role=patient`; may read/write only
  their own records.
* **Doctor / physiotherapist** – authenticated account with `role=doctor`; may
  read a patient's records only when an `active` `patient_doctor_links` row
  exists.
* **Unauthenticated visitor** – can reach only `/health` and the four
  `/auth/*` public endpoints.
* **Malicious / authenticated attacker** – a registered account attempting
  horizontal/vertical privilege escalation (IDOR, role abuse).
* **External attacker** – pre-auth abuse of login, registration and password
  reset.
* **Developer / operator** – holds deployment secrets and runs
  `backend/scripts/verify_doctor.py` out of band.
* **Hosting provider** – Railway/Vercel operate the runtime and terminate TLS.
* **Third-party service** – SMTP provider (when configured), Google Fonts,
  MediaPipe CDN, YouTube.

## Entry Points

Public, unauthenticated:

| Method | Path | File |
| --- | --- | --- |
| GET | `/health` | `backend/app/main.py` |
| POST | `/auth/register` | `backend/app/api/routes/auth.py` |
| POST | `/auth/login` | `backend/app/api/routes/auth.py` |
| POST | `/auth/forgot-password` | `backend/app/api/routes/auth.py` |
| POST | `/auth/verify-reset-code` | `backend/app/api/routes/auth.py` |
| POST | `/auth/reset-password` | `backend/app/api/routes/auth.py` |

Authenticated entry points (see `attack-surface.md` for the full list): `/auth/me`,
`/auth/logout`, `/users/me/*`, `/doctors`, `/doctors/{id}`, `/patients`,
`/sessions`, `/progress/daily`, `/diet`, `/notes`, `/appointments`, `/messages`,
`/alerts`, `/reports`, `/reports/{id}`, `/reports/generate`.

Frontend inputs feeding the backend include registration and login fields, the
patient medical profile, exercise session payloads, diet records, notes,
appointment bookings/updates, telehealth messages, and the three-step password
reset. There are **no file-upload entry points** (no `UploadFile`/`File`/`Form`
parameter exists anywhere in `backend/app`).

## Threats

Realistic threats for this architecture:

1. **Credential attacks** – credential stuffing / password spraying against
   `/auth/login`, and reset-code brute force against `/auth/verify-reset-code`.
2. **Account enumeration** – probing login, forgot-password or registration to
   learn whether an address is registered.
3. **Broken object-level authorization (IDOR/BOLA)** – requesting another
   patient's sessions, reports, notes, diet, messages or alerts by id.
4. **JWT forgery / confusion** – presenting a token with the wrong algorithm, a
   foreign issuer, an elevated `role`, or a password-reset token as a session.
5. **Self-service privilege escalation** – registering as `role=doctor` to obtain
   clinical authority.
6. **Session theft via XSS** – a stolen bearer token being replayed.
7. **Resource exhaustion** – unbounded report generation, list endpoints and
   free-text fields.
8. **Clinical-integrity tampering** – submitting fabricated metrics as real
   measurements.
9. **Sensitive-data exposure in the repository** – committed health data /
   credential material.
10. **Abuse of unaudited access** – no record of who read which patient record.

## Attack Scenarios

**A. JWT forgery.** *Path:* attacker crafts a token and calls an authenticated
endpoint. *Asset:* every account. *Capability:* network access. *Existing
protection:* HS256/384/512 allow-list, required claims (`sub`, `exp`, `iat`,
`iss`, `purpose`), pinned issuer, `purpose=access` required, and `token_version`
comparison against the database (`app/core/security.py`, `app/api/deps.py`; tests
in `tests/test_jwt_security.py`). *Remaining risk:* **Low** — forging requires
`SECRET_KEY`, which has no default and is validated at startup.

**B. IDOR on patient data.** *Path:* authenticated patient requests another
patient's `patient_account_id` or object id. *Asset:* patient records. *Capability:*
any authenticated account. *Existing protection:* `resolve_patient_scope` and
`ensure_patient_access` force patients to their own id and require an active link
for doctors (`app/services/access.py`; tests in `tests/test_authorization.py`,
`tests/test_reports.py`). *Remaining risk:* **Low**, with the note-level caveat in
finding **TM‑4**.

**C. Password-spraying login.** *Path:* many passwords across many addresses.
*Asset:* account integrity. *Capability:* network access. *Existing protection:*
per-email failure counter (`auth_throttles`, scope `login`) **and** a per-source
budget (`login_max_requests_per_ip_per_window`, scope `auth_ip`) enforced in
`auth_service.authenticate`; generic error messages. *Remaining risk:* **Low‑Medium**
— DB-backed counters are effective, but there is no CAPTCHA/proof-of-work.

**D. Reset-code brute force.** *Path:* repeatedly guess a 6-digit code. *Asset:*
account takeover. *Capability:* network access. *Existing protection:* code
bcrypt-hashed, 10-minute TTL, 5-attempt cap per reset row, per-source limit across
the whole flow (`password_reset_service.py`). *Remaining risk:* **Low**.

**E. Doctor self-registration.** *Path:* register with `role=doctor`, appear in
the directory and be booked. *Asset:* patient records across the platform.
*Capability:* network access. *Existing protection:* `doctor_profiles.is_verified`
defaults to `false`; unverified profiles are excluded from `GET /doctors`,
cannot be booked or messaged, are non-enumerable, and cannot acquire links
(`app/services/access.py`, `doctor_service.py`, `communication_service.py`,
`appointment_service.py`; tests in `tests/test_doctor_verification.py`).
*Remaining risk:* **Low**, subject to the operational availability of
`backend/scripts/verify_doctor.py`.

**F. XSS → token theft.** *Path:* injected script reads
`localStorage['physio_access_token']`. *Asset:* the session. *Capability:* an XSS
sink. *Existing protection:* React escaping, no `dangerouslySetInnerHTML` or
`innerHTML` in `src/`, and no untrusted content is injected; a strict CSP is **not**
set (`app/main.py` explains this is deferred). *Remaining risk:* **Medium** — the
token is in `localStorage`, so any future XSS yields a usable session; see
**TM‑2**.

**G. Resource exhaustion.** *Path:* repeated `POST /reports/generate` (which loads
every session a patient owns) or oversized free-text payloads. *Asset:*
availability. *Capability:* any authenticated account. *Existing protection:*
bounded numeric fields and a 2000-char cap on session notes; DB-backed throttling
on the auth surface only. *Remaining risk:* **Medium** — see **TM‑3**.

**H. Historic health-data exposure.** *Path:* cloning the repository (or reading
its git history) and extracting `aiphysio.db`. *Asset:* identifiable health data
and an unsalted SHA-256 credential digest. *Capability:* read access to the repo
or its history. *Existing protection:* the file is now git-ignored and absent from
HEAD. *Remaining risk:* **High** — the blob remains reachable in history; see
**TM‑1**.

## Impact

Consequences are, in decreasing order of severity: full authentication bypass
(secret compromise), account takeover (credential/reset abuse), cross-patient
health-data disclosure (IDOR), exposure of identifiable health data and a
crackable credential digest through the repository history, denial of service
through unbounded work, and loss of clinical trust from fabricated measurements
being presented as real. Data considered sensitive here includes medical history,
pain profile, emergency contacts, session metrics, telehealth messages and
password-reset material.

## Likelihood

* JWT forgery (with a properly configured secret): **Low**
* IDOR on patient-scoped endpoints: **Low**
* Credential stuffing / spraying: **Medium**
* Reset-code brute force: **Low**
* Unverified-doctor escalation: **Low**
* Token theft through a future XSS: **Medium**
* Resource exhaustion: **Medium**
* Repository history exposure of legacy health data: **Medium** (occurred if the
  repository is public)

## Risk Severity

| ID | Finding | Severity | Classification |
| --- | --- | --- | --- |
| TM‑1 | Legacy `aiphysio.db` remains in git history | High | Confirmed historical exposure |
| TM‑2 | Access token stored in `localStorage` | Medium | Security weakness |
| TM‑3 | Unbounded `/reports/generate` and free-text fields | Medium | Security weakness |
| TM‑4 | Cross-clinician note visibility | Medium | Potential risk requiring verification |
| TM‑5 | No CSP header | Medium | Missing control |
| TM‑6 | Registration account-enumeration oracle (409) | Low | Security weakness |
| TM‑7 | Doctor directory exposes contact details; `/doctors/{id}` has no relationship check | Low | Security weakness |
| TM‑8 | Booking silently creates a permanent access link | Medium | Security weakness |
| TM‑9 | No access-audit log / retention policy | Medium | Missing control |
| TM‑10 | No verified encryption-at-rest or backup configuration | Medium | Potential risk requiring verification |
| TM‑11 | Failed report-provenance tests reduce assurance | Low | Confirmed weakness (current test run) |

## Existing Mitigations

Controls that **actually exist** in the current tree:

* **Fail-fast configuration** – `app/core/config.py` refuses to start without a
  strong `SECRET_KEY`, rejects placeholder/low-entropy keys, restricts JWT to
  HMAC, and in production rejects localhost/wildcard CORS and a non-SMTP mail
  transport (unless the explicit opt-out is set). Tested in
  `tests/test_security_config.py`.
* **Password hashing** – per-hash bcrypt (`app/core/security.py`); legacy
  unsalted SHA-256 login is disabled by default
  (`allow_legacy_password_login = False`).
* **JWT hardening** – algorithm allow-list, required claims, issuer pinning,
  `purpose` separation, `token_version` invalidation on password change.
* **Reset-flow hardening** – CSPRNG codes stored hashed, single-use, expiring,
  attempt-capped, cooldown + per-email and per-source limits, generic responses.
* **Rate limiting** – DB-backed `auth_throttles` shared across workers/restarts.
* **Object-level authorization** – central `access.py` used by every
  patient-scoped route.
* **Role enforcement** – `require_patient` / `require_doctor`; `role` is resolved
  from the database, never from the token claim.
* **Doctor verification gate** – deny-by-default for clinicians.
* **Security headers** – `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, `Permissions-Policy`, and HSTS in production (`app/main.py`).
* **API docs disabled in production** – `docs_url`/`redoc_url`/`openapi_url` are
  `None` when `ENVIRONMENT=production`.
* **No secrets in the working tree** – only `.env.example` templates; a scan of
  tracked files found no hardcoded credentials.
* **No SQL string building** – SQLAlchemy ORM/`select()` only; the sole raw
  `sqlite3` use is the read-only legacy importer.
* **No upload surface** – no multipart route exists.
* **Privacy-preserving pose** – frames and landmarks never leave the device.
* **Build-time guards** – `vite.config.ts` aborts a production build on a
  missing, local or placeholder `VITE_API_URL`.
* **Reset-code non-logging** – email bodies are never logged; verified in
  `tests/test_password_reset_logging.py`.

## Missing Mitigations

Relevant controls that are **not** currently present:

* Content-Security-Policy (deliberately deferred pending browser verification).
* httpOnly cookie / refresh-token transport for the access token.
* A distributed rate limit and request-size/body limits outside the auth surface.
* Pagination on list endpoints.
* Access-audit logging and a documented data-retention policy.
* Consent capture and a patient-facing view/revocation of clinician links.
* A configured SMTP provider (so password reset currently cannot complete in the
  production configuration).
* A root README describing security expectations, and CI running the test suite.

## Findings

Each item uses the required structure.

### TM‑1 — Legacy health database remains reachable in git history

* **Severity:** High
* **Classification:** Confirmed historical exposure
* **Evidence:** `git rev-list --all --objects` lists the blob
  `aiphysio.db` (object `60a37bb…`); `git log --all --diff-filter=AD -- aiphysio.db`
  shows it added in `912e837` and `7b35e9f` and deleted in `ea2a631` and
  `a132eaa`. It is absent from HEAD and is covered by `*.db` / `aiphysio.db` in
  `.gitignore`, but a copy still exists on disk in the working tree.
* **Risk:** Identifiable health information and an unsalted SHA-256 password
  digest remain extractable from history by anyone who can clone the repository.
* **Affected location:** Repository git history; working-tree file `aiphysio.db`
  (untracked).
* **Why it matters:** Health data is among the most sensitive personal data; a
  digest in the legacy format is crackable offline.
* **Recommended fix:** Treat the data as disclosed; rewrite history to purge the
  blob (e.g. `git filter-repo`/BFG) and force-push, or, if history rewrite is not
  acceptable, rotate the affected credentials and require a password reset; keep
  the legacy database only outside the repository and load it via
  `backend/scripts/migrate_legacy.py`.

### TM‑2 — Access token stored in `localStorage`

* **Severity:** Medium
* **Classification:** Security weakness
* **Evidence:** `src/services/api.ts` reads/writes `localStorage` key
  `physio_access_token`; `docs/DEPLOYMENT.md` §6 lists this as a known gap.
* **Risk:** A successful XSS (or any script running in origin) can exfiltrate a
  usable bearer token.
* **Affected location:** `src/services/api.ts`.
* **Why it matters:** The token is the sole session credential and is valid for
  the configured access-token lifetime (default 7 days).
* **Recommended fix:** Move to httpOnly, `Secure`, `SameSite` cookies with a
  short-lived access token and a refresh flow; keep `token_version` invalidation.

### TM‑3 — Unbounded report generation and unbounded free-text fields

* **Severity:** Medium
* **Classification:** Security weakness
* **Evidence:** `report_service.generate_report` selects **all** of a patient's
  sessions (no month filter in SQL); `POST /reports/generate` has no rate limit.
  `NoteCreate.note_text`, `MessageCreate.message`, `AppointmentCreate.reason` and
  `AlertCreate.message` have no `max_length` (`app/schemas`).
* **Risk:** A single account can trigger repeated expensive recomputations or
  store arbitrarily large strings (resource consumption / API4).
* **Affected location:** `backend/app/services/report_service.py`,
  `backend/app/api/routes/reports.py`, `backend/app/schemas/{note,alert,appointment}.py`.
* **Why it matters:** Availability of a shared backend is affected by any single
  authenticated caller.
* **Recommended fix:** Filter the month in SQL, add per-patient throttling on
  `/reports/generate`, add `max_length` bounds, and paginate list endpoints.

### TM‑4 — Cross-clinician note visibility is not scoped to the author

* **Severity:** Medium
* **Classification:** Potential risk requiring verification
* **Evidence:** `note_service.list_notes` filters only on `patient_account_id`,
  whereas `communication_service.list_messages` additionally filters by
  `doctor_profile_id`. There is no `doctor_profile_id` filter or test covering two
  doctors sharing a patient.
* **Risk:** Any clinician linked to a patient can read notes authored by other
  clinicians.
* **Affected location:** `backend/app/services/note_service.py`.
* **Why it matters:** It can expose one clinician's assessment to another, which
  may or may not be clinically intended.
* **Recommended fix:** Decide the intended visibility explicitly and encode it
  (author-only, or shared-with-audit); add a two-doctor test.

### TM‑5 — No Content-Security-Policy

* **Severity:** Medium
* **Classification:** Missing control
* **Evidence:** `backend/app/main.py` sets several headers but explicitly omits
  CSP, documenting that the SPA loads Google Fonts and YouTube iframes and that a
  policy must be verified in a browser first. `vercel.json` adds no CSP.
* **Risk:** No defence-in-depth against script injection.
* **Affected location:** `backend/app/main.py`, `vercel.json`.
* **Why it matters:** CSP mitigates the impact of any future XSS, which is
  amplified by TM‑2.
* **Recommended fix:** Author and test a CSP against the real bundle; set it at
  the edge (Vercel) and/or via middleware.

### TM‑6 — Registration reveals whether an email is registered

* **Severity:** Low
* **Classification:** Security weakness
* **Evidence:** `auth_service.register_account` raises
  `ConflictError("An account with this email already exists.")` when the email
  already exists (HTTP 409).
* **Risk:** Registration is a reliable account-enumeration oracle, unlike login
  and forgot-password, which are deliberately generic.
* **Affected location:** `backend/app/services/auth_service.py`.
* **Why it matters:** Enumeration feeds targeted credential attacks.
* **Recommended fix:** Return a generic outcome (e.g. always "check your email")
  and complete the flow out of band.

### TM‑7 — Doctor directory breadth and unconstrained profile reads

* **Severity:** Low
* **Classification:** Security weakness
* **Evidence:** `doctor_service.list_doctor_profiles` returns all verified
  profiles (including `email`, `whatsapp`, `contact`); `GET /doctors/{id}`
  (`doctor_service.get_profile`) performs no relationship check for verified
  profiles.
* **Risk:** Any authenticated user can enumerate verified clinicians one id at a
  time and harvest contact details.
* **Affected location:** `backend/app/services/doctor_service.py`,
  `backend/app/api/routes/doctors.py`.
* **Why it matters:** Unnecessary disclosure of clinician contact data.
* **Recommended fix:** Expose only directory fields; paginate; consider requiring
  a relationship for detail views.

### TM‑8 — Booking silently creates a durable access link

* **Severity:** Medium
* **Classification:** Security weakness
* **Evidence:** `appointment_service.create_appointment` calls `ensure_link`,
  creating an `active` `patient_doctor_links` row with no clinician acceptance and
  no in-product revocation.
* **Risk:** A single booking grants a clinician ongoing read access to the
  patient's entire record.
* **Affected location:** `backend/app/services/appointment_service.py`,
  `backend/app/services/access.py`.
* **Why it matters:** Consent and revocation expectations for health data.
* **Recommended fix:** Make linkage explicit/consented and revocable, and show the
  patient who can see their data.

### TM‑9 — No access audit log or retention policy

* **Severity:** Medium
* **Classification:** Missing control
* **Evidence:** No audit/access-log model or middleware exists; deletes are hard
  deletes; no retention job exists for `auth_throttles` or
  `password_reset_tokens`.
* **Risk:** "Who accessed this record, when?" cannot be answered; data is retained
  indefinitely.
* **Affected location:** `backend/app/models`, `backend/app/services`.
* **Why it matters:** Basic accountability for health data.
* **Recommended fix:** Add an append-only access log and a documented retention
  policy.

### TM‑10 — Encryption at rest and backups are not verifiable from the repository

* **Severity:** Medium
* **Classification:** Potential risk requiring verification
* **Evidence:** No backup or disk-encryption configuration is present in the
  repository; the managed database is external.
* **Risk:** If the managed database has no encryption/backups, a provider-side
  incident could expose or destroy patient data.
* **Affected location:** Deployment/platform configuration (outside the repo).
* **Why it matters:** Health data requires durable, protected storage.
* **Recommended fix:** Confirm encryption-at-rest, automated backups and restore
  testing in the hosting provider's dashboard; do not assume them.

### TM‑11 — Report-provenance tests currently fail

* **Severity:** Low
* **Classification:** Confirmed weakness (observed test run)
* **Evidence:** `backend/tests/test_data_persistence.py::test_report_generation_uses_persisted_sessions`
  and three tests in `backend/tests/test_pose_metrics_provenance.py` fail in the
  current tree (a run on 2026-10-06 reported 4 failed, 0 errors).
* **Risk:** The automated assurance for measurement provenance and report
  aggregation is currently red, so those guarantees are not being continuously
  verified.
* **Affected location:** `backend/tests/`.
* **Why it matters:** Clinical-integrity behaviour is exactly what these tests
  cover.
* **Recommended fix:** Investigate and fix the failing assertions (documentation
  only here; no code was changed).

## Overall assessment

The architecture applies a coherent set of server-side controls — fail-fast
secret validation, bcrypt, hardened JWTs, DB-backed throttling, a central
authorization helper and a clinician verification gate — and no third-party API
credentials exist to steal. The principal residual risks are the historical
health-data exposure (**TM‑1**), token storage in `localStorage` (**TM‑2**),
unbounded work and fields (**TM‑3**), and the absence of audit/retention and a
CSP. This is an assessment of the current implementation only; it does not claim
that production security is complete, that the system is fully secure, or that
any regulatory or medical compliance exists.
