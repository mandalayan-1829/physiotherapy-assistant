# Security Checklist

> **Scope and method.** A category-by-category assessment of the repository as
> it exists at the time of writing (branch `security-documentation`, HEAD
> `a132eaa`, 2026-10-06). Every status is based on the current source,
> configuration and tests. Nothing was implemented — recommendations are
> documented only. Where a claim is not verifiable from the repository it is
> marked **Requires verification**; where the category does not apply it is
> **Not applicable**.
>
> Status legend: **Implemented**, **Partially implemented**, **Missing**,
> **Requires verification**, **Not applicable**.

## Authentication

* **Status:** Implemented
* **Evidence:** `POST /auth/register`, `/auth/login`, `/auth/logout`, `/auth/me`
  (`backend/app/api/routes/auth.py`); bcrypt hashing (`app/core/security.py`);
  server-side password policy enforced on register and reset; login failure
  counter and generic errors (`app/services/auth_service.py`, scope `login` in
  `auth_throttles`).
* **Finding:** Authentication is complete and server-verified. Registration
  reveals whether an email is already registered (409).
* **Risk:** Account enumeration via registration.
* **Recommendation:** Document only — return a generic registration outcome.

**Finding A1**
* Severity → **Low**
* Evidence → `auth_service.register_account` raises `ConflictError("An account with this email already exists.")`
* Risk → reliable account-enumeration oracle
* Affected location → `backend/app/services/auth_service.py`
* Why it matters → feeds targeted credential attacks
* Recommended fix → generic response regardless of email existence

## Authorization / Role-Based Access Control

* **Status:** Implemented
* **Evidence:** Central `app/services/access.py` (`resolve_patient_scope`,
  `ensure_patient_access`, `link_exists`); `require_patient`/`require_doctor`
  (`app/api/deps.py`); role resolved from the DB; object-ownership checks on
  delete/patch routes; tests in `tests/test_authorization.py`,
  `tests/test_doctor_verification.py`, `tests/test_reports.py`.
* **Finding:** BOLA/IDOR is well handled for patient-scoped routes. Note-level
  visibility is broader than `messages` (author is not scoped); doctor profile
  detail reads are not relationship-scoped.
* **Risk:** Cross-clinician note exposure; directory harvesting.
* **Recommendation:** Document only — explicitly define note visibility and
  tighten doctor-profile detail exposure.

**Finding A2**
* Severity → **Medium**
* Evidence → `note_service.list_notes` filters only on `patient_account_id`, unlike `communication_service.list_messages` which filters by `doctor_profile_id`; no cross-doctor note test exists
* Risk → a linked clinician can read notes authored by other clinicians
* Affected location → `backend/app/services/note_service.py`
* Why it matters → may disclose one clinician's assessment to another
* Recommended fix → decide and enforce visibility; add a two-doctor test

**Finding A3**
* Severity → **Low**
* Evidence → `doctor_service.list_doctor_profiles` returns all verified profiles incl. email/whatsapp/contact; `get_profile` has no relationship check for verified profiles
* Risk → any authenticated user can enumerate verified clinicians and contact details
* Affected location → `backend/app/services/doctor_service.py`, `backend/app/api/routes/doctors.py`
* Why it matters → unnecessary clinician contact disclosure
* Recommended fix → restrict directory fields / paginate / relationship-gate detail views

## Session / Token Security

* **Status:** Partially implemented
* **Evidence:** JWT with required claims `sub`,`exp`,`iat`,`iss`,`purpose`; issuer
  pinning; HMAC algorithm allow-list; `purpose=access` required for API calls;
  `token_version` invalidation on password change
  (`app/core/security.py`, `app/api/deps.py`); refresh-token tests in
  `tests/test_jwt_security.py`.
* **Finding:** Token verification is strong, but the token is stored in
  `localStorage` and there is no refresh/rotation; `/auth/logout` is a client-side
  no-op and default lifetime is 7 days.
* **Risk:** Token theft through any XSS; a leaked token is valid until expiry.
* **Recommendation:** Document only — move to httpOnly cookies with short-lived
  access tokens and refresh; add server-side revocation.

**Finding S1**
* Severity → **Medium**
* Evidence → `src/services/api.ts` stores the JWT in `localStorage['physio_access_token']`; `docs/DEPLOYMENT.md` §6 records the gap; `app/routes/auth.py` logout is a 204 no-op
* Risk → exfiltration/replay of a long-lived bearer token
* Affected location → `src/services/api.ts`
* Why it matters → the token is the sole session credential
* Recommended fix → httpOnly/Secure/SameSite cookie transport + refresh flow

## Input Validation

* **Status:** Partially implemented
* **Evidence:** Pydantic v2 models on every body (`backend/app/schemas`), with
  `Field` bounds on numeric fields, `Literal` enums for roles/statuses/categories/
  alert types, `EmailStr`, and a `month_key` regex. Server-side password policy.
* **Finding:** Numeric and enum inputs are well bounded; several free-text fields
  have no `max_length` (`NoteCreate.note_text`, `MessageCreate.message`,
  `AppointmentCreate.reason`, `AlertCreate.message`), and appointment
  `date`/`time` are unbounded-format strings.
* **Risk:** Oversized payloads; inconsistent date handling.
* **Recommendation:** Document only — add `max_length` (and a date type) and a
  request body-size limit at the proxy.

**Finding I1**
* Severity → **Low**
* Evidence → `app/schemas/note.py`, `alert.py`, `appointment.py` (no `max_length` on free text)
* Risk → storage-consumption / oversized requests (OWASP API4)
* Affected location → `backend/app/schemas/{note,alert,appointment}.py`
* Why it matters → availability of a shared backend
* Recommended fix → add `max_length` bounds and a body-size limit

## Injection

* **Status:** Implemented
* **Evidence:** 100% SQLAlchemy 2.0 ORM/`select()` in `backend/app` — no raw SQL
  string building; the only `sqlite3` usage is the read-only legacy importer
  (`backend/scripts/migrate_legacy.py`). No `subprocess`, `os.system`, `eval` or
  shell invocation. No filesystem path derived from user input.
* **Finding:** None.
* **Risk:** None identified.
* **Recommendation:** None.

## Cross-Site Scripting (XSS)

* **Status:** Partially implemented
* **Evidence:** React escapes by default; no `dangerouslySetInnerHTML`, `innerHTML`
  or `eval` in `src/`; the YouTube iframe `videoId` comes from static
  `src/data/exercises.ts`, not user input. Reset codes/identity stay in JSX.
* **Finding:** No XSS sink is present, but no Content-Security-Policy is set
  (`app/main.py` defers it deliberately) and the token is in `localStorage`.
* **Risk:** A future injection would yield a usable token; no CSP defence-in-depth.
* **Recommendation:** Document only — add a verified CSP.

**Finding X1**
* Severity → **Medium**
* Evidence → `app/main.py` sets headers but explicitly omits CSP; `vercel.json` adds no CSP
* Risk → reduced defence-in-depth against script injection
* Affected location → `backend/app/main.py`, `vercel.json`
* Why it matters → amplifies the impact of S1
* Recommended fix → author/test a CSP against the real bundle

## Cross-Site Request Forgery (CSRF)

* **Status:** Not applicable / Implemented-by-design
* **Evidence:** The API authenticates with a `Authorization: Bearer` header read
  from `localStorage` (`src/services/api.ts`); there are no cookie-based sessions
  and no `SameSite`-based auth, so a cross-site form cannot attach credentials
  automatically. CORS is an explicit origin allow-list with credentials
  (`app/main.py`).
* **Finding:** CSRF does not apply to the current bearer-token model.
* **Risk:** None in the current design.
* **Recommendation:** Document only — if cookie-based auth is adopted, add CSRF
  protection (anti-CSRF token or `SameSite` cookies).

## API Security

* **Status:** Partially implemented
* **Evidence:** Explicit `response_model` on routes (no `password_hash`/
  `token_version` leakage); consistent error mapping
  (`app/services/errors.py`, `app/main.py`); interactive docs disabled in
  production; CORS restricted; security headers set.
* **Finding:** No global rate limit, no pagination, and no request body-size
  limit; `/reports/generate` performs unbounded work.
* **Risk:** Resource consumption (OWASP API4).
* **Recommendation:** Document only — add a global limiter, pagination and
  body-size limits, and bound report generation to a month.

**Finding API1**
* Severity → **Medium**
* Evidence → `report_service.generate_report` selects every session for the patient (no SQL month filter); `POST /reports/generate` unthrottled; list endpoints unpaginated
* Risk → denial-of-service / resource exhaustion by one authenticated account
* Affected location → `backend/app/services/report_service.py`, `backend/app/api/routes/reports.py`
* Why it matters → shared backend availability
* Recommended fix → SQL month filter, per-patient throttle, pagination

## Rate Limiting / Abuse

* **Status:** Implemented
* **Evidence:** DB-backed `auth_throttles` used across the auth surface
  (`app/services/rate_limit.py`): register 5/h per source, login 30 per source +
  5 per email/15 min, forgot/verify/reset per source. Counters are rows in the
  shared database, so they hold across workers/restarts. Tests in
  `tests/test_rate_limiting.py`.
* **Finding:** Limits cover the unauthenticated auth surface only; authenticated
  write endpoints (reports, booking, messages, alerts) are unlimited.
* **Risk:** Authenticated abuse (report/booking/message spam).
* **Recommendation:** Document only — add per-account limits to authenticated
  write endpoints.

## File Uploads

* **Status:** Not applicable
* **Evidence:** No `UploadFile`/`File`/`Form` parameter or static file mount
  exists in `backend/app`; `python-multipart` is listed but unreachable.
* **Finding:** No upload surface exists, so the upload attack class does not
  apply. The unused dependency is pinned.
* **Risk:** None currently.
* **Recommendation:** Document only — if uploads are added, apply the standard
  controls (allow-list + content sniff, size caps, generated names, out-of-root
  storage, authorized retrieval, malware scan, retention).

## Database Security

* **Status:** Partially implemented
* **Evidence:** SQLAlchemy ORM with parameter binding; FKs with explicit
  `ondelete` (`CASCADE`/`SET NULL`); indexes on FK/`WHERE` columns; unique
  constraints on `accounts.email`, profiles, `patient_doctor_links`,
  `auth_throttles(scope,key)` and `reports(patient_account_id, month_key)`
  (migrations `c4c7db25443c`, `b0411f7bc699`, `c8d1e4a72b90`, `d2f5a3c1e847`);
  credentials from `DATABASE_URL`.
* **Finding:** Migrations now add columns with server defaults and are tested
  against a non-empty database (`tests/test_migrations.py`). Plaintext clinical
  fields are stored without application-level encryption; retention/cleanup jobs
  are absent.
* **Risk:** Exposure if the managed database lacks encryption/backups; unbounded
  growth of throttle/reset tables.
* **Recommendation:** Document only — confirm encryption-at-rest + backups with
  the provider; add cleanup/retention jobs.

**Finding D1**
* Severity → **Medium**
* Evidence → no encryption-at-rest or backup configuration in the repo; managed DB is external
* Risk → provider-side exposure/loss of health data
* Affected location → deployment platform (outside the repo)
* Why it matters → health data requires protected, durable storage
* Recommended fix → verify (do not assume) encryption and backups; test restore

## Secrets

* **Status:** Implemented (current tree)
* **Evidence:** `SECRET_KEY` has no default and is validated at startup
  (`app/core/config.py`, `tests/test_security_config.py`); only `.env.example`
  templates are tracked; `git grep` found no hardcoded credentials;
  `.railway/railway.ts` uses `preserve()` for secrets.
* **Finding:** No hardcoded secret exists, but the legacy `aiphysio.db` remains in
  git history (health data + unsalted SHA-256 digest).
* **Risk:** Historical health-data and credential-digest exposure.
* **Recommendation:** Document only — purge history / treat as disclosed; see
  `secrets.md` S‑1.

**Finding SEC1**
* Severity → **High**
* Evidence → `git rev-list --all --objects` lists `aiphysio.db` (blob `60a37bb…`); added `912e837`/`7b35e9f`, deleted `ea2a631`/`a132eaa`
* Risk → retrievable identifiable health data + crackable digest
* Affected location → git history; working-tree `aiphysio.db`
* Why it matters → high-sensitivity data and a crackable credential
* Recommended fix → rewrite history + force-push, or treat as disclosed and rotate/force resets

## Encryption

* **Status:** Requires verification
* **Evidence:** Passwords and reset codes are bcrypt-hashed
  (`app/core/security.py`). Transport is HTTPS-terminated by the hosting platform
  (`docs/DEPLOYMENT.md`, `/.railway/railway.ts`); HSTS is set in production
  (`app/main.py`). No application-level encryption of stored clinical fields and
  no at-rest encryption configuration are present in the repository.
* **Finding:** Hashing exists; transit encryption is assumed via the platform;
  at-rest encryption is not configured in the repo.
* **Risk:** Unverified protection of stored health data.
* **Recommendation:** Document only — verify platform TLS and at-rest encryption;
  do not claim encryption exists until confirmed.

## Cross-Origin Resource Sharing (CORS)

* **Status:** Implemented
* **Evidence:** `CORSMiddleware` with `allow_origins=settings.cors_origin_list`,
  `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`
  (`app/main.py`). Production startup rejects localhost/wildcard and empty lists
  (`app/core/config.py`, `tests/test_security_config.py`). The IaC writes out the
  exact deployed Vue/React origin.
* **Finding:** Origin allow-list is enforced; method/header allow-lists remain
  wildcard.
* **Risk:** Low — wildcard methods/headers with a fixed origin is common; tighten
  if the surface grows.
* **Recommendation:** Document only — narrow methods/headers where practical.

## Dependency Security

* **Status:** Requires verification
* **Evidence:** Backend pins are exact in `backend/requirements.txt` (`PyJWT`
  upgraded to 2.13.0; `python-multipart` to 0.0.31); a test asserts the patched
  `PyJWT` floor (`tests/test_jwt_security.py`). Frontend has a `package-lock.json`.
* **Finding:** No advisory scan was run as part of this documentation, so CVE
  status is unverified for both ecosystems.
* **Risk:** Unknown.
* **Recommendation:** Document only — run `pip-audit` and `npm audit` in CI and
  track results.

## Logging / Monitoring

* **Status:** Partially implemented
* **Evidence:** Standard `logging`; reset codes/bodies are never logged
  (`app/services/email_service.py`, `tests/test_password_reset_logging.py`); JWT
  `SECRET_KEY` errors never include the value. No structured logging, request IDs,
  access/audit log, redaction policy or monitoring/alerting configuration.
* **Finding:** Secret-safety in logs is good; there is no access audit trail and
  no monitoring.
* **Risk:** No accountability for record access; no operational detection.
* **Recommendation:** Document only — add structured logging with redaction,
  an append-only access log, and monitoring/alerting.

## Admin Security

* **Status:** Not applicable (no admin surface)
* **Evidence:** `role` is constrained to `patient`/`doctor`; registration rejects
  other roles (`tests/test_doctor_verification.py::test_there_are_still_exactly_two_roles`);
  clinician verification is an out-of-band script
  (`backend/scripts/verify_doctor.py`), not an API/UI surface.
* **Finding:** There is no administrator role or admin interface to secure.
* **Risk:** None from an admin surface; the trade-off is that verification depends
  on operational discipline.
* **Recommendation:** Document only — keep verification out of band and control
  script access.

## Deployment / Infrastructure

* **Status:** Partially implemented
* **Evidence:** `/.railway/railway.ts` (Railway IaC: managed PostgreSQL, pre-deploy
  `alembic upgrade head`, `/health` healthcheck, `ENVIRONMENT=production`,
  `ENABLE_API_DOCS=false`, `TRUST_PROXY_HEADERS=true`, `preserve()` for secrets);
  `vercel.json` (Vite SPA); `docs/DEPLOYMENT.md` runbook; `vite.config.ts` aborts a
  production build on a missing/local/placeholder `VITE_API_URL`.
* **Finding:** Fail-fast config and a documented runbook exist. There is no CI
  workflow (`.github/` absent), no Dockerfile (documented as unnecessary), and the
  production configuration currently disables password reset
  (`EMAIL_BACKEND=console` + `ALLOW_PRODUCTION_WITHOUT_EMAIL=true`). The repo uses
  Railway for the backend (visible in the current configuration).
* **Risk:** No automated gate on push; password reset unavailable.
* **Recommendation:** Document only — add CI (tests, type-check, build, dependency
  audit) and configure an SMTP provider.

## Data Privacy

* **Status:** Partially implemented
* **Evidence:** On-device pose processing — no frames or landmarks are uploaded
  (`src/utils/poseEstimator.ts`, `src/components/PoseTracker.tsx`,
  `app/models/session.py`); no third-party analytics/AI; `.gitignore` excludes
  research artefacts and databases; report/diet/notes serve from the backend, not
  browser storage. `metadata.json` carries a stale Gemini capability claim.
* **Finding:** The runtime privacy model is sound; retention policy, consent
  record and a patient-facing export/deletion path are absent; a stale metadata
  claim implies non-existent AI egress.
* **Risk:** Health data retained indefinitely without consent/accountability.
* **Recommendation:** Document only — add a retention/consent policy,
  export/deletion paths, and correct the metadata.

**Finding P1**
* Severity → **Medium**
* Evidence → no retention policy, consent field or export/deletion endpoint in `backend/app`
* Risk → health data retained indefinitely; no data-subject controls
* Affected location → `backend/app/models`, `backend/app/api/routes`
* Why it matters → health-data regimes generally require these controls
* Recommended fix → define retention, add consent and export/deletion paths (documentation only here)

## Error Handling

* **Status:** Implemented
* **Evidence:** Domain errors map to consistent JSON via `ServiceError` handlers
  (`app/services/errors.py`, `app/main.py`); FastAPI runs with `debug=False`; no
  traceback/path leakage; SMTP failures log only the transport error.
* **Finding:** Error handling is uniform and avoids internal-detail disclosure.
* **Risk:** Low — `ApiError` surfaces backend `detail` strings verbatim, which can
  confirm object existence on some paths.
* **Recommendation:** Document only — pair uniform 404s where enumeration matters.

## Backup / Recovery

* **Status:** Requires verification
* **Evidence:** No backup, restore or retention configuration exists in the
  repository; `docs/DEPLOYMENT.md` mentions backups as a pre-deployment
  requirement but no mechanism is implemented or documented as active. Alembic
  provides schema roll-forward (`alembic upgrade head`).
* **Finding:** Schema migration is versioned; data backup/recovery is not
  configured in the repository.
* **Risk:** Potential data loss with no verified restore path.
* **Recommendation:** Document only — confirm and document provider backups,
  retention and a tested restore procedure.

---

## Overall checklist result

| Category | Status |
| --- | --- |
| Authentication | Implemented |
| Authorization / RBAC | Implemented |
| Session / token security | Partially implemented |
| Input validation | Partially implemented |
| Injection | Implemented |
| Cross-Site Scripting | Partially implemented |
| CSRF | Not applicable (bearer-header auth) |
| API security | Partially implemented |
| Rate limiting / abuse | Implemented |
| File uploads | Not applicable |
| Database security | Partially implemented |
| Secrets | Implemented (current tree); historical exposure outstanding |
| Encryption | Requires verification |
| CORS | Implemented |
| Dependency security | Requires verification |
| Logging / monitoring | Partially implemented |
| Admin security | Not applicable |
| Deployment / infrastructure | Partially implemented |
| Data privacy | Partially implemented |
| Error handling | Implemented |
| Backup / recovery | Requires verification |

No recommendation in this document was implemented. This checklist does not
assert that the system is fully secure, that production security is complete, or
that any regulatory or medical compliance exists.
