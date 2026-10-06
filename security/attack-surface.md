# Attack Surface

> **Scope and method.** This maps the externally reachable surface of the
> repository as it exists at the time of writing (branch `security-documentation`,
> HEAD `a132eaa`, 2026-10-06). Endpoints, roles and inputs are taken from the
> current FastAPI routes and Pydantic schemas; nothing was changed. No exploit
> payloads, private URLs or credentials are included.

## Public Endpoints

Unauthenticated, externally reachable:

| Method | Path | Purpose | Rate limit |
| --- | --- | --- | --- |
| GET | `/health` | Liveness probe; returns service name and version | none |
| POST | `/auth/register` | Create a patient or (pending) doctor account; returns a token | per-source: `register_max_requests_per_hour` (default 5/h) |
| POST | `/auth/login` | Authenticate into the selected portal | per-email failures + per-source: `login_max_requests_per_ip_per_window` (default 30) |
| POST | `/auth/forgot-password` | Start a reset; emails a code | per-email cooldown + hourly cap, per-source cap |
| POST | `/auth/verify-reset-code` | Exchange a code for a short-lived reset token | per-source cap across the flow |
| POST | `/auth/reset-password` | Set a new password and end existing sessions | per-source cap across the flow |

Interactive API documentation (`/docs`, `/redoc`, `/openapi.json`) is served only
when `enable_api_docs` is true **and** the environment is not `production`
(`app/core/config.py`). In the current production configuration
(`ENABLE_API_DOCS=false`), the OpenAPI document is not exposed.

## Authenticated Endpoints

All require `Authorization: Bearer <JWT>` resolving to an active account
(`app/api/deps.py`).

| Method | Path | Roles | Object-level check |
| --- | --- | --- | --- |
| POST | `/auth/logout` | any authenticated | n/a (stateless no-op) |
| GET | `/auth/me` | any authenticated | self |
| GET, PUT | `/users/me/profile` | patient | self only |
| GET | `/users/me/doctor-profile` | any authenticated | self |
| GET | `/doctors` | any authenticated | verified clinicians only |
| GET | `/doctors/{doctor_profile_id}` | any authenticated | verified only; no relationship check |
| GET | `/patients` | doctor | link-scoped roster |
| GET | `/sessions` (`?patient_account_id`) | both | `resolve_patient_scope` |
| POST | `/sessions` | patient (forced to self) | forced to requester |
| GET | `/progress/daily` (`?patient_account_id`) | both | `resolve_patient_scope` |
| GET | `/diet` (`?patient_account_id`) | both | `resolve_patient_scope` |
| POST | `/diet` | patient (forced to self) | forced to requester |
| DELETE | `/diet/{record_id}` | patient | ownership → 404 |
| GET | `/notes` (`?patient_account_id`) | both | `resolve_patient_scope` |
| POST | `/notes` | both | patient→self; doctor→must be linked |
| DELETE | `/notes/{note_id}` | both | authorship check |
| GET | `/appointments` (`?patient_account_id`) | both | patient→own; doctor→own+linked |
| POST | `/appointments` | patient (forced to self) | forced to requester; creates link |
| PATCH | `/appointments/{id}` | both | owner / assigned doctor |
| GET | `/messages` (`?patient_account_id`) | both | `ensure_patient_access` (+ doctor filter) |
| POST | `/messages` | both | patient→self + `ensure_link`; doctor→own id only |
| GET | `/alerts` (`?patient_account_id`) | both | `resolve_patient_scope` |
| POST | `/alerts` | patient (forced to self) | forced to requester |
| GET | `/reports` (`?patient_account_id`) | both | `resolve_patient_scope` |
| GET | `/reports/{report_id}` | both | owning patient resolved from the row, then scope-checked |
| POST | `/reports/generate` | both | `resolve_patient_scope` (doctor must be linked) |

Router prefixes are declared in each `backend/app/api/routes/*.py` and the routers
are mounted in `backend/app/main.py`.

## Authorization / Role-Based Access

* **Two roles only.** `patient` and `doctor`, enforced by a
  `CheckConstraint("role IN ('patient','doctor')")` on `accounts` and by
  `Literal["patient","doctor"]` in the schemas. A `doctor` account represents a
  doctor/physiotherapist.
* **Role comes from the database, not the token.** `get_current_account`
  (`app/api/deps.py`) loads the `Account` by `sub` and rejects inactive accounts
  and stale `token_version`s; the `role` claim in the JWT is advisory only.
  `require_patient` / `require_doctor` gate role-specific routes with 403.
* **Patient permissions.** A patient is forced to their own record:
  `resolve_patient_scope` rejects a `patient_account_id` that is not the
  requester (**403**), and `ensure_patient_access` does the same for non-scoped
  routes.
* **Doctor permissions.** A doctor must supply an explicit `patient_account_id`
  and hold an **active** `patient_doctor_links` row; otherwise **403**. An
  unverified clinician cannot acquire a link or read any records
  (`ensure_doctor_verified`).
* **Object-level authorization.** `DELETE /diet/{id}`, `DELETE /notes/{id}`,
  `PATCH /appointments/{id}`, `POST /notes` and `POST /messages` compare the row's
  owner to the caller. `GET /reports/{id}` resolves the owning patient from the
  row (the path carries no patient id), closing the parameter-tampering path.
* **Frontend route guards are convenience only.** `src/App.tsx` renders patient
  or doctor views by role, but every rule above is enforced server-side.
* **Known gap.** `note_service.list_notes` filters only by `patient_account_id`,
  so a linked clinician can read notes authored by other clinicians (see
  `threat-model.md` TM‑4). `GET /doctors/{id}` performs no relationship check for
  verified profiles (TM‑7).

Authorization behaviour is covered by `backend/tests/test_authorization.py`,
`test_doctor_verification.py` and `test_reports.py`.

## User Inputs

| Input | Fields | Validation |
| --- | --- | --- |
| Login | `email`, `password`, `role` | `EmailStr`; `role` is a `Literal` |
| Registration | `full_name`, `email`, `password`, `role`, `specialization?`, `qualification?`, `license_number?` | `EmailStr`, length bounds; password policy enforced server-side |
| Patient profile | ~35 medical fields | typed; `age` 0–130, `pain_intensity` 0–10, `daily_sitting_hours` 0–24, heights/weights ≥0 |
| Exercise session | `exercise`, `exercise_label`, `reps`, `target_reps`, `form_accuracy`, `duration_sec`, `notes`, `performed_at`, `metrics_source` | `reps` ≤10 000, `form_accuracy` 0–100, `duration_sec` ≤86 400, `notes` ≤2000, `metrics_source` is a `Literal` |
| Diet record | `meal`, `calories`, `protein`, `carbs`, `fats`, `recorded_at` | `meal` length bounds, `calories` 0–20 000, macros ≥0 |
| Note | `patient_account_id?`, `note_text`, `category` | `category` is a `Literal`; `note_text` has a minimum length but **no `max_length`** |
| Appointment | `doctor_profile_id`, `date`, `time`, `reason` | `date`/`time` are bounded **strings** (no date format); `reason` has no `max_length` |
| Message | `doctor_profile_id`, `patient_account_id?`, `message` | `message` has a minimum length but **no `max_length`** |
| Alert | `alert_type`, `message`, `sent_to` | `alert_type` is a `Literal`; `message` has no `max_length` |
| Report generate | `month_key`, `patient_account_id?` | `month_key` matches `^\d{4}-\d{2}$` |
| Password reset | `email`, `verification_code`, `reset_token`, `new_password`, `confirm_password` | format/length bounded; password policy enforced server-side |

Every body is parsed by Pydantic v2 before it reaches a handler.

## File Uploads

**There are no file-upload endpoints.** No route in `backend/app` declares an
`UploadFile`, `File(...)` or `Form(...)` parameter, and there is no static file
mount or storage bucket in the application. The dependency
`python-multipart` is present in `backend/requirements.txt` (FastAPI imports it
when form parsing is used) but is **not reachable** from any current route. If
uploads are added later, the standard controls (extension allow-list plus content
sniffing, size caps, server-generated names, storage outside the web root,
authorized/signed retrieval, malware scanning, retention) would apply.

## Database Access

* **Technology:** SQLite for local development (`DATABASE_URL=sqlite:///./physioai.db`)
  and PostgreSQL for production. `backend/requirements.txt` pins the
  `psycopg[binary]` driver, and `app/db/session.py` applies SQLite-only
  `connect_args` conditionally.
* **Access path:** the backend only, through SQLAlchemy 2.0 (`select()`/ORM);
  there is no raw-SQL string building in `backend/app`. The single `sqlite3` use
  is the read-only legacy importer (`backend/scripts/migrate_legacy.py`).
* **ORM:** SQLAlchemy 2.0.36 with a declarative `Base` and a naming convention
  (`app/db/base.py`).
* **Credentials:** supplied via the `DATABASE_URL` environment variable; not
  present in tracked files.
* **Public exposure:** the application does not expose the database. Whether the
  managed instance is reachable only from the private network is **Potential risk
  requiring verification** (platform configuration, not visible in the repo).

## Storage

* **Frontend storage:** `localStorage['physio_access_token']` (the JWT) and a
  legacy client-side alert log `physio_guardian_alert_log`
  (`src/utils/storage.ts`). Reports are no longer stored client-side — they are
  served by the backend (`src/views/ReportsView.tsx` uses `listReports` /
  `generateReport`). No `sessionStorage` or cookies are used by the application.
* **Backend storage:** the relational database only (the tables listed in
  `backend/app/models`). Reset codes are stored hashed in
  `password_reset_tokens`; throttle counters in `auth_throttles`.
* **Database storage:** accounts, patient/doctor profiles, sessions, diet,
  appointments, notes, messages, alerts, links, reports.
* **Local files:** the development SQLite files `physioai.db` /
  `backend/physioai.db` and the legacy `aiphysio.db` (all git-ignored). The
  MediaPipe WASM runtime is staged into `public/mediapipe/` at install/build time
  and is git-ignored.
* **Generated reports:** JSON aggregates in the `reports` table; no files are
  generated on disk for reports.
* **Research data:** `.gitignore` excludes `data/raw`, `data/processed`,
  `data/exports`, `data/participants`, `results/` and `experiments/**/*.{json,csv}`,
  indicating an intended location for research artefacts; no research pipeline is
  implemented in the current source.
* **Browser storage:** see frontend storage above.

## Webhooks

**There are no webhooks.** No inbound webhook route exists, and the application
does not register outbound webhooks with any provider.

## Third-Party Integrations

| Integration | How it is used | Secret? | Location |
| --- | --- | --- | --- |
| MediaPipe Tasks Vision (WASM/WebGL) | On-device pose estimation; runtime self-hosted from `node_modules` into `public/mediapipe/wasm` | No | `src/utils/poseEstimator.ts`, `scripts/copy-mediapipe-assets.mjs` |
| MediaPipe pose model CDN | Default model URL (version-pinned), overridable to self-host | No | `src/utils/poseEstimator.ts`, `scripts/fetch-pose-model.mjs` |
| SMTP / email | Password-reset code delivery (currently unconfigured) | Yes (`SMTP_*`) | `backend/app/services/email_service.py` |
| Google Fonts | Stylesheet in the page head | No | `index.html` |
| YouTube-nocookie | Exercise demo iframe | No | `src/views/ExerciseSelectionView.tsx` |
| WhatsApp click-to-chat | Patient-initiated "share with guardian" link | No | `src/views/DashboardHome.tsx`, `src/views/MedicalProfileView.tsx` |
| Railway | Backend hosting / managed PostgreSQL (IaC) | Yes (platform vars) | `.railway/railway.ts` |
| Vercel | Frontend static hosting | No | `vercel.json` |

No AI/analytics/payment/maps API is integrated.

## Admin Surfaces

**There is no administrator role and no administration interface.** The `accounts`
table constrains `role` to `patient` or `doctor`, and registration rejects any
other role. Clinician verification is deliberately an **out-of-band operations
action** performed with `backend/scripts/verify_doctor.py --email <address>`
(and `--revoke`), which writes directly to the database — it is not exposed
through the API. This is an operational distinction: the script runs with
server/database access, outside the HTTP surface.

## Network / Deployment Exposure

* **Public frontend:** the React SPA is served as static assets from Vercel
  (`vercel.json` pins framework, build command and output directory; an SPA
  rewrite sends all paths to `index.html`).
* **Public backend:** the FastAPI service is reachable over HTTPS on the hosting
  platform (`.railway/railway.ts` starts `uvicorn app.main:app --host 0.0.0.0
  --port $PORT`). Only the declared routes are reachable.
* **HTTPS:** the application relies on the platform terminating TLS and, in
  production, sets HSTS via middleware (`app/main.py`). The application itself
  does not terminate TLS. Whether TLS is enforced end-to-end is **Potential risk
  requiring verification**.
* **CORS:** `CORSMiddleware` with an explicit origin list from `CORS_ORIGINS`,
  `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`.
  Production startup rejects localhost origins and an empty list. There is no
  wildcard origin.
* **Reverse proxy:** `TRUST_PROXY_HEADERS` defaults to `false`; the client IP used
  for throttling is `request.client.host` unless the operator explicitly enables
  proxy-header trust (production sets it `true` because the platform terminates
  TLS and sets `X-Forwarded-For`). `get_client_ip` documents that trusting the
  header otherwise would let a client reset its own throttle bucket.
* **Health endpoint:** `GET /health` is unauthenticated and returns the service
  name and version (minor fingerprinting). It is used as the platform healthcheck
  (`/.railway/railway.ts`, `docs/DEPLOYMENT.md`).
* **Security headers:** the middleware sets `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`,
  `Permissions-Policy: camera=(self), microphone=(), geolocation=()` and (in
  production) HSTS. No CSP is set.

## Dependencies

Security-relevant dependencies from the current manifests (no CVE is claimed
without evidence):

* **Backend** (`backend/requirements.txt`): `fastapi==0.141.1`,
  `uvicorn[standard]==0.52.4`, `SQLAlchemy==2.0.36`, `alembic==1.14.0`,
  `psycopg[binary]==3.3.6`, `pydantic==2.13.5`, `pydantic-settings==2.15.0`,
  `email-validator==2.2.0`, `bcrypt==4.2.1`, `PyJWT==2.13.0`,
  `python-multipart==0.0.31`; test-only `pytest`, `httpx`, `packaging`. The
  manifest comments note that `PyJWT` was upgraded from 2.10.1 (advisories on the
  token-decode path) and `python-multipart` from 0.0.20 (advisories on parsing,
  currently unreachable). A test asserts the pinned `PyJWT` is at least the
  patched version (`backend/tests/test_jwt_security.py`).
* **Frontend** (`package.json`): `react`/`react-dom` ^19.3.0,
  `@mediapipe/tasks-vision` ^1.0.1, `lucide-react` ^1.16.0, `canvas-confetti`
  ^1.9.4; build tooling `vite` ^6.4.3, `typescript` ^5.7.0, `@vitejs/plugin-react`,
  `tailwindcss`/`@tailwindcss/vite` ^4.0.0, and a `railway` devDependency used by
  the IaC file. A `package-lock.json` is present.

The frontend dependency tree is pinned by `package-lock.json`; no advisory scan
was run as part of this documentation, so **dependency CVE status is
unverified**.

## Summary of the surface

The reachable surface is a small REST API behind a JWT: six unauthenticated
endpoints, the rest gated by `get_current_account` plus role/object checks; no
file uploads, no webhooks, no admin interface, no server-side AI endpoints; and
on-device pose processing that never transmits frames. The most notable surface
concerns are the account-enumeration behaviour of registration, unbounded
report generation and free-text fields, the breadth of the doctor directory, and
the deliberate absence of a CSP.
