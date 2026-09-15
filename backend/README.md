# PhysioAI Backend (FastAPI)

Server-side foundation for the PhysioAI physiotherapy platform: authentication,
a persistent relational database, and role-based authorization for exactly two
roles — **patient** and **doctor / physiotherapist**.

## Layout

```
backend/
  app/
    main.py            FastAPI app, CORS, error mapping
    core/              settings + password/JWT security
    db/                declarative base, engine/session, table bootstrap
    models/            SQLAlchemy models (accounts, profiles, sessions, ...)
    schemas/           Pydantic request/response models
    services/          business logic + authorization helpers
    api/
      deps.py          auth dependencies and role guards
      routes/          auth, users, doctors, sessions, progress, diet,
                       appointments, notes, reports, communication
  alembic/             migration environment + initial schema revision
  scripts/
    migrate_legacy.py  one-way import from the old aiphysio.db
  tests/               pytest suite
  requirements.txt
  .env.example
```

## Run locally

```bash
cd backend
python -m venv ../.venv                 # or reuse an existing environment
../.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source ../.venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env                    # then set SECRET_KEY
python -m uvicorn app.main:app --reload --port 8000
```

* Health check: `GET http://localhost:8000/health`
* Interactive docs: `http://localhost:8000/docs`

Tables are created automatically on startup. Point `DATABASE_URL` at
PostgreSQL for production — no code changes are required.

## Import existing data (optional)

The previous architecture's database (`../aiphysio.db`) is **never modified**.
To carry its rows into the new schema:

```bash
cd backend
python scripts/migrate_legacy.py
```

Migrated: the existing patient account (its legacy password digest is kept so
the owner can sign in once, after which the password is re-hashed with bcrypt),
the two clinician directory profiles, appointments, diet records and any
sessions/notes that can be resolved.

## Migrations

```bash
cd backend
python -m alembic upgrade head
python -m alembic revision --autogenerate -m "describe change"
```

## Tests

```bash
cd backend
python -m pytest
```

Covers health, registration, duplicate email rejection, password hashing,
valid/invalid login, both roles, unauthenticated rejection, patient data
isolation and doctor authorization.

## Password reset (forgot password)

```
POST /auth/forgot-password    { email, role? }  -> generic message
POST /auth/verify-reset-code  { email, verification_code, role? } -> reset_token
POST /auth/reset-password     { reset_token, new_password, confirm_password }
```

* The 6-digit code is generated server-side with `secrets` (CSPRNG), stored
  only as a bcrypt hash, expires after `PASSWORD_RESET_CODE_TTL_MINUTES`
  (default 10) and is single-use.
* A resend always supersedes the previous code; requests are rate-limited by a
  60 second cooldown plus an hourly cap.
* Codes that fail `PASSWORD_RESET_MAX_ATTEMPTS` (default 5) times invalidate the
  whole request.
* Unknown addresses, addresses in the wrong portal, and existing accounts all
  receive the **identical** response so neither account existence nor role is
  disclosed.
* A successful reset increments `accounts.token_version`, which invalidates
  every access token issued before the change.
* Failed sign-ins are counted server-side. Attempts 1-5 return
  `Invalid email or password` together with `failed_attempts` and
  `show_forgot_password`; after the fifth failure the address is rate-limited
  (HTTP 429).

### Configuring email delivery

Until SMTP is configured, the **console** transport is used. It records the
message server-side and logs a clear `EMAIL NOT SENT` warning (with the message
body, so the flow can be completed locally) — it never reports success.

For real delivery set these in `backend/.env`:

```
EMAIL_BACKEND=smtp
SMTP_HOST=smtp.your-provider.com
SMTP_PORT=587
SMTP_USERNAME=your-smtp-username
SMTP_PASSWORD=your-smtp-password
SMTP_FROM_EMAIL=no-reply@your-domain.com
SMTP_FROM_NAME=PhysioAI
SMTP_USE_TLS=true
```

Any SMTP provider works (SendGrid, SES, Postmark, Mailgun, a corporate relay,
…). `EMAIL_BACKEND=auto` (the default) selects SMTP automatically once
`SMTP_HOST` is set. Never commit real credentials.

## Roles and authorization

* `POST /auth/register` accepts `role: "patient"` or `role: "doctor"` only.
* A patient can only read and write their own records.
* A doctor can only read patients they hold an active link with. Links are
  created when a patient books an appointment with that clinician.
* Authorization is enforced on the server; hiding UI routes is not relied upon.
