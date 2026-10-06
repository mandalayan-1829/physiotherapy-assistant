# Secrets

> **Scope and method.** This is an assessment of the repository as it exists at
> the time of writing (branch `security-documentation`, HEAD `a132eaa`,
> 2026-10-06). It is based on the current source, the two `.env.example`
> templates, `.gitignore`, the deployment configuration and a git-history scan.
>
> **No secret value is reproduced anywhere in this document.** Only variable
> names, file locations and secret *types* are identified. No secret was added,
> changed or rotated.

## Where Secrets Are Expected

Secrets exist only as **backend/runtime environment variables** — never in the
frontend bundle and never in tracked files.

| Secret (type) | Variable name(s) | Where it is read | Notes |
| --- | --- | --- | --- |
| JWT signing key | `SECRET_KEY` | `backend/app/core/config.py`, used by `backend/app/core/security.py` | Required; no default; validated at startup |
| Database credentials | `DATABASE_URL` | `backend/app/db/session.py`, `backend/alembic/env.py` | May embed a username/password for PostgreSQL |
| SMTP password | `SMTP_PASSWORD` | `backend/app/services/email_service.py` | Optional; needed only when `EMAIL_BACKEND=smtp` |
| SMTP username | `SMTP_USERNAME` | `backend/app/services/email_service.py` | Optional / provider-specific |
| Legacy DB location | `LEGACY_DATABASE_PATH` | `backend/scripts/migrate_legacy.py` | A path, not a secret, but points at sensitive data |

Deployment-side handling: `.railway/railway.ts` uses `preserve()` for
`SECRET_KEY` and all `SMTP_*` variables, which means "keep the value already set
on the platform" — so no real secret is ever written into the repository. The
file also writes out non-secret values (`ENVIRONMENT`, `ENABLE_API_DOCS`,
`TRUST_PROXY_HEADERS`, `EMAIL_BACKEND`, `ALLOW_PRODUCTION_WITHOUT_EMAIL`,
`CORS_ORIGINS`).

No third-party API keys (payment, analytics, AI, maps, etc.) exist in the
project. `metadata.json` advertises `MAJOR_CAPABILITY_SERVER_SIDE_GEMINI_API`,
but there is **no** Gemini/Google GenAI SDK, import, endpoint or key anywhere in
the code — it is a stale declaration, not a secret.

## Environment / Configuration Handling

* **`.env` handling.** `app/core/config.py` uses `pydantic-settings`
  (`SettingsConfigDict(env_file=".env", extra="ignore")`), so the backend reads a
  local `backend/.env` in development and process/host environment variables
  otherwise. No `.env` file is tracked (only `.env.example`).
* **`.env.example` handling.** There are exactly two templates, both tracked:
  `.env.example` (frontend) at the repository root and `backend/.env.example`.
  Both are marked `[TEMPLATE]` and contain placeholders only. The frontend
  template explicitly documents that `VITE_*` values are public. The backend
  template documents that `SECRET_KEY` has no default and shows how to generate
  one, and that `ALLOW_LEGACY_PASSWORD_LOGIN` must stay off outside a migration.
* **`VITE_*` behavior.** Only `VITE_*` variables reach the browser. The frontend
  reads `VITE_API_URL`, `VITE_POSE_WASM_PATH` and `VITE_POSE_MODEL_URL`
  (`src/services/api.ts`, `src/utils/poseEstimator.ts`). These are **public
  configuration**, not secrets. `vite.config.ts` inlines them at build time.
* **Backend environment variables.** All server-side settings — including
  `SECRET_KEY`, `DATABASE_URL`, `CORS_ORIGINS`, `JWT_*`, threshold/reset settings,
  `EMAIL_*`/`SMTP_*`, `TRUST_PROXY_HEADERS` and `ALLOW_*` flags — are defined in
  `app/core/config.py` and documented in `backend/.env.example`.
* **Production configuration.** `ENVIRONMENT=production` triggers additional
  validation: a strong `SECRET_KEY`, no localhost/wildcard CORS, a real mail
  transport (or the explicit `ALLOW_PRODUCTION_WITHOUT_EMAIL` opt-out) and
  disabled interactive API docs. The current `.railway/railway.ts` selects
  `ENVIRONMENT=production`, `ENABLE_API_DOCS=false`, `TRUST_PROXY_HEADERS=true`,
  `EMAIL_BACKEND=console` and `ALLOW_PRODUCTION_WITHOUT_EMAIL=true`.

### Secrets vs. public browser configuration

| Value | Public to the browser? | Secret? |
| --- | --- | --- |
| `VITE_API_URL` | Yes (inlined into the JS bundle) | No |
| `VITE_POSE_WASM_PATH` | Yes | No |
| `VITE_POSE_MODEL_URL` | Yes | No |
| `SECRET_KEY` | No (backend only) | **Yes** |
| `DATABASE_URL` | No | **Yes** (may contain credentials) |
| `SMTP_PASSWORD` / `SMTP_USERNAME` | No | **Yes** |
| `CORS_ORIGINS`, `ENVIRONMENT`, throttle/reset settings | No | No (configuration) |

## Hardcoded-Secret Risks

A scan of the working tree for credentials (`git grep` across `*.py`, `*.ts`,
`*.tsx`, `*.json`, `*.env*`, excluding the lockfile) found **no hardcoded
secrets** in the current tree. Specifically:

* `SECRET_KEY` has **no default value** in `app/core/config.py`; the API refuses
  to start without a valid one.
* No `.env` file is tracked; only the two `.env.example` templates.
* `backend/.env.example` contains obvious placeholders (`SMTP_HOST=smtp.example.com`,
  `SMTP_USERNAME=your-smtp-username`, `SMTP_PASSWORD=your-smtp-password`) — these
  are documentation placeholders, not credentials.
* No private keys, tokens or connection strings appear in tracked files.

**No hardcoded secret was found in the current working tree.**

## Client-Side Exposure Risks

* `VITE_*` variables are inlined into the bundle by design. The only ones used
  are a backend URL and two asset paths — none is a secret. `vite.config.ts`
  comments explicitly warn against putting a secret behind a `VITE_` prefix.
* **`localStorage`** holds the JWT under key `physio_access_token`
  (`src/services/api.ts`). This is a credential-at-rest exposure vector (readable
  by any script in origin) rather than a secret leak, and it is documented as a
  known gap in `docs/DEPLOYMENT.md` §6. A legacy, non-server alert log is also
  kept under `physio_guardian_alert_log` (`src/utils/storage.ts`).
* `GET /auth/me` and other responses are filtered by explicit Pydantic response
  models (`app/schemas`); `password_hash`, `legacy_password_hash` and
  `token_version` are never serialized.
* No secrets are placed in API responses. Password-reset responses are generic
  and never return the code or the short-lived reset token except to the caller
  who verified a code.
* No third-party scripts, analytics or error-reporting SDKs are present that
  could exfiltrate browser data.

## Git / History Exposure Risks

* **`.env` files:** a history scan shows `.env` was **never** committed to this
  repository — only `.env.example`.
* **`aiphysio.db` (HIGH):** the legacy SQLite database was committed in commits
  `912e837` and `7b35e9f` and removed from tracking in `ea2a631` and `a132eaa`.
  It is absent from HEAD and now covered by `.gitignore` (`*.db`, `aiphysio.db`),
  **but the blob remains reachable in git history** (`git rev-list --all
  --objects` lists object `60a37bb…` for `aiphysio.db`).
  * Type of sensitive material exposed: identifiable patient health information
    (a `users` row with demographics, medical history and pain profile,
    emergency/guardian contacts), directory clinician records, and an **unsalted
    SHA-256 password digest** in the legacy `users` table.
  * Where: repository git history (all clones/forks inherit it); a copy also
    exists in the working tree, untracked.
  * Does it remain in history: **yes.**
  * Security implication: disclosure of health data plus a crackable credential
    digest. Treat as disclosed if the repository is or was ever public.
* The full value is **not** reproduced here.

No other secret material was found in history.

## Logging / Error Exposure Risks

* **Reset codes are never logged.** `app/services/email_service.py` documents and
  enforces that message bodies (which contain the reset code) are never logged by
  any transport. The `console` transport records the message in memory and logs
  only a warning that no mail was sent, withholding the body.
  `tests/test_password_reset_logging.py` asserts this.
* **Passwords and tokens are never logged.** No log statement writes a password,
  reset token, JWT or `SECRET_KEY`; `validate_secret_key` errors never include the
  offending value (`tests/test_security_config.py` asserts this).
* **PII in logs (informational).** `password_reset_service._send_code_email`
  logs the recipient email on successful delivery and a transport-level error on
  SMTP failure; `logger.info` also records a completed reset with the account id.
  These are operational signals, not credentials, but they do place an email
  address and account id in logs.
* **Database credentials in errors.** SMTP failures log `str(exc)`, which is the
  transport error, not a connection string. With `debug=False`, unhandled
  exceptions produce a plain 500 with no traceback or path leakage. Production
  was not inspected at runtime, so this is **Potential risk requiring
  verification** for the deployed environment.

## Secret Rotation Requirements

| Secret | If exposed, rotate? | Impact of rotation |
| --- | --- | --- |
| `SECRET_KEY` | Yes | Invalidates **every** issued access and reset token — all users must sign in again. Must be a random value ≥32 chars, not a placeholder. |
| `SMTP_PASSWORD` / `SMTP_USERNAME` | Yes | The mail provider may lock the credential; password-reset delivery stops until updated. |
| `DATABASE_URL` credentials | Yes | Requires a coordinated rotation on the managed database and the service. |
| Legacy SHA-256 digests in `aiphysio.db` | Yes (per affected account) | Unsaltsed SHA-256 cannot be "rotated" as a key; the only remediation is to force a password reset for affected accounts once/if they are migrated. |

Because a password change increments `Account.token_version`, resetting one
account's password already invalidates that account's previously issued tokens.
Rotating `SECRET_KEY` invalidates all tokens globally.

## Actual Findings

### S‑1 — Legacy database with health data and an unsalted digest in git history

* **Severity:** High
* **Classification:** Confirmed historical exposure
* **Evidence:** `git log --all --diff-filter=AD -- aiphysio.db` (added `912e837`,
  `7b35e9f`; deleted `ea2a631`, `a132eaa`); `git rev-list --all --objects` still
  lists the `aiphysio.db` blob. The file is untracked and git-ignored now but
  present on disk.
* **Risk:** Identifiable health data and a crackable credential digest are
  retrievable from history.
* **Affected location:** Git history; working-tree `aiphysio.db`.
* **Why it matters:** Health data is high-sensitivity, and unsalted SHA-256 is
  offline-crackable.
* **Recommended fix:** Purge the blob from history (history rewrite + force-push)
  and, if that is not possible, treat the data as disclosed and force password
  resets; keep legacy data outside the repository.

### S‑2 — Bearer token stored in `localStorage`

* **Severity:** Medium
* **Classification:** Security weakness (credential-at-rest in the browser)
* **Evidence:** `src/services/api.ts` (`TOKEN_KEY = 'physio_access_token'`);
  `docs/DEPLOYMENT.md` §6.
* **Risk:** Any script executing in origin can read and replay the token.
* **Affected location:** `src/services/api.ts`.
* **Why it matters:** It is the only session credential, valid for the token
  lifetime (default 7 days).
* **Recommended fix:** httpOnly/Secure/SameSite cookies with short-lived access
  tokens and a refresh flow.

### S‑3 — No `.env` was ever committed; no hardcoded secret found

* **Severity:** Informational
* **Classification:** Verified control (no exposure)
* **Evidence:** history scan for `*.env`; working-tree `git grep` for credential
  patterns returned only documentation/identifier matches.
* **Risk:** None identified in the current tree.
* **Affected location:** n/a.
* **Why it matters:** Confirms the primary secret-hygiene property holds.
* **Recommended fix:** Keep the `.env.example`-only policy; add a CI secret scan
  to preserve it.

### S‑4 — PII may reach application logs

* **Severity:** Low
* **Classification:** Security weakness
* **Evidence:** `password_reset_service.py` logs recipient email/account id;
  `email_service.py` logs the SMTP exception string.
* **Risk:** Email addresses and account ids accumulate in logs; log aggregation
  broadens exposure.
* **Affected location:** `backend/app/services/password_reset_service.py`,
  `backend/app/services/email_service.py`.
* **Why it matters:** Logs are frequently copied and widely readable.
* **Recommended fix:** Adopt structured logging with PII redaction and a
  retention/access policy for logs.

### S‑5 — Production mail transport and secrets are currently unset

* **Severity:** Low
* **Classification:** Confirmed configuration state
* **Evidence:** `.railway/railway.ts` sets `EMAIL_BACKEND=console` and
  `ALLOW_PRODUCTION_WITHOUT_EMAIL=true`, with `SMTP_*` and `SECRET_KEY` expected
  to be supplied via `preserve()`.
* **Risk:** Password reset cannot complete until SMTP is configured; if
  `SECRET_KEY` were missing the API would refuse to start (by design), so this is
  a functionality/operability note rather than a leak.
* **Affected location:** `.railway/railway.ts`, `backend/.env.example`.
* **Why it matters:** Users cannot reset passwords; operators must remember to
  set the secrets.
* **Recommended fix:** Configure one SMTP provider, set `EMAIL_BACKEND=smtp`,
  remove the opt-out, and confirm `SECRET_KEY` is set on the platform.

### S‑6 — Stale capability declaration in `metadata.json`

* **Severity:** Informational
* **Classification:** Documentation/consistency issue
* **Evidence:** `metadata.json` declares
  `MAJOR_CAPABILITY_SERVER_SIDE_GEMINI_API`; no Gemini SDK, import or key exists.
* **Risk:** Misleading metadata could imply a third-party integration (and a
  secret) that does not exist.
* **Affected location:** `metadata.json`.
* **Why it matters:** Avoids implying data egress or a key that is not present.
* **Recommended fix:** Remove the claim or correct it.

## Summary

There are **no hardcoded secrets** and **no committed `.env` files** in the
current tree; `SECRET_KEY` has no default and is validated at startup; and reset
codes are never logged. The one confirmed exposure is historical: the legacy
`aiphysio.db` remains reachable in git history and must be treated as disclosed
(**S‑1**). Rotation guidance is above. This document does not claim that all
secrets are safe, that encryption exists, or that monitoring exists — those were
not verified.
