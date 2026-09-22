# Deployment runbook

**Status: nothing in this document has been executed.** No Railway account, no
Vercel project and no PostgreSQL instance were touched while writing it, because
the credentials for them live with the operator. Everything below is the exact
procedure to run, plus the commands to verify each step.

Target architecture:

```
Vercel (Vite SPA)  --HTTPS-->  Railway (FastAPI)  -->  Railway PostgreSQL
```

---

## 0. What must be true before deploying

| Requirement | Enforced by |
| --- | --- |
| `SECRET_KEY` is a random value, >= 32 chars, not a placeholder | `backend/app/core/config.py` raises `ConfigurationError` at import |
| `CORS_ORIGINS` is the real frontend origin and contains no `localhost` | same validator, production branch |
| `EMAIL_BACKEND=smtp` with a reachable `SMTP_HOST`, **or** a deliberate `ALLOW_PRODUCTION_WITHOUT_EMAIL=true` | same validator |
| The frontend build knows the backend URL | `vite.config.ts` aborts the build if `VITE_API_URL` is missing, local, **or a documentation placeholder** |

All four are fail-fast: a misconfigured deploy does not start rather than
starting insecurely. That is deliberate, so expect a *crash* rather than a partly
working service if a variable is wrong.

---

## 1. Repository layout the platform must understand

| Path | Role |
| --- | --- |
| `/` | The Vite frontend. `package.json`, `src/`, `index.html`, `vercel.json`. |
| `/backend` | The FastAPI app. `app/main.py`, `requirements.txt`, `alembic/`. |

There is exactly **one** Python manifest in the repository
(`backend/requirements.txt`) and exactly **one** JavaScript manifest
(`package.json`). The retired Streamlit prototype - `core/`, `dashboard/app.py`,
`tests/camerapose_test.py` and a root `requirements.txt` that pinned Streamlit,
OpenCV and MediaPipe - was removed, so a platform cannot pick up the wrong
application by accident.

The Railway service still sets `rootDirectory: "backend"`, because that is where
the FastAPI package lives. If you configure Railway by hand instead, set the
service's Root Directory to `backend` in Settings.

---

## 2. Railway backend

### 2.1 Provision

Railway's Config as Code (`railway.toml`) is **deprecated** and is not read for new
services. This repository uses Infrastructure as Code instead. The definition lives
in [`../.railway/railway.ts`](../.railway/railway.ts) and declares the `api` service,
a managed PostgreSQL resource, the start command, the pre-deploy migration and the
healthcheck path.

```bash
# from the repository root
npm install                      # provides the `railway` SDK used by .railway/railway.ts
npm install -g @railway/cli
railway login
railway link                     # select the project + environment to manage

railway config plan              # preview only; changes nothing
railway config apply             # apply after confirming
```

### 2.2 Then set the variables the file deliberately does not carry

`railway.ts` uses `preserve()` for these, which means "keep the value already on
Railway" - no secret is ever written into the repository. Set them once, in the
Railway dashboard or with `railway variables --set`:

| Variable | Required | Notes |
| --- | --- | --- |
| `SECRET_KEY` | **yes** | `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Changing it logs every user out. |
| `SMTP_HOST` | yes\* | Any standards-compliant provider (Resend, Brevo, SendGrid, Postmark, SES, corporate relay). \*Not required while the opt-out in 2.2.1 is in place, but password reset stays broken until it is set. |
| `SMTP_PORT` | yes | `587` with STARTTLS, or `465` for implicit TLS. |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | yes | Provider credentials. |
| `SMTP_FROM_EMAIL` | yes | Must be a sender the provider has authorised. |
| `SMTP_USE_TLS` | no | Defaults to `true`. |

`DATABASE_URL`, `ENVIRONMENT`, `ENABLE_API_DOCS`, `TRUST_PROXY_HEADERS`,
`EMAIL_BACKEND`, `ALLOW_PRODUCTION_WITHOUT_EMAIL` and **`CORS_ORIGINS`** are set by
`.railway/railway.ts`; `DATABASE_URL` is injected from the managed PostgreSQL
resource.

### 2.2.1 Deploying without a mail provider (current interim state)

The `SMTP_*` rows above are **not** in effect yet. No provider has been chosen, so
`railway.ts` selects the `console` transport together with
`ALLOW_PRODUCTION_WITHOUT_EMAIL=true`. The API refuses to start in production
without one or the other, so the opt-out has to be explicit - it can never happen
by forgetting to set a variable.

What this means in practice:

* Registration, login, sessions, reports and every other flow work normally.
* **`/auth/forgot-password` does not work.** The code is generated but cannot be
delivered, so no user can complete a reset.
* Reset codes are still never written to logs and never returned by the API. The
  console transport refuses to log the message body, so the opt-out does not leak
  credentials - it only removes a capability.
* The flag is narrow: it does not relax `SECRET_KEY`, `CORS_ORIGINS` or the JWT
  algorithm check. A startup `WARNING` records that the flow is unavailable.

To restore password reset: set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`,
`SMTP_PASSWORD`, `SMTP_FROM_EMAIL`, switch `EMAIL_BACKEND` to `smtp`, and remove
`ALLOW_PRODUCTION_WITHOUT_EMAIL`.

`CORS_ORIGINS` is written out literally in `railway.ts` as
`https://physiotherapy-assistant-seven.vercel.app`, because it is public knowledge
and the API refuses to start in production without it. Add further origins
comma-separated (a custom domain, a temporary preview hostname). It must never
contain `localhost` and must never be `*`: the API allows credentialed requests,
and the browser rejects a wildcard alongside credentials anyway.

> There is no `FRONTEND_URL` setting. The backend does not build links back to
the frontend, so setting one would do nothing.

The complete variable list, including the non-deployment defaults, is in
[`backend/.env.example`](../backend/.env.example).

### 2.3 Effective production settings

| Setting | Value | Why |
| --- | --- | --- |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` | Railway injects `$PORT`; hardcoding 8000 would make the service unreachable. |
| Pre-deploy command | `alembic upgrade head` | Migrations run against the real database before the new container takes traffic. A failure fails the deploy. |
| Healthcheck | `/health` | Unauthenticated, lightweight, leaks nothing. |
| Healthcheck timeout | 120 s | Allows for a cold start plus migrations. |
| Restart policy | Platform default (`ON_FAILURE`) | No custom restart loop is implemented or wanted. |

### 2.4 Why a Dockerfile is not used

Railway's standard Python build already detects `backend/requirements.txt` and the
pre-deploy/healthcheck hooks cover the parts of a Dockerfile that would matter
here. A Dockerfile would add a second place to keep the Python version and system
packages in sync with no benefit. This can be revisited if a native dependency ever
becomes necessary.

### 2.5 Verify

```bash
curl -sS https://<api-host>/health
# {"status":"ok","service":"PhysioAI API","version":"0.2.0"}
```

Then confirm migrations actually applied, rather than assuming:

```bash
railway run alembic current      # should print the head revision
railway run alembic heads        # should match it
```

---

## 3. Vercel frontend

1. Import `mandalayan-1829/physiotherapy-assistant`. Vercel detects Vite;
   [`vercel.json`](../vercel.json) pins the framework, build command and output
   directory so detection cannot drift.
2. Project -> Settings -> Environment Variables, then add for **Production**:

   | Name | Value |
   | --- | --- |
   | `VITE_API_URL` | the **actual** Railway HTTPS origin from step 2.5, e.g. `https://<api-service>.up.railway.app` - no trailing slash, no path |

3. **Deploy again.** `VITE_*` values are inlined into the bundle *at build time*;
   changing the variable does nothing to an existing deployment. Vercel only
   picks it up when it runs a new build.

`VITE_API_URL` is **public** - it is inlined into the browser bundle. Never put a
secret behind a `VITE_` prefix.

### 3.1 Why the deployed site said "Cannot reach the PhysioAI backend at https://your-fastapi-backend-url.com"

That message is the client's own connection-error text, and the URL in it is the
inline `VITE_API_URL`. The placeholder was **not** in the source or in git - it
was set as the Vercel Production environment variable and baked into the bundle
at build time. Confirm it in the deployed artifact at any time:

```bash
curl -sS https://<web-host>/ | grep -oE '/assets/index-[A-Za-z0-9_-]+\.js'
curl -sS https://<web-host>/assets/index-XXXX.js | grep -o 'https://[a-z0-9.-]*backend[a-z0-9.-]*'
```

Three guards make this class of outage impossible from now on:

| Guard | Where | Catches |
| --- | --- | --- |
| Build aborts on a missing URL | `vite.config.ts` | forgot to set the variable |
| Build aborts on a local URL | `vite.config.ts` | `localhost` copy-pasted into production |
| Build aborts on a placeholder | `vite.config.ts` + `src/config/apiUrl.ts` | `your-fastapi-backend-url.com`, `api.example.com`, `example.up.railway.app`, ... |

`src/services/api.ts` repeats the placeholder check at runtime as defence in
depth for a bundle built outside `npm run build`.

If a deploy fails with *"VITE_API_URL is set to a placeholder"*, that is the
guard working: set the real Railway origin in Vercel and redeploy. Nothing on
localhost can be reached from a visitor's browser, so there is deliberately no
fallback.

> **CORS and preview deployments:** preview builds get unique `*.vercel.app`
> hostnames. Because the API enables credentialed CORS, a wildcard origin is not
> allowed, so previews will be blocked unless every preview hostname is listed in
> `CORS_ORIGINS`. Simplest fix for now: test against Production, or add the
> specific preview origin temporarily.

---

## 4. Post-deploy smoke test

Run in order; stop at the first failure.

```bash
API=https://<api-host>
WEB=https://<web-host>
```

| # | Check | Command / action | Expected |
| --- | --- | --- | --- |
| 1 | Health over HTTPS | `curl -sS $API/health` | `200`, JSON body, no secrets |
| 2 | Interactive docs disabled | `curl -sS -o /dev/null -w '%{http_code}' $API/docs` | `404` |
| 3 | CORS allows the frontend | `curl -sS -D- -o /dev/null -X OPTIONS $API/auth/login -H "Origin: $WEB" -H "Access-Control-Request-Method: POST" \| grep -i access-control-allow-origin` | the exact `$WEB` origin, not `*` |
| 4 | CORS rejects a foreign origin | same, with `-H "Origin: https://evil.example"` | no `access-control-allow-origin` header |
| 5 | Registration | register a throwaway patient in the UI | account created, signed in |
| 6 | Authenticated read | reload the dashboard | profile loads from `GET /auth/me`, not from browser storage |
| 7 | Camera pose tracking | open an exercise, choose *Camera analysis* | pose skeleton tracks; no "tracking unavailable" banner |
| 8 | Session persistence | complete a set | row appears in `GET /sessions` (verify in the UI history view) |
| 9 | Report generation | generate a monthly report | report served by `GET /reports`, not from localStorage |
| 10 | Restart resilience | `railway redeploy`, then `curl $API/health` | back to `200` without manual intervention |

Step 7 is the one that exercises the self-hosted MediaPipe runtime. If it fails
with "The pose-estimation runtime could not be initialised", confirm the deploy
actually contains the assets:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' $WEB/mediapipe/wasm/vision_wasm_internal.js   # expect 200
```

---

## 5. Local development

The backend and frontend are independent. Use the current interpreter directly -
no virtual environment is required, and any pre-existing `.venv` in this repository
predates the current dependency manifest and must not be reused.

```bash
# --- backend ---
cd backend
python -m pip install -r requirements.txt   # or use a fresh env: python -m venv .venv
cp .env.example .env                 # then set SECRET_KEY (>= 32 chars)
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`alembic` and `uvicorn` are run as `python -m ...` so the command works whether
or not a virtual environment is activated, and regardless of `PATH`.

```bash
# --- frontend (separate terminal, from the repository root) ---
npm install                          # also copies the MediaPipe WASM runtime
# .env: VITE_API_URL=http://localhost:8000
npm run dev                          # http://localhost:3000
```

`npm install`, `npm run dev` and `npm run build` all run
`scripts/copy-mediapipe-assets.mjs`, which stages the WASM runtime from
`node_modules` into `public/mediapipe/wasm`. It is git-ignored because it is
reproducible from `package-lock.json`.

### Tests

```bash
cd backend && python -m pytest -q     # backend suite
```

There is currently **no frontend test runner**; `npm run lint` runs `tsc --noEmit`.

---

## 6. Known gaps

These are unresolved and are **not** claimed as done:

1. **Access tokens are stored in `localStorage`.** Mitigated by a short-ish token
   lifetime and server-side `token_version` invalidation, but a successful XSS
   still yields a usable token. Migrating to httpOnly cookies with a refresh-token
   flow is the correct fix and has not been done.
2. **No frontend test suite.** The pose engine and API mappers have no automated
   coverage.
3. **No research data model.** Sessions store aggregate totals only; per-repetition
   records, ground truth, experiment metadata, export and the evaluation pipeline
   are not implemented.
4. **Reports are generated by the backend but the UI still reads/writes drafts in
   `localStorage`** (`src/utils/storage.ts`, marked `TODO(reports)`).
5. **No mail provider is configured, so password reset is unavailable.** The
   production service currently starts via the explicit
   `ALLOW_PRODUCTION_WITHOUT_EMAIL=true` opt-out (see 2.2.1). Choosing an SMTP
   provider closes this gap.
