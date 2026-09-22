/**
 * Railway Infrastructure as Code for the PhysioAI backend.
 *
 * Why this file and not `railway.toml`
 * ------------------------------------
 * Railway deprecated Config as Code (`railway.toml` / `railway.json`): it is no
 * longer read for new services and stops being read entirely on 2026-12-01. The
 * supported mechanism is Infrastructure as Code, evaluated by the Railway CLI:
 *
 *     npm install                 # provides the `railway` SDK used below
 *     npm install -g @railway/cli
 *     railway login
 *     railway link                # pick this project + environment
 *     railway config plan         # preview, changes nothing
 *     railway config apply        # apply after confirming
 *
 * Scope: this single repository holds every service, so one file describes the
 * whole environment and no `partial` export is used.
 *
 * Secrets are never written here. `preserve()` means "keep the value already set
 * on Railway", so the real SECRET_KEY / SMTP password stay in Railway and never
 * reach git.
 */

import { defineRailway, github, group, postgres, preserve, project, service } from 'railway/iac';

/** The deployed FastAPI app, run from the `backend/` directory of this repo. */
const BACKEND_ROOT = 'backend';

export default defineRailway(() => {
  // Managed PostgreSQL. Railway injects DATABASE_URL for the service below.
  const db = postgres('postgres');

  const api = service('api', {
    source: github('mandalayan-1829/physiotherapy-assistant', {
      branch: 'main',
      // The FastAPI package lives here, and so do its requirements.txt and
      // alembic.ini. Setting the root directory means the build, the pre-deploy
      // migration and the start command all run with `backend/` as the working
      // directory. The repository root holds no Python manifest.
      rootDirectory: BACKEND_ROOT,
    }),

    // `$PORT` is provided by Railway; a deployed service must never hardcode
    // 8000, and it must bind 0.0.0.0 so the platform can reach it.
    start: 'uvicorn app.main:app --host 0.0.0.0 --port $PORT',

    // Runs after a successful build and before the new container takes traffic.
    // A failing migration fails the deploy, so schema and code ship together and
    // a partially-migrated database is never served.
    preDeploy: 'alembic upgrade head',

    // The API serves GET /health with no auth and no secrets, so it is a safe
    // liveness probe.
    healthcheck: '/health',
    healthcheckTimeout: 120,

    env: {
      // --- Database ------------------------------------------------------
      DATABASE_URL: db.env.DATABASE_URL,

      // --- Runtime -------------------------------------------------------
      // Selects production behaviour: interactive docs are disabled, localhost
      // CORS origins are rejected, and a real mail transport becomes mandatory.
      ENVIRONMENT: 'production',
      ENABLE_API_DOCS: 'false',
      // Railway terminates TLS at its edge proxy and sets X-Forwarded-For, so
      // the header can be trusted for per-IP throttling. Leaving this false
      // would make every request appear to come from the proxy, collapsing all
      // per-source rate limits into one shared bucket.
      TRUST_PROXY_HEADERS: 'true',
      // Mail transport. No SMTP provider has been chosen yet, so the console
      // transport is selected deliberately together with the explicit opt-out
      // below: the API refuses to start in production without a real transport
      // unless that opt-out is present and true.
      //
      // While this is in place, /auth/forgot-password cannot deliver its code, so
      // PASSWORD RESET DOES NOT WORK. Reset codes are still never logged and
      // never returned to the client. To restore the flow: set the SMTP_*
      // variables below, change this to 'smtp', and drop the opt-out.
      EMAIL_BACKEND: 'console',
      ALLOW_PRODUCTION_WITHOUT_EMAIL: 'true',

      // --- Operator-supplied values (set these on Railway, not in git) ----
      // SECRET_KEY: JWT signing key. REQUIRED - the API refuses to start
      //   without it. Generate with:
      //     python -c "import secrets; print(secrets.token_urlsafe(48))"
      //   Changing it invalidates every issued token.
      SECRET_KEY: preserve(),
      // CORS_ORIGINS: this is the deployed Vercel production origin, written out
      //   rather than preserved because it is public knowledge (it is in the
      //   browser's address bar), the API refuses to start in production unless
      //   it is set, and it must never contain localhost.
      //
      //   Add further origins comma-separated - e.g. a custom domain. Preview
      //   deployments get a unique *.vercel.app hostname each time; because the
      //   API allows credentialed requests, a wildcard is not an option, so add
      //   a specific preview origin temporarily if you need to test one.
      CORS_ORIGINS: 'https://physiotherapy-assistant-seven.vercel.app',
      // SMTP_*: transactional mail used only for password-reset codes.
      SMTP_HOST: preserve(),
      SMTP_PORT: preserve(),
      SMTP_USERNAME: preserve(),
      SMTP_PASSWORD: preserve(),
      SMTP_FROM_EMAIL: preserve(),
      SMTP_USE_TLS: preserve(),
    },
  });

  const backend = group('Backend', [api, db]);

  return project('physiotherapy-assistant', {
    resources: [backend],
  });
});
