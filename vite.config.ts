import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * The frontend talks to the FastAPI backend through `VITE_API_URL`.
 *
 * Two failure modes are worth designing against, because both are silent:
 *
 *   1. An unset `VITE_API_URL` in a production build used to fall back to
 *      `http://localhost:8000`. The bundle built fine, deployed fine, and then
 *      failed for every real user with a connection error - and worse, would
 *      happily send credentials to whatever runs on the visitor's own machine.
 *   2. A `VITE_*` name is *inlined into the browser bundle*. Putting a secret
 *      there publishes it.
 *   3. A *placeholder* URL passes every check that only rejects empty/local
 *      values. `VITE_API_URL=https://your-fastapi-backend-url.com` builds,
 *      deploys and serves, and then fails for every visitor with "Cannot reach
 *      the PhysioAI backend" - the exact production outage this file now stops.
 *
 * So the production build refuses to run without an explicit API URL, refuses
 * to run if the URL looks local, and refuses to run if it is a documentation
 * placeholder. `npm run dev` keeps the localhost convenience, because there is
 * no ambiguity about where a dev bundle runs.
 */
import { isPlaceholderApiUrl } from './src/config/apiUrl';

const API_URL_VAR = 'VITE_API_URL';

/** Hosts that can only ever be reachable from the developer's own machine. */
const LOCAL_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0', '::1'];

function assertProductionApiUrl(mode: string): void {
  // Vite merges `.env*` files with the process environment, so this picks up
  // both `VITE_API_URL=...` in a shell and a committed `.env.production`.
  const env = { ...loadEnv(mode, process.cwd(), 'VITE_'), ...process.env };
  const value = (env[API_URL_VAR] ?? '').trim();

  if (!value) {
    throw new Error(
      `\n\n${API_URL_VAR} is not set, so the production build was aborted.\n\n` +
        'A production bundle must never fall back to localhost: the app would\n' +
        'build and deploy successfully, then fail for every visitor.\n\n' +
        'Set it to the deployed backend URL and rebuild, for example:\n' +
        `  ${API_URL_VAR}=https://<api-service>.up.railway.app npm run build\n\n` +
        'On Vercel, add it under Project -> Settings -> Environment Variables.\n' +
        'See .env.example for the full list.\n\n',
    );
  }

  let host: string;
  try {
    host = new URL(value).hostname.toLowerCase();
  } catch {
    throw new Error(
      `\n\n${API_URL_VAR} is not a valid absolute URL: ${JSON.stringify(value)}\n\n` +
        'It must include the scheme and host, e.g. https://<api-service>.up.railway.app\n' +
        '(no trailing slash required, no path).\n\n',
    );
  }

  if (LOCAL_HOSTS.includes(host)) {
    throw new Error(
      `\n\n${API_URL_VAR} points at ${host}, which is a local address, so the\n` +
        'production build was aborted.\n\n' +
        'A deployed bundle cannot reach a server on the visitor\'s machine.\n' +
        'Point it at the deployed backend URL, or use `npm run dev` for local\n' +
        'development.\n\n',
    );
  }

  if (isPlaceholderApiUrl(value)) {
    throw new Error(
      `\n\n${API_URL_VAR} is set to a placeholder: ${JSON.stringify(value)}\n\n` +
        'That name appears in documentation, not on a real backend, so the\n' +
        'production build was aborted. Shipping it would deploy a bundle that\n' +
        'fails for every visitor with "Cannot reach the PhysioAI backend".\n\n' +
        'Set the environment variable to the actual deployed backend URL - for\n' +
        'example the Railway service domain - then deploy again:\n' +
        `  ${API_URL_VAR}=https://<api-service>.up.railway.app npm run build\n\n`,
    );
  }
}

export default defineConfig(({ command, mode }) => {
  if (command === 'build') {
    assertProductionApiUrl(mode);
  }

  return {
    plugins: [
      react(),
      tailwindcss(),
    ],
    server: {
      host: '0.0.0.0',
      port: 3000,
    },
  };
});
