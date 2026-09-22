/**
 * Shared rules for the backend base URL (`VITE_API_URL`).
 *
 * Two different callers need the same answer to "is this a real backend?":
 *
 *   * `vite.config.ts` runs at build time and must refuse to produce a bundle
 *     that cannot reach a backend.
 *   * `src/services/api.ts` runs in the browser and refuses to aim requests at
 *     a host that was never a backend.
 *
 * They must agree, so the rule lives here once. The module is deliberately
 * dependency-free so it can be imported from the Vite config (Node) and from
 * application code (browser) alike.
 *
 * Why this exists: a placeholder does not fail loudly. A bundle built with
 * `VITE_API_URL=https://your-fastapi-backend-url.com` compiles, deploys and
 * serves perfectly - and then every request dies with "Cannot reach the
 * PhysioAI backend". That is an outage disguised as a successful deploy, so the
 * names used in documentation are rejected outright.
 */

/**
 * Substrings that appear only in documentation examples, never in a real
 * deployed backend hostname.
 *
 * Note what is *absent*: `up.railway.app` is Railway's genuine domain suffix,
 * so rejecting it would block the real deployment. Only the fake labels around
 * it (`your-`, `example`) are rejected.
 */
export const PLACEHOLDER_HOST_TOKENS: readonly string[] = [
  'your-',
  'your_',
  'your.',
  'example',
  'placeholder',
  'change-me',
  'change_me',
  'changeme',
  'replace-me',
  'replace_me',
  'todo',
  'xxxx',
];

/**
 * True when `value` is an absolute URL whose host is a documentation
 * placeholder.
 *
 * An empty value is *not* a placeholder - whether an unset value is acceptable
 * depends on the environment, so that decision belongs to the caller. Values
 * that are not absolute URLs also return `false`; they are reported separately
 * as malformed, which is a different (and more obvious) failure.
 */
export function isPlaceholderApiUrl(value: string | undefined | null): boolean {
  const candidate = (value ?? '').trim().toLowerCase();
  if (!candidate) return false;

  let host: string;
  try {
    host = new URL(candidate).hostname.toLowerCase();
  } catch {
    return false;
  }

  return PLACEHOLDER_HOST_TOKENS.some((token) => host.includes(token));
}
