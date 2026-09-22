/**
 * Centralized HTTP client for the PhysioAI FastAPI backend.
 *
 * Every backend call in the application goes through this module so that the
 * base URL, authentication header and error handling live in one place.
 *
 * Configuration: set `VITE_API_URL` (see `.env.example`). It must never be
 * hardcoded at call sites.
 */

import { isPlaceholderApiUrl } from '../config/apiUrl';

/**
 * Resolve the backend base URL.
 *
 * `npm run build` aborts when `VITE_API_URL` is missing, points at a local host
 * or is a documentation placeholder (see `vite.config.ts`), so a production
 * bundle always carries a real URL. The localhost default below therefore exists
 * *only* for `npm run dev`; it is never reachable in a production build.
 */
function resolveBaseUrl(): string {
  const configured = (import.meta.env.VITE_API_URL ?? '').trim();

  if (configured) {
    // Defence in depth for a bundle produced outside `npm run build`: sending
    // requests to a documentation hostname is never useful, in any environment.
    if (isPlaceholderApiUrl(configured)) {
      throw new Error(
        `VITE_API_URL is set to the placeholder ${JSON.stringify(configured)}, which is ` +
          'not a real backend. Point it at the deployed backend URL.',
      );
    }
    return configured.replace(/\/+$/, '');
  }

  if (import.meta.env.PROD) {
    // Should be unreachable: the build fails first. Kept as a guard so a bundle
    // produced by some other toolchain cannot silently target localhost.
    throw new Error(
      'VITE_API_URL is not set. This build cannot reach the backend; refusing to ' +
        'silently fall back to localhost.',
    );
  }

  return 'http://localhost:8000';
}

const BASE_URL = resolveBaseUrl();

/**
 * The access token is a credential, nothing more. It is the only thing kept in
 * browser storage. Identity, role and medical data are *never* read from
 * storage — they are always resolved from the backend via `GET /auth/me`.
 */
const TOKEN_KEY = 'physio_access_token';

export class ApiError extends Error {
  readonly status: number;
  /** Parsed error body, so callers can read server-provided metadata. */
  readonly body: Record<string, unknown>;

  constructor(status: number, message: string, body: Record<string, unknown> = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }

  /** Number of consecutive failed sign-in attempts (401/429 from login). */
  get failedAttempts(): number | undefined {
    const value = this.body.failed_attempts;
    return typeof value === 'number' ? value : undefined;
  }

  /** The backend decides when the UI should offer "Forgot password?". */
  get showForgotPassword(): boolean {
    return this.body.show_forgot_password === true;
  }

  get retryAfterSeconds(): number | undefined {
    const value = this.body.retry_after_seconds;
    return typeof value === 'number' ? value : undefined;
  }

  /** Set when the reset request must be restarted (e.g. attempts exhausted). */
  get restartRequired(): boolean {
    return this.body.restart_required === true;
  }
}

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (typeof window === 'undefined') return;
  if (token) {
    window.localStorage.setItem(TOKEN_KEY, token);
  } else {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

export function apiBaseUrl(): string {
  return BASE_URL;
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: unknown;
  /** Set to false for public endpoints such as /health and /auth/login. */
  auth?: boolean;
  signal?: AbortSignal;
}

async function extractError(response: Response): Promise<{ message: string; body: Record<string, unknown> }> {
  try {
    const data = (await response.json()) as Record<string, unknown>;
    const detail = data?.detail;
    if (typeof detail === 'string') return { message: detail, body: data };
    if (Array.isArray(detail)) {
      const message = detail
        .map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item))
        .join('; ');
      return { message, body: data };
    }
    return { message: `Request failed with status ${response.status}`, body: data ?? {} };
  } catch {
    return {
      message: `Request failed with status ${response.status}`,
      body: {},
    };
  }
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = true, signal } = options;

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch {
    throw new ApiError(0, `Cannot reach the PhysioAI backend at ${BASE_URL}. Is it running?`);
  }

  if (!response.ok) {
    const { message, body } = await extractError(response);
    throw new ApiError(response.status, message, body);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

/** Liveness probe for the backend. */
export function checkHealth(): Promise<{ status: string; service: string; version: string }> {
  return request('/health', { auth: false });
}
