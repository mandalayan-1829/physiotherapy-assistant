/**
 * Centralized API client for AI Physio.
 * All HTTP requests go through this module.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const USE_MOCK = import.meta.env.VITE_USE_MOCK_API === "true";

export function isMockMode(): boolean {
  return USE_MOCK;
}

function getToken(): string | null {
  return localStorage.getItem("auth_token");
}

export function setToken(token: string) {
  localStorage.setItem("auth_token", token);
}

export function clearToken() {
  localStorage.removeItem("auth_token");
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

export function getWsUrl(sessionId: string): string {
  const base = import.meta.env.VITE_WS_BASE_URL || "ws://localhost:8000";
  return `${base}/ws/session/${sessionId}`;
}
