import { apiRequest, isMockMode } from "./api";
import type { Session } from "@/types/session";

export async function createSession(exercise: string, targetReps: number): Promise<Session> {
  if (isMockMode()) {
    return {
      id: Date.now(),
      user_id: 1,
      exercise,
      reps: 0,
      form_accuracy: 0,
      duration_sec: 0,
      notes: "",
      date: new Date().toISOString(),
    };
  }
  return apiRequest<Session>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ exercise, target_reps: targetReps }),
  });
}

export async function endSession(
  sessionId: number,
  reps: number,
  formAccuracy: number,
  durationSec: number,
  notes: string = ""
): Promise<void> {
  if (isMockMode()) return;
  await apiRequest(`/api/sessions/${sessionId}/end`, {
    method: "PUT",
    body: JSON.stringify({
      session_id: sessionId,
      reps,
      form_accuracy: formAccuracy,
      duration_sec: durationSec,
      notes,
    }),
  });
}

export async function fetchSessions(): Promise<Session[]> {
  if (isMockMode()) {
    return [
      { id: 1, user_id: 1, exercise: "squat", reps: 12, form_accuracy: 85, duration_sec: 180, notes: "", date: "2025-01-15 10:30:00" },
      { id: 2, user_id: 1, exercise: "lunges", reps: 10, form_accuracy: 90, duration_sec: 150, notes: "", date: "2025-01-14 09:15:00" },
      { id: 3, user_id: 1, exercise: "shoulder_raises", reps: 15, form_accuracy: 75, duration_sec: 120, notes: "", date: "2025-01-13 11:00:00" },
    ];
  }
  return apiRequest<Session[]>("/api/sessions");
}

export async function fetchRecentSessions(): Promise<Session[]> {
  if (isMockMode()) return fetchSessions();
  return apiRequest<Session[]>("/api/sessions/recent");
}
