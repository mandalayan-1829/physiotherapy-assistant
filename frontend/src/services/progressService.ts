import { apiRequest, isMockMode } from "./api";
import type { ProgressData, ProgressSummary } from "@/types/session";

export async function fetchProgress(): Promise<ProgressData> {
  if (isMockMode()) {
    return {
      total_sessions: 24,
      total_reps: 240,
      avg_form_accuracy: 82.5,
      exercises_done: 6,
      summary: [
        { exercise: "squat", total_sessions: 8, total_reps: 96, avg_form: 85 },
        { exercise: "lunges", total_sessions: 5, total_reps: 50, avg_form: 88 },
        { exercise: "shoulder_raises", total_sessions: 4, total_reps: 60, avg_form: 78 },
        { exercise: "calf_raises", total_sessions: 3, total_reps: 45, avg_form: 90 },
        { exercise: "knee_raises", total_sessions: 2, total_reps: 20, avg_form: 80 },
        { exercise: "tree_pose", total_sessions: 2, total_reps: 10, avg_form: 82 },
      ],
      recent_sessions: [
        { id: 1, user_id: 1, exercise: "squat", reps: 12, form_accuracy: 85, duration_sec: 180, notes: "", date: "2025-01-15 10:30:00" },
        { id: 2, user_id: 1, exercise: "lunges", reps: 10, form_accuracy: 90, duration_sec: 150, notes: "", date: "2025-01-14 09:15:00" },
        { id: 3, user_id: 1, exercise: "shoulder_raises", reps: 15, form_accuracy: 75, duration_sec: 120, notes: "", date: "2025-01-13 11:00:00" },
      ],
    };
  }
  return apiRequest<ProgressData>("/api/progress");
}

export async function fetchProgressSummary(): Promise<ProgressSummary> {
  if (isMockMode()) {
    return {
      total_sessions: 24,
      total_reps: 240,
      avg_form_accuracy: 82.5,
      exercises_done: 6,
      current_streak: 3,
    };
  }
  return apiRequest<ProgressSummary>("/api/progress/summary");
}
