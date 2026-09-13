export interface Session {
  id: number;
  user_id: number;
  exercise: string;
  reps: number;
  form_accuracy: number;
  duration_sec: number;
  notes: string;
  date: string | null;
}

export interface SessionSummary {
  exercise: string;
  total_sessions: number;
  total_reps: number;
  avg_form: number;
}

export interface ProgressData {
  total_sessions: number;
  total_reps: number;
  avg_form_accuracy: number;
  exercises_done: number;
  summary: SessionSummary[];
  recent_sessions: Session[];
}

export interface ProgressSummary {
  total_sessions: number;
  total_reps: number;
  avg_form_accuracy: number;
  exercises_done: number;
  current_streak: number;
}
