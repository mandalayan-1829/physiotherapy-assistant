export interface Exercise {
  id: string;
  label: string;
  icon: string;
  target: string;
  type: "physio" | "yoga";
  video_id: string;
  form_checks: string[];
  tip: string;
  limitations: string[];
  difficulty: string;
  recommended_reps: number;
  description: string;
}

export interface PoseState {
  rep_count: number;
  angle: number;
  stage: string;
  feedback: string;
  form_status: "good" | "warning" | "incorrect" | "no_pose";
  form_errors: string[];
  form_ok: boolean;
  exercise: string;
  hold_count: number;
  alarm_active: boolean;
}

export interface SessionStartPayload {
  exercise: string;
  target_reps: number;
}

export interface SessionEndPayload {
  session_id: number;
  reps: number;
  form_accuracy: number;
  duration_sec: number;
  notes: string;
}
