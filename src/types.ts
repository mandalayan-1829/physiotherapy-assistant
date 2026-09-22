export type ExerciseType = 'physio' | 'yoga';

export interface Exercise {
  id: string;
  label: string;
  icon: string;
  target: string;
  type: ExerciseType;
  videoId: string;
  formChecks: string[];
  tip: string;
  limitations: string[];
  defaultTargetReps: number;
  description: string;
  primaryJoint: string;
  idealAngleRange: string;
}

export type ExerciseStage = 'up' | 'down' | 'holding' | 'cow' | 'cat';

export interface ExerciseState {
  exercise: string;
  reps: number;
  targetReps: number;
  stage: ExerciseStage;
  angle: number;
  feedback: string;
  formErrors: string[];
  formOk: boolean;
  holdCount: number;
  holdSecondsRequired: number;
  isComplete: boolean;
}

export interface User {
  id: number;
  name: string;
  age: number;
  gender: string;
  dob?: string;
  email: string;
  contactNumber?: string;
  bloodGroup: string;
  heightCm: number;
  weightKg: number;
  occupation?: string;
  currentProblem: string;
  problemStartDate?: string;
  problemCause?: string;
  previousInjuries?: string;
  pastSurgeries?: string;
  medicalConditions: string;
  painLocation: string;
  painIntensity: number; // 0 to 10
  painType: string;
  painTriggers?: string;
  painDuration?: string;
  dailySittingHours?: number;
  activityLevel?: string;
  exerciseHabits?: string;
  currentMedications: string;
  movementRestrictions: string;
  rehabGoals: string;
  allergies?: string;
  precautions?: string;
  exerciseLimitations: string;
  emergencyContactName: string;
  emergencyContactPhone: string;
  guardianWhatsapp: string;
  doctorName: string;
  createdAt: string;
}

/**
 * How a session's metrics were produced.
 *
 *  - `pose_inference` real on-device pose estimation measured the movement
 *  - `simulated`      the retired demo tracker fabricated the landmarks
 *  - `manual`         the patient logged the set themselves; nothing was measured
 *
 * Only `pose_inference` sessions contribute to a form-accuracy average. This is
 * enforced by the backend, not merely displayed here.
 */
export type MetricsSource = 'pose_inference' | 'simulated' | 'manual';

export interface Session {
  id: number;
  userId: number;
  exercise: string;
  exerciseLabel: string;
  reps: number;
  targetReps: number;
  /** 0-100. Only meaningful when `metricsSource` is `pose_inference`. */
  formAccuracy: number;
  durationSec: number;
  notes: string;
  metricsSource: MetricsSource;
  date: string;
}

export interface DietEntry {
  id: number;
  userId: number;
  meal: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  date: string;
}

export interface Note {
  id: number;
  userId: number;
  noteText: string;
  category: 'clinical' | 'exercise' | 'symptom' | 'general';
  date: string;
}

export interface Doctor {
  id: number;
  name: string;
  specialization: string;
  experience: number;
  qualification: string;
  availableDays: string;
  timings: string;
  about: string;
  contact: string;
  whatsapp: string;
  email: string;
  hospital?: string;
  rating?: number;
  bio?: string;
  availableSlots?: string[];
  avatarUrl?: string;
}

export type UserRole = 'patient' | 'doctor';

export type AppointmentStatus = 
  | 'scheduled'
  | 'confirmed'
  | 'ready'
  | 'completed'
  | 'cancelled'
  | 'pending'
  | 'approved';

export interface MonthlyReport {
  id: string;
  monthKey: string; // e.g. "2026-09"
  monthName: string; // e.g. "September 2026"
  generatedDate: string;
  patientId: number;
  patientName: string;
  patientEmail: string;
  assignedDoctorName?: string;
  assignedDoctorEmail?: string;
  hasUpcomingCheckup: boolean;
  totalSessions: number;
  /** Sessions that included a measured movement (pose inference). */
  measuredSessions: number;
  totalReps: number;
  completedExercises: number;
  /**
   * Mean form score over measured sessions only. Zero when nothing was
   * measured - never an estimate.
   */
  avgAccuracy: number;
  avgScore: number;
  /**
   * Share of elapsed days in the month with at least one recorded session.
   * A defined figure derived from session dates, not an estimated adherence.
   */
  activeDaysPercent: number;
  exerciseBreakdown: {
    exerciseLabel: string;
    sessions: number;
    reps: number;
    avgAccuracy: number;
  }[];
  progressTrend: string;
  /** Report delivery is not implemented; the status records that fact. */
  emailStatus: 'Sent' | 'Not sent' | 'Draft';
  emailSentDate?: string;
  recipients: string[];
}

export interface DailyHistoryEntry {
  date: string; // YYYY-MM-DD
  displayDate: string;
  sessionsCount: number;
  /** Sessions on this day that included a measured movement. */
  measuredSessions: number;
  totalReps: number;
  /** Repetitions across measured sessions only. */
  measuredReps: number;
  /** Always 0 when the day has no measured session. */
  correctReps: number;
  incorrectReps: number;
  /** Mean form score over measured sessions; 0 when nothing was measured. */
  avgFormScore: number;
  totalDurationSec: number;
  exercises: string[];
  sessions: Session[];
}

export interface Appointment {
  id: number;
  userId: number;
  doctorId: number;
  doctorName: string;
  specialization: string;
  patientName: string;
  email: string;
  date: string;
  time: string;
  reason: string;
  status: AppointmentStatus;
  adminNote?: string;
  createdAt: string;
}

export interface Message {
  id: number;
  userId: number;
  doctorId: number;
  sender: 'user' | 'doctor';
  message: string;
  timestamp: string;
}

export interface GuardianAlert {
  id: number;
  userId: number;
  alertType: 'pain_spike' | 'workout_milestone' | 'emergency_help' | 'missed_routine';
  message: string;
  sentTo: string;
  timestamp: string;
}

export interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}
