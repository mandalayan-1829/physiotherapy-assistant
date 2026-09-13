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

export interface Session {
  id: number;
  userId: number;
  exercise: string;
  exerciseLabel: string;
  reps: number;
  targetReps: number;
  formAccuracy: number; // 0-100%
  durationSec: number;
  notes: string;
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
  avatarUrl?: string;
}

export type AppointmentStatus = 'pending' | 'approved' | 'completed' | 'cancelled';

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
