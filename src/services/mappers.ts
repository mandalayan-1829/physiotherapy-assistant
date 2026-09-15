/**
 * Translation layer between the backend's snake_case DTOs and the existing
 * frontend view models in `src/types.ts`.
 *
 * Keeping this mapping isolated means the current views did not need to be
 * rewritten to change data source.
 */

import type {
  Appointment,
  AppointmentStatus,
  DietEntry,
  Doctor,
  Message,
  Note,
  Session,
  User,
} from '../types';
import type { Account, DoctorProfileDTO, PatientProfileDTO } from './auth';

const pad = (value: number) => String(value).padStart(2, '0');

/** ISO timestamp -> "YYYY-MM-DD HH:MM" (matches the legacy view model). */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`
  );
}

/** ISO timestamp -> "YYYY-MM-DD". */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso.slice(0, 10);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

// --- Patient profile --------------------------------------------------------

export function profileToUser(account: Account, profile: PatientProfileDTO | null): User {
  return {
    id: account.id,
    name: account.full_name,
    email: account.email,
    age: profile?.age ?? 0,
    gender: profile?.gender ?? '',
    dob: profile?.dob ?? '',
    contactNumber: profile?.contact_number ?? '',
    bloodGroup: profile?.blood_group ?? '',
    heightCm: profile?.height_cm ?? 0,
    weightKg: profile?.weight_kg ?? 0,
    occupation: profile?.occupation ?? '',
    currentProblem: profile?.current_problem ?? '',
    problemStartDate: profile?.problem_start_date ?? '',
    problemCause: profile?.problem_cause ?? '',
    previousInjuries: profile?.previous_injuries ?? '',
    pastSurgeries: profile?.past_surgeries ?? '',
    medicalConditions: profile?.medical_conditions ?? '',
    currentMedications: profile?.current_medications ?? '',
    allergies: profile?.allergies ?? '',
    precautions: profile?.precautions ?? '',
    exerciseLimitations: profile?.exercise_limitations ?? '',
    painLocation: profile?.pain_location ?? '',
    painIntensity: profile?.pain_intensity ?? 0,
    painType: profile?.pain_type ?? '',
    painTriggers: profile?.pain_triggers ?? '',
    painDuration: profile?.pain_duration ?? '',
    dailySittingHours: profile?.daily_sitting_hours ?? 0,
    activityLevel: profile?.activity_level ?? '',
    exerciseHabits: profile?.exercise_habits ?? '',
    movementRestrictions: profile?.movement_restrictions ?? '',
    rehabGoals: profile?.rehab_goals ?? '',
    emergencyContactName: profile?.emergency_contact_name ?? '',
    emergencyContactPhone: profile?.emergency_contact_phone ?? '',
    guardianWhatsapp: profile?.guardian_whatsapp ?? '',
    doctorName: profile?.doctor_name ?? '',
    createdAt: '',
  };
}

/** Frontend user view model -> backend press payload (snake_case). */
export function userToProfilePayload(user: User): Partial<PatientProfileDTO> {
  return {
    age: user.age,
    gender: user.gender,
    dob: user.dob ?? '',
    contact_number: user.contactNumber ?? '',
    blood_group: user.bloodGroup,
    height_cm: user.heightCm,
    weight_kg: user.weightKg,
    occupation: user.occupation ?? '',
    current_problem: user.currentProblem,
    problem_start_date: user.problemStartDate ?? '',
    problem_cause: user.problemCause ?? '',
    previous_injuries: user.previousInjuries ?? '',
    past_surgeries: user.pastSurgeries ?? '',
    medical_conditions: user.medicalConditions,
    current_medications: user.currentMedications,
    allergies: user.allergies ?? '',
    precautions: user.precautions ?? '',
    exercise_limitations: user.exerciseLimitations,
    pain_location: user.painLocation,
    pain_intensity: user.painIntensity,
    pain_type: user.painType,
    pain_triggers: user.painTriggers ?? '',
    pain_duration: user.painDuration ?? '',
    daily_sitting_hours: user.dailySittingHours ?? 0,
    activity_level: user.activityLevel ?? '',
    exercise_habits: user.exerciseHabits ?? '',
    movement_restrictions: user.movementRestrictions,
    rehab_goals: user.rehabGoals,
    emergency_contact_name: user.emergencyContactName,
    emergency_contact_phone: user.emergencyContactPhone,
    guardian_whatsapp: user.guardianWhatsapp,
    doctor_name: user.doctorName,
  };
}

// --- Doctors ----------------------------------------------------------------

export interface DoctorDTO {
  id: number;
  name: string;
  specialization: string;
  experience: number;
  qualification: string;
  available_days: string;
  timings: string;
  about: string;
  contact: string;
  whatsapp: string;
  email: string;
  hospital: string;
  rating: number;
}

export function doctorToView(dto: DoctorDTO | DoctorProfileDTO): Doctor {
  return {
    id: dto.id,
    name: dto.name,
    specialization: dto.specialization,
    experience: dto.experience,
    qualification: dto.qualification,
    availableDays: dto.available_days,
    timings: dto.timings,
    about: dto.about,
    contact: dto.contact,
    whatsapp: dto.whatsapp,
    email: dto.email,
    hospital: dto.hospital,
    rating: dto.rating,
    avatarUrl: undefined,
  };
}

// --- Sessions ---------------------------------------------------------------

export interface WorkoutSessionDTO {
  id: number;
  patient_account_id: number;
  exercise: string;
  exercise_label: string;
  reps: number;
  target_reps: number;
  form_accuracy: number;
  duration_sec: number;
  notes: string;
  performed_at: string;
}

export function sessionToView(dto: WorkoutSessionDTO): Session {
  return {
    id: dto.id,
    userId: dto.patient_account_id,
    exercise: dto.exercise,
    exerciseLabel: dto.exercise_label,
    reps: dto.reps,
    targetReps: dto.target_reps,
    formAccuracy: dto.form_accuracy,
    durationSec: dto.duration_sec,
    notes: dto.notes,
    date: formatDateTime(dto.performed_at),
  };
}

// --- Diet -------------------------------------------------------------------

export interface DietDTO {
  id: number;
  patient_account_id: number;
  meal: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  recorded_at: string;
}

export function dietToView(dto: DietDTO): DietEntry {
  return {
    id: dto.id,
    userId: dto.patient_account_id,
    meal: dto.meal,
    calories: dto.calories,
    protein: dto.protein,
    carbs: dto.carbs,
    fats: dto.fats,
    date: formatDate(dto.recorded_at),
  };
}

// --- Notes ------------------------------------------------------------------

export interface NoteDTO {
  id: number;
  patient_account_id: number;
  doctor_profile_id: number | null;
  note_text: string;
  category: Note['category'];
  created_at: string;
}

export function noteToView(dto: NoteDTO): Note {
  return {
    id: dto.id,
    userId: dto.patient_account_id,
    noteText: dto.note_text,
    category: dto.category,
    date: formatDate(dto.created_at),
  };
}

// --- Appointments -----------------------------------------------------------

export interface AppointmentDTO {
  id: number;
  patient_account_id: number;
  doctor_profile_id: number;
  patient_name: string;
  doctor_name: string;
  specialization: string;
  email: string;
  date: string;
  time: string;
  reason: string;
  status: AppointmentStatus;
  clinician_note: string;
  created_at: string;
}

export function appointmentToView(dto: AppointmentDTO): Appointment {
  return {
    id: dto.id,
    userId: dto.patient_account_id,
    doctorId: dto.doctor_profile_id,
    doctorName: dto.doctor_name,
    specialization: dto.specialization,
    patientName: dto.patient_name,
    email: dto.email,
    date: dto.date,
    time: dto.time,
    reason: dto.reason,
    status: dto.status,
    adminNote: dto.clinician_note || undefined,
    createdAt: formatDateTime(dto.created_at),
  };
}

// --- Messages ---------------------------------------------------------------

export interface MessageDTO {
  id: number;
  patient_account_id: number;
  doctor_profile_id: number;
  sender: 'patient' | 'doctor';
  message: string;
  created_at: string;
}

export function messageToView(dto: MessageDTO): Message {
  return {
    id: dto.id,
    userId: dto.patient_account_id,
    doctorId: dto.doctor_profile_id,
    sender: dto.sender === 'patient' ? 'user' : 'doctor',
    message: dto.message,
    timestamp: formatDateTime(dto.created_at),
  };
}

/** Frontend sender values are mapped back to the API vocabulary. */
export function senderToApi(sender: 'user' | 'doctor'): 'patient' | 'doctor' {
  return sender === 'user' ? 'patient' : 'doctor';
}
