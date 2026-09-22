/**
 * Domain API functions. All application data now flows through these calls;
 * views receive plain props from the app shell exactly as before.
 */

import { request } from './api';
import {
  appointmentToView,
  dietToView,
  doctorToView,
  messageToView,
  noteToView,
  reportToView,
  sessionToView,
  senderToApi,
} from './mappers';
import type {
  AppointmentDTO,
  DietDTO,
  DoctorDTO,
  MessageDTO,
  NoteDTO,
  ReportDTO,
  ReportPresentationContext,
  WorkoutSessionDTO,
} from './mappers';
import type {
  Appointment,
  AppointmentStatus,
  DietEntry,
  Doctor,
  GuardianAlert,
  Message,
  MetricsSource,
  MonthlyReport,
  Note,
  Session,
} from '../types';

function query(params: Record<string, number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

// --- Patient roster (doctor portal) ----------------------------------------

export interface PatientSummary {
  account_id: number;
  name: string;
  email: string;
  current_problem: string;
  pain_intensity: number;
}

export function listPatients(): Promise<PatientSummary[]> {
  return request<PatientSummary[]>('/patients');
}

// --- Doctors ----------------------------------------------------------------

export async function listDoctors(): Promise<Doctor[]> {
  const data = await request<DoctorDTO[]>('/doctors');
  return data.map(doctorToView);
}

// --- Exercise sessions ------------------------------------------------------

export async function listSessions(patientAccountId?: number): Promise<Session[]> {
  const data = await request<WorkoutSessionDTO[]>(`/sessions${query({ patient_account_id: patientAccountId })}`);
  return data.map(sessionToView);
}

export interface NewSession {
  exercise: string;
  exerciseLabel: string;
  reps: number;
  targetReps: number;
  formAccuracy: number;
  durationSec: number;
  notes: string;
  /**
   * States how the metrics were produced. Required, so a caller cannot
   * accidentally have a manually logged session treated as a measurement.
   */
  metricsSource: MetricsSource;
}

export async function createSession(input: NewSession): Promise<Session> {
  const data = await request<WorkoutSessionDTO>('/sessions', {
    method: 'POST',
    body: {
      exercise: input.exercise,
      exercise_label: input.exerciseLabel,
      reps: input.reps,
      target_reps: input.targetReps,
      form_accuracy: input.formAccuracy,
      duration_sec: input.durationSec,
      notes: input.notes,
      metrics_source: input.metricsSource,
    },
  });
  return sessionToView(data);
}

// --- Monthly reports --------------------------------------------------------
//
// Reports live in the backend. Previously the browser computed these aggregates
// itself and persisted them to localStorage, which meant two devices could show
// different figures for the same month and the numbers were not part of the
// patient's record at all. Only the presentation context (attending clinician,
// whether a checkup is booked) is assembled client-side.

export async function listReports(
  context: ReportPresentationContext,
  patientAccountId?: number,
): Promise<MonthlyReport[]> {
  const data = await request<ReportDTO[]>(
    `/reports${query({ patient_account_id: patientAccountId })}`,
  );
  return data.map((dto) => reportToView(dto, context));
}

/**
 * Generate (or refresh) the report for a month and return the stored result.
 *
 * Regenerating an existing month updates the same row instead of creating a
 * second one, so callers do not need to de-duplicate.
 */
export async function generateReport(
  monthKey: string,
  context: ReportPresentationContext,
  patientAccountId?: number,
): Promise<MonthlyReport> {
  const data = await request<ReportDTO>('/reports/generate', {
    method: 'POST',
    body: { month_key: monthKey, patient_account_id: patientAccountId },
  });
  return reportToView(data, context);
}

/** Retrieve a single stored report by its id. */
export async function getReport(
  reportId: string | number,
  context: ReportPresentationContext,
): Promise<MonthlyReport> {
  const data = await request<ReportDTO>(`/reports/${reportId}`);
  return reportToView(data, context);
}

// --- Diet -------------------------------------------------------------------

export async function listDiet(patientAccountId?: number): Promise<DietEntry[]> {
  const data = await request<DietDTO[]>(`/diet${query({ patient_account_id: patientAccountId })}`);
  return data.map(dietToView);
}

export interface NewDietEntry {
  meal: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
}

export async function addDiet(input: NewDietEntry): Promise<DietEntry> {
  const data = await request<DietDTO>('/diet', { method: 'POST', body: input });
  return dietToView(data);
}

export function deleteDiet(recordId: number): Promise<void> {
  return request<void>(`/diet/${recordId}`, { method: 'DELETE' });
}

// --- Clinical notes ---------------------------------------------------------

export async function listNotes(patientAccountId?: number): Promise<Note[]> {
  const data = await request<NoteDTO[]>(`/notes${query({ patient_account_id: patientAccountId })}`);
  return data.map(noteToView);
}

export async function addPatientNote(
  patientAccountId: number,
  noteText: string,
  category: Note['category'],
): Promise<Note> {
  const data = await request<NoteDTO>('/notes', {
    method: 'POST',
    body: { patient_account_id: patientAccountId, note_text: noteText, category },
  });
  return noteToView(data);
}

export function deleteNote(noteId: number): Promise<void> {
  return request<void>(`/notes/${noteId}`, { method: 'DELETE' });
}

// --- Appointments -----------------------------------------------------------

export async function listAppointments(patientAccountId?: number): Promise<Appointment[]> {
  const data = await request<AppointmentDTO[]>(
    `/appointments${query({ patient_account_id: patientAccountId })}`,
  );
  return data.map(appointmentToView);
}

export interface NewAppointment {
  doctorProfileId: number;
  date: string;
  time: string;
  reason: string;
}

export async function bookAppointment(input: NewAppointment): Promise<Appointment> {
  const data = await request<AppointmentDTO>('/appointments', {
    method: 'POST',
    body: {
      doctor_profile_id: input.doctorProfileId,
      date: input.date,
      time: input.time,
      reason: input.reason,
    },
  });
  return appointmentToView(data);
}

export async function updateAppointment(
  appointmentId: number,
  patch: { status?: AppointmentStatus; clinician_note?: string },
): Promise<Appointment> {
  const data = await request<AppointmentDTO>(`/appointments/${appointmentId}`, {
    method: 'PATCH',
    body: patch,
  });
  return appointmentToView(data);
}

// --- Messages ---------------------------------------------------------------

export async function listMessages(patientAccountId?: number): Promise<Message[]> {
  const data = await request<MessageDTO[]>(
    `/messages${query({ patient_account_id: patientAccountId })}`,
  );
  return data.map(messageToView);
}

export async function sendMessage(input: {
  doctorProfileId: number;
  message: string;
  patientAccountId?: number;
  sender: 'user' | 'doctor';
}): Promise<Message> {
  const data = await request<MessageDTO>('/messages', {
    method: 'POST',
    body: {
      doctor_profile_id: input.doctorProfileId,
      patient_account_id: input.patientAccountId,
      message: input.message,
    },
  });
  // `sender` is authoritative from the server; it is passed in only for clarity.
  void senderToApi(input.sender);
  return messageToView(data);
}

// --- Safety alerts ----------------------------------------------------------

export interface AlertDTO {
  id: number;
  patient_account_id: number;
  alert_type: GuardianAlert['alertType'];
  message: string;
  sent_to: string;
  created_at: string;
}

export async function listAlerts(patientAccountId?: number): Promise<GuardianAlert[]> {
  const data = await request<AlertDTO[]>(`/alerts${query({ patient_account_id: patientAccountId })}`);
  return data.map((dto) => ({
    id: dto.id,
    userId: dto.patient_account_id,
    alertType: dto.alert_type,
    message: dto.message,
    sentTo: dto.sent_to,
    timestamp: dto.created_at,
  }));
}

export function createAlert(input: {
  alertType: GuardianAlert['alertType'];
  message: string;
  sentTo: string;
}): Promise<void> {
  return request<void>('/alerts', {
    method: 'POST',
    body: { alert_type: input.alertType, message: input.message, sent_to: input.sentTo },
  });
}
