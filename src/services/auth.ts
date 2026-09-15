/**
 * Authentication service.
 *
 * Credentials are verified by the backend. The frontend stores only the
 * returned access token (see `api.ts`); the authenticated identity and role
 * always come from `GET /auth/me`.
 */

import { request, setToken } from './api';
import type { User, UserRole } from '../types';

export interface Account {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
}

export interface PatientProfileDTO {
  id: number;
  account_id: number;
  age: number | null;
  gender: string;
  dob: string;
  contact_number: string;
  blood_group: string;
  height_cm: number;
  weight_kg: number;
  occupation: string;
  current_problem: string;
  problem_start_date: string;
  problem_cause: string;
  previous_injuries: string;
  past_surgeries: string;
  medical_conditions: string;
  current_medications: string;
  allergies: string;
  precautions: string;
  exercise_limitations: string;
  pain_location: string;
  pain_intensity: number;
  pain_type: string;
  pain_triggers: string;
  pain_duration: string;
  daily_sitting_hours: number;
  activity_level: string;
  exercise_habits: string;
  movement_restrictions: string;
  rehab_goals: string;
  emergency_contact_name: string;
  emergency_contact_phone: string;
  guardian_whatsapp: string;
  doctor_name: string;
}

export interface DoctorProfileDTO {
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

export interface SessionPayload {
  access_token: string;
  token_type: string;
  expires_in: number;
  account: Account;
}

export interface MeResponse {
  account: Account;
  profile: PatientProfileDTO | null;
  doctor_profile: DoctorProfileDTO | null;
}

export interface RegisterInput {
  fullName: string;
  email: string;
  password: string;
  role: UserRole;
  specialization?: string;
  qualification?: string;
  licenseNumber?: string;
}

/** Resolved authentication state used by the app shell. */
export interface AuthSession {
  account: Account;
  role: UserRole;
  profile: PatientProfileDTO | null;
  doctorProfile: DoctorProfileDTO | null;
}

function toSession(response: SessionPayload, me: MeResponse | null = null): AuthSession {
  return {
    account: response.account,
    role: response.account.role,
    profile: me?.profile ?? null,
    doctorProfile: me?.doctor_profile ?? null,
  };
}

export async function register(input: RegisterInput): Promise<AuthSession> {
  const response = await request<SessionPayload>('/auth/register', {
    method: 'POST',
    auth: false,
    body: {
      full_name: input.fullName,
      email: input.email,
      password: input.password,
      role: input.role,
      specialization: input.specialization,
      qualification: input.qualification,
      license_number: input.licenseNumber,
    },
  });
  setToken(response.access_token);
  return toSession(response);
}

/**
 * Sign in against the selected portal. `role` must match the account's role,
 * otherwise the backend rejects the attempt.
 */
export async function login(email: string, password: string, role: UserRole): Promise<AuthSession> {
  const response = await request<SessionPayload>('/auth/login', {
    method: 'POST',
    auth: false,
    body: { email, password, role },
  });
  setToken(response.access_token);
  return toSession(response);
}

/** Restore the session from a stored token. Returns null when unauthenticated. */
export async function fetchCurrentSession(): Promise<AuthSession | null> {
  const me = await request<MeResponse>('/auth/me');
  return {
    account: me.account,
    role: me.account.role,
    profile: me.profile,
    doctorProfile: me.doctor_profile,
  };
}

export async function logout(): Promise<void> {
  try {
    await request<void>('/auth/logout', { method: 'POST' });
  } catch {
    // The token may already be invalid; clearing local state is what matters.
  } finally {
    setToken(null);
  }
}

// --- Password reset ---------------------------------------------------------
//
// Every step is performed by the backend. No verification code, reset token or
// reset state is ever kept in browser storage.

export interface ForgotPasswordResult {
  message: string;
}

/**
 * Request a verification code. The backend returns the same message whether or
 * not the address belongs to an account, so nothing is inferred here.
 */
export function requestPasswordReset(email: string, role: UserRole): Promise<ForgotPasswordResult> {
  return request<ForgotPasswordResult>('/auth/forgot-password', {
    method: 'POST',
    auth: false,
    body: { email, role },
  });
}

export interface VerifyCodeResult {
  reset_token: string;
  token_type: string;
  expires_in: number;
}

/** Exchange a valid code for a short-lived password-reset authorisation. */
export function verifyResetCode(
  email: string,
  verificationCode: string,
  role: UserRole,
): Promise<VerifyCodeResult> {
  return request<VerifyCodeResult>('/auth/verify-reset-code', {
    method: 'POST',
    auth: false,
    body: { email, verification_code: verificationCode, role },
  });
}

export interface ResetPasswordResult {
  message: string;
}

/** Set the new password. This ends every existing session for the account. */
export function resetPassword(
  resetToken: string,
  newPassword: string,
  confirmPassword: string,
): Promise<ResetPasswordResult> {
  return request<ResetPasswordResult>('/auth/reset-password', {
    method: 'POST',
    auth: false,
    body: {
      reset_token: resetToken,
      new_password: newPassword,
      confirm_password: confirmPassword,
    },
  });
}

/** Persist the patient's medical profile. */
export function saveProfile(profile: Partial<PatientProfileDTO>): Promise<PatientProfileDTO> {
  return request<PatientProfileDTO>('/users/me/profile', { method: 'PUT', body: profile });
}

export function getProfile(): Promise<PatientProfileDTO> {
  return request<PatientProfileDTO>('/users/me/profile');
}

// Re-exported for the app shell which keeps a `User` view model.
export type { User };
