export interface Doctor {
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
}

export interface Appointment {
  id: number;
  user_id: number;
  doctor_id: number;
  doctor_name: string | null;
  specialization: string | null;
  patient_name: string | null;
  date: string;
  time: string;
  reason: string;
  status: string;
  admin_note: string;
  created_at: string | null;
}

export interface Note {
  id: number;
  user_id: number;
  note_text: string;
  date: string | null;
}

export interface Message {
  id: number;
  user_id: number;
  doctor_id: number;
  sender: string;
  message: string;
  timestamp: string | null;
}
