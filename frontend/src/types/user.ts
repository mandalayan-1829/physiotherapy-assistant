export interface User {
  id: number;
  name: string;
  age: number | null;
  gender: string;
  email: string;
  blood_group: string;
  height_cm: number;
  weight_kg: number;
  medical_conditions: string;
  exercise_limitations: string;
  rehab_goals: string;
  created_at: string | null;
}

export interface UserCreate {
  name: string;
  age: number;
  email: string;
  password: string;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface AuthResponse {
  success: boolean;
  message: string;
  user: User | null;
}
