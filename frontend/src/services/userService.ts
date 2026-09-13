import { apiRequest, isMockMode, setToken, clearToken } from "./api";
import type { User, AuthResponse } from "@/types/user";

export async function login(email: string, password: string): Promise<AuthResponse> {
  if (isMockMode()) {
    const mockUser: User = {
      id: 1,
      name: "Demo User",
      age: 30,
      gender: "male",
      email,
      blood_group: "O+",
      height_cm: 175,
      weight_kg: 70,
      medical_conditions: "",
      exercise_limitations: "",
      rehab_goals: "",
      created_at: new Date().toISOString(),
    };
    setToken("mock-token-1");
    return { success: true, message: "Login successful", user: mockUser };
  }
  const res = await apiRequest<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (res.user) {
    setToken(String(res.user.id));
  }
  return res;
}

export async function signup(
  name: string,
  age: number,
  email: string,
  password: string
): Promise<AuthResponse> {
  if (isMockMode()) {
    return { success: true, message: "Account created! Please login.", user: null };
  }
  return apiRequest<AuthResponse>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ name, age, email, password }),
  });
}

export async function getProfile(): Promise<User> {
  if (isMockMode()) {
    return {
      id: 1,
      name: "Demo User",
      age: 30,
      gender: "male",
      email: "demo@aiphysio.com",
      blood_group: "O+",
      height_cm: 175,
      weight_kg: 70,
      medical_conditions: "",
      exercise_limitations: "",
      rehab_goals: "",
      created_at: new Date().toISOString(),
    };
  }
  return apiRequest<User>("/api/users/me");
}

export async function updateProfile(data: Partial<User>): Promise<User> {
  if (isMockMode()) {
    return { ...data, id: 1, name: "Demo User", email: "demo@aiphysio.com" } as User;
  }
  return apiRequest<User>("/api/users/me", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export function logout() {
  clearToken();
}
