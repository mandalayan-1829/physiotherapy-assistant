import { apiRequest, isMockMode } from "./api";
import type { Exercise } from "@/types/exercise";
import { MOCK_EXERCISES } from "./mockData";

export async function fetchExercises(): Promise<Exercise[]> {
  if (isMockMode()) return MOCK_EXERCISES;
  const res = await apiRequest<{ exercises: Exercise[]; total: number }>("/api/exercises");
  return res.exercises;
}

export async function fetchExercise(id: string): Promise<Exercise> {
  if (isMockMode()) {
    return MOCK_EXERCISES.find((e) => e.id === id) || MOCK_EXERCISES[0];
  }
  return apiRequest<Exercise>(`/api/exercises/${id}`);
}
