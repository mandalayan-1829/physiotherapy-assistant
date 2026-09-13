import { useState, useEffect } from "react";
import type { Exercise } from "@/types/exercise";
import { fetchExercises } from "@/services/exerciseService";

export function useExercises() {
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExercises()
      .then(setExercises)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return { exercises, loading, error };
}
