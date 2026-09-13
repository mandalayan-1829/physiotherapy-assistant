import { useState, useEffect } from "react";
import type { ProgressData } from "@/types/session";
import { fetchProgress } from "@/services/progressService";

export function useProgress() {
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchProgress()
      .then(setProgress)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return { progress, loading, error };
}
