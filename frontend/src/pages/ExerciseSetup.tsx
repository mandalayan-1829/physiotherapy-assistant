import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Play } from "lucide-react";
import { fetchExercise } from "@/services/exerciseService";
import type { Exercise } from "@/types/exercise";
import LoadingSpinner from "@/components/common/LoadingSpinner";

export default function ExerciseSetup() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(true);
  const [targetReps, setTargetReps] = useState(10);

  useEffect(() => {
    if (id) {
      fetchExercise(id)
        .then((ex) => {
          setExercise(ex);
          setTargetReps(ex.recommended_reps);
        })
        .catch(() => navigate("/exercises"))
        .finally(() => setLoading(false));
    }
  }, [id]);

  if (loading) return <LoadingSpinner />;
  if (!exercise) return null;

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <button
        onClick={() => navigate(`/exercises/${exercise.id}`)}
        className="flex items-center gap-2 text-surface-500 hover:text-surface-900 text-sm font-medium mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to {exercise.label}
      </button>

      <div className="text-center mb-10">
        <div className="text-5xl mb-3">{exercise.icon}</div>
        <h1 className="text-2xl font-extrabold text-surface-900 mb-1">
          Get Ready for {exercise.label}
        </h1>
        <p className="text-surface-500">Set up your session parameters</p>
      </div>

      <div className="bg-white rounded-2xl border border-surface-200 p-8 mb-8">
        <div className="mb-8">
          <label className="block text-sm font-semibold text-surface-700 mb-3">
            Target Reps: <span className="text-primary-500">{targetReps}</span>
          </label>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={targetReps}
            onChange={(e) => setTargetReps(Number(e.target.value))}
            className="w-full h-2 bg-surface-200 rounded-full appearance-none cursor-pointer accent-primary-500"
          />
          <div className="flex justify-between text-xs text-surface-400 mt-1">
            <span>5</span>
            <span>50</span>
          </div>
        </div>

        <div className="mb-6">
          <h3 className="font-bold text-surface-900 mb-3">✅ Form Checks</h3>
          <div className="space-y-2">
            {exercise.form_checks.map((check) => (
              <div key={check} className="flex items-center gap-2 text-sm text-surface-600">
                <div className="w-5 h-5 rounded-full bg-emerald-50 flex items-center justify-center">
                  <span className="text-emerald-500 text-xs">✓</span>
                </div>
                {check}
              </div>
            ))}
          </div>
        </div>

        <div className="bg-primary-50 rounded-xl p-4">
          <p className="text-primary-700 text-sm">
            💡 <strong>Tip:</strong> {exercise.tip}
          </p>
        </div>
      </div>

      <button
        onClick={() => navigate(`/exercises/${exercise.id}/session`, { state: { targetReps } })}
        className="w-full py-4 bg-primary-500 text-white rounded-2xl font-bold text-base hover:bg-primary-600 transition-all shadow-lg shadow-primary-200 flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Start Exercise ({targetReps} reps)
      </button>
    </div>
  );
}
