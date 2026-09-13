import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Play, AlertTriangle } from "lucide-react";
import { fetchExercise } from "@/services/exerciseService";
import type { Exercise } from "@/types/exercise";
import LoadingSpinner from "@/components/common/LoadingSpinner";

export default function ExerciseDetails() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      fetchExercise(id)
        .then(setExercise)
        .catch(() => navigate("/exercises"))
        .finally(() => setLoading(false));
    }
  }, [id]);

  if (loading) return <LoadingSpinner />;
  if (!exercise) return null;

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Back */}
      <button
        onClick={() => navigate("/exercises")}
        className="flex items-center gap-2 text-surface-500 hover:text-surface-900 text-sm font-medium mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Exercises
      </button>

      {/* Header */}
      <div className="text-center mb-10">
        <div className="text-6xl mb-4">{exercise.icon}</div>
        <h1 className="text-3xl font-extrabold text-surface-900 mb-2">{exercise.label}</h1>
        <p className="text-primary-500 font-medium">{exercise.target}</p>
      </div>

      {/* Video */}
      <div className="bg-surface-100 rounded-2xl overflow-hidden mb-8">
        <iframe
          width="100%"
          height="360"
          src={`https://www.youtube.com/embed/${exercise.video_id}`}
          frameBorder="0"
          allowFullScreen
          className="w-full"
        />
      </div>

      {/* Info grid */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-2xl border border-surface-200 p-6">
          <h3 className="font-bold text-surface-900 mb-3">About</h3>
          <p className="text-surface-600 text-sm leading-relaxed">{exercise.description}</p>
        </div>

        <div className="bg-white rounded-2xl border border-surface-200 p-6">
          <h3 className="font-bold text-surface-900 mb-3">Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-surface-500">Type</span>
              <span className="font-medium text-surface-900 capitalize">{exercise.type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-surface-500">Difficulty</span>
              <span className="font-medium text-surface-900 capitalize">{exercise.difficulty}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-surface-500">Recommended Reps</span>
              <span className="font-medium text-surface-900">{exercise.recommended_reps}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Form checks + Tips */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-2xl border border-surface-200 p-6">
          <h3 className="font-bold text-surface-900 mb-3">✅ Form Checks</h3>
          <ul className="space-y-2">
            {exercise.form_checks.map((check) => (
              <li key={check} className="flex items-center gap-2 text-sm text-surface-600">
                <div className="w-2 h-2 bg-emerald-500 rounded-full" />
                {check}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-primary-50 rounded-2xl border border-primary-100 p-6">
          <h3 className="font-bold text-primary-900 mb-3">💡 Pro Tip</h3>
          <p className="text-primary-700 text-sm leading-relaxed">{exercise.tip}</p>
        </div>
      </div>

      {/* CTA */}
      <button
        onClick={() => navigate(`/exercises/${exercise.id}/setup`)}
        className="w-full py-4 bg-primary-500 text-white rounded-2xl font-bold text-base hover:bg-primary-600 transition-all shadow-lg shadow-primary-200 flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Start Session
      </button>
    </div>
  );
}
