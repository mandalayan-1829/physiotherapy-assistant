import { useNavigate, useLocation } from "react-router-dom";
import { CheckCircle, RotateCcw, BarChart3, Home } from "lucide-react";
import Card from "@/components/common/Card";

export default function SessionResults() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as {
    reps: number;
    targetReps: number;
    formAccuracy: number;
    duration: number;
    exercise: any;
  } | null;

  const reps = state?.reps || 0;
  const targetReps = state?.targetReps || 10;
  const formAccuracy = state?.formAccuracy || 0;
  const duration = state?.duration || 0;
  const exercise = state?.exercise;

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}m ${s}s`;
  };

  return (
    <div className="p-8 max-w-2xl mx-auto text-center">
      <div className="mb-8">
        <div className="w-20 h-20 bg-emerald-50 rounded-full flex items-center justify-center mx-auto mb-4">
          <CheckCircle className="w-10 h-10 text-emerald-500" />
        </div>
        <h1 className="text-3xl font-extrabold text-surface-900 mb-2">Session Complete!</h1>
        <p className="text-surface-500">
          {exercise?.icon} {exercise?.label} — Great work!
        </p>
      </div>

      {/* Results */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <Card className="text-center">
          <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Reps Completed</p>
          <p className="text-3xl font-bold text-surface-900">
            {reps} <span className="text-lg text-surface-400">/ {targetReps}</span>
          </p>
        </Card>
        <Card className="text-center">
          <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Form Score</p>
          <p className={`text-3xl font-bold ${
            formAccuracy >= 80 ? "text-emerald-600" : formAccuracy >= 50 ? "text-amber-600" : "text-red-600"
          }`}>
            {formAccuracy}%
          </p>
        </Card>
        <Card className="text-center">
          <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Duration</p>
          <p className="text-2xl font-bold text-surface-900">{formatTime(duration)}</p>
        </Card>
        <Card className="text-center">
          <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Completion</p>
          <p className="text-2xl font-bold text-surface-900">
            {Math.round((reps / targetReps) * 100)}%
          </p>
        </Card>
      </div>

      {/* Actions */}
      <div className="space-y-3">
        <button
          onClick={() => navigate(`/exercises/${exercise?.id}/setup`)}
          className="w-full py-3.5 bg-primary-500 text-white rounded-xl font-semibold text-sm hover:bg-primary-600 transition-all flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Start Another Session
        </button>
        <button
          onClick={() => navigate("/progress")}
          className="w-full py-3.5 bg-surface-100 text-surface-700 rounded-xl font-semibold text-sm hover:bg-surface-200 transition-all flex items-center justify-center gap-2"
        >
          <BarChart3 className="w-4 h-4" />
          View Progress
        </button>
        <button
          onClick={() => navigate("/dashboard")}
          className="w-full py-3.5 bg-surface-100 text-surface-700 rounded-xl font-semibold text-sm hover:bg-surface-200 transition-all flex items-center justify-center gap-2"
        >
          <Home className="w-4 h-4" />
          Back to Dashboard
        </button>
      </div>
    </div>
  );
}
