import { ArrowLeft, Dumbbell, Sparkles } from 'lucide-react';
import { Exercise, Session } from '../types';
import { PoseTracker } from '../components/PoseTracker';

interface TrackingViewProps {
  exercise: Exercise;
  targetReps: number;
  onSessionComplete: (session: Omit<Session, 'id'>) => void;
  onBack: () => void;
}

export function TrackingView({ exercise, targetReps, onSessionComplete, onBack }: TrackingViewProps) {
  return (
    <div className="space-y-6">
      
      {/* Top back navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-white border border-slate-200 text-xs font-medium text-slate-700 hover:text-slate-900 hover:bg-slate-50 transition-colors cursor-pointer shadow-xs"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Exercises</span>
        </button>

        {/* Neutral statement of fact: whether analysis is actually running is
            reported by the tracker itself, which knows which mode was chosen. */}
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Pose processing runs on this device</span>
        </div>
      </div>

      {/* Main Pose Tracker Component */}
      <PoseTracker
        exercise={exercise}
        targetReps={targetReps}
        onSessionComplete={onSessionComplete}
        onCancel={onBack}
      />

      {/* Exercise Clinical Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-blue-700 flex items-center gap-1.5 mb-2">
            <Sparkles className="w-3.5 h-3.5 text-blue-600" />
            <span>Target Anatomical Objectives</span>
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            {exercise.description}
          </p>
          <div className="mt-3 pt-2.5 border-t border-slate-200 text-xs text-slate-500">
            <span className="font-semibold text-slate-700">Monitored Kinetic Chain:</span> {exercise.primaryJoint}
          </div>
        </div>

        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs">
          <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-700 flex items-center gap-1.5 mb-2">
            <Dumbbell className="w-3.5 h-3.5 text-emerald-600" />
            <span>Key Biomechanical Form Checks</span>
          </h3>
          <ul className="space-y-1.5 text-xs text-slate-600">
            {exercise.formChecks.map((chk, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>{chk}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3 pt-2.5 border-t border-slate-200 text-xs text-blue-700 italic">
            "{exercise.tip}"
          </div>
        </div>
      </div>

    </div>
  );
}
