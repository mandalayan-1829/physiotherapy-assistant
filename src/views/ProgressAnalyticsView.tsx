import { useState } from 'react';
import { 
  Activity, 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  Dumbbell, 
  HeartPulse, 
  ShieldCheck, 
  TrendingUp,
  Filter
} from 'lucide-react';
import { Session } from '../types';
import { EXERCISES } from '../data/exercises';

interface ProgressAnalyticsViewProps {
  sessions: Session[];
}

export function ProgressAnalyticsView({ sessions }: ProgressAnalyticsViewProps) {
  const [filterExercise, setFilterExercise] = useState<string>('all');

  // 4 Expandable Sections
  const [openSections, setOpenSections] = useState<{
    overall: boolean;
    performance: boolean;
    trends: boolean;
    history: boolean;
  }>({
    overall: true,
    performance: true,
    trends: true,
    history: true,
  });

  const toggleSection = (key: keyof typeof openSections) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const totalSessions = sessions.length;
  const totalReps = sessions.reduce((acc, s) => acc + s.reps, 0);
  const totalMinutes = Math.round(sessions.reduce((acc, s) => acc + s.durationSec, 0) / 60);
  const avgAccuracy = sessions.length > 0
    ? Math.round(sessions.reduce((acc, s) => acc + s.formAccuracy, 0) / sessions.length)
    : 0;

  // Breakdown by exercise
  const exerciseStats: Record<string, { count: number; totalReps: number; avgAcc: number; type: string }> = {};
  sessions.forEach((s) => {
    if (!exerciseStats[s.exercise]) {
      const info = EXERCISES[s.exercise];
      exerciseStats[s.exercise] = { 
        count: 0, 
        totalReps: 0, 
        avgAcc: 0,
        type: info?.type || 'physio'
      };
    }
    exerciseStats[s.exercise].count += 1;
    exerciseStats[s.exercise].totalReps += s.reps;
    exerciseStats[s.exercise].avgAcc += s.formAccuracy;
  });

  Object.keys(exerciseStats).forEach((k) => {
    exerciseStats[k].avgAcc = Math.round(exerciseStats[k].avgAcc / exerciseStats[k].count);
  });

  const filteredSessions = filterExercise === 'all'
    ? sessions
    : sessions.filter((s) => s.exercise === filterExercise);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
            Biometric Telemetry & Outcomes
          </span>
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
          <HeartPulse className="w-5 h-5 text-blue-600" />
          <span>Rehabilitation Biometrics & Analytics</span>
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1">
          Historical record of joint angular precision, kinematic stability, and routine adherence.
        </p>
      </div>

      {/* Expandable Sections with Clean Dividers (No large boxed cards) */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* 1. Overall Progress */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('overall')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Activity className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Overall Progress Summary
                </h2>
                <span className="text-xs text-slate-500">
                  {totalSessions} sessions completed • {totalReps} total reps • {avgAccuracy}% avg accuracy
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.overall ? 'Collapse' : 'Expand'}
              </span>
              {openSections.overall ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.overall && (
            <div className="pb-6 pt-2 px-2">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-3 border-b border-slate-200 text-xs">
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Total Sessions</span>
                  <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{totalSessions}</span>
                  <span className="text-[11px] text-slate-400">Recorded workouts</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Cumulative Repetitions</span>
                  <span className="text-2xl font-bold font-mono text-emerald-600 mt-1 block">{totalReps}</span>
                  <span className="text-[11px] text-slate-400">Target repetitions</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Average Form Accuracy</span>
                  <span className="text-2xl font-bold font-mono text-blue-600 mt-1 block">{avgAccuracy}%</span>
                  <span className="text-[11px] text-slate-400">Computer vision score</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Active Exercise Time</span>
                  <span className="text-2xl font-bold font-mono text-slate-800 mt-1 block">{totalMinutes}m</span>
                  <span className="text-[11px] text-slate-400">Under physical tracking</span>
                </div>
              </div>

              <div className="pt-3 flex items-center justify-between text-xs text-slate-700">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  <span>Clinical Adherence: Patient is currently on track with the prescribed rehabilitation frequency.</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 2. Exercise Performance */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('performance')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Dumbbell className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-600 transition-colors">
                  Exercise Performance & Joint Accuracy
                </h2>
                <span className="text-xs text-slate-500">
                  Per-movement breakdown of volume, frequency, and alignment fidelity
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.performance ? 'Collapse' : 'Expand'}
              </span>
              {openSections.performance ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.performance && (
            <div className="pb-6 pt-2 px-2">
              {Object.keys(exerciseStats).length === 0 ? (
                <p className="text-xs text-slate-500 py-4 text-center">No workout data logged yet.</p>
              ) : (
                <div className="divide-y divide-slate-200">
                  {Object.entries(exerciseStats).map(([exId, stat]) => {
                    const info = EXERCISES[exId] || { label: exId, target: 'Rehab' };
                    return (
                      <div key={exId} className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-slate-900 text-sm">{info.label}</span>
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-100 text-slate-600 border border-slate-200">
                              {stat.type}
                            </span>
                          </div>
                          <p className="text-slate-500 text-xs mt-0.5">{info.target}</p>
                        </div>

                        <div className="flex items-center gap-6 shrink-0">
                          <div className="text-right">
                            <span className="text-slate-400 block text-[11px]">Volume</span>
                            <span className="text-slate-800 font-mono font-medium">{stat.count} sessions • {stat.totalReps} reps</span>
                          </div>

                          <div className="w-32">
                            <div className="flex justify-between text-[11px] mb-1">
                              <span className="text-slate-400">Score</span>
                              <span className="font-mono font-bold text-emerald-600">{stat.avgAcc}%</span>
                            </div>
                            <div className="w-full h-1.5 rounded-full bg-slate-100 overflow-hidden">
                              <div 
                                className="h-full bg-emerald-500 rounded-full" 
                                style={{ width: `${stat.avgAcc}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 3. Weekly Trends */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('trends')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <TrendingUp className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Weekly Trends & Kinematic Consistency
                </h2>
                <span className="text-xs text-slate-500">
                  Historical progression curve across successive workout dates
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.trends ? 'Collapse' : 'Expand'}
              </span>
              {openSections.trends ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.trends && (
            <div className="pb-6 pt-2 px-2 text-xs space-y-4">
              {sessions.length === 0 ? (
                <p className="text-slate-500 text-center py-4">No workout history available yet.</p>
              ) : (
                <div className="space-y-3">
                  <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-2">
                      Recent Session Accuracy Curve
                    </span>
                    <div className="space-y-2">
                      {sessions.slice(0, 6).map((s, idx) => (
                        <div key={s.id || idx} className="flex items-center gap-3 text-xs">
                          <span className="w-24 text-slate-500 font-mono text-[11px] truncate">{s.date.split(' ')[0]}</span>
                          <span className="w-36 text-slate-900 font-medium truncate">{s.exerciseLabel}</span>
                          <div className="flex-1 h-2 rounded-full bg-slate-200 overflow-hidden">
                            <div
                              className="h-full bg-blue-600 rounded-full"
                              style={{ width: `${s.formAccuracy}%` }}
                            />
                          </div>
                          <span className="w-12 text-right font-mono font-bold text-emerald-600">{s.formAccuracy}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 4. Session History */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('history')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Calendar className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Complete Session History & Clinical Log
                </h2>
                <span className="text-xs text-slate-500">
                  {filteredSessions.length} recorded session entries
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.history ? 'Collapse' : 'Expand'}
              </span>
              {openSections.history ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.history && (
            <div className="pb-6 pt-2 px-2 text-xs space-y-3">
              
              {/* Filter controls */}
              <div className="flex items-center justify-between gap-3 pb-2 border-b border-slate-200">
                <span className="text-slate-600 text-xs">Filter by exercise:</span>
                <select
                  value={filterExercise}
                  onChange={(e) => setFilterExercise(e.target.value)}
                  className="px-2.5 py-1.5 bg-white border border-slate-200 rounded-lg text-xs text-slate-800 focus:outline-none focus:border-blue-500 cursor-pointer"
                >
                  <option value="all">All Movements ({sessions.length})</option>
                  {Object.values(EXERCISES).map((ex) => (
                    <option key={ex.id} value={ex.id}>{ex.label}</option>
                  ))}
                </select>
              </div>

              {filteredSessions.length === 0 ? (
                <p className="text-slate-500 py-6 text-center text-xs">No records found for the chosen filter.</p>
              ) : (
                <div className="divide-y divide-slate-200">
                  {filteredSessions.map((session) => (
                    <div
                      key={session.id}
                      className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 hover:bg-slate-50 px-1 rounded transition-colors"
                    >
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-900 text-xs sm:text-sm">{session.exerciseLabel}</span>
                          <span className="text-slate-400 font-mono text-[11px]">• {session.date}</span>
                        </div>
                        <p className="text-slate-500 text-[11px]">{session.notes || 'Routine session completed.'}</p>
                      </div>

                      <div className="flex items-center gap-4 text-xs shrink-0 self-end sm:self-center">
                        <span className="text-slate-500 font-mono">{session.reps} reps / {Math.round(session.durationSec)}s</span>
                        <span className={`px-2 py-0.5 rounded font-mono font-bold text-xs ${
                          session.formAccuracy >= 90
                            ? 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                            : session.formAccuracy >= 80
                            ? 'bg-blue-50 text-blue-700 border border-blue-200'
                            : 'bg-amber-50 text-amber-800 border border-amber-200'
                        }`}>
                          {session.formAccuracy}% accuracy
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
