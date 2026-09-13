import { useState } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  Filter, 
  History, 
  Search, 
  ShieldCheck 
} from 'lucide-react';
import { DailyHistoryEntry, Session, User } from '../types';
import { getHistoricalDailyEntries, getTodayDateString } from '../utils/storage';

interface HistoryViewProps {
  user: User;
  sessions: Session[];
}

export function HistoryView({ user, sessions }: HistoryViewProps) {
  const [expandedDates, setExpandedDates] = useState<{ [date: string]: boolean }>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedExerciseFilter, setSelectedExerciseFilter] = useState('all');

  const dailyEntries = getHistoricalDailyEntries(sessions);
  const todayStr = getTodayDateString();

  const toggleExpand = (dateStr: string) => {
    setExpandedDates((prev) => ({
      ...prev,
      [dateStr]: !prev[dateStr],
    }));
  };

  // Collect all unique exercises across history
  const allExercises = Array.from(new Set(sessions.map((s) => s.exerciseLabel)));

  // Filter daily entries
  const filteredEntries = dailyEntries.filter((entry) => {
    const matchesSearch = 
      entry.date.includes(searchQuery) ||
      entry.displayDate.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.exercises.some((e) => e.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesExercise = 
      selectedExerciseFilter === 'all' ||
      entry.exercises.includes(selectedExerciseFilter);

    return matchesSearch && matchesExercise;
  });

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Permanent Clinical Records
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <History className="w-5 h-5 text-blue-600" />
              <span>Rehabilitation History & Daily Archives</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Separate historical tracker preserving every day's completed workouts, reps, accuracy, and clinical notes.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 font-mono font-semibold border border-slate-200">
              {dailyEntries.length} Total Active Days Recorded
            </span>
          </div>
        </div>
      </div>

      {/* Filter & Search Toolbar */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search dates or exercises..."
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
          <Filter className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs text-slate-500 font-medium">Exercise:</span>
          <select
            value={selectedExerciseFilter}
            onChange={(e) => setSelectedExerciseFilter(e.target.value)}
            className="px-2.5 py-1.5 text-xs rounded-lg border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Exercises ({allExercises.length})</option>
            {allExercises.map((ex) => (
              <option key={ex} value={ex}>{ex}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Notice on Daily Reset vs Permanent Storage */}
      <div className="p-3.5 bg-[#F0F7FF] rounded-xl border border-blue-200 flex items-center gap-3 text-xs text-blue-900">
        <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0" />
        <div>
          <span className="font-semibold">Persistent Rehabilitation Archive: </span>
          <span>
            Daily counters on the home dashboard reset each morning, but all historical sessions remain permanently stored here for clinical auditing and monthly report generation.
          </span>
        </div>
      </div>

      {/* Daily Entries List */}
      <div className="space-y-3">
        {filteredEntries.length === 0 ? (
          <div className="p-10 text-center bg-white rounded-xl border border-slate-200 text-slate-500 text-xs">
            <History className="w-8 h-8 mx-auto text-slate-300 mb-2" />
            <p className="font-semibold text-slate-700">No historical records match your filter.</p>
            <p className="mt-1">Clear your search query to view all historical dates.</p>
          </div>
        ) : (
          filteredEntries.map((day) => {
            const isToday = day.date === todayStr;
            const isExpanded = !!expandedDates[day.date];

            return (
              <div
                key={day.date}
                className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden transition-all hover:border-slate-300"
              >
                {/* Date Header Row (Clickable) */}
                <button
                  onClick={() => toggleExpand(day.date)}
                  className="w-full p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-left cursor-pointer hover:bg-slate-50/70 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-slate-100 text-slate-600 border border-slate-200">
                      <Calendar className="w-4 h-4 text-slate-700" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-slate-900">{day.displayDate}</span>
                        {isToday && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                            Today's Log
                          </span>
                        )}
                        <span className="text-xs text-slate-400 font-mono">({day.date})</span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 mt-1">
                        {day.exercises.map((ex) => (
                          <span
                            key={ex}
                            className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200"
                          >
                            {ex}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Summary Badges on the Right */}
                  <div className="flex items-center gap-4 sm:gap-6 self-end sm:self-center text-xs font-mono">
                    <div className="text-right">
                      <span className="text-[10px] text-slate-400 block font-sans">REPETITIONS</span>
                      <span className="font-bold text-slate-900">{day.totalReps}</span>
                    </div>

                    <div className="text-right">
                      <span className="text-[10px] text-slate-400 block font-sans">FORM SCORE</span>
                      <span className="font-bold text-emerald-600">{day.avgFormScore}%</span>
                    </div>

                    <div className="text-right hidden sm:block">
                      <span className="text-[10px] text-slate-400 block font-sans">DURATION</span>
                      <span className="text-slate-700">{Math.floor(day.totalDurationSec / 60)}m {day.totalDurationSec % 60}s</span>
                    </div>

                    <div className="text-slate-400 pl-2">
                      {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    </div>
                  </div>
                </button>

                {/* Expanded Session Breakdown */}
                {isExpanded && (
                  <div className="p-4 bg-slate-50/70 border-t border-slate-200 space-y-3">
                    
                    {/* Day Metric Pills */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                        <span className="text-slate-500 text-[10px] block">Workout Sessions</span>
                        <span className="font-bold text-slate-900 mt-0.5 block">{day.sessionsCount} session(s)</span>
                      </div>
                      <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                        <span className="text-slate-500 text-[10px] block">Valid Form Reps</span>
                        <span className="font-bold text-emerald-600 mt-0.5 block">{day.correctReps} reps</span>
                      </div>
                      <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                        <span className="text-slate-500 text-[10px] block">Needs Improvement</span>
                        <span className="font-bold text-amber-600 mt-0.5 block">{day.incorrectReps} reps</span>
                      </div>
                      <div className="bg-white p-2.5 rounded-lg border border-slate-200">
                        <span className="text-slate-500 text-[10px] block">Safety Events</span>
                        <span className="font-bold text-slate-900 mt-0.5 block">
                          {day.warningsCount === 0 ? '0 Warnings' : `${day.warningsCount} Notice`}
                        </span>
                      </div>
                    </div>

                    {/* Detailed Session Items */}
                    <div className="space-y-2 pt-1">
                      <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
                        Individual Workouts Logged
                      </span>

                      {day.sessions.map((s) => (
                        <div
                          key={s.id}
                          className="bg-white p-3 rounded-lg border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                        >
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-slate-900">{s.exerciseLabel}</span>
                              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200 font-mono">
                                {s.formAccuracy}% Accuracy
                              </span>
                              <span className="text-slate-400 font-mono text-[11px]">
                                {s.date.split(' ')[1] || s.date}
                              </span>
                            </div>
                            <p className="text-slate-600 mt-1">{s.notes || 'Posture alignment verified within therapeutic guidelines.'}</p>
                          </div>

                          <div className="flex items-center gap-4 text-xs font-mono shrink-0">
                            <div>
                              <span className="text-slate-400 block text-[10px]">GOAL / REPS</span>
                              <span className="font-bold text-slate-900">{s.reps} / {s.targetReps} reps</span>
                            </div>
                            <div>
                              <span className="text-slate-400 block text-[10px]">TIME</span>
                              <span className="text-slate-900">{s.durationSec}s</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

    </div>
  );
}
