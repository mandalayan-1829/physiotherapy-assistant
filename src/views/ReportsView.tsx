import { useCallback, useEffect, useMemo, useState } from 'react';
import { 
  Activity, 
  AlertCircle, 
  AlertTriangle, 
  ArrowDown, 
  ArrowUp, 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  Download, 
  FileSpreadsheet, 
  FileText, 
  History, 
  Mail, 
  RefreshCw, 
  Send, 
  ShieldCheck, 
  Stethoscope, 
  TrendingUp, 
  User as UserIcon, 
  X 
} from 'lucide-react';
import { Appointment, Doctor, MonthlyReport, Session, User } from '../types';
import { getHistoricalDailyEntries, getTodayDateString, getTodaySessions } from '../utils/storage';
import { generateReport, listReports } from '../services/physio';
import type { ReportPresentationContext } from '../services/mappers';

interface ReportsViewProps {
  user: User;
  sessions: Session[];
  appointments: Appointment[];
  doctors: Doctor[];
  onNavigate?: (tab: any) => void;
}

export function ReportsView({
  user,
  sessions,
  appointments,
  doctors,
  onNavigate,
}: ReportsViewProps) {
  const [activeSubTab, setActiveSubTab] = useState<'daily' | 'weekly' | 'monthly' | 'history'>('monthly');
  const [selectedMonth, setSelectedMonth] = useState<string>(getTodayDateString().slice(0, 7));
  // Reports are read from the backend, so the view has to model the request
  // lifecycle: loading, failure and "this month has no report yet".
  const [reportsList, setReportsList] = useState<MonthlyReport[]>([]);
  const [reportsLoading, setReportsLoading] = useState<boolean>(true);
  const [reportsError, setReportsError] = useState<string | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);
  const [emailStatusFeedback, setEmailStatusFeedback] = useState<string | null>(null);
  const [selectedExerciseFilter, setSelectedExerciseFilter] = useState<string>('all');

  // Today and historical aggregates
  const todaySessions = getTodaySessions(sessions);
  const dailyEntries = getHistoricalDailyEntries(sessions);
  // Repetitions and form scores are only reported for sessions that actually
  // measured a movement. Nothing is extrapolated from an unmeasured session.
  const measuredToday = todaySessions.filter((s) => s.metricsSource === 'pose_inference');
  const todayFormScore = measuredToday.length
    ? Math.round(measuredToday.reduce((acc, s) => acc + s.formAccuracy, 0) / measuredToday.length)
    : 0;
  const todayMeasuredReps = measuredToday.reduce((acc, s) => acc + s.reps, 0);
  const todayGoodReps = measuredToday.length
    ? Math.round(todayMeasuredReps * (todayFormScore / 100))
    : 0;
  const todayEntry = dailyEntries.find((d) => d.date === getTodayDateString()) || {
    date: getTodayDateString(),
    displayDate: 'Today',
    sessionsCount: todaySessions.length,
    measuredSessions: measuredToday.length,
    totalReps: todaySessions.reduce((acc, s) => acc + s.reps, 0),
    measuredReps: todayMeasuredReps,
    correctReps: todayGoodReps,
    incorrectReps: measuredToday.length ? Math.max(0, todayMeasuredReps - todayGoodReps) : 0,
    avgFormScore: todayFormScore,
    totalDurationSec: todaySessions.reduce((acc, s) => acc + s.durationSec, 0),
    exercises: Array.from(new Set(todaySessions.map((s) => s.exerciseLabel))),
    sessions: todaySessions,
  };

  // Weekly calculations (last 7 days)
  const weekCutoff = new Date();
  weekCutoff.setDate(weekCutoff.getDate() - 7);
  const cutoffStr = weekCutoff.toISOString().split('T')[0];
  const weekSessions = sessions.filter((s) => s.date.split(' ')[0] >= cutoffStr);
  const weekMeasured = weekSessions.filter((s) => s.metricsSource === 'pose_inference');
  const weekTotalReps = weekSessions.reduce((acc, s) => acc + s.reps, 0);
  // A window with no measured session reports no score. It does not report a
  // default or "typical" percentage.
  const weekAvgAccuracy = weekMeasured.length > 0
    ? Math.round(weekMeasured.reduce((acc, s) => acc + s.formAccuracy, 0) / weekMeasured.length)
    : null;
  const weekTargetSessions = 7;
  const weekActiveDays = new Set(weekSessions.map((s) => s.date.split(' ')[0])).size;

  /**
   * Presentation-only context the backend does not own.
   *
   * The backend computes every measurement in a report from persisted sessions.
   * Whether a checkup is booked, and which clinician is attending, are assembled
   * here from the appointment and directory data this view already holds.
   */
  const reportContext: ReportPresentationContext = useMemo(() => {
    const upcoming = appointments.find(
      (appointment) =>
        (appointment.status === 'confirmed' ||
          appointment.status === 'scheduled' ||
          appointment.status === 'ready' ||
          appointment.status === 'approved') &&
        appointment.date >= getTodayDateString(),
    );
    const assignedDoc = doctors.find((doctor) => doctor.name === user.doctorName);
    return {
      assignedDoctorName: assignedDoc?.name,
      assignedDoctorEmail: assignedDoc?.email,
      hasUpcomingCheckup: !!upcoming,
    };
  }, [appointments, doctors, user.doctorName]);

  const loadReports = useCallback(async () => {
    setReportsLoading(true);
    setReportsError(null);
    try {
      setReportsList(await listReports(reportContext));
    } catch (error) {
      setReportsError(
        error instanceof Error ? error.message : 'Reports could not be loaded.',
      );
    } finally {
      setReportsLoading(false);
    }
  }, [reportContext]);

  useEffect(() => {
    void loadReports();
  }, [loadReports]);

  // The month is only "active" if the backend actually holds a report for it.
  // Nothing is generated as a side effect of rendering.
  const activeMonthlyReport = reportsList.find((r) => r.monthKey === selectedMonth) ?? null;

  const handleGenerateReport = async () => {
    setGenerating(true);
    setReportsError(null);
    try {
      const fresh = await generateReport(selectedMonth, reportContext);
      await loadReports();
      setReportsList((previous) => {
        const withoutMonth = previous.filter((r) => r.monthKey !== fresh.monthKey);
        return [fresh, ...withoutMonth].sort((a, b) => b.monthKey.localeCompare(a.monthKey));
      });
      setEmailStatusFeedback(
        `Report for ${fresh.monthName} generated from ${fresh.measuredSessions} measured session(s) of ${fresh.totalSessions} recorded.`,
      );
      setTimeout(() => setEmailStatusFeedback(null), 5000);
    } catch (error) {
      setReportsError(
        error instanceof Error ? error.message : 'The report could not be generated.',
      );
    } finally {
      setGenerating(false);
    }
  };

  const handleSendEmail = (report: MonthlyReport) => {
    // No mail transport is configured for reports, so nothing is dispatched. Say
    // so rather than recording a delivery that did not happen - and do not
    // persist anything locally to imply one.
    setEmailStatusFeedback(
      `Report e-mail delivery is not configured, so no message was sent. Intended recipients: ${report.recipients.join(', ')}.`,
    );
    setTimeout(() => setEmailStatusFeedback(null), 5000);
  };

  const handleExportReport = (report: MonthlyReport) => {
    const reportText = `====================================================
PHYSIOAI CLINICAL REHABILITATION REPORT
====================================================
Report ID: ${report.id}
Month: ${report.monthName}
Generated On: ${report.generatedDate}
Patient: ${report.patientName} (${report.patientEmail})
Attending Physician: ${report.assignedDoctorName || 'None assigned'} (${report.assignedDoctorEmail || 'N/A'})
Upcoming Scheduled Checkup: ${report.hasUpcomingCheckup ? 'Yes (Clinical review pending)' : 'No active appointment'}

CLINICAL REHABILITATION SUMMARY:
- Total Workouts / Sessions: ${report.totalSessions}
- Cumulative Repetitions: ${report.totalReps}
- Sessions With Measured Movement: ${report.measuredSessions} of ${report.totalSessions}
- Average Form Score (measured sessions only): ${report.measuredSessions > 0 ? `${report.avgAccuracy}%` : 'Not measured'}
- Days Active In Month: ${report.activeDaysPercent}% of elapsed days

EXERCISE BREAKDOWN:
${report.exerciseBreakdown.map((ex) => `• ${ex.exerciseLabel}: ${ex.reps} reps across ${ex.sessions} sessions (Avg Form: ${ex.avgAccuracy}%)`).join('\n')}

PHYSICIAN TRAJECTORY NOTE:
${report.progressTrend}

DELIVERY STATUS:
- Status: ${report.emailStatus}
${report.emailStatus === 'Not sent'
  ? '- NOTE: report e-mail delivery is not configured, so no message was transmitted.'
  : `- Intended Recipients: ${report.recipients.join(', ')}`}
${report.emailSentDate ? `- Sent Date: ${report.emailSentDate}` : ''}
====================================================`;

    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PhysioAI_Report_${report.monthKey}_${user.name.replace(/\s+/g, '_')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header & Sub-Tabs */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Clinical Reports & Rehabilitation History
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-blue-600" />
              <span>Rehabilitation Reports</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Comprehensive daily workout logs, weekly recovery trends, automated monthly physician reports, and historical archives.
            </p>
          </div>

          {/* Report Type Selector Tabs */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveSubTab('daily')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeSubTab === 'daily'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Daily Report
            </button>
            <button
              onClick={() => setActiveSubTab('weekly')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeSubTab === 'weekly'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Weekly Trend
            </button>
            <button
              onClick={() => setActiveSubTab('monthly')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeSubTab === 'monthly'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Monthly Report
            </button>
            <button
              onClick={() => setActiveSubTab('history')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeSubTab === 'history'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Report History ({reportsList.length})
            </button>
          </div>
        </div>
      </div>

      {emailStatusFeedback && (
        <div className="p-3.5 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-xs text-[#065F46] flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600" />
            <span className="font-medium">{emailStatusFeedback}</span>
          </div>
          <button
            onClick={() => setEmailStatusFeedback(null)}
            className="text-slate-400 hover:text-slate-600"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ================================================================ */}
      {/* SECTION A: DAILY REPORT */}
      {/* ================================================================ */}
      {activeSubTab === 'daily' && (
        <div className="space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-4 rounded-xl border border-slate-200">
            <div>
              <span className="text-xs font-mono uppercase text-slate-500">Active Day Log</span>
              <h2 className="text-lg font-bold text-slate-900">Today's Rehabilitation Summary ({todayEntry.displayDate})</h2>
              <p className="text-xs text-slate-600">
                Exercises performed today with computer vision kinematics form scoring and repetition validity.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                {todaySessions.length > 0 ? `${todaySessions.length} Sessions Logged` : 'Ready for Workout'}
              </span>
            </div>
          </div>

          {/* Daily Metric Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[11px] font-mono uppercase text-slate-500 block">Total Repetitions</span>
              <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{todayEntry.totalReps}</span>
              <span className="text-xs text-slate-500 mt-0.5 block">Across all exercises</span>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[11px] font-mono uppercase text-slate-500 block">Form Accuracy</span>
              <span className="text-2xl font-bold font-mono text-emerald-600 mt-1 block">
                {todayEntry.avgFormScore > 0 ? `${todayEntry.avgFormScore}%` : 'N/A'}
              </span>
              <span className="text-xs text-slate-500 mt-0.5 block">Joint alignment score</span>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[11px] font-mono uppercase text-slate-500 block">Correct vs Needs Work</span>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-2xl font-bold font-mono text-emerald-600">{todayEntry.correctReps}</span>
                <span className="text-xs text-slate-400">/</span>
                <span className="text-sm font-bold font-mono text-amber-600">{todayEntry.incorrectReps}</span>
              </div>
              <span className="text-xs text-slate-500 mt-0.5 block">Valid form reps</span>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[11px] font-mono uppercase text-slate-500 block">Active Therapy Time</span>
              <span className="text-2xl font-bold font-mono text-blue-600 mt-1 block">
                {Math.floor(todayEntry.totalDurationSec / 60)}m {todayEntry.totalDurationSec % 60}s
              </span>
              <span className="text-xs text-slate-500 mt-0.5 block">Time under tension</span>
            </div>
          </div>

          {/* Today's Individual Session Logs */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
            <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Today's Executed Exercises</span>
              <span className="text-xs text-slate-500 font-mono">{todaySessions.length} record(s)</span>
            </div>

            {todaySessions.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs">
                <Activity className="w-8 h-8 mx-auto text-slate-300 mb-2" />
                <p className="font-semibold text-slate-700">No exercise sessions logged today yet.</p>
                <p className="mt-1">Daily counters reset each morning. Start a live camera session to record today's rehabilitation.</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {todaySessions.map((s) => (
                  <div key={s.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-slate-50/60">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-slate-900">{s.exerciseLabel}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                          {s.metricsSource === 'pose_inference' ? `${s.formAccuracy}% form score` : 'Not measured'}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">{s.date.split(' ')[1] || s.date}</span>
                      </div>
                      <p className="text-xs text-slate-600">
                        {s.notes ||
                          (s.metricsSource === 'pose_inference'
                            ? 'Session recorded with pose-inference measurement.'
                            : 'Session recorded without a measurement.')}
                      </p>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-mono shrink-0">
                      <div className="text-right">
                        <span className="text-slate-400 block text-[10px]">REPETITIONS</span>
                        <span className="text-slate-900 font-bold">{s.reps} / {s.targetReps} reps</span>
                      </div>
                      <div className="text-right">
                        <span className="text-slate-400 block text-[10px]">DURATION</span>
                        <span className="text-slate-900 font-bold">{s.durationSec}s</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* SECTION B: WEEKLY REPORT */}
      {/* ================================================================ */}
      {activeSubTab === 'weekly' && (
        <div className="space-y-5">
          <div className="bg-white p-4 rounded-xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <span className="text-xs font-mono uppercase text-slate-500">7-Day Rolling Window</span>
              <h2 className="text-lg font-bold text-slate-900">Weekly Kinematic & Recovery Trend</h2>
              <p className="text-xs text-slate-600">
                Cumulative repetitions, consistency rate, and joint stability across the last 7 calendar days.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                Active days: {weekActiveDays} of {weekTargetSessions}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-xs text-slate-500 block">Weekly Workout Sessions</span>
              <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{weekSessions.length} / 7</span>
              <span className="text-xs text-slate-500 mt-1 flex items-center gap-1 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Recorded in the last 7 days</span>
              </span>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-xs text-slate-500 block">Total Weekly Repetitions</span>
              <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{weekTotalReps}</span>
              <span className="text-xs text-slate-500 mt-1 block">Prescribed rehabilitation sets</span>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
              <span className="text-xs text-slate-500 block">Average Form Score</span>
              <span className="text-2xl font-bold font-mono text-emerald-600 mt-1 block">
                {weekAvgAccuracy === null ? 'N/A' : `${weekAvgAccuracy}%`}
              </span>
              <span className="text-xs text-slate-500 mt-1 block">
                {weekAvgAccuracy === null ? 'No measured session in this window' : 'From pose-inference sessions only'}
              </span>
            </div>
          </div>

          {/* Clean Day-by-Day Activity Visualization */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-4">
              Daily Repetition Volume & Quality
            </h3>
            
            <div className="space-y-3">
              {dailyEntries.slice(0, 7).map((day) => {
                const barWidth = Math.min(100, Math.round((day.totalReps / 30) * 100));
                return (
                  <div key={day.date} className="flex items-center gap-3 text-xs">
                    <span className="w-28 font-mono text-slate-600 shrink-0 truncate">{day.date}</span>
                    <div className="flex-1 bg-slate-100 h-5 rounded-md overflow-hidden relative">
                      <div
                        className="bg-blue-600 h-full rounded-md transition-all duration-300"
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <span className="w-20 text-right font-mono font-bold text-slate-900 shrink-0">
                      {day.totalReps} reps
                    </span>
                    <span className="w-14 text-right font-mono font-semibold text-emerald-600 shrink-0">
                      {day.avgFormScore}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* SECTION C: MONTHLY REPORT (Requirements 7, 8, 9) */}
      {/* ================================================================ */}
      {/* Loading / failure / empty states. A month only renders a report document
          once the backend confirms it holds one. */}
      {activeSubTab === 'monthly' && reportsLoading && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center text-slate-500 text-sm">
          <RefreshCw className="w-6 h-6 mx-auto text-slate-300 mb-2 animate-spin" />
          <p className="font-semibold text-slate-700">Loading stored reports…</p>
        </div>
      )}

      {activeSubTab === 'monthly' && !reportsLoading && reportsError && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start justify-between gap-3">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600 mt-0.5" />
            <div>
              <p className="font-semibold">Reports could not be loaded</p>
              <p className="mt-0.5 leading-relaxed">{reportsError}</p>
            </div>
          </div>
          <button
            onClick={() => void loadReports()}
            className="px-3 py-1.5 rounded-lg bg-white border border-rose-200 text-rose-700 font-semibold shrink-0 cursor-pointer hover:bg-rose-100"
          >
            Retry
          </button>
        </div>
      )}

      {activeSubTab === 'monthly' && !reportsLoading && !reportsError && !activeMonthlyReport && (
        <div className="bg-white rounded-xl border border-slate-200 p-10 text-center">
          <FileText className="w-8 h-8 mx-auto text-slate-300 mb-2" />
          <p className="font-semibold text-slate-700">No report stored for this month</p>
          <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto leading-relaxed">
            A report is generated from the sessions recorded in the month you select.
            Nothing is created until you ask for it.
          </p>
          <button
            onClick={() => void handleGenerateReport()}
            disabled={generating}
            className="mt-4 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-white text-xs font-bold inline-flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${generating ? 'animate-spin' : ''}`} />
            <span>{generating ? 'Generating…' : `Generate report for ${selectedMonth}`}</span>
          </button>
        </div>
      )}

      {activeSubTab === 'monthly' && !reportsLoading && !reportsError && activeMonthlyReport && (
        <div className="space-y-5">
          
          {/* Controls Bar: Month Selector, Generate, Export, Email */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xs">
            <div className="flex items-center gap-3">
              <div>
                <label className="block text-[10px] font-mono uppercase text-slate-500 font-semibold mb-0.5">
                  Reporting Month
                </label>
                <select
                  value={selectedMonth}
                  onChange={(e) => setSelectedMonth(e.target.value)}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white text-xs font-semibold text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="2026-09">September 2026 (Current)</option>
                  <option value="2026-08">August 2026</option>
                  <option value="2026-07">July 2026</option>
                </select>
              </div>

              <button
                onClick={handleGenerateReport}
                className="px-3 py-1.5 mt-4 md:mt-0 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer border border-slate-200"
              >
                <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${generating ? 'animate-spin' : ''}`} />
                <span>{generating ? 'Generating…' : 'Re-Generate'}</span>
              </button>
            </div>

            <div className="flex items-center gap-2 self-end md:self-auto">
              <button
                onClick={() => handleExportReport(activeMonthlyReport)}
                className="px-3.5 py-1.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold flex items-center gap-1.5 border border-slate-300 transition-colors cursor-pointer shadow-xs"
              >
                <Download className="w-3.5 h-3.5 text-slate-500" />
                <span>Export / Download</span>
              </button>

              <button
                onClick={() => handleSendEmail(activeMonthlyReport)}
                className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
              >
                <Send className="w-3.5 h-3.5 fill-white" />
                <span>Send Monthly Report Email</span>
              </button>
            </div>
          </div>

          {/* Monthly Report Document Card */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            
            {/* Clinical Header Banner */}
            <div className="p-6 border-b border-slate-200 bg-slate-50/50">
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-blue-100 text-blue-800 border border-blue-200 font-semibold">
                      Clinical Rehabilitation Record
                    </span>
                    <span className="text-xs text-slate-400">•</span>
                    <span className="text-xs font-mono text-slate-500">{activeMonthlyReport.id}</span>
                  </div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1.5">
                    Monthly Rehabilitation Assessment: {activeMonthlyReport.monthName}
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Generated: {activeMonthlyReport.generatedDate} • {activeMonthlyReport.measuredSessions} of {activeMonthlyReport.totalSessions} sessions measured by pose inference
                  </p>
                </div>

                {/* Delivery & Email Status Badge (Requirement 8) */}
                <div className="text-right sm:self-start">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border"
                    style={{
                      backgroundColor: activeMonthlyReport.emailStatus === 'Sent' ? '#ECFDF3' : '#FFF8E6',
                      borderColor: activeMonthlyReport.emailStatus === 'Sent' ? '#A7F3D0' : '#FDE68A',
                      color: activeMonthlyReport.emailStatus === 'Sent' ? '#065F46' : '#92400E'
                    }}
                  >
                    <Mail className="w-3.5 h-3.5" />
                    <span>Email Status: {activeMonthlyReport.emailStatus}</span>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1">
                    {activeMonthlyReport.emailSentDate
                      ? `Marked sent ${activeMonthlyReport.emailSentDate}`
                      : 'Delivery not configured - no message is transmitted'}
                  </p>
                </div>
              </div>

              {/* Patient & Doctor Recipient Card */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4 pt-4 border-t border-slate-200/80 text-xs">
                <div className="bg-white p-3 rounded-lg border border-slate-200">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">PATIENT PARTICULARS</span>
                  <span className="font-bold text-slate-900 text-sm mt-0.5 block">{activeMonthlyReport.patientName}</span>
                  <span className="text-slate-600 block">{activeMonthlyReport.patientEmail}</span>
                  <span className="text-slate-500 text-[11px] block mt-1">Prescribed Focus: {user.currentProblem}</span>
                </div>

                <div className="bg-white p-3 rounded-lg border border-slate-200">
                  <span className="text-[10px] font-mono uppercase text-slate-400 block">ATTENDING CLINICIAN & RECIPIENT</span>
                  <span className="font-bold text-slate-900 text-sm mt-0.5 block">
                    {activeMonthlyReport.assignedDoctorName || 'No clinician assigned'}
                  </span>
                  <span className="text-slate-600 block">
                    {activeMonthlyReport.assignedDoctorEmail || 'Not on file'}
                  </span>
                  <div className="mt-1 flex items-center gap-1.5">
                    {activeMonthlyReport.hasUpcomingCheckup ? (
                      <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                        Upcoming Checkup Scheduled (Doctor receives report copy)
                      </span>
                    ) : (
                      <span className="text-[11px] text-slate-500">
                        No active checkup (Sent to patient only)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Core Metrics Strip */}
            <div className="p-6 border-b border-slate-200">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
                1. Monthly Kinematic Summary
              </h3>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">Total Sessions</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-1 block">
                    {activeMonthlyReport.totalSessions}
                  </span>
                  <span className="text-[11px] text-slate-500">Completed workouts</span>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-[10px] font-mono uppercase text-slate-500 block">Total Repetitions</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-1 block">
                    {activeMonthlyReport.totalReps}
                  </span>
                  <span className="text-[11px] text-slate-500">Logged across month</span>
                </div>

                <div className="p-3 bg-[#ECFDF3]/60 rounded-lg border border-[#A7F3D0]">
                  <span className="text-[10px] font-mono uppercase text-[#065F46] block">Average Form Score</span>
                  <span className="text-xl font-bold font-mono text-emerald-700 mt-1 block">
                    {activeMonthlyReport.measuredSessions > 0 ? `${activeMonthlyReport.avgAccuracy}%` : 'N/A'}
                  </span>
                  <span className="text-[11px] text-emerald-800">
                    {activeMonthlyReport.measuredSessions > 0
                      ? `From ${activeMonthlyReport.measuredSessions} measured session(s)`
                      : 'No measured session this month'}
                  </span>
                </div>

                <div className="p-3 bg-[#F0F7FF]/60 rounded-lg border border-blue-200">
                  <span className="text-[10px] font-mono uppercase text-blue-700 block">Days Active</span>
                  <span className="text-xl font-bold font-mono text-blue-700 mt-1 block">
                    {activeMonthlyReport.activeDaysPercent}%
                  </span>
                  <span className="text-[11px] text-blue-800">of elapsed days had a session</span>
                </div>
              </div>
            </div>

            {/* Exercise-Wise Breakdown Table */}
            <div className="p-6 border-b border-slate-200">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
                2. Exercise-Wise Performance Breakdown
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border border-slate-200 rounded-lg overflow-hidden">
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Prescribed Exercise</th>
                      <th className="py-2.5 px-3">Sessions</th>
                      <th className="py-2.5 px-3">Total Reps</th>
                      <th className="py-2.5 px-3">Kinematic Form Score</th>
                      <th className="py-2.5 px-3">Clinical Evaluation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {activeMonthlyReport.exerciseBreakdown.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="py-2.5 px-3 font-semibold text-slate-900">{item.exerciseLabel}</td>
                        <td className="py-2.5 px-3 font-mono text-slate-700">{item.sessions}</td>
                        <td className="py-2.5 px-3 font-mono font-bold text-slate-900">{item.reps}</td>
                        <td className="py-2.5 px-3 font-mono font-bold text-emerald-600">{item.avgAccuracy}%</td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                            Therapeutic Range
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Improvement Trend & Safety Notes */}
            <div className="p-6 bg-slate-50/50 space-y-4">
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-1.5">
                  3. Kinematic Trajectory & Progress Trend
                </h3>
                <p className="text-xs text-slate-700 leading-relaxed bg-white p-3.5 rounded-lg border border-slate-200">
                  {activeMonthlyReport.progressTrend}
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-white rounded-lg border border-slate-200">
                  <span className="font-semibold text-slate-800 block">Measurement Coverage</span>
                  <p className="text-slate-600 mt-0.5">
                    {activeMonthlyReport.measuredSessions === activeMonthlyReport.totalSessions
                      ? 'Every session this month included a measured movement.'
                      : `${activeMonthlyReport.totalSessions - activeMonthlyReport.measuredSessions} of ${activeMonthlyReport.totalSessions} session(s) recorded no measurement. No safety-event or adverse-anomaly classification is produced.`}
                  </p>
                </div>

                <div className="p-3 bg-white rounded-lg border border-slate-200">
                  <span className="font-semibold text-slate-800 block">Report Recipients</span>
                  <p className="text-slate-600 mt-0.5 font-mono text-[11px]">
                    {activeMonthlyReport.recipients.join(' • ')}
                  </p>
                </div>
              </div>
            </div>

          </div>

        </div>
      )}

      {/* ================================================================ */}
      {/* SECTION D: REPORT HISTORY (Requirement 9) */}
      {/* ================================================================ */}
      {activeSubTab === 'history' && (
        <div className="space-y-4">
          <div className="bg-white p-4 rounded-xl border border-slate-200 flex items-center justify-between">
            <div>
              <span className="text-xs font-mono uppercase text-slate-500">Historical Archive</span>
              <h2 className="text-lg font-bold text-slate-900">Generated Monthly Reports History</h2>
              <p className="text-xs text-slate-600">
                View past monthly rehabilitation records, dispatch status, and download records.
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500 font-semibold">
              {reportsList.length} Archived Reports
            </span>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 divide-y divide-slate-100 shadow-xs overflow-hidden">
            {reportsList.map((rpt) => (
              <div key={rpt.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-900">{rpt.monthName} Report</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                      rpt.emailStatus === 'Sent'
                        ? 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]'
                        : 'bg-[#FFF8E6] text-[#92400E] border-[#FDE68A]'
                    }`}>
                      {rpt.emailStatus === 'Sent' ? 'Marked sent' : 'Draft - not sent'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Generated: {rpt.generatedDate} • {rpt.totalSessions} sessions ({rpt.totalReps} total reps) • Form score: {rpt.measuredSessions > 0 ? `${rpt.avgAccuracy}% from ${rpt.measuredSessions} measured` : 'not measured'}
                  </p>
                  <p className="text-[11px] text-slate-600 font-mono">
                    Recipients (intended): {rpt.recipients.join(', ')}
                  </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => {
                      setSelectedMonth(rpt.monthKey);
                      setActiveSubTab('monthly');
                    }}
                    className="px-3 py-1.5 rounded-lg bg-[#F0F7FF] hover:bg-blue-100 text-blue-700 text-xs font-semibold transition-colors cursor-pointer border border-blue-200"
                  >
                    View Report
                  </button>

                  <button
                    onClick={() => handleExportReport(rpt)}
                    className="p-1.5 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-slate-200 cursor-pointer"
                    title="Download Report File"
                  >
                    <Download className="w-4 h-4" />
                  </button>

                  {rpt.emailStatus !== 'Sent' && (
                    <button
                      onClick={() => handleSendEmail(rpt)}
                      className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer"
                    >
                      <Send className="w-3 h-3 fill-white" />
                      <span>Send Email</span>
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
