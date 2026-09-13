import { useState } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  Dumbbell, 
  FileSpreadsheet, 
  HeartPulse, 
  History, 
  Phone, 
  Play, 
  ShieldAlert, 
  ShieldCheck, 
  TrendingUp, 
  UserCheck, 
  Video 
} from 'lucide-react';
import { Appointment, Exercise, Session, User } from '../types';
import { EXERCISES, getRecommendedExercises, isExerciseSafe } from '../data/exercises';
import { getTodayDateString, getTodaySessions, logGuardianAlert } from '../utils/storage';

interface DashboardHomeProps {
  user: User;
  sessions: Session[];
  appointments?: Appointment[];
  onStartExercise: (exercise: Exercise, targetReps?: number) => void;
  onNavigate: (tab: any) => void;
}

export function DashboardHome({ 
  user, 
  sessions, 
  appointments = [], 
  onStartExercise, 
  onNavigate 
}: DashboardHomeProps) {
  // Accordion sections state
  const [openSections, setOpenSections] = useState<{
    rehab: boolean;
    todayProgress: boolean;
    appointment: boolean;
    safety: boolean;
  }>({
    rehab: true,
    todayProgress: true,
    appointment: true,
    safety: true,
  });

  const [sosSent, setSosSent] = useState<boolean>(false);

  const toggleSection = (sectionKey: keyof typeof openSections) => {
    setOpenSections((prev) => ({ ...prev, [sectionKey]: !prev[sectionKey] }));
  };

  const recommendedExercises = getRecommendedExercises(user);

  // STRICT REQUIREMENT 10: Today's activity only on dashboard!
  const todaySessions = getTodaySessions(sessions);
  const todayReps = todaySessions.reduce((acc, s) => acc + s.reps, 0);
  const todayAvgAccuracy = todaySessions.length > 0
    ? Math.round(todaySessions.reduce((acc, s) => acc + s.formAccuracy, 0) / todaySessions.length)
    : 0;
  const todayDurationMinutes = Math.round(todaySessions.reduce((acc, s) => acc + s.durationSec, 0) / 60);

  // Exercises completed today
  const completedExerciseIdsToday = new Set(todaySessions.map((s) => s.exercise));
  const completedCountToday = recommendedExercises.filter((ex) => completedExerciseIdsToday.has(ex.id)).length;

  // Next upcoming appointment
  const nextAppointment = appointments.find(
    (a) => a.status === 'confirmed' || a.status === 'ready' || a.status === 'approved' || a.status === 'scheduled'
  ) || appointments[0];

  // Emergency SOS trigger
  const handleTriggerSos = () => {
    const phone = user.guardianWhatsapp || user.emergencyContactPhone || '+91 98765 43210';
    const cleanPhone = phone.replace(/[^0-9]/g, '');
    const message = encodeURIComponent(
      `[PhysioAI Alert] ${user.name} reported pain spike (${user.painIntensity}/10) during rehabilitation routine. Emergency contact request.`
    );
    const link = `https://api.whatsapp.com/send?phone=${cleanPhone}&text=${message}`;

    logGuardianAlert({
      userId: user.id,
      alertType: 'emergency_help',
      message: `Emergency notification dispatched to guardian (${user.emergencyContactName || 'Emergency Contact'})`,
      sentTo: phone,
    });

    setSosSent(true);
    setTimeout(() => setSosSent(false), 5000);
    window.open(link, '_blank');
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Patient Rehabilitation Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-xs font-mono tracking-wider uppercase text-slate-500">
                Today's Rehabilitation Overview
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-1">
              Welcome, {user.name}
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-2xl">
              {user.currentProblem 
                ? `Clinical Protocol: ${user.currentProblem}.` 
                : "Active personalized recovery plan with live biomechanical angle verification."}
            </p>
          </div>

          <div className="flex items-center gap-2.5 self-start sm:self-auto">
            <button
              onClick={() => onStartExercise(recommendedExercises[0] || EXERCISES.squat)}
              className="px-4 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
            >
              <Play className="w-3.5 h-3.5 fill-white" />
              <span>Start Routine ({recommendedExercises[0]?.label || 'Squat'})</span>
            </button>
            <button
              onClick={() => onNavigate('exercises')}
              className="px-3 py-2.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 text-xs font-medium transition-colors cursor-pointer shadow-xs"
            >
              Exercises
            </button>
          </div>
        </div>
      </div>

      {/* QUICK ACCESS ACTION STRIP (Requirement 10) */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <button
          onClick={() => onNavigate('telehealth')}
          className="p-3.5 rounded-xl bg-white hover:bg-[#F0F7FF]/60 border border-slate-200 hover:border-blue-300 transition-all text-left flex items-center gap-3 cursor-pointer shadow-xs group"
        >
          <div className="p-2.5 rounded-lg bg-blue-50 text-blue-600 group-hover:bg-blue-100 transition-colors">
            <Video className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold text-slate-900 block group-hover:text-blue-700">
              Telehealth Consultations
            </span>
            <span className="text-[11px] text-slate-500 block">Doctor visits, queue & video reviews</span>
          </div>
        </button>

        <button
          onClick={() => onNavigate('reports')}
          className="p-3.5 rounded-xl bg-white hover:bg-[#ECFDF3]/60 border border-slate-200 hover:border-emerald-300 transition-all text-left flex items-center gap-3 cursor-pointer shadow-xs group"
        >
          <div className="p-2.5 rounded-lg bg-emerald-50 text-emerald-600 group-hover:bg-emerald-100 transition-colors">
            <FileSpreadsheet className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold text-slate-900 block group-hover:text-emerald-700">
              Clinical Reports
            </span>
            <span className="text-[11px] text-slate-500 block">Daily, weekly & monthly email delivery</span>
          </div>
        </button>

        <button
          onClick={() => onNavigate('history')}
          className="p-3.5 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 hover:border-slate-300 transition-all text-left flex items-center gap-3 cursor-pointer shadow-xs group"
        >
          <div className="p-2.5 rounded-lg bg-slate-100 text-slate-700 group-hover:bg-slate-200 transition-colors">
            <History className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-bold text-slate-900 block group-hover:text-slate-900">
              Historical Activity
            </span>
            <span className="text-[11px] text-slate-500 block">Permanent daily logs & past reps</span>
          </div>
        </button>
      </div>

      {/* Main Accordion Sections */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* SECTION 1: TODAY'S REHABILITATION & PRESCRIBED EXERCISES */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('rehab')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <Dumbbell className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
                  Today's Rehabilitation Exercises
                </h2>
                <span className="text-xs text-slate-500">
                  {recommendedExercises.length} prescribed movements for your current protocol ({completedCountToday} completed today)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.rehab ? 'Hide' : 'Show Details'}
              </span>
              {openSections.rehab ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.rehab && (
            <div className="pb-4 pt-1 space-y-1">
              {recommendedExercises.map((ex) => {
                const safety = isExerciseSafe(ex.id, user);
                const hasCompletedToday = todaySessions.some((s) => s.exercise === ex.id);

                return (
                  <div
                    key={ex.id}
                    className="py-3 px-3 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-[#F0F7FF]/40 border border-transparent hover:border-blue-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-slate-900">{ex.label}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-100 text-slate-600 border border-slate-200">
                          {ex.type}
                        </span>
                        {hasCompletedToday ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0] flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                            <span>Done Today</span>
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                            Ready
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Target: <span className="text-slate-700 font-medium">{ex.target}</span> • Joint: {ex.primaryJoint}
                      </p>
                      {!safety.safe && (
                        <p className="text-[11px] text-amber-800 bg-[#FFF8E6] px-2.5 py-1 rounded border border-amber-200 mt-1.5 flex items-center gap-1.5 inline-flex">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                          <span>Safety Notice: {safety.warning}</span>
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-3 shrink-0 self-end sm:self-center">
                      <div className="text-right text-xs">
                        <span className="text-slate-500 block text-[11px]">Goal Reps</span>
                        <span className="text-slate-900 font-mono font-bold">{ex.defaultTargetReps} reps</span>
                      </div>
                      <button
                        onClick={() => onStartExercise(ex)}
                        className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                      >
                        <Play className="w-3 h-3 fill-white" />
                        <span>Start</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 2: TODAY'S PROGRESS (Only today's counters - resets daily) */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('todayProgress')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-700 transition-colors">
                  Today's Progress
                </h2>
                <span className="text-xs text-slate-500">
                  Daily live counters (resets every morning; historical data saved in History)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.todayProgress ? 'Hide' : 'Show Details'}
              </span>
              {openSections.todayProgress ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.todayProgress && (
            <div className="pb-4 pt-1">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-3 px-2 border-b border-slate-200 text-xs">
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Today's Repetitions</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-0.5 block">{todayReps}</span>
                  <span className="text-[11px] text-slate-500">Executed today</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Today's Form Score</span>
                  <span className="text-xl font-bold font-mono text-emerald-600 mt-0.5 block">
                    {todayAvgAccuracy > 0 ? `${todayAvgAccuracy}%` : 'N/A'}
                  </span>
                  <span className="text-[11px] text-slate-500">Computer vision alignment</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Today's Active Time</span>
                  <span className="text-xl font-bold font-mono text-blue-600 mt-0.5 block">{todayDurationMinutes} min</span>
                  <span className="text-[11px] text-slate-500">{todaySessions.length} sessions logged</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Daily Goal Completion</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-0.5 block">
                    {completedCountToday} / {recommendedExercises.length}
                  </span>
                  <span className="text-[11px] text-slate-500">Exercises finished</span>
                </div>
              </div>

              <div className="pt-3 px-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <p className="text-slate-600">
                  <span className="font-semibold text-slate-800">Daily Reset Notice:</span> Today's counters refresh at 00:00. To inspect prior days' reps and monthly assessments, check the permanent History and Reports sections.
                </p>
                <button
                  onClick={() => onNavigate('reports')}
                  className="text-xs text-blue-600 hover:text-blue-800 font-semibold underline underline-offset-4 cursor-pointer shrink-0"
                >
                  View Monthly Reports →
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 3: UPCOMING APPOINTMENT */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('appointment')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <Calendar className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
                  Upcoming Appointment
                </h2>
                <span className="text-xs text-slate-500">
                  {nextAppointment ? `${nextAppointment.doctorName} • ${nextAppointment.date}` : 'No upcoming visits'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.appointment ? 'Hide' : 'Show Details'}
              </span>
              {openSections.appointment ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.appointment && (
            <div className="pb-4 pt-1">
              {nextAppointment ? (
                <div className="py-3.5 px-3.5 rounded-lg bg-[#F0F7FF]/50 border border-blue-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-slate-900">{nextAppointment.doctorName}</span>
                      <span className="text-slate-600">• {nextAppointment.specialization}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        nextAppointment.status === 'ready'
                          ? 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                          : nextAppointment.status === 'confirmed' || nextAppointment.status === 'approved'
                          ? 'bg-[#F0F7FF] text-blue-700 border border-blue-200'
                          : 'bg-[#FFF8E6] text-amber-800 border border-amber-200'
                      }`}>
                        {nextAppointment.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-slate-600">
                      <span>Scheduled: <strong className="text-slate-900">{nextAppointment.date}</strong> at <strong className="text-slate-900">{nextAppointment.time}</strong></span>
                    </div>
                    <p className="text-slate-500 text-[11px]">Reason: {nextAppointment.reason}</p>
                    {nextAppointment.adminNote && (
                      <p className="text-blue-900 text-[11px] bg-blue-50/80 p-2 rounded border border-blue-200">
                        Physician Note: {nextAppointment.adminNote}
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => onNavigate('telehealth')}
                      className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition-colors cursor-pointer shadow-xs"
                    >
                      {nextAppointment.status === 'ready' ? 'Join Consultation Room' : 'Open in Telehealth'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="py-3 px-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-xs">
                  <span className="text-slate-500">No active appointment booked.</span>
                  <button
                    onClick={() => onNavigate('telehealth')}
                    className="text-xs text-blue-600 hover:text-blue-800 font-semibold cursor-pointer"
                  >
                    Schedule Specialist Consultation →
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 4: TODAY'S SAFETY STATUS & EMERGENCY CONTACT */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('safety')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <ShieldCheck className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
                  Today's Safety Status & Emergency Contact
                </h2>
                <span className="text-xs text-slate-500">
                  Pain rating: {user.painIntensity}/10 • Emergency Guardian: {user.emergencyContactName}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.safety ? 'Hide' : 'Show Details'}
              </span>
              {openSections.safety ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.safety && (
            <div className="pb-4 pt-1 space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                
                {/* Clinical Safety Metrics */}
                <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-700">Pain Level Status</span>
                    <span className="font-mono font-bold text-slate-900">{user.painIntensity} / 10</span>
                  </div>
                  <p className="text-[11px] text-slate-600">
                    Limitation: <span className="font-medium text-slate-800">{user.exerciseLimitations || 'None reported'}</span>
                  </p>
                  <p className="text-[11px] text-slate-600">
                    Precaution: <span className="font-medium text-slate-800">{user.precautions || 'Maintain adequate warm-up'}</span>
                  </p>
                </div>

                {/* Emergency Contact Card */}
                <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200 space-y-2 flex flex-col justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-slate-400 block">DESIGNATED GUARDIAN / CONTACT</span>
                    <span className="font-bold text-slate-900 text-sm block mt-0.5">{user.emergencyContactName}</span>
                    <span className="text-slate-600 block text-xs">{user.emergencyContactPhone}</span>
                  </div>

                  <div className="pt-2 flex items-center gap-2">
                    <button
                      onClick={handleTriggerSos}
                      className="w-full py-2 px-3 rounded-lg bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 font-semibold text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                    >
                      <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
                      <span>{sosSent ? 'Emergency Alert Sent!' : 'Dispatch Guardian SOS Alert'}</span>
                    </button>
                  </div>
                </div>

              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
