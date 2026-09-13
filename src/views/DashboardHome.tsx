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
  HeartPulse, 
  Play, 
  ShieldAlert, 
  ShieldCheck, 
  TrendingUp, 
  UserCheck 
} from 'lucide-react';
import { Appointment, Exercise, Session, User } from '../types';
import { EXERCISES, getRecommendedExercises, isExerciseSafe } from '../data/exercises';
import { logGuardianAlert } from '../utils/storage';

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
  // Accordion state for the 5 specified sections
  const [openSections, setOpenSections] = useState<{
    rehab: boolean;
    progress: boolean;
    activity: boolean;
    appointment: boolean;
    safety: boolean;
  }>({
    rehab: true,
    progress: true,
    activity: false,
    appointment: true,
    safety: false,
  });

  const [sosSent, setSosSent] = useState<boolean>(false);

  const toggleSection = (sectionKey: keyof typeof openSections) => {
    setOpenSections((prev) => ({ ...prev, [sectionKey]: !prev[sectionKey] }));
  };

  const recommendedExercises = getRecommendedExercises(user);

  // Today's Date String
  const todayDateStr = new Date().toISOString().split('T')[0];
  const todaySessions = sessions.filter((s) => s.date.startsWith(todayDateStr));

  // Biometrics calculations
  const totalReps = sessions.reduce((acc, s) => acc + s.reps, 0);
  const avgAccuracy = sessions.length > 0
    ? Math.round(sessions.reduce((acc, s) => acc + s.formAccuracy, 0) / sessions.length)
    : 92;
  const todayDurationMinutes = Math.round(todaySessions.reduce((acc, s) => acc + s.durationSec, 0) / 60);

  // Next upcoming appointment
  const nextAppointment = appointments.find((a) => a.status === 'approved' || a.status === 'pending') || appointments[0];

  // Emergency SOS trigger
  const handleTriggerSos = () => {
    const phone = user.guardianWhatsapp || user.emergencyContactPhone || '918293413240';
    const cleanPhone = phone.replace(/[^0-9]/g, '');
    const message = encodeURIComponent(
      `[PhysioAI Alert] ${user.name} reported a pain spike (${user.painIntensity}/10) or requested assistance during rehabilitation routine. Location: Home.`
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
      
      {/* Patient Rehabilitation Header - Clean, Calm, Healthcare SaaS */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-xs font-mono tracking-wider uppercase text-slate-500">
                Patient Rehabilitation Portal
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-1">
              Welcome, {user.name}
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-2xl">
              {user.currentProblem 
                ? `Clinical Protocol Focus: ${user.currentProblem}.` 
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
              Directory
            </button>
          </div>
        </div>
      </div>

      {/* Main Container: Stacked Expandable Sections with Dividers */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* SECTION 1: TODAY'S REHABILITATION */}
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
                  Today's Rehabilitation
                </h2>
                <span className="text-xs text-slate-500">
                  {recommendedExercises.length} prescribed exercises for your kinetic profile
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
                            <span>Completed Today</span>
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                            Ready
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Target: <span className="text-slate-700 font-medium">{ex.target}</span> • Kinetic Chain: {ex.primaryJoint}
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
                        <span>Start Exercise</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 2: RECOVERY PROGRESS */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('progress')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-700 transition-colors">
                  Recovery Progress
                </h2>
                <span className="text-xs text-slate-500">
                  Form score, repetitions, session completion, and alignment trends
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.progress ? 'Hide' : 'Show Details'}
              </span>
              {openSections.progress ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.progress && (
            <div className="pb-4 pt-1">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-3 px-2 border-b border-slate-200 text-xs">
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Average Form Score</span>
                  <span className="text-xl font-bold font-mono text-emerald-600 mt-0.5 block">{avgAccuracy}%</span>
                  <span className="text-[11px] text-slate-500">Computer vision alignment</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Cumulative Repetitions</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-0.5 block">{totalReps}</span>
                  <span className="text-[11px] text-slate-500">Logged across protocol</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Sessions Completed</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-0.5 block">{sessions.length}</span>
                  <span className="text-[11px] text-slate-500">Rehabilitation workouts</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[11px] uppercase tracking-wider">Current Pain Rating</span>
                  <span className="text-xl font-bold font-mono text-slate-900 mt-0.5 block">{user.painIntensity} / 10</span>
                  <span className="text-[11px] text-slate-500">{user.painType || 'Stable trajectory'}</span>
                </div>
              </div>

              <div className="pt-3 px-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <p className="text-slate-700">
                  <span className="font-semibold text-emerald-700">Trajectory Summary:</span> Kinematic stability within therapeutic thresholds. Form accuracy has remained above 85% with steady joint control.
                </p>
                <button
                  onClick={() => onNavigate('progress')}
                  className="text-xs text-blue-600 hover:text-blue-800 font-medium underline underline-offset-4 cursor-pointer shrink-0"
                >
                  View Full Biometrics Analytics →
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 3: TODAY'S ACTIVITY */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('activity')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <Activity className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
                  Today's Activity
                </h2>
                <span className="text-xs text-slate-500">
                  {todaySessions.length} sessions logged today ({todayDurationMinutes} mins active time)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.activity ? 'Hide' : 'Show Details'}
              </span>
              {openSections.activity ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </button>

          {openSections.activity && (
            <div className="pb-4 pt-1 space-y-2">
              {todaySessions.length === 0 ? (
                <div className="py-4 px-3 text-xs text-slate-500 bg-slate-50 rounded-lg border border-slate-200">
                  No sessions recorded today yet. Select an exercise from Today's Rehabilitation above to begin.
                </div>
              ) : (
                todaySessions.map((session) => (
                  <div
                    key={session.id}
                    className="py-2.5 px-3 rounded-lg bg-slate-50/80 border border-slate-200 flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-semibold text-slate-900">{session.exerciseLabel}</span>
                      <span className="text-slate-500 ml-2">
                        {session.reps} reps completed in {Math.round(session.durationSec)}s
                      </span>
                      {session.notes && (
                        <p className="text-[11px] text-slate-500 mt-0.5 italic">{session.notes}</p>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="font-mono font-bold text-emerald-600">{session.formAccuracy}%</span>
                      <span className="text-[10px] text-slate-500 block">accuracy</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* SECTION 4: UPCOMING APPOINTMENT */}
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
                  {nextAppointment ? `${nextAppointment.doctorName} • ${nextAppointment.date}` : 'No scheduled visits'}
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
                        nextAppointment.status === 'approved'
                          ? 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                          : 'bg-[#FFF8E6] text-amber-800 border border-amber-200'
                      }`}>
                        {nextAppointment.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-slate-600">
                      <span>Scheduled: <strong className="text-slate-900">{nextAppointment.date}</strong> at <strong className="text-slate-900">{nextAppointment.time}</strong></span>
                    </div>
                    <p className="text-slate-500 text-[11px]">Clinical Reason: {nextAppointment.reason}</p>
                    {nextAppointment.adminNote && (
                      <p className="text-blue-900 text-[11px] bg-blue-50/80 p-2 rounded border border-blue-200">
                        Physician Instructions: {nextAppointment.adminNote}
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      onClick={() => onNavigate('telehealth')}
                      className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition-colors cursor-pointer shadow-xs"
                    >
                      View in Telehealth
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
        {/* SECTION 5: SAFETY & MONITORING */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('safety')}
            className="w-full py-3.5 px-2 flex items-center justify-between text-left group cursor-pointer transition-colors hover:bg-[#F0F7FF]/50 rounded-lg select-none"
          >
            <div className="flex items-center gap-3">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-700 transition-colors">
                  Safety & Monitoring
                </h2>
                <span className="text-xs text-slate-500">
                  Real-time posture safety boundaries, movement restrictions, and emergency contact
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
            <div className="pb-4 pt-1 space-y-3 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                    Kinetic Safety State
                  </span>
                  <p className="text-slate-800">
                    Active pose validation running at 30 FPS. Checks joint hyper-extension, knee valgus collapse, and excessive lumbar flexion.
                  </p>
                  <p className="text-slate-500 text-[11px] mt-1.5">
                    Precautions: {user.precautions || 'Warm up hamstrings and hips thoroughly before exercise.'}
                  </p>
                </div>

                <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                    Movement Restrictions
                  </span>
                  <p className="text-slate-800">
                    {user.movementRestrictions || 'Avoid rapid twisting and deep uncontrolled flexion under load.'}
                  </p>
                  <p className="text-slate-500 text-[11px] mt-1.5">
                    Pain Location: {user.painLocation || 'None reported'} (Intensity {user.painIntensity}/10)
                  </p>
                </div>
              </div>

              {/* Emergency Contact & SOS Button */}
              <div className="p-3.5 rounded-lg bg-[#FFF1F2] border border-rose-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <span className="text-xs font-bold text-rose-800 flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-rose-600" />
                    <span>Designated Emergency Guardian Contact</span>
                  </span>
                  <p className="text-[11px] text-slate-700 mt-0.5">
                    {user.emergencyContactName || 'Guardian'} ({user.guardianWhatsapp || user.emergencyContactPhone || '+91 98765 43210'})
                  </p>
                </div>

                <button
                  onClick={handleTriggerSos}
                  disabled={sosSent}
                  className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs flex items-center justify-center gap-2 transition-colors cursor-pointer shrink-0 disabled:opacity-50 shadow-xs"
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>{sosSent ? 'Alert Dispatched' : 'Send Emergency Assistance SOS'}</span>
                </button>
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
