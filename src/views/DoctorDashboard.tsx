import { useEffect, useState } from 'react';
import { 
  Activity, 
  AlertCircle, 
  Calendar, 
  CheckCircle2, 
  Clock, 
  FileSpreadsheet, 
  FileText, 
  History, 
  Mail, 
  MessageSquare, 
  Search, 
  Send, 
  ShieldAlert, 
  Stethoscope, 
  UserCheck, 
  User as UserIcon, 
  Users, 
  Video 
} from 'lucide-react';
import { Appointment, AppointmentStatus, Doctor, Session, User } from '../types';
import { getHistoricalDailyEntries } from '../utils/storage';
import { listReports } from '../services/physio';

export interface DoctorPatientSummary {
  account_id: number;
  name: string;
  email: string;
  current_problem: string;
  pain_intensity: number;
}

interface DoctorDashboardProps {
  user: User;
  doctors: Doctor[];
  appointments: Appointment[];
  sessions: Session[];
  /** Patients linked to this clinician on the server. */
  patients?: DoctorPatientSummary[];
  onNavigate: (tab: any) => void;
  onUpdateAppointmentStatus: (id: number, status: AppointmentStatus, note?: string) => void;
}

export function DoctorDashboard({
  user,
  doctors,
  appointments,
  sessions,
  patients: linkedPatients = [],
  onNavigate,
  onUpdateAppointmentStatus,
}: DoctorDashboardProps) {
  const [selectedStatusFilter, setSelectedStatusFilter] = useState<string>('all');
  const [searchPatient, setSearchPatient] = useState<string>('');
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  // Active doctor profile (the signed-in clinician, or the first in the directory)
  const currentDoctor = doctors[0] || {
    id: 0,
    name: 'Clinician',
    specialization: 'Not recorded',
    qualification: '',
    experience: 0,
    availableDays: '',
    timings: '',
    about: '',
    contact: '',
    whatsapp: '',
    email: '',
    hospital: '',
  };

  // Reports are stored per patient on the server. This counts them for the
  // patients actually linked to this clinician.
  //
  // The previous implementation read `getMonthlyReports()` from localStorage,
  // which counted report drafts saved by whoever was using this browser - not
  // reports belonging to these patients. The figure was device-local and
  // clinically meaningless.
  const [reportCount, setReportCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const accountIds = linkedPatients.map((patient) => patient.account_id);

    if (accountIds.length === 0) {
      setReportCount(0);
      return;
    }

    void (async () => {
      try {
        // Presentation context is irrelevant to a count, so nothing patient
        // specific is passed.
        const counts = await Promise.all(
          accountIds.map((accountId) =>
            listReports({ hasUpcomingCheckup: false }, accountId).then((rows) => rows.length),
          ),
        );
        if (!cancelled) setReportCount(counts.reduce((total, n) => total + n, 0));
      } catch {
        // `null` renders as an em dash: an unavailable figure, never a zero that
        // would read as "this patient has no reports".
        if (!cancelled) setReportCount(null);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [linkedPatients]);

  // Roster derived from the patients actually linked to this clinician.
  // No simulated patient rows are generated.
  const todayStr = new Date().toISOString().split('T')[0];
  const patients = linkedPatients.map((patient) => {
    const patientSessions = sessions.filter((s) => s.userId === patient.account_id);
    const activeDays = new Set(patientSessions.map((s) => s.date.split(' ')[0]));
    // Days with a recorded session inside the last 7 days. Counting every
    // distinct day the patient has ever trained would report a long-standing
    // patient as permanently "100% adherent".
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 6);
    const windowStartStr = sevenDaysAgo.toISOString().split('T')[0];
    const recentActiveDays = new Set(
      patientSessions
        .map((s) => s.date.split(' ')[0])
        .filter((day) => day >= windowStartStr && day <= todayStr),
    );
    const adherence = Math.min(100, Math.round((recentActiveDays.size / 7) * 100));
    const latest = patientSessions
      .map((s) => s.date)
      .sort((a, b) => b.localeCompare(a))[0];

    let lastSession = 'No sessions recorded';
    if (latest) {
      if (latest.startsWith(todayStr)) lastSession = 'Today';
      else if (activeDays.has(todayStr)) lastSession = 'Today';
      else lastSession = latest.split(' ')[0];
    }

    return {
      id: patient.account_id,
      name: patient.name,
      email: patient.email,
      condition: patient.current_problem || 'Not recorded',
      painScore: patient.pain_intensity,
      adherence,
      lastSession,
      status: patient.pain_intensity >= 5 ? 'Review Required' : adherence > 0 ? 'Active Protocol' : 'New Patient',
    };
  });

  // Only sessions with an actual measurement contribute to the form average.
  const measuredSessions = sessions.filter((s) => s.metricsSource === 'pose_inference');
  const overallFormAverage =
    measuredSessions.length > 0
      ? Math.round(measuredSessions.reduce((acc, s) => acc + s.formAccuracy, 0) / measuredSessions.length)
      : null;

  const filteredAppointments = appointments.filter((appt) => {
    const matchesStatus = selectedStatusFilter === 'all' || appt.status === selectedStatusFilter;
    const matchesQuery = 
      appt.patientName.toLowerCase().includes(searchPatient.toLowerCase()) ||
      appt.reason.toLowerCase().includes(searchPatient.toLowerCase());
    return matchesStatus && matchesQuery;
  });

  const handleStatusChange = (id: number, status: AppointmentStatus, message: string) => {
    onUpdateAppointmentStatus(id, status, message);
    setActionFeedback(`Appointment #${id} marked as "${status.toUpperCase()}".`);
    setTimeout(() => setActionFeedback(null), 3500);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Clinician Management Console
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <Stethoscope className="w-7 h-7 text-blue-600" />
              <span>{currentDoctor.name}</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              {currentDoctor.specialization} • {currentDoctor.hospital} • Telehealth Consultation Queue
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <button
              onClick={() => onNavigate('telehealth')}
              className="px-4 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
            >
              <Video className="w-3.5 h-3.5" />
              <span>Open Telehealth Queue</span>
            </button>
            <button
              onClick={() => onNavigate('reports')}
              className="px-3.5 py-2.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200 cursor-pointer shadow-xs"
            >
              Review Reports
            </button>
          </div>
        </div>
      </div>

      {actionFeedback && (
        <div className="p-3 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-xs text-[#065F46] flex items-center gap-2 font-medium">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{actionFeedback}</span>
        </div>
      )}

      {/* Clinician KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-[11px] font-mono uppercase text-slate-500 block">Total Patients</span>
          <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{patients.length} Active</span>
          <span className="text-xs text-slate-500 mt-0.5 block">Orthopedic protocols</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-[11px] font-mono uppercase text-slate-500 block">Upcoming Consults</span>
          <span className="text-2xl font-bold font-mono text-blue-600 mt-1 block">
            {appointments.filter((a) => a.status !== 'completed' && a.status !== 'cancelled').length}
          </span>
          <span className="text-xs text-slate-500 mt-0.5 block">Scheduled video sessions</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-[11px] font-mono uppercase text-slate-500 block">Monthly Reports</span>
          <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">
            {reportCount === null ? '—' : reportCount}
          </span>
          <span className="text-xs text-slate-500 mt-0.5 block">Stored on the server for your patients</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-[11px] font-mono uppercase text-slate-500 block">Form Score Avg</span>
          <span className="text-2xl font-bold font-mono text-emerald-600 mt-1 block">
            {overallFormAverage === null ? '—' : `${overallFormAverage}%`}
          </span>
          <span className="text-xs text-slate-500 mt-0.5 block">
            {measuredSessions.length === 0
              ? 'No measured session yet'
              : `From ${measuredSessions.length} of ${sessions.length} session(s)`}
          </span>
        </div>
      </div>

      {/* SECTION 1: TELEHEALTH APPOINTMENT QUEUE & APPROVALS (Requirement 3: Doctor Flow) */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              Telehealth Consultations & Booking Requests
            </h2>
            <p className="text-xs text-slate-500">
              Review patient appointment requests, confirm slots, mark consultation ready, and initiate video sessions.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={selectedStatusFilter}
              onChange={(e) => setSelectedStatusFilter(e.target.value)}
              className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-300 bg-white"
            >
              <option value="all">All Appointments ({appointments.length})</option>
              <option value="scheduled">Scheduled / Pending</option>
              <option value="confirmed">Confirmed</option>
              <option value="ready">Consultation Ready</option>
              <option value="completed">Completed</option>
            </select>
          </div>
        </div>

        {filteredAppointments.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs">
            <Calendar className="w-8 h-8 mx-auto text-slate-300 mb-2" />
            <p className="font-semibold text-slate-700">No appointments found matching this criteria.</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {filteredAppointments.map((appt) => {
              const isScheduled = appt.status === 'scheduled' || appt.status === 'pending';
              const isConfirmed = appt.status === 'confirmed' || appt.status === 'approved';
              const isReady = appt.status === 'ready';

              return (
                <div key={appt.id} className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-slate-50/70">
                  <div className="space-y-1.5 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-slate-900 text-sm">{appt.patientName}</span>
                      <span className="text-xs text-slate-500 font-mono">({appt.email})</span>
                      
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        isReady
                          ? 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                          : isConfirmed
                          ? 'bg-[#F0F7FF] text-blue-700 border border-blue-200'
                          : 'bg-[#FFF8E6] text-amber-800 border border-amber-200'
                      }`}>
                        {appt.status}
                      </span>
                    </div>

                    <div className="flex items-center gap-4 text-xs text-slate-600">
                      <span>Date: <strong className="text-slate-800">{appt.date}</strong> at <strong className="text-slate-800">{appt.time}</strong></span>
                      <span>Doctor: {appt.doctorName}</span>
                    </div>

                    <p className="text-xs text-slate-700 bg-slate-100/70 p-2 rounded border border-slate-200">
                      <span className="font-semibold">Patient Note: </span>{appt.reason}
                    </p>
                  </div>

                  {/* Clinician Action Buttons */}
                  <div className="flex flex-wrap items-center gap-2 shrink-0 self-end md:self-center">
                    
                    {isScheduled && (
                      <button
                        onClick={() => handleStatusChange(appt.id, 'confirmed', 'Confirmed by clinician.')}
                        className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold cursor-pointer shadow-xs"
                      >
                        Accept & Confirm
                      </button>
                    )}

                    {!isReady && appt.status !== 'completed' && (
                      <button
                        onClick={() => handleStatusChange(appt.id, 'ready', 'Clinician marked consultation room ready.')}
                        className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold cursor-pointer shadow-xs flex items-center gap-1"
                      >
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Open Consultation Room</span>
                      </button>
                    )}

                    {isReady && (
                      <button
                        onClick={() => onNavigate('telehealth')}
                        className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold cursor-pointer shadow-xs flex items-center gap-1 animate-pulse"
                      >
                        <Video className="w-3.5 h-3.5" />
                        <span>Enter Live Room</span>
                      </button>
                    )}

                    <button
                      onClick={() => onNavigate('telehealth')}
                      className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium cursor-pointer border border-slate-200"
                    >
                      Message Patient
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* SECTION 2: PATIENT ROSTER & CLINICAL STATUS */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              Assigned Patients Roster
            </h2>
            <p className="text-xs text-slate-500">
              Patients linked to your account, with their recorded session activity and
              self-reported pain score.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">{patients.length} registered patients</span>
        </div>

        <div className="divide-y divide-slate-100">
          {patients.map((pat) => (
            <div key={pat.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-slate-50">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 text-sm">{pat.name}</span>
                  <span className="text-xs text-slate-500">{pat.email}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                    {pat.status}
                  </span>
                </div>
                <p className="text-xs text-slate-600">Protocol: {pat.condition}</p>
                <div className="flex items-center gap-4 text-xs text-slate-500 mt-1">
                  <span>Pain Level: <strong className="text-slate-800">{pat.painScore}/10</strong></span>
                  <span>Active days (7d): <strong className="text-emerald-600">{pat.adherence}%</strong></span>
                  <span>Last Active: {pat.lastSession}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                <button
                  onClick={() => onNavigate('reports')}
                  className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium border border-slate-200 cursor-pointer"
                >
                  Clinical Report
                </button>
                <button
                  onClick={() => onNavigate('telehealth')}
                  className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold cursor-pointer shadow-xs"
                >
                  Consult
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
