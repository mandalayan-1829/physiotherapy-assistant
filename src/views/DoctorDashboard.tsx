import { useState } from 'react';
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
import { Appointment, AppointmentStatus, Doctor, MonthlyReport, Session, User } from '../types';
import { getHistoricalDailyEntries, getMonthlyReports, getTodaySessions } from '../utils/storage';

interface DoctorDashboardProps {
  user: User;
  doctors: Doctor[];
  appointments: Appointment[];
  sessions: Session[];
  onNavigate: (tab: any) => void;
  onUpdateAppointmentStatus: (id: number, status: AppointmentStatus, note?: string) => void;
}

export function DoctorDashboard({
  user,
  doctors,
  appointments,
  sessions,
  onNavigate,
  onUpdateAppointmentStatus,
}: DoctorDashboardProps) {
  const [selectedStatusFilter, setSelectedStatusFilter] = useState<string>('all');
  const [searchPatient, setSearchPatient] = useState<string>('');
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  // Active doctor profile (Dr. Aarav Patel or first doctor)
  const currentDoctor = doctors[0] || {
    id: 1,
    name: 'Dr. Aarav Patel',
    specialization: 'Orthopedic Physiotherapy',
    hospital: 'Apollo Physical Therapy Center',
  };

  const monthlyReports = getMonthlyReports();
  const todaySessions = getTodaySessions(sessions);

  // Patient roster simulated from user + sessions
  const patients = [
    {
      id: user.id,
      name: user.name,
      email: user.email,
      condition: user.currentProblem || 'Knee Osteoarthritis Rehabilitation',
      painScore: user.painIntensity,
      adherence: 88,
      lastSession: todaySessions.length > 0 ? 'Today' : 'Yesterday',
      status: 'Active Protocol',
    },
    {
      id: 'patient-2',
      name: 'Sarah Jenkins',
      email: 'sarah.j@example.com',
      condition: 'Post-ACL Reconstruction (Week 6)',
      painScore: 3,
      adherence: 94,
      lastSession: 'Today',
      status: 'Improving',
    },
    {
      id: 'patient-3',
      name: 'David Kumar',
      email: 'david.k@example.com',
      condition: 'Cervical Spondylosis & Posture',
      painScore: 5,
      adherence: 72,
      lastSession: '2 days ago',
      status: 'Review Required',
    },
  ];

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
          <span className="text-2xl font-bold font-mono text-slate-900 mt-1 block">{monthlyReports.length}</span>
          <span className="text-xs text-slate-500 mt-0.5 block">Clinical evaluations compiled</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-[11px] font-mono uppercase text-slate-500 block">Kinematic Form Avg</span>
          <span className="text-2xl font-bold font-mono text-emerald-600 mt-1 block">91%</span>
          <span className="text-xs text-slate-500 mt-0.5 block">Across patient routines</span>
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
              Monitor compliance, pain trajectories, and latest exercise activity.
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
                  <span>Adherence: <strong className="text-emerald-600">{pat.adherence}%</strong></span>
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
