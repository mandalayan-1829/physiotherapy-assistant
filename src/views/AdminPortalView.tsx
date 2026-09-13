import { useState } from 'react';
import { 
  CheckCircle2, 
  Clock, 
  Lock, 
  Mail, 
  MessageSquare, 
  Phone, 
  Plus, 
  Send, 
  ShieldAlert, 
  ShieldCheck, 
  Trash2, 
  UserCheck, 
  X,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Calendar,
  Stethoscope
} from 'lucide-react';
import { Appointment, Doctor, GuardianAlert, Message } from '../types';

interface AdminPortalViewProps {
  doctors: Doctor[];
  appointments: Appointment[];
  messages: Message[];
  guardianAlerts: GuardianAlert[];
  isAdminLoggedIn: boolean;
  onAdminLogin: () => void;
  onAdminLogout: () => void;
  onUpdateAppointmentStatus: (id: number, status: Appointment['status'], adminNote?: string) => void;
  onReplyMessage: (doctorId: number, text: string) => void;
  onAddDoctor: (doc: Omit<Doctor, 'id'>) => void;
}

export function AdminPortalView({
  doctors,
  appointments,
  messages,
  guardianAlerts,
  isAdminLoggedIn,
  onAdminLogin,
  onAdminLogout,
  onUpdateAppointmentStatus,
  onReplyMessage,
  onAddDoctor,
}: AdminPortalViewProps) {
  const [passcode, setPasscode] = useState<string>('');
  const [loginError, setLoginError] = useState<string>('');
  
  // Note dialog
  const [editingApptId, setEditingApptId] = useState<number | null>(null);
  const [noteInput, setNoteInput] = useState<string>('');

  // Doctor reply
  const [replyInput, setReplyInput] = useState<string>('');
  const [replyDoctorId, setReplyDoctorId] = useState<number>(doctors[0]?.id || 1);

  // Expandable sections
  const [openSections, setOpenSections] = useState<{
    appointments: boolean;
    chat: boolean;
    alerts: boolean;
    registry: boolean;
  }>({
    appointments: true,
    chat: false,
    alerts: true,
    registry: false,
  });

  const toggleSection = (key: keyof typeof openSections) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // New Doctor Form
  const [showAddDoc, setShowAddDoc] = useState<boolean>(false);
  const [docName, setDocName] = useState<string>('');
  const [docSpec, setDocSpec] = useState<string>('');
  const [docExp, setDocExp] = useState<number>(5);
  const [docQual, setDocQual] = useState<string>('MBBS, MPT');
  const [docDays, setDocDays] = useState<string>('Mon, Wed, Fri');
  const [docTimings, setDocTimings] = useState<string>('09:00 AM - 01:00 PM');
  const [docAbout, setDocAbout] = useState<string>('');
  const [docPhone, setDocPhone] = useState<string>('');
  const [docEmail, setDocEmail] = useState<string>('');

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (passcode.trim() === 'admin@physio123' || passcode.trim() === 'admin') {
      onAdminLogin();
      setLoginError('');
    } else {
      setLoginError('Invalid clinical credentials. Try: admin@physio123');
    }
  };

  const handleAddDoctorSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!docName.trim() || !docSpec.trim()) return;

    onAddDoctor({
      name: docName.trim(),
      specialization: docSpec.trim(),
      experience: docExp,
      qualification: docQual,
      availableDays: docDays,
      timings: docTimings,
      about: docAbout || 'Dedicated orthopedic rehabilitation therapist.',
      contact: docPhone || '+91 82934 13240',
      whatsapp: docPhone.replace(/[^0-9]/g, '') || '918293413240',
      email: docEmail || 'doctor@physioai.health',
      avatarUrl: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=200&auto=format&fit=crop&q=80',
    });

    setShowAddDoc(false);
    setDocName('');
    setDocSpec('');
  };

  const handleSendDoctorReply = (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyInput.trim()) return;
    onReplyMessage(replyDoctorId, replyInput.trim());
    setReplyInput('');
  };

  if (!isAdminLoggedIn) {
    return (
      <div className="max-w-md mx-auto my-12 p-8 bg-white border border-slate-200 rounded-2xl shadow-xl space-y-6">
        <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 border border-blue-200 flex items-center justify-center mx-auto">
          <Lock className="w-6 h-6" />
        </div>

        <div className="text-center">
          <h2 className="text-xl font-bold text-slate-900">Physician & Clinic Portal</h2>
          <p className="text-xs text-slate-500 mt-1">
            Restricted area for attending doctors and clinical administrators.
          </p>
        </div>

        {loginError && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs">
            {loginError}
          </div>
        )}

        <form onSubmit={handleLoginSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">Access Passcode</label>
            <input
              type="password"
              required
              value={passcode}
              onChange={(e) => setPasscode(e.target.value)}
              placeholder="Enter passcode (e.g. admin@physio123)"
              className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-none focus:border-blue-500"
            />
          </div>

          <button
            type="submit"
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-all shadow-xs cursor-pointer"
          >
            Authenticate Portal Access
          </button>
        </form>

        <div className="pt-2 border-t border-slate-200 text-center">
          <button
            type="button"
            onClick={() => {
              setPasscode('admin@physio123');
              onAdminLogin();
            }}
            className="text-[11px] text-blue-600 hover:text-blue-700 font-semibold cursor-pointer"
          >
            Demo One-Click Login (admin@physio123)
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Top Banner */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Clinical Administration
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-600" />
              <span>Doctor & Clinic Management Portal</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Review appointments, dispatch physician clinical advice, and audit emergency guardian alerts.
            </p>
          </div>

          <button
            onClick={onAdminLogout}
            className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200 transition-colors cursor-pointer self-start sm:self-auto shadow-2xs"
          >
            Sign Out of Portal
          </button>
        </div>
      </div>

      {/* Expandable Sections Container */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* 1. APPOINTMENTS MANAGEMENT */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('appointments')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Clock className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Patient Appointment Requests
                </h2>
                <span className="text-xs text-slate-500">
                  {appointments.length} total appointments in clinical schedule
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.appointments ? 'Collapse' : 'Expand'}
              </span>
              {openSections.appointments ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.appointments && (
            <div className="pb-6 pt-2 px-2 text-xs">
              {appointments.length === 0 ? (
                <p className="text-xs text-slate-500 py-4 text-center">No appointment requests in queue.</p>
              ) : (
                <div className="divide-y divide-slate-200">
                  {appointments.map((appt) => (
                    <div key={appt.id} className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-900 text-sm">{appt.patientName}</span>
                          <span className="text-slate-500">({appt.email})</span>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                            appt.status === 'approved'
                              ? 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                              : appt.status === 'completed'
                              ? 'bg-blue-50 text-blue-700 border border-blue-200'
                              : appt.status === 'cancelled'
                              ? 'bg-rose-50 text-rose-700 border border-rose-200'
                              : 'bg-amber-50 text-amber-800 border border-amber-200'
                          }`}>
                            {appt.status}
                          </span>
                        </div>

                        <p className="text-slate-700">
                          Assigned: {appt.doctorName} • Date: {appt.date} at {appt.time}
                        </p>
                        <p className="text-slate-500 text-[11px]">Chief Reason: {appt.reason}</p>
                        {appt.adminNote && (
                          <p className="text-blue-800 bg-blue-50 p-1.5 rounded border border-blue-200 text-[11px] mt-1">
                            Doctor's Note: {appt.adminNote}
                          </p>
                        )}
                      </div>

                      {/* Status Update Actions */}
                      <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                        <button
                          onClick={() => onUpdateAppointmentStatus(appt.id, 'approved', appt.adminNote || 'Confirmed by clinical team.')}
                          className="px-2.5 py-1.5 rounded-md bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0] text-[11px] font-semibold hover:bg-emerald-100 transition-colors cursor-pointer"
                        >
                          Approve
                        </button>

                        <button
                          onClick={() => onUpdateAppointmentStatus(appt.id, 'completed')}
                          className="px-2.5 py-1.5 rounded-md bg-blue-50 text-blue-700 border border-blue-200 text-[11px] font-semibold hover:bg-blue-100 transition-colors cursor-pointer"
                        >
                          Mark Done
                        </button>

                        <button
                          onClick={() => onUpdateAppointmentStatus(appt.id, 'cancelled')}
                          className="px-2.5 py-1.5 rounded-md bg-rose-50 text-rose-700 border border-rose-200 text-[11px] font-semibold hover:bg-rose-100 transition-colors cursor-pointer"
                        >
                          Decline
                        </button>

                        <button
                          onClick={() => {
                            setEditingApptId(appt.id);
                            setNoteInput(appt.adminNote || '');
                          }}
                          className="px-2.5 py-1.5 rounded-md bg-white text-slate-700 border border-slate-200 text-[11px] font-semibold hover:bg-slate-50 transition-colors cursor-pointer"
                        >
                          Add Note
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 2. DOCTOR REPLY TO PATIENT CHAT */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('chat')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <MessageSquare className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-600 transition-colors">
                  Physician Direct Chat Dispatch
                </h2>
                <span className="text-xs text-slate-500">
                  Send patient advice directly into clinical conversation thread
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.chat ? 'Collapse' : 'Expand'}
              </span>
              {openSections.chat ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.chat && (
            <div className="pb-6 pt-2 px-2 text-xs">
              <form onSubmit={handleSendDoctorReply} className="space-y-3">
                <div className="flex items-center gap-3">
                  <label className="text-slate-600 font-semibold">Reply As:</label>
                  <select
                    value={replyDoctorId}
                    onChange={(e) => setReplyDoctorId(parseInt(e.target.value) || 1)}
                    className="px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-xs text-slate-800 focus:outline-none focus:border-blue-500 cursor-pointer"
                  >
                    {doctors.map((d) => (
                      <option key={d.id} value={d.id}>{d.name} ({d.specialization})</option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    required
                    value={replyInput}
                    onChange={(e) => setReplyInput(e.target.value)}
                    placeholder="Send clinical response or exercise adjustment to patient..."
                    className="flex-1 px-3.5 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500"
                  />
                  <button
                    type="submit"
                    className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>Send Reply</span>
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 3. GUARDIAN ALERT AUDIT LOG */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('alerts')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-rose-500" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-rose-600 transition-colors">
                  Guardian SOS Alert Audit Log
                </h2>
                <span className="text-xs text-slate-500">
                  {guardianAlerts.length} emergency alerts recorded
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.alerts ? 'Collapse' : 'Expand'}
              </span>
              {openSections.alerts ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.alerts && (
            <div className="pb-6 pt-2 px-2 text-xs">
              {guardianAlerts.length === 0 ? (
                <p className="text-slate-500 py-3 text-center">No emergency alerts logged. All patient routines operating within safety parameters.</p>
              ) : (
                <div className="divide-y divide-slate-200">
                  {guardianAlerts.map((alert) => (
                    <div key={alert.id} className="py-3 flex items-center justify-between text-xs">
                      <div>
                        <p className="font-semibold text-rose-700">{alert.message}</p>
                        <p className="text-[11px] text-slate-500 mt-0.5">
                          Target: {alert.sentTo} • Timestamp: {alert.timestamp}
                        </p>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-50 text-rose-700 border border-rose-200">
                        {alert.alertType}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 4. DOCTORS DIRECTORY MANAGEMENT */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('registry')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <UserCheck className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Clinic Doctors Registry & Staff
                </h2>
                <span className="text-xs text-slate-500">
                  {doctors.length} registered specialists on staff
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.registry ? 'Collapse' : 'Expand'}
              </span>
              {openSections.registry ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.registry && (
            <div className="pb-6 pt-2 px-2 text-xs space-y-4">
              <div className="flex justify-end">
                <button
                  onClick={() => setShowAddDoc(!showAddDoc)}
                  className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer shadow-xs"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>{showAddDoc ? 'Close Form' : 'Register New Doctor'}</span>
                </button>
              </div>

              {/* Add Doctor Form */}
              {showAddDoc && (
                <form onSubmit={handleAddDoctorSubmit} className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-3">
                  <h3 className="font-semibold text-slate-900">Register Clinical Specialist</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                    <div>
                      <label className="text-slate-700 block mb-1 font-semibold">Doctor Name</label>
                      <input
                        type="text"
                        required
                        value={docName}
                        onChange={(e) => setDocName(e.target.value)}
                        placeholder="Dr. Full Name"
                        className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="text-slate-700 block mb-1 font-semibold">Specialization</label>
                      <input
                        type="text"
                        required
                        value={docSpec}
                        onChange={(e) => setDocSpec(e.target.value)}
                        placeholder="e.g. Spine & Posture Specialist"
                        className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="text-slate-700 block mb-1 font-semibold">Experience (Years)</label>
                      <input
                        type="number"
                        min="1"
                        value={docExp}
                        onChange={(e) => setDocExp(parseInt(e.target.value) || 1)}
                        className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>

                  <div className="flex justify-end gap-2 pt-2">
                    <button
                      type="button"
                      onClick={() => setShowAddDoc(false)}
                      className="px-3 py-1.5 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900 font-semibold"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-bold shadow-xs"
                    >
                      Save Specialist
                    </button>
                  </div>
                </form>
              )}

              {/* Doctors List */}
              <div className="divide-y divide-slate-200">
                {doctors.map((doc) => (
                  <div key={doc.id} className="py-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <img src={doc.avatarUrl} alt={doc.name} className="w-8 h-8 rounded-lg object-cover border border-slate-200" />
                      <div>
                        <span className="font-semibold text-slate-900">{doc.name}</span>
                        <span className="text-slate-500 ml-2">({doc.specialization})</span>
                        <p className="text-[11px] text-slate-500">{doc.availableDays} • {doc.timings}</p>
                      </div>
                    </div>
                    <span className="text-slate-600 font-mono text-[11px]">{doc.contact}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

      </div>

      {/* Note modal */}
      {editingApptId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4">
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-xl p-6 shadow-xl space-y-3">
            <h3 className="text-sm font-bold text-slate-900">Attach Physician Clinical Instruction</h3>
            <textarea
              rows={3}
              value={noteInput}
              onChange={(e) => setNoteInput(e.target.value)}
              placeholder="e.g. Please wear loose athletic apparel and bring recent MRI report..."
              className="w-full p-3 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 resize-none"
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setEditingApptId(null)}
                className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const appt = appointments.find((a) => a.id === editingApptId);
                  if (appt) {
                    onUpdateAppointmentStatus(appt.id, appt.status, noteInput);
                  }
                  setEditingApptId(null);
                }}
                className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold cursor-pointer shadow-xs"
              >
                Save Instruction
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
