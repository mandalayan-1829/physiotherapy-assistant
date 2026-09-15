import { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  AlertCircle, 
  Calendar, 
  Camera, 
  CameraOff, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  FileText, 
  Heart, 
  Mail, 
  MessageSquare, 
  Mic, 
  MicOff, 
  Phone, 
  Plus, 
  Send, 
  ShieldCheck, 
  Stethoscope, 
  UserCheck, 
  User as UserIcon, 
  Video, 
  VideoOff, 
  X 
} from 'lucide-react';
import { Appointment, AppointmentStatus, Doctor, Message, User, UserRole } from '../types';

interface TelehealthViewProps {
  user: User;
  doctors: Doctor[];
  appointments: Appointment[];
  messages: Message[];
  userRole?: UserRole;
  onBookAppointment: (input: {
    doctorId: number;
    date: string;
    time: string;
    reason: string;
  }) => void | Promise<void>;
  onSendMessage: (doctorId: number, message: string) => void | Promise<void>;
  onUpdateAppointmentStatus?: (id: number, status: AppointmentStatus, note?: string) => void;
}

export function TelehealthView({
  user,
  doctors,
  appointments,
  messages,
  userRole = 'patient',
  onBookAppointment,
  onSendMessage,
  onUpdateAppointmentStatus,
}: TelehealthViewProps) {
  // Navigation tabs within Telehealth
  const [activeTab, setActiveTab] = useState<'directory' | 'appointments' | 'chat'>('appointments');
  const [selectedDoctorId, setSelectedDoctorId] = useState<number>(doctors[0]?.id || 1);
  const [chatInput, setChatInput] = useState<string>('');
  
  // Booking modal state
  const [bookingDoctor, setBookingDoctor] = useState<Doctor | null>(null);
  const [bookingDate, setBookingDate] = useState<string>('');
  const [bookingTime, setBookingTime] = useState<string>('10:00 AM');
  const [bookingReason, setBookingReason] = useState<string>('');
  const [bookingSuccess, setBookingSuccess] = useState<boolean>(false);

  // Active Live Consultation Room
  const [activeConsultationAppt, setActiveConsultationAppt] = useState<Appointment | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(true);
  const [micActive, setMicActive] = useState<boolean>(true);
  const [consultationNotes, setConsultationNotes] = useState<string>('');
  const [roomMessage, setRoomMessage] = useState<string>('');

  // Doctor availability inspection modal/expansion
  const [expandedDoctorId, setExpandedDoctorId] = useState<number | null>(doctors[0]?.id || null);

  const currentDoctor = doctors.find((d) => d.id === selectedDoctorId) || doctors[0];
  const activeDoctorMessages = messages.filter(
    (m) => m.doctorId === currentDoctor.id && m.userId === user.id
  );

  // Auto select first appointment if in consultation ready state
  useEffect(() => {
    const readyAppt = appointments.find((a) => a.status === 'ready');
    if (readyAppt && appointments.length > 0 && activeTab === 'directory') {
      // Keep user informed
    }
  }, [appointments]);

  const handleSendChat = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    onSendMessage(currentDoctor.id, chatInput.trim());
    setChatInput('');
  };

  const handleCreateBooking = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!bookingDoctor || !bookingDate) return;

    // The booking is persisted by the backend; the confirmation only shows
    // once the server has accepted it.
    await onBookAppointment({
      doctorId: bookingDoctor.id,
      date: bookingDate,
      time: bookingTime,
      reason:
        bookingReason ||
        `Physical assessment & review${user.currentProblem ? ` for ${user.currentProblem}` : ''}`,
    });

    setBookingSuccess(true);
    setTimeout(() => {
      setBookingSuccess(false);
      setBookingDoctor(null);
      setActiveTab('appointments');
    }, 1500);
  };

  const getStatusBadge = (status: AppointmentStatus) => {
    switch (status) {
      case 'ready':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0] flex items-center gap-1.5 animate-pulse">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Consultation Ready</span>
          </span>
        );
      case 'confirmed':
      case 'approved':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#F0F7FF] text-blue-700 border border-blue-200 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
            <span>Confirmed by Doctor</span>
          </span>
        );
      case 'completed':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-slate-500" />
            <span>Completed</span>
          </span>
        );
      case 'cancelled':
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-red-50 text-red-700 border border-red-200">
            Cancelled
          </span>
        );
      case 'scheduled':
      case 'pending':
      default:
        return (
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#FFF8E6] text-[#92400E] border border-[#FDE68A] flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-amber-600" />
            <span>Scheduled (Review Pending)</span>
          </span>
        );
    }
  };

  const handleStartConsultation = (appt: Appointment) => {
    setActiveConsultationAppt(appt);
  };

  const handleSendInRoomMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!roomMessage.trim() || !activeConsultationAppt) return;
    onSendMessage(activeConsultationAppt.doctorId, `[In-Call Consultation Note] ${roomMessage.trim()}`);
    setRoomMessage('');
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header & Sub-Navigation */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Interactive Telehealth Consultations
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <Video className="w-5 h-5 text-blue-600" />
              <span>Telehealth & Physician Video Reviews</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Connect with registered physiotherapists, manage appointment approvals, launch consultation rooms, and coordinate rehabilitation protocols.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('appointments')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'appointments'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Consultations ({appointments.length})
            </button>
            <button
              onClick={() => setActiveTab('directory')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'directory'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Doctor Directory & Availability
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'chat'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Direct Messaging
            </button>
          </div>
        </div>
      </div>

      {/* ================================================================ */}
      {/* TAB 1: CONSULTATIONS & UPCOMING APPOINTMENTS */}
      {/* ================================================================ */}
      {activeTab === 'appointments' && (
        <div className="space-y-4">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-4 rounded-xl border border-slate-200">
            <div>
              <h2 className="text-base font-bold text-slate-900">Scheduled Consultations & Visits</h2>
              <p className="text-xs text-slate-500">
                Track status across "Scheduled", "Confirmed", and "Consultation Ready".
              </p>
            </div>

            <button
              onClick={() => setActiveTab('directory')}
              className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs self-start sm:self-auto"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Book New Consultation</span>
            </button>
          </div>

          {appointments.length === 0 ? (
            <div className="p-12 text-center bg-white rounded-xl border border-slate-200 text-slate-500 text-xs">
              <Calendar className="w-8 h-8 mx-auto text-slate-300 mb-2" />
              <p className="font-semibold text-slate-700">No scheduled consultations right now.</p>
              <p className="mt-1">Browse the doctor directory and book a session with a licensed specialist.</p>
              <button
                onClick={() => setActiveTab('directory')}
                className="mt-4 px-4 py-2 rounded-lg bg-blue-600 text-white font-semibold text-xs inline-flex items-center gap-1.5"
              >
                <span>Browse Doctor Availability</span>
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {appointments.map((appt) => {
                const isReady = appt.status === 'ready';
                const isConfirmed = appt.status === 'confirmed' || appt.status === 'approved';

                return (
                  <div
                    key={appt.id}
                    className={`bg-white rounded-xl border p-5 shadow-xs transition-all ${
                      isReady 
                        ? 'border-emerald-300 bg-[#ECFDF3]/20 ring-2 ring-emerald-500/20' 
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      
                      <div className="space-y-2 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-base font-bold text-slate-900">{appt.doctorName}</span>
                          <span className="text-xs text-slate-500 font-medium">({appt.specialization})</span>
                          {getStatusBadge(appt.status)}
                        </div>

                        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
                          <span className="flex items-center gap-1 font-medium text-slate-800">
                            <Calendar className="w-3.5 h-3.5 text-blue-600" />
                            <span>{appt.date}</span>
                          </span>
                          <span className="flex items-center gap-1 font-medium text-slate-800">
                            <Clock className="w-3.5 h-3.5 text-blue-600" />
                            <span>{appt.time}</span>
                          </span>
                          <span className="text-slate-500">
                            Patient: <span className="font-medium text-slate-700">{appt.patientName}</span>
                          </span>
                        </div>

                        <p className="text-xs text-slate-700 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80">
                          <span className="font-semibold text-slate-900">Clinical Purpose: </span>
                          {appt.reason}
                        </p>

                        {appt.adminNote && (
                          <p className="text-[11px] text-blue-800 bg-[#F0F7FF] px-2.5 py-1.5 rounded border border-blue-200">
                            <span className="font-semibold">Physician Clinic Note: </span>{appt.adminNote}
                          </p>
                        )}
                      </div>

                      {/* Action Controls */}
                      <div className="flex flex-col sm:flex-row md:flex-col items-end gap-2 shrink-0 self-end md:self-center">
                        
                        {/* Consultation Ready -> Join Room */}
                        {(isReady || isConfirmed) && (
                          <button
                            onClick={() => handleStartConsultation(appt)}
                            className={`w-full sm:w-auto px-4 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs ${
                              isReady
                                ? 'bg-emerald-600 hover:bg-emerald-700 text-white ring-2 ring-emerald-400/40'
                                : 'bg-blue-600 hover:bg-blue-700 text-white'
                            }`}
                          >
                            <Video className="w-3.5 h-3.5" />
                            <span>{isReady ? 'Join Consultation Room Now' : 'Enter Consultation Room'}</span>
                          </button>
                        )}

                        <button
                          onClick={() => {
                            setSelectedDoctorId(appt.doctorId);
                            setActiveTab('chat');
                          }}
                          className="w-full sm:w-auto px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer border border-slate-200"
                        >
                          <MessageSquare className="w-3.5 h-3.5 text-slate-500" />
                          <span>Contact Attending Doctor</span>
                        </button>

                        {/* Clinician Management Controls (if Doctor role or quick admin testing) */}
                        <div className="flex items-center gap-1.5 pt-1">
                          {appt.status === 'scheduled' && (
                            <button
                              onClick={() => onUpdateAppointmentStatus?.(appt.id, 'confirmed', 'Confirmed by clinic attending physician.')}
                              className="px-2 py-1 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 cursor-pointer"
                            >
                              Confirm
                            </button>
                          )}
                          {appt.status !== 'ready' && appt.status !== 'completed' && (
                            <button
                              onClick={() => onUpdateAppointmentStatus?.(appt.id, 'ready', 'Consultation room open and doctor is on standby.')}
                              className="px-2 py-1 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 cursor-pointer"
                            >
                              Mark Ready
                            </button>
                          )}
                          {appt.status !== 'completed' && (
                            <button
                              onClick={() => onUpdateAppointmentStatus?.(appt.id, 'completed', 'Consultation successfully concluded.')}
                              className="px-2 py-1 rounded text-[10px] font-semibold bg-slate-100 text-slate-700 border border-slate-200 hover:bg-slate-200 cursor-pointer"
                            >
                              Mark Complete
                            </button>
                          )}
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

      {/* ================================================================ */}
      {/* TAB 2: DOCTOR DIRECTORY & AVAILABILITY */}
      {/* ================================================================ */}
      {activeTab === 'directory' && (
        <div className="space-y-4">
          <div className="bg-white p-4 rounded-xl border border-slate-200">
            <h2 className="text-base font-bold text-slate-900">Licensed Physiotherapists & Clinical Availability</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Select a specialized physiotherapist to review available weekly consultation slots and book assessment reviews.
            </p>
          </div>

          <div className="space-y-3">
            {doctors.map((doc) => {
              const isExpanded = expandedDoctorId === doc.id;

              return (
                <div
                  key={doc.id}
                  className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden transition-all"
                >
                  <div className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-start gap-3.5">
                      <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-base shrink-0 border border-blue-200">
                        {doc.name.replace('Dr. ', '').charAt(0)}
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <h3 className="text-base font-bold text-slate-900">{doc.name}</h3>
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                            Verified Specialist
                          </span>
                        </div>
                        <p className="text-xs font-semibold text-blue-600">{doc.specialization}</p>
                        <p className="text-xs text-slate-500">
                          {doc.hospital} • {doc.experience} experience • Rating: {doc.rating} ★
                        </p>
                        <p className="text-xs text-slate-600 mt-2 max-w-2xl leading-relaxed">{doc.bio}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0 self-end md:self-center">
                      <button
                        onClick={() => setExpandedDoctorId(isExpanded ? null : doc.id)}
                        className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors cursor-pointer border border-slate-200"
                      >
                        {isExpanded ? 'Hide Slots' : 'View Availability'}
                      </button>

                      <button
                        onClick={() => {
                          setBookingDoctor(doc);
                          setBookingDate(new Date().toISOString().split('T')[0]);
                        }}
                        className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                      >
                        <Calendar className="w-3.5 h-3.5" />
                        <span>Book Consultation</span>
                      </button>
                    </div>
                  </div>

                  {/* Expanded Availability Section */}
                  {isExpanded && (
                    <div className="p-4 bg-slate-50 border-t border-slate-200 space-y-3">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
                          Weekly Clinical Consultation Slots
                        </span>
                        <span className="text-slate-500 font-mono">Standard 30-min Video Assessment</span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {doc.availableSlots && doc.availableSlots.length > 0 ? (
                          doc.availableSlots.map((slot, idx) => (
                            <div
                              key={idx}
                              onClick={() => {
                                setBookingDoctor(doc);
                                setBookingTime(slot);
                                setBookingDate(new Date().toISOString().split('T')[0]);
                              }}
                              className="p-2.5 bg-white rounded-lg border border-slate-200 text-center hover:border-blue-500 hover:bg-blue-50/50 cursor-pointer transition-colors"
                            >
                              <span className="text-xs font-mono font-bold text-slate-900 block">{slot}</span>
                              <span className="text-[10px] text-emerald-600 font-medium block mt-0.5">Available</span>
                            </div>
                          ))
                        ) : (
                          ['09:30 AM', '11:00 AM', '02:30 PM', '04:00 PM'].map((slot, idx) => (
                            <div
                              key={idx}
                              onClick={() => {
                                setBookingDoctor(doc);
                                setBookingTime(slot);
                                setBookingDate(new Date().toISOString().split('T')[0]);
                              }}
                              className="p-2.5 bg-white rounded-lg border border-slate-200 text-center hover:border-blue-500 hover:bg-blue-50/50 cursor-pointer transition-colors"
                            >
                              <span className="text-xs font-mono font-bold text-slate-900 block">{slot}</span>
                              <span className="text-[10px] text-emerald-600 font-medium block mt-0.5">Available</span>
                            </div>
                          ))
                        )}
                      </div>

                      <div className="pt-2 text-[11px] text-slate-500 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                        <span>Consultations conduct orthopedic form evaluation, kinematics review, and exercise calibration.</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* TAB 3: DIRECT MESSAGING */}
      {/* ================================================================ */}
      {activeTab === 'chat' && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden flex flex-col h-[520px]">
          
          {/* Chat Header with Doctor Selector */}
          <div className="p-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-sm">
                {currentDoctor.name.replace('Dr. ', '').charAt(0)}
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900">{currentDoctor.name}</h3>
                <span className="text-xs text-slate-500">{currentDoctor.specialization} • Direct Clinical Channel</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <select
                value={selectedDoctorId}
                onChange={(e) => setSelectedDoctorId(parseInt(e.target.value, 10))}
                className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-300 bg-white"
              >
                {doctors.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Messages Log */}
          <div className="flex-1 p-4 overflow-y-auto space-y-3 bg-slate-50/40">
            {activeDoctorMessages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 text-xs">
                <MessageSquare className="w-8 h-8 text-slate-300 mb-2" />
                <p>No messages exchanged with {currentDoctor.name} yet.</p>
                <p className="mt-1">Send a query regarding your prescribed exercises or symptoms.</p>
              </div>
            ) : (
              activeDoctorMessages.map((msg) => {
                const isUser = msg.sender === 'user';
                return (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-xl px-4 py-2.5 text-xs leading-relaxed shadow-2xs ${
                        isUser
                          ? 'bg-blue-600 text-white rounded-br-xs'
                          : 'bg-white text-slate-800 border border-slate-200 rounded-bl-xs'
                      }`}
                    >
                      <p>{msg.message}</p>
                    </div>
                    <span className="text-[10px] text-slate-400 mt-1 px-1 font-mono">
                      {msg.timestamp}
                    </span>
                  </div>
                );
              })
            )}
          </div>

          {/* Message Input Bar */}
          <form onSubmit={handleSendChat} className="p-3 border-t border-slate-200 bg-white flex items-center gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder={`Send clinical query to ${currentDoctor.name}...`}
              className="flex-1 px-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            />
            <button
              type="submit"
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Send className="w-3.5 h-3.5 fill-white" />
              <span>Send</span>
            </button>
          </form>

        </div>
      )}

      {/* ================================================================ */}
      {/* BOOKING MODAL */}
      {/* ================================================================ */}
      {bookingDoctor && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl border border-slate-200 shadow-xl max-w-md w-full p-6 space-y-4 animate-in fade-in zoom-in-95 duration-150">
            
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div>
                <span className="text-[10px] font-mono uppercase text-blue-600 font-semibold">Telehealth Booking</span>
                <h3 className="text-base font-bold text-slate-900">Schedule Video Consultation</h3>
              </div>
              <button
                onClick={() => setBookingDoctor(null)}
                className="text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {bookingSuccess ? (
              <div className="py-8 text-center space-y-2">
                <CheckCircle2 className="w-10 h-10 text-emerald-600 mx-auto" />
                <h4 className="text-base font-bold text-slate-900">Consultation Scheduled!</h4>
                <p className="text-xs text-slate-600">
                  Appointment confirmed with {bookingDoctor.name} for {bookingDate} at {bookingTime}.
                </p>
              </div>
            ) : (
              <form onSubmit={handleCreateBooking} className="space-y-3.5 text-xs">
                
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                  <span className="text-slate-500 block text-[10px] font-mono uppercase">CLINICIAN</span>
                  <span className="font-bold text-slate-900 block">{bookingDoctor.name}</span>
                  <span className="text-slate-600 block">{bookingDoctor.specialization} • {bookingDoctor.hospital}</span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-700 font-semibold mb-1">Consultation Date</label>
                    <input
                      type="date"
                      required
                      value={bookingDate}
                      onChange={(e) => setBookingDate(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-700 font-semibold mb-1">Select Time Slot</label>
                    <select
                      value={bookingTime}
                      onChange={(e) => setBookingTime(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 bg-white"
                    >
                      <option value="09:30 AM">09:30 AM</option>
                      <option value="10:00 AM">10:00 AM</option>
                      <option value="11:00 AM">11:00 AM</option>
                      <option value="02:30 PM">02:30 PM</option>
                      <option value="04:00 PM">04:00 PM</option>
                      <option value="05:30 PM">05:30 PM</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-slate-700 font-semibold mb-1">Chief Reason for Consultation</label>
                  <textarea
                    rows={3}
                    value={bookingReason}
                    onChange={(e) => setBookingReason(e.target.value)}
                    placeholder={`e.g. Check range of motion for ${user.currentProblem}, form alignment check...`}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div className="pt-2 flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setBookingDoctor(null)}
                    className="px-4 py-2 rounded-lg border border-slate-300 text-slate-700 font-medium hover:bg-slate-50 cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold flex items-center gap-1.5 cursor-pointer shadow-xs"
                  >
                    <Calendar className="w-3.5 h-3.5" />
                    <span>Confirm Booking</span>
                  </button>
                </div>

              </form>
            )}

          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* DEDICATED CONSULTATION ROOM (Requirement 3: Clean consultation interface ready for integration) */}
      {/* ================================================================ */}
      {activeConsultationAppt && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-4xl w-full overflow-hidden flex flex-col max-h-[90vh]">
            
            {/* Room Header */}
            <div className="bg-slate-900 text-white px-5 py-3.5 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold tracking-tight">
                      Live Telehealth Consultation Room
                    </h3>
                    <span className="px-1.5 py-0.2 rounded text-[10px] font-mono uppercase bg-emerald-900/80 text-emerald-300 border border-emerald-700">
                      Encrypted Channel
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Patient: {activeConsultationAppt.patientName} • Attending: {activeConsultationAppt.doctorName}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setActiveConsultationAppt(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 cursor-pointer transition-colors"
                title="Exit Consultation Room"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Ready Notice Banner */}
            <div className="bg-[#F0F7FF] border-b border-blue-200 px-4 py-2 text-xs text-blue-900 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-blue-600 shrink-0" />
                <span>
                  Live consultation room active. Video carrier integration stub ready for WebRTC/HIPAA carrier.
                </span>
              </div>
              <span className="font-mono text-[11px] text-blue-700 font-semibold">{activeConsultationAppt.time}</span>
            </div>

            {/* Room Body: Video preview + Patient context + In-room messaging */}
            <div className="flex-1 grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-200 overflow-y-auto">
              
              {/* Left 2 Cols: Clinical Video Viewport & Camera/Mic Preview */}
              <div className="md:col-span-2 p-4 sm:p-5 flex flex-col justify-between space-y-4 bg-slate-950 text-white">
                
                <div className="relative aspect-video rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center overflow-hidden group">
                  {cameraActive ? (
                    <div className="text-center p-6 space-y-2">
                      <div className="w-16 h-16 rounded-full bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center mx-auto mb-2 animate-pulse">
                        <Activity className="w-8 h-8" />
                      </div>
                      <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
                        Computer Vision Sensor Active
                      </span>
                      <p className="text-sm font-semibold text-slate-200">
                        {activeConsultationAppt.doctorName} & {activeConsultationAppt.patientName}
                      </p>
                      <p className="text-xs text-slate-400 max-w-sm mx-auto">
                        Kinematic joint angle overlay streaming ready. Microphones calibrated at 48kHz.
                      </p>
                    </div>
                  ) : (
                    <div className="text-center text-slate-400 text-xs">
                      <CameraOff className="w-8 h-8 mx-auto mb-2 text-slate-500" />
                      <span>Camera Paused by User</span>
                    </div>
                  )}

                  {/* Audio/Video Indicators */}
                  <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-slate-900/80 px-2.5 py-1 rounded-full text-[10px] font-mono border border-slate-700">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span>AUDIO FEED OK</span>
                  </div>

                  <div className="absolute bottom-3 right-3 bg-slate-900/80 px-2.5 py-1 rounded text-[10px] text-slate-300 border border-slate-700">
                    Kinematic Joint Tracking: Active
                  </div>
                </div>

                {/* Consultation Room Call Controls */}
                <div className="flex items-center justify-center gap-3 py-1">
                  <button
                    onClick={() => setCameraActive(!cameraActive)}
                    className={`p-3 rounded-full transition-colors cursor-pointer ${
                      cameraActive ? 'bg-slate-800 hover:bg-slate-700 text-white' : 'bg-red-600 text-white'
                    }`}
                    title={cameraActive ? 'Turn off camera' : 'Turn on camera'}
                  >
                    {cameraActive ? <Camera className="w-4 h-4" /> : <CameraOff className="w-4 h-4" />}
                  </button>

                  <button
                    onClick={() => setMicActive(!micActive)}
                    className={`p-3 rounded-full transition-colors cursor-pointer ${
                      micActive ? 'bg-slate-800 hover:bg-slate-700 text-white' : 'bg-red-600 text-white'
                    }`}
                    title={micActive ? 'Mute microphone' : 'Unmute microphone'}
                  >
                    {micActive ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
                  </button>

                  <button
                    onClick={() => {
                      onUpdateAppointmentStatus?.(activeConsultationAppt.id, 'completed', 'Completed during live video session.');
                      setActiveConsultationAppt(null);
                    }}
                    className="px-4 py-2.5 rounded-full bg-red-600 hover:bg-red-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <span>Conclude & Leave Room</span>
                  </button>
                </div>

              </div>

              {/* Right Col: Consultation Details & In-Call Messages */}
              <div className="p-4 flex flex-col justify-between bg-white text-slate-800 space-y-4">
                
                <div className="space-y-3">
                  <div className="pb-2 border-b border-slate-100">
                    <span className="text-[10px] font-mono uppercase text-slate-400 block">CLINICAL AGENDA</span>
                    <h4 className="text-xs font-bold text-slate-900 mt-0.5">{activeConsultationAppt.reason}</h4>
                    <p className="text-[11px] text-slate-500 mt-1">Prescribed Focus: {user.currentProblem}</p>
                  </div>

                  <div>
                    <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
                      IN-CONSULTATION CLINICAL NOTES
                    </span>
                    <textarea
                      rows={3}
                      value={consultationNotes}
                      onChange={(e) => setConsultationNotes(e.target.value)}
                      placeholder="Type real-time physician guidance or patient observations..."
                      className="w-full text-xs p-2 rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                {/* In-Call Quick Chat Drawer */}
                <div className="space-y-2 border-t border-slate-100 pt-3">
                  <span className="text-[10px] font-bold text-slate-700 uppercase tracking-wider block">
                    In-Call Direct Messages
                  </span>

                  <form onSubmit={handleSendInRoomMessage} className="flex items-center gap-1.5">
                    <input
                      type="text"
                      value={roomMessage}
                      onChange={(e) => setRoomMessage(e.target.value)}
                      placeholder="Direct instruction..."
                      className="flex-1 px-2.5 py-1.5 text-xs rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      type="submit"
                      className="p-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 cursor-pointer"
                    >
                      <Send className="w-3.5 h-3.5 fill-white" />
                    </button>
                  </form>
                </div>

              </div>

            </div>

          </div>
        </div>
      )}

    </div>
  );
}
