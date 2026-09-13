import { useState } from 'react';
import { 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Clock, 
  MessageSquare, 
  Phone, 
  Plus, 
  Send, 
  Stethoscope, 
  UserCheck, 
  X,
  User as UserIcon,
  Video
} from 'lucide-react';
import { Appointment, Doctor, Message, User } from '../types';

interface TelehealthViewProps {
  user: User;
  doctors: Doctor[];
  appointments: Appointment[];
  messages: Message[];
  onBookAppointment: (appointment: Omit<Appointment, 'id' | 'createdAt'>) => void;
  onSendMessage: (doctorId: number, message: string) => void;
}

export function TelehealthView({
  user,
  doctors,
  appointments,
  messages,
  onBookAppointment,
  onSendMessage,
}: TelehealthViewProps) {
  const [activeTab, setActiveTab] = useState<'doctors' | 'appointments' | 'chat'>('doctors');
  const [selectedDoctorId, setSelectedDoctorId] = useState<number>(doctors[0]?.id || 1);
  const [chatInput, setChatInput] = useState<string>('');

  // Expandable doctor rows
  const [expandedDoctorId, setExpandedDoctorId] = useState<number | null>(doctors[0]?.id || null);

  // Booking Modal
  const [bookingDoctor, setBookingDoctor] = useState<Doctor | null>(null);
  const [bookingDate, setBookingDate] = useState<string>('');
  const [bookingTime, setBookingTime] = useState<string>('10:00 AM');
  const [bookingReason, setBookingReason] = useState<string>('');
  const [bookingSuccess, setBookingSuccess] = useState<boolean>(false);

  const currentDoctor = doctors.find((d) => d.id === selectedDoctorId) || doctors[0];
  const doctorMessages = messages.filter(
    (m) => m.doctorId === currentDoctor.id && m.userId === user.id
  );

  const toggleDoctorExpand = (id: number) => {
    setExpandedDoctorId((prev) => (prev === id ? null : id));
  };

  const handleSendChat = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    onSendMessage(currentDoctor.id, chatInput.trim());
    setChatInput('');
  };

  const handleCreateBooking = (e: React.FormEvent) => {
    e.preventDefault();
    if (!bookingDoctor || !bookingDate) return;

    onBookAppointment({
      userId: user.id,
      doctorId: bookingDoctor.id,
      doctorName: bookingDoctor.name,
      specialization: bookingDoctor.specialization,
      patientName: user.name,
      email: user.email,
      date: bookingDate,
      time: bookingTime,
      reason: bookingReason || `Physical assessment & review for ${user.currentProblem}`,
      status: 'pending',
    });

    setBookingSuccess(true);
    setTimeout(() => {
      setBookingSuccess(false);
      setBookingDoctor(null);
      setActiveTab('appointments');
    }, 1800);
  };

  const openWhatsApp = (doc: Doctor) => {
    const cleanNumber = doc.contact.replace(/[^0-9]/g, '');
    const message = encodeURIComponent(
      `Hello ${doc.name}, I am ${user.name}, currently undergoing physiotherapy for ${user.currentProblem}. I would like to schedule a consultation review.`
    );
    window.open(`https://api.whatsapp.com/send?phone=${cleanNumber}&text=${message}`, '_blank');
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
                Clinical Telehealth & Consultations
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <Stethoscope className="w-5 h-5 text-blue-600" />
              <span>Physiotherapy Specialists & Telehealth</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Connect with registered physiotherapists, schedule video reviews, or send rehabilitation queries.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('doctors')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'doctors'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
              }`}
            >
              Specialist Directory
            </button>
            <button
              onClick={() => setActiveTab('appointments')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'appointments'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
              }`}
            >
              Scheduled Visits ({appointments.length})
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'chat'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
              }`}
            >
              Clinical Chat
            </button>
          </div>
        </div>
      </div>

      {/* ================================================================ */}
      {/* TAB 1: SPECIALIST DIRECTORY (Clean expandable rows, NOT a wall of cards) */}
      {/* ================================================================ */}
      {activeTab === 'doctors' && (
        <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
          {doctors.map((doc) => {
            const isExpanded = expandedDoctorId === doc.id;

            return (
              <div key={doc.id} className="transition-colors">
                
                {/* Horizontal row trigger: Dr. Name | Specialization | View Details > */}
                <button
                  onClick={() => toggleDoctorExpand(doc.id)}
                  aria-expanded={isExpanded}
                  className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
                >
                  <div className="flex items-center gap-3 min-w-0 pr-4">
                    <img
                      src={doc.avatarUrl}
                      alt={doc.name}
                      className="w-10 h-10 rounded-lg object-cover border border-slate-200 shrink-0"
                    />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                          {doc.name}
                        </span>
                        <span className="text-slate-400 text-xs hidden sm:inline-block">•</span>
                        <span className="text-xs text-blue-600 font-medium">
                          {doc.specialization}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 truncate">
                        {doc.experience} yrs experience • Available: {doc.availableDays}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-xs text-slate-500 hidden sm:inline-block">
                      {isExpanded ? 'Hide Details' : 'View Details'}
                    </span>
                    <div className="w-6 h-6 rounded flex items-center justify-center text-slate-400 group-hover:text-slate-700">
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 text-blue-600" />
                      ) : (
                        <ChevronRight className="w-4 h-4" />
                      )}
                    </div>
                  </div>
                </button>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="pb-6 pt-1 px-3 space-y-4 text-xs">
                    <p className="text-slate-700 leading-relaxed max-w-3xl">
                      {doc.about}
                    </p>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 py-3 border-t border-b border-slate-200">
                      <div className="flex items-center gap-2 text-slate-600">
                        <Calendar className="w-4 h-4 text-blue-600 shrink-0" />
                        <span>Days: <strong className="text-slate-900 font-medium">{doc.availableDays}</strong></span>
                      </div>
                      <div className="flex items-center gap-2 text-slate-600">
                        <Clock className="w-4 h-4 text-blue-600 shrink-0" />
                        <span>Hours: <strong className="text-slate-900 font-medium">{doc.timings}</strong></span>
                      </div>
                      <div className="flex items-center gap-2 text-slate-600">
                        <Phone className="w-4 h-4 text-blue-600 shrink-0" />
                        <span>Direct: <strong className="text-slate-900 font-mono font-medium">{doc.contact}</strong></span>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                      <button
                        onClick={() => {
                          setSelectedDoctorId(doc.id);
                          setActiveTab('chat');
                        }}
                        className="text-xs text-blue-600 hover:text-blue-700 font-medium underline underline-offset-4 cursor-pointer"
                      >
                        Send Clinical Inquiry in Chat →
                      </button>

                      <div className="flex items-center gap-2.5">
                        <button
                          onClick={() => openWhatsApp(doc)}
                          className="px-3 py-2 rounded-lg bg-[#ECFDF3] hover:bg-[#D1FAE5] border border-[#A7F3D0] text-[#065F46] text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                        >
                          <MessageSquare className="w-3.5 h-3.5" />
                          <span>WhatsApp Consultation</span>
                        </button>
                        <button
                          onClick={() => setBookingDoctor(doc)}
                          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                        >
                          <Calendar className="w-3.5 h-3.5" />
                          <span>Book Assessment</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ================================================================ */}
      {/* TAB 2: APPOINTMENTS LIST (Clean list rows, NOT cards) */}
      {/* ================================================================ */}
      {activeTab === 'appointments' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Your Scheduled Consultations
            </span>
            <button
              onClick={() => setActiveTab('doctors')}
              className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer shadow-xs"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Book Appointment</span>
            </button>
          </div>

          {appointments.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500 border border-slate-200 rounded-lg bg-slate-50">
              You have no upcoming or past appointments scheduled.
            </div>
          ) : (
            <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
              {appointments.map((appt) => (
                <div key={appt.id} className="py-4 px-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-900 text-sm">{appt.doctorName}</span>
                      <span className="text-slate-500">• {appt.specialization}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
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

                    <div className="flex items-center gap-4 text-slate-700">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3.5 h-3.5 text-blue-600" />
                        <span>Date: <strong>{appt.date}</strong> at <strong>{appt.time}</strong></span>
                      </span>
                    </div>

                    <p className="text-slate-500 text-[11px]">Reason: {appt.reason}</p>

                    {appt.adminNote && (
                      <p className="text-blue-800 text-[11px] bg-blue-50 p-2 rounded border border-blue-200 mt-1">
                        Doctor Instructions: {appt.adminNote}
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                    <button
                      onClick={() => {
                        setSelectedDoctorId(appt.doctorId);
                        setActiveTab('chat');
                      }}
                      className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-medium cursor-pointer shadow-xs"
                    >
                      Chat with Doctor
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ================================================================ */}
      {/* TAB 3: CLINICAL CHAT */}
      {/* ================================================================ */}
      {activeTab === 'chat' && (
        <div className="grid grid-cols-1 md:grid-cols-12 border border-slate-200 rounded-lg overflow-hidden min-h-[480px] bg-white">
          
          {/* Doctor Selector Sidebar (4 cols) */}
          <div className="md:col-span-4 border-b md:border-b-0 md:border-r border-slate-200 p-3 space-y-1 bg-slate-50">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 block px-2 mb-2">
              Select Physician
            </span>
            {doctors.map((doc) => (
              <button
                key={doc.id}
                onClick={() => setSelectedDoctorId(doc.id)}
                className={`w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-colors cursor-pointer ${
                  selectedDoctorId === doc.id
                    ? 'bg-blue-50 border border-blue-200 text-blue-900'
                    : 'hover:bg-white text-slate-700'
                }`}
              >
                <img
                  src={doc.avatarUrl}
                  alt={doc.name}
                  className="w-9 h-9 rounded-lg object-cover border border-slate-200 shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold truncate text-slate-900">{doc.name}</p>
                  <p className="text-[11px] text-slate-500 truncate">{doc.specialization}</p>
                </div>
              </button>
            ))}
          </div>

          {/* Chat Messages and Input (8 cols) */}
          <div className="md:col-span-8 flex flex-col justify-between p-4 bg-white">
            
            {/* Chat Header */}
            <div className="pb-3 border-b border-slate-200 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">{currentDoctor.name}</h3>
                <p className="text-[11px] text-[#065F46] flex items-center gap-1.5 font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span>Online for clinical guidance</span>
                </p>
              </div>

              <button
                onClick={() => openWhatsApp(currentDoctor)}
                className="px-2.5 py-1 rounded bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0] text-xs font-semibold flex items-center gap-1 cursor-pointer"
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span>WhatsApp</span>
              </button>
            </div>

            {/* Message Thread */}
            <div className="my-4 space-y-3 max-h-[320px] overflow-y-auto pr-2">
              {doctorMessages.length === 0 ? (
                <div className="text-center py-10">
                  <p className="text-xs text-slate-500">No messages yet with {currentDoctor.name}. Inquire about exercise modifications or form feedback below.</p>
                </div>
              ) : (
                doctorMessages.map((msg) => {
                  const isUser = msg.sender === 'user';
                  return (
                    <div
                      key={msg.id}
                      className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                    >
                      <div
                        className={`max-w-md p-3 rounded-lg text-xs leading-relaxed ${
                          isUser
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-50 text-slate-800 border border-slate-200'
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

            {/* Input form */}
            <form onSubmit={handleSendChat} className="pt-3 border-t border-slate-200 flex items-center gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={`Ask ${currentDoctor.name} about exercise adjustments, pain, or rehabilitation...`}
                className="flex-1 px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500"
              />
              <button
                type="submit"
                className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-colors cursor-pointer shadow-xs"
                title="Send Message"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Booking Appointment Modal */}
      {bookingDoctor && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-xl w-full max-w-md overflow-hidden shadow-xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-200">
              <div>
                <h3 className="text-sm font-bold text-slate-900">Schedule Clinical Visit</h3>
                <p className="text-xs text-slate-500">With {bookingDoctor.name} ({bookingDoctor.specialization})</p>
              </div>
              <button
                onClick={() => setBookingDoctor(null)}
                className="p-1 rounded text-slate-400 hover:text-slate-600 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {bookingSuccess ? (
              <div className="p-8 text-center space-y-2">
                <CheckCircle2 className="w-10 h-10 text-emerald-600 mx-auto" />
                <h4 className="text-sm font-bold text-slate-900">Appointment Request Submitted</h4>
                <p className="text-xs text-slate-500">
                  Your appointment request has been recorded and submitted to {bookingDoctor.name}.
                </p>
              </div>
            ) : (
              <form onSubmit={handleCreateBooking} className="p-4 space-y-3.5 text-xs">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Preferred Consultation Date</label>
                  <input
                    type="date"
                    required
                    value={bookingDate}
                    onChange={(e) => setBookingDate(e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Consultation Time Slot</label>
                  <select
                    value={bookingTime}
                    onChange={(e) => setBookingTime(e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500"
                  >
                    <option value="09:00 AM">09:00 AM</option>
                    <option value="10:00 AM">10:00 AM</option>
                    <option value="11:30 AM">11:30 AM</option>
                    <option value="02:00 PM">02:00 PM</option>
                    <option value="04:00 PM">04:00 PM</option>
                    <option value="05:30 PM">05:30 PM</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Reason for Consultation / Symptoms</label>
                  <textarea
                    rows={3}
                    value={bookingReason}
                    onChange={(e) => setBookingReason(e.target.value)}
                    placeholder={`e.g. Form review and pain check for ${user.currentProblem}`}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 resize-none"
                  />
                </div>

                <div className="pt-2 flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setBookingDoctor(null)}
                    className="px-3.5 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-colors cursor-pointer shadow-xs"
                  >
                    Confirm Appointment
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
