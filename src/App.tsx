import { useState, useEffect } from 'react';
import { Sidebar, AppNavTab, PatientNavTab, DoctorNavTab } from './components/Sidebar';
import { PortalAuth } from './components/PortalAuth';
import { AccountProfileModal } from './components/AccountProfileModal';
import { DashboardHome } from './views/DashboardHome';
import { DoctorDashboard } from './views/DoctorDashboard';
import { ExerciseSelectionView } from './views/ExerciseSelectionView';
import { TrackingView } from './views/TrackingView';
import { TelehealthView } from './views/TelehealthView';
import { ReportsView } from './views/ReportsView';
import { HistoryView } from './views/HistoryView';
import { DietTrackerView } from './views/DietTrackerView';
import { ProgressAnalyticsView } from './views/ProgressAnalyticsView';
import { NotesView } from './views/NotesView';
import { MedicalProfileView } from './views/MedicalProfileView';
import { AdminPortalView } from './views/AdminPortalView';

import { 
  Appointment, 
  AppointmentStatus, 
  DietEntry, 
  Doctor, 
  Exercise, 
  GuardianAlert, 
  Message, 
  Note, 
  Session, 
  User, 
  UserRole 
} from './types';
import { 
  addDietEntry, 
  addDoctor, 
  addNote, 
  addSession, 
  bookAppointment, 
  clearStoredAuthRole, 
  deleteDietEntry, 
  deleteNote, 
  getAppointments, 
  getDietEntries, 
  getDoctors, 
  getGuardianAlerts, 
  getMessages, 
  getNotes, 
  getSessions, 
  getStoredAuthRole, 
  getUserProfile, 
  initializeStorage, 
  saveUserProfile, 
  sendMessage, 
  setStoredAuthRole, 
  updateAppointmentStatus 
} from './utils/storage';
import { EXERCISES } from './data/exercises';

export function App() {
  // Authentication Role: 'patient' | 'doctor' | null (if null, show dual PortalAuth screen)
  const [authRole, setAuthRole] = useState<UserRole | null>(() => {
    // If not set yet, defaults to 'patient' for immediate preview, but user can click switch domain anytime
    return getStoredAuthRole() || 'patient';
  });

  const [currentTab, setCurrentTab] = useState<AppNavTab>('home');
  
  // Profile modal state
  const [profileModalOpen, setProfileModalOpen] = useState<boolean>(false);
  const [profileModalTab, setProfileModalTab] = useState<'general' | 'medical' | 'settings'>('general');

  // Active workout state
  const [activeExercise, setActiveExercise] = useState<Exercise | null>(null);
  const [activeTargetReps, setActiveTargetReps] = useState<number>(10);

  // App data state
  const [user, setUser] = useState<User>(getUserProfile);
  const [sessions, setSessions] = useState<Session[]>(getSessions);
  const [dietEntries, setDietEntries] = useState<DietEntry[]>(getDietEntries);
  const [notes, setNotes] = useState<Note[]>(getNotes);
  const [doctors, setDoctors] = useState<Doctor[]>(getDoctors);
  const [appointments, setAppointments] = useState<Appointment[]>(getAppointments);
  const [messages, setMessages] = useState<Message[]>(getMessages);
  const [guardianAlerts, setGuardianAlerts] = useState<GuardianAlert[]>(getGuardianAlerts);
  const [isAdminLoggedIn, setIsAdminLoggedIn] = useState<boolean>(false);

  // Notification toast
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  useEffect(() => {
    initializeStorage();
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // Auth Handlers
  const handleLogin = (role: UserRole) => {
    setStoredAuthRole(role);
    setAuthRole(role);
    setCurrentTab(role === 'patient' ? 'home' : 'doctor_dashboard');
    showToast(`Signed into ${role === 'patient' ? 'Patient Portal' : 'Doctor / Clinician Console'}`);
  };

  const handleLogout = () => {
    clearStoredAuthRole();
    setAuthRole(null);
    showToast('Logged out. Select a portal domain to continue.');
  };

  const handleOpenProfileModal = (tab: 'general' | 'medical' | 'settings' = 'general') => {
    setProfileModalTab(tab);
    setProfileModalOpen(true);
  };

  // Exercise tracking handlers
  const handleStartExercise = (exercise: Exercise, targetReps: number = 10) => {
    setActiveExercise(exercise);
    setActiveTargetReps(targetReps || exercise.defaultTargetReps);
    setCurrentTab('session');
  };

  const handleSessionComplete = (sessionData: Omit<Session, 'id'>) => {
    const created = addSession(sessionData);
    setSessions([created, ...sessions]);
    showToast(`Saved session: ${sessionData.exerciseLabel} (${sessionData.formAccuracy}% accuracy)`);
    setActiveExercise(null);
    setCurrentTab('history');
  };

  const handleSaveProfile = (updatedUser: User) => {
    saveUserProfile(updatedUser);
    setUser(updatedUser);
    showToast('Medical profile updated successfully');
  };

  const handleAddDietEntry = (entry: Omit<DietEntry, 'id'>) => {
    const created = addDietEntry(entry);
    setDietEntries([created, ...dietEntries]);
    showToast(`Logged meal: ${entry.meal}`);
  };

  const handleDeleteDietEntry = (id: number) => {
    deleteDietEntry(id);
    setDietEntries(dietEntries.filter((e) => e.id !== id));
  };

  const handleAddNote = (text: string, category: Note['category']) => {
    const created = addNote(text, category);
    setNotes([created, ...notes]);
    showToast('Clinical journal note saved');
  };

  const handleDeleteNote = (id: number) => {
    deleteNote(id);
    setNotes(notes.filter((n) => n.id !== id));
  };

  const handleBookAppointment = (data: Omit<Appointment, 'id' | 'createdAt' | 'status'>) => {
    const created = bookAppointment(data);
    setAppointments([created, ...appointments]);
    showToast(`Appointment requested with ${data.doctorName}`);
  };

  const handleSendMessage = (doctorId: number, text: string) => {
    const created = sendMessage(doctorId, 'user', text);
    setMessages([...messages, created]);
    showToast('Message sent to physician');
  };

  const handleDoctorReply = (doctorId: number, text: string) => {
    const created = sendMessage(doctorId, 'doctor', text);
    setMessages([...messages, created]);
    showToast('Clinical advice dispatched to patient');
  };

  const handleUpdateAppointment = (id: number, status: AppointmentStatus, adminNote?: string) => {
    updateAppointmentStatus(id, status, adminNote);
    setAppointments(getAppointments());
    showToast(`Appointment status updated to "${status.toUpperCase()}"`);
  };

  const handleAddDoctor = (doc: Omit<Doctor, 'id'>) => {
    const created = addDoctor(doc);
    setDoctors([...doctors, created]);
    showToast(`Registered specialist: ${doc.name}`);
  };

  // If user is not authenticated into a specific domain, show the 2-Domain Portal Auth screen!
  if (!authRole) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
        <PortalAuth onLogin={handleLogin} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-800 flex flex-col md:flex-row font-sans selection:bg-blue-100 selection:text-blue-900">
      
      {/* LEFT SIDEBAR NAVIGATION WITH PROFILE DROPDOWN (Requirement 1 & 2) */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          if (tab !== 'session') {
            setActiveExercise(null);
          }
          setCurrentTab(tab);
        }}
        userRole={authRole}
        user={user}
        activeDoctor={doctors[0]}
        onLogout={handleLogout}
        onOpenProfileModal={handleOpenProfileModal}
      />

      {/* Global Toast Alert */}
      {toastMessage && (
        <div className="fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold shadow-xl flex items-center gap-2 border border-slate-800 animate-in slide-in-from-bottom-2 fade-in">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Account Profile & Settings Modal */}
      <AccountProfileModal
        isOpen={profileModalOpen}
        onClose={() => setProfileModalOpen(false)}
        userRole={authRole}
        user={user}
        activeDoctor={doctors[0]}
        initialTab={profileModalTab}
        onSaveProfile={handleSaveProfile}
        onSwitchPortal={() => {
          setProfileModalOpen(false);
          setAuthRole(null);
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          
          {/* ================================================================ */}
          {/* PATIENT PORTAL VIEWS */}
          {/* ================================================================ */}
          {authRole === 'patient' && (
            <>
              {/* Home Dashboard: Today's rehabilitation only (Requirement 10) */}
              {currentTab === 'home' && (
                <DashboardHome
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  onStartExercise={handleStartExercise}
                  onNavigate={(tab) => {
                    if (tab === 'telehealth') setCurrentTab('telehealth');
                    else if (tab === 'reports') setCurrentTab('reports');
                    else if (tab === 'history') setCurrentTab('history');
                    else if (tab === 'exercises') setCurrentTab('exercises');
                    else if (tab === 'progress') setCurrentTab('progress');
                    else setCurrentTab('home');
                  }}
                />
              )}

              {/* Exercises Directory */}
              {currentTab === 'exercises' && (
                <ExerciseSelectionView
                  user={user}
                  onSelectExercise={handleStartExercise}
                />
              )}

              {/* Live Exercise Session */}
              {currentTab === 'session' && (
                <TrackingView
                  exercise={activeExercise || EXERCISES.squat}
                  targetReps={activeTargetReps}
                  onSessionComplete={handleSessionComplete}
                  onBack={() => {
                    setActiveExercise(null);
                    setCurrentTab('exercises');
                  }}
                />
              )}

              {/* Dedicated Reports View (Requirements 4, 7, 8, 9) */}
              {currentTab === 'reports' && (
                <ReportsView
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  doctors={doctors}
                  onNavigate={(tab) => setCurrentTab(tab)}
                />
              )}

              {/* Dedicated History View (Requirement 6) */}
              {currentTab === 'history' && (
                <HistoryView
                  user={user}
                  sessions={sessions}
                />
              )}

              {/* Telehealth & Appointments (Requirement 3) */}
              {(currentTab === 'telehealth' || currentTab === 'appointments') && (
                <TelehealthView
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  messages={messages}
                  userRole={authRole}
                  onBookAppointment={handleBookAppointment}
                  onSendMessage={handleSendMessage}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {/* Progress Biometrics */}
              {currentTab === 'progress' && (
                <ProgressAnalyticsView sessions={sessions} />
              )}

              {/* Diet Tracker */}
              {currentTab === 'diet' && (
                <DietTrackerView
                  dietEntries={dietEntries}
                  onAddEntry={handleAddDietEntry}
                  onDeleteEntry={handleDeleteDietEntry}
                />
              )}

              {/* Notes Journal */}
              {currentTab === 'notes' && (
                <NotesView
                  notes={notes}
                  onAddNote={handleAddNote}
                  onDeleteNote={handleDeleteNote}
                />
              )}

              {/* Safety & SOS Section */}
              {currentTab === 'safety' && (
                <DashboardHome
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  onStartExercise={handleStartExercise}
                  onNavigate={(tab) => setCurrentTab(tab as AppNavTab)}
                />
              )}
            </>
          )}

          {/* ================================================================ */}
          {/* DOCTOR / PHYSIOTHERAPIST PORTAL VIEWS */}
          {/* ================================================================ */}
          {authRole === 'doctor' && (
            <>
              {/* Doctor Dashboard (Requirement 3 & 12) */}
              {currentTab === 'doctor_dashboard' && (
                <DoctorDashboard
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  sessions={sessions}
                  onNavigate={(tab) => {
                    if (tab === 'telehealth') setCurrentTab('telehealth');
                    else if (tab === 'reports') setCurrentTab('doctor_reports');
                    else setCurrentTab(tab as AppNavTab);
                  }}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {/* Patient Roster */}
              {currentTab === 'doctor_patients' && (
                <DoctorDashboard
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  sessions={sessions}
                  onNavigate={(tab) => setCurrentTab(tab as AppNavTab)}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {/* Patient Reports for Clinician */}
              {currentTab === 'doctor_reports' && (
                <ReportsView
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  doctors={doctors}
                  onNavigate={(tab) => setCurrentTab(tab)}
                />
              )}

              {/* Patient Progress */}
              {currentTab === 'doctor_progress' && (
                <ProgressAnalyticsView sessions={sessions} />
              )}

              {/* Alerts & SOS */}
              {currentTab === 'doctor_alerts' && (
                <AdminPortalView
                  doctors={doctors}
                  appointments={appointments}
                  messages={messages}
                  guardianAlerts={guardianAlerts}
                  isAdminLoggedIn={true}
                  onAdminLogin={() => setIsAdminLoggedIn(true)}
                  onAdminLogout={handleLogout}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                  onReplyMessage={handleDoctorReply}
                  onAddDoctor={handleAddDoctor}
                />
              )}

              {/* Doctor Telehealth & Appointments */}
              {(currentTab === 'telehealth' || currentTab === 'doctor_appointments' || currentTab === 'doctor_messages') && (
                <TelehealthView
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  messages={messages}
                  userRole={authRole}
                  onBookAppointment={handleBookAppointment}
                  onSendMessage={handleSendMessage}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}
            </>
          )}

        </main>

        {/* Clean Footer */}
        <footer className="border-t border-slate-200 bg-white py-3.5 px-4 sm:px-8 text-center text-xs text-slate-500">
          <p className="flex flex-wrap items-center justify-center gap-2">
            <span>PhysioAI Clinical Physical Therapy Suite</span>
            <span>•</span>
            <span>Client-Side Kinematic Angle Verification</span>
            <span>•</span>
            <button
              onClick={() => handleOpenProfileModal('settings')}
              className="text-blue-600 hover:text-blue-800 font-medium cursor-pointer"
            >
              Switch Role ({authRole.toUpperCase()})
            </button>
          </p>
        </footer>
      </div>

    </div>
  );
}

export default App;
