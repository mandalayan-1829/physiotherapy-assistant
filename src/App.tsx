import { useState, useEffect } from 'react';
import { Navbar, NavTab } from './components/Navbar';
import { DashboardHome } from './views/DashboardHome';
import { ExerciseSelectionView } from './views/ExerciseSelectionView';
import { TrackingView } from './views/TrackingView';
import { TelehealthView } from './views/TelehealthView';
import { DietTrackerView } from './views/DietTrackerView';
import { ProgressAnalyticsView } from './views/ProgressAnalyticsView';
import { NotesView } from './views/NotesView';
import { MedicalProfileView } from './views/MedicalProfileView';
import { AdminPortalView } from './views/AdminPortalView';

import { 
  Appointment, 
  DietEntry, 
  Doctor, 
  Exercise, 
  GuardianAlert, 
  Message, 
  Note, 
  Session, 
  User 
} from './types';
import { 
  addDietEntry, 
  addDoctor, 
  addNote, 
  addSession, 
  bookAppointment, 
  deleteDietEntry, 
  deleteNote, 
  getAppointments, 
  getDietEntries, 
  getDoctors, 
  getGuardianAlerts, 
  getMessages, 
  getNotes, 
  getSessions, 
  getUserProfile, 
  initializeStorage, 
  saveUserProfile, 
  sendMessage, 
  updateAppointmentStatus 
} from './utils/storage';
import { EXERCISES } from './data/exercises';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
  
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

  // Handlers
  const handleStartExercise = (exercise: Exercise, targetReps: number = 10) => {
    setActiveExercise(exercise);
    setActiveTargetReps(targetReps || exercise.defaultTargetReps);
    setCurrentTab('tracking');
  };

  const handleSessionComplete = (sessionData: Omit<Session, 'id'>) => {
    const created = addSession(sessionData);
    setSessions([created, ...sessions]);
    showToast(`Saved session: ${sessionData.exerciseLabel} (${sessionData.formAccuracy}% accuracy)`);
    setActiveExercise(null);
    setCurrentTab('progress');
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

  const handleUpdateAppointment = (id: number, status: Appointment['status'], adminNote?: string) => {
    updateAppointmentStatus(id, status, adminNote);
    setAppointments(getAppointments());
    showToast(`Appointment status changed to ${status}`);
  };

  const handleAddDoctor = (doc: Omit<Doctor, 'id'>) => {
    const created = addDoctor(doc);
    setDoctors([...doctors, created]);
    showToast(`Registered specialist: ${doc.name}`);
  };

  return (
    <div className="min-h-screen bg-white text-slate-800 flex flex-col font-sans selection:bg-blue-100 selection:text-blue-900">
      
      {/* Top App Bar */}
      <Navbar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          if (tab !== 'tracking') {
            setActiveExercise(null);
          }
          setCurrentTab(tab);
        }}
        user={user}
        unreadCount={0}
        isAdminLoggedIn={isAdminLoggedIn}
      />

      {/* Global Toast Alert */}
      {toastMessage && (
        <div className="fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold shadow-xl flex items-center gap-2 border border-slate-800 animate-in slide-in-from-bottom-2 fade-in">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'dashboard' && (
          <DashboardHome
            user={user}
            sessions={sessions}
            appointments={appointments}
            onStartExercise={handleStartExercise}
            onNavigate={(tab) => setCurrentTab(tab)}
          />
        )}

        {currentTab === 'exercises' && (
          <ExerciseSelectionView
            user={user}
            onSelectExercise={handleStartExercise}
          />
        )}

        {currentTab === 'tracking' && (
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

        {currentTab === 'telehealth' && (
          <TelehealthView
            user={user}
            doctors={doctors}
            appointments={appointments}
            messages={messages}
            onBookAppointment={handleBookAppointment}
            onSendMessage={handleSendMessage}
          />
        )}

        {currentTab === 'diet' && (
          <DietTrackerView
            dietEntries={dietEntries}
            onAddEntry={handleAddDietEntry}
            onDeleteEntry={handleDeleteDietEntry}
          />
        )}

        {currentTab === 'progress' && (
          <ProgressAnalyticsView sessions={sessions} />
        )}

        {currentTab === 'notes' && (
          <NotesView
            notes={notes}
            onAddNote={handleAddNote}
            onDeleteNote={handleDeleteNote}
          />
        )}

        {currentTab === 'profile' && (
          <MedicalProfileView
            user={user}
            onSaveProfile={handleSaveProfile}
          />
        )}

        {currentTab === 'admin' && (
          <AdminPortalView
            doctors={doctors}
            appointments={appointments}
            messages={messages}
            guardianAlerts={guardianAlerts}
            isAdminLoggedIn={isAdminLoggedIn}
            onAdminLogin={() => setIsAdminLoggedIn(true)}
            onAdminLogout={() => setIsAdminLoggedIn(false)}
            onUpdateAppointmentStatus={handleUpdateAppointment}
            onReplyMessage={handleDoctorReply}
            onAddDoctor={handleAddDoctor}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-slate-50/80 py-4 text-center text-xs text-slate-500">
        <p>PhysioAI • Real-Time AI Rehabilitation & Postural Kinematics • Clinical Physical Therapy Suite</p>
      </footer>

    </div>
  );
}
export default App;
