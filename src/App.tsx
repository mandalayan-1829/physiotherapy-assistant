import { useCallback, useEffect, useState } from 'react';
import { Sidebar, AppNavTab } from './components/Sidebar';
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
import { NotImplementedView } from './views/NotImplementedView';

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
  UserRole,
} from './types';
import { EXERCISES } from './data/exercises';
import { ApiError } from './services/api';
import type { AuthSession } from './services/auth';
import { fetchCurrentSession, logout as apiLogout, saveProfile } from './services/auth';
import { doctorToView, profileToUser, userToProfilePayload } from './services/mappers';
import {
  addDiet,
  addPatientNote,
  bookAppointment,
  createSession,
  deleteDiet,
  deleteNote,
  listAppointments,
  listDiet,
  listDoctors,
  listMessages,
  listNotes,
  listPatients,
  listSessions,
  sendMessage,
  updateAppointment as apiUpdateAppointment,
  type PatientSummary,
} from './services/physio';

/** Build a view-model `User` for a doctor account (no medical profile). */
function doctorAccountToUser(session: AuthSession): User {
  return {
    id: session.account.id,
    name: session.account.full_name,
    email: session.account.email,
    age: 0,
    gender: '',
    bloodGroup: '',
    heightCm: 0,
    weightKg: 0,
    currentProblem: '',
    medicalConditions: '',
    painLocation: '',
    painIntensity: 0,
    painType: '',
    currentMedications: '',
    movementRestrictions: '',
    rehabGoals: '',
    exerciseLimitations: '',
    emergencyContactName: '',
    emergencyContactPhone: '',
    guardianWhatsapp: '',
    doctorName: '',
    createdAt: '',
  };
}

export function App() {
  // --- Authentication ------------------------------------------------------
  const [session, setSession] = useState<AuthSession | null>(null);
  const [isRestoringSession, setIsRestoringSession] = useState<boolean>(true);

  const authRole: UserRole | null = session?.role ?? null;

  // --- UI state ------------------------------------------------------------
  const [currentTab, setCurrentTab] = useState<AppNavTab>('home');
  const [profileModalOpen, setProfileModalOpen] = useState<boolean>(false);
  const [profileModalTab, setProfileModalTab] = useState<'general' | 'medical' | 'settings'>('general');
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);

  // --- Active workout ------------------------------------------------------
  const [activeExercise, setActiveExercise] = useState<Exercise | null>(null);
  const [activeTargetReps, setActiveTargetReps] = useState<number>(10);

  // --- Application data (all loaded from the backend) ----------------------
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [dietEntries, setDietEntries] = useState<DietEntry[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [guardianAlerts, setGuardianAlerts] = useState<GuardianAlert[]>([]);
  const [patients, setPatients] = useState<PatientSummary[]>([]);

  const showToast = useCallback((message: string) => {
    setToastMessage(message);
    setTimeout(() => setToastMessage(null), 3500);
  }, []);

  const reportError = useCallback(
    (error: unknown, fallback: string) => {
      if (error instanceof ApiError) {
        showToast(error.message);
      } else {
        showToast(fallback);
      }
    },
    [showToast],
  );

  // --- Session restore -----------------------------------------------------
  // Identity always comes from the backend; nothing about the user is assumed
  // from browser storage.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const restored = await fetchCurrentSession();
        if (!cancelled) setSession(restored);
      } catch {
        if (!cancelled) setSession(null);
      } finally {
        if (!cancelled) setIsRestoringSession(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- Data loading --------------------------------------------------------
  useEffect(() => {
    if (!session) {
      setUser(null);
      setSessions([]);
      setDietEntries([]);
      setNotes([]);
      setDoctors([]);
      setAppointments([]);
      setMessages([]);
      setGuardianAlerts([]);
      setPatients([]);
      return;
    }

    let cancelled = false;

    (async () => {
      setIsLoadingData(true);
      try {
        if (session.role === 'patient') {
          setUser(profileToUser(session.account, session.profile));
          const [loadedSessions, loadedDiet, loadedNotes, loadedDoctors, loadedAppointments, loadedMessages] =
            await Promise.all([
              listSessions(),
              listDiet(),
              listNotes(),
              listDoctors(),
              listAppointments(),
              listMessages(),
            ]);
          if (cancelled) return;
          setSessions(loadedSessions);
          setDietEntries(loadedDiet);
          setNotes(loadedNotes);
          setDoctors(loadedDoctors);
          setAppointments(loadedAppointments);
          setMessages(loadedMessages);
        } else {
          setUser(doctorAccountToUser(session));
          const [loadedAppointments, roster, directory] = await Promise.all([
            listAppointments(),
            listPatients(),
            listDoctors(),
          ]);
          if (cancelled) return;
          setAppointments(loadedAppointments);
          setPatients(roster);
          setDoctors(directory);

          const [sessionLists, messageLists] = await Promise.all([
            Promise.all(roster.map((patient) => listSessions(patient.account_id))),
            Promise.all(roster.map((patient) => listMessages(patient.account_id))),
          ]);
          if (cancelled) return;
          setSessions(sessionLists.flat());
          setMessages(messageLists.flat());
        }
      } catch (error) {
        if (!cancelled) reportError(error, 'Could not load your data from the server.');
      } finally {
        if (!cancelled) setIsLoadingData(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [session, reportError]);

  // --- Auth handlers -------------------------------------------------------
  const handleAuthenticated = (authenticated: AuthSession) => {
    setSession(authenticated);
    setCurrentTab(authenticated.role === 'patient' ? 'home' : 'doctor_dashboard');
    showToast(
      authenticated.role === 'patient'
        ? `Signed in as ${authenticated.account.full_name}`
        : `Signed in to the Clinical Console as ${authenticated.account.full_name}`,
    );
  };

  const handleLogout = async () => {
    await apiLogout();
    setSession(null);
    setActiveExercise(null);
    setCurrentTab('home');
    showToast('Signed out. Select a portal domain to continue.');
  };

  const handleOpenProfileModal = (tab: 'general' | 'medical' | 'settings' = 'general') => {
    setProfileModalTab(tab);
    setProfileModalOpen(true);
  };

  // --- Exercise tracking ---------------------------------------------------
  const handleStartExercise = (exercise: Exercise, targetReps: number = 10) => {
    setActiveExercise(exercise);
    setActiveTargetReps(targetReps || exercise.defaultTargetReps);
    setCurrentTab('session');
  };

  const handleSessionComplete = async (sessionData: Omit<Session, 'id'>) => {
    try {
      const created = await createSession({
        exercise: sessionData.exercise,
        exerciseLabel: sessionData.exerciseLabel,
        reps: sessionData.reps,
        targetReps: sessionData.targetReps,
        formAccuracy: sessionData.formAccuracy,
        durationSec: sessionData.durationSec,
        notes: sessionData.notes,
        metricsSource: sessionData.metricsSource,
      });
      setSessions((previous) => [created, ...previous]);
      // A session without a real measurement must not be announced as a score.
      showToast(
        created.metricsSource === 'pose_inference'
          ? `Saved session: ${created.exerciseLabel} (${created.formAccuracy}% measured form)`
          : `Saved session: ${created.exerciseLabel} (logged manually, no form score)`,
      );
      setActiveExercise(null);
      setCurrentTab('history');
    } catch (error) {
      reportError(error, 'Could not save this session to the server.');
    }
  };

  // --- Profile -------------------------------------------------------------
  const handleSaveProfile = async (updatedUser: User) => {
    try {
      const saved = await saveProfile(userToProfilePayload(updatedUser));
      setUser((previous) =>
        previous ? profileToUser(session!.account, saved) : previous,
      );
      showToast('Medical profile updated successfully');
    } catch (error) {
      reportError(error, 'Could not save your medical profile.');
    }
  };

  // --- Diet ----------------------------------------------------------------
  const handleAddDietEntry = async (entry: Omit<DietEntry, 'id' | 'userId'>) => {
    try {
      const created = await addDiet({
        meal: entry.meal,
        calories: entry.calories,
        protein: entry.protein,
        carbs: entry.carbs,
        fats: entry.fats,
      });
      setDietEntries((previous) => [created, ...previous]);
      showToast(`Logged meal: ${created.meal}`);
    } catch (error) {
      reportError(error, 'Could not save this meal.');
    }
  };

  const handleDeleteDietEntry = async (id: number) => {
    try {
      await deleteDiet(id);
      setDietEntries((previous) => previous.filter((entry) => entry.id !== id));
    } catch (error) {
      reportError(error, 'Could not delete this meal.');
    }
  };

  // --- Notes ---------------------------------------------------------------
  const handleAddNote = async (text: string, category: Note['category']) => {
    if (!session) return;
    try {
      const created = await addPatientNote(session.account.id, text, category);
      setNotes((previous) => [created, ...previous]);
      showToast('Journal note saved');
    } catch (error) {
      reportError(error, 'Could not save this note.');
    }
  };

  const handleDeleteNote = async (id: number) => {
    try {
      await deleteNote(id);
      setNotes((previous) => previous.filter((note) => note.id !== id));
    } catch (error) {
      reportError(error, 'Could not delete this note.');
    }
  };

  // --- Appointments --------------------------------------------------------
  const handleBookAppointment = async (data: {
    doctorId: number;
    date: string;
    time: string;
    reason: string;
  }) => {
    try {
      const created = await bookAppointment({
        doctorProfileId: data.doctorId,
        date: data.date,
        time: data.time,
        reason: data.reason,
      });
      setAppointments((previous) => [created, ...previous]);
      showToast(`Appointment requested with ${created.doctorName}`);
    } catch (error) {
      reportError(error, 'Could not book this appointment.');
    }
  };

  const handleUpdateAppointment = async (
    id: number,
    status: AppointmentStatus,
    clinicianNote?: string,
  ) => {
    try {
      const updated = await apiUpdateAppointment(id, {
        status,
        ...(clinicianNote !== undefined ? { clinician_note: clinicianNote } : {}),
      });
      setAppointments((previous) =>
        previous.map((appointment) => (appointment.id === id ? updated : appointment)),
      );
      showToast(`Appointment status updated to "${status.toUpperCase()}"`);
    } catch (error) {
      reportError(error, 'Could not update this appointment.');
    }
  };

  // --- Messages ------------------------------------------------------------
  const handleSendMessage = async (doctorId: number, text: string) => {
    if (!session) return;
    if (session.role !== 'patient') {
      // Clinician replies need a selected patient context, which the console
      // does not provide yet.
      showToast('Sending a clinician reply requires selecting a patient (not implemented yet).');
      return;
    }
    try {
      const created = await sendMessage({
        doctorProfileId: doctorId,
        message: text,
        sender: 'user',
      });
      setMessages((previous) => [...previous, created]);
      showToast('Message sent to your clinician');
    } catch (error) {
      reportError(error, 'Could not send this message.');
    }
  };

  // --- Render --------------------------------------------------------------

  if (isRestoringSession) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center space-y-2">
          <div className="w-10 h-10 mx-auto rounded-lg bg-blue-600 animate-pulse" />
          <p className="text-sm text-slate-600">Restoring your session…</p>
        </div>
      </div>
    );
  }

  // Anyone who is not authenticated only ever sees the portal chooser.
  if (!session) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
        <PortalAuth onAuthenticated={handleAuthenticated} />
      </div>
    );
  }

  const activeDoctor = session.role === 'doctor' && session.doctorProfile
    ? doctorToView(session.doctorProfile)
    : doctors[0];

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-800 flex flex-col md:flex-row font-sans selection:bg-blue-100 selection:text-blue-900">

      {/* LEFT SIDEBAR NAVIGATION WITH PROFILE DROPDOWN */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          if (tab !== 'session') {
            setActiveExercise(null);
          }
          setCurrentTab(tab);
        }}
        userRole={session.role}
        user={user ?? doctorAccountToUser(session)}
        activeDoctor={activeDoctor}
        onLogout={handleLogout}
        onOpenProfileModal={handleOpenProfileModal}
      />

      {/* Global Toast Alert */}
      {toastMessage && (
        <div className="fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-semibold shadow-xl flex items-center gap-2 border border-slate-800">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Account Profile & Settings Modal */}
      {user && (
        <AccountProfileModal
          isOpen={profileModalOpen}
          onClose={() => setProfileModalOpen(false)}
          userRole={session.role}
          user={user}
          activeDoctor={activeDoctor}
          initialTab={profileModalTab}
          onSaveProfile={handleSaveProfile}
          onSwitchPortal={() => {
            setProfileModalOpen(false);
            void handleLogout();
          }}
        />
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">

          {isLoadingData && (
            <div className="mb-4 p-2.5 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-600 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span>Syncing with the PhysioAI server…</span>
            </div>
          )}

          {/* ================================================================ */}
          {/* PATIENT PORTAL VIEWS */}
          {/* ================================================================ */}
          {session.role === 'patient' && user && (
            <>
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

              {currentTab === 'exercises' && (
                <ExerciseSelectionView user={user} onSelectExercise={handleStartExercise} />
              )}

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

              {currentTab === 'reports' && (
                <ReportsView
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  doctors={doctors}
                  onNavigate={(tab) => setCurrentTab(tab)}
                />
              )}

              {currentTab === 'history' && <HistoryView user={user} sessions={sessions} />}

              {(currentTab === 'telehealth' || currentTab === 'appointments') && (
                <TelehealthView
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  messages={messages}
                  userRole={session.role}
                  onBookAppointment={handleBookAppointment}
                  onSendMessage={handleSendMessage}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {currentTab === 'progress' && <ProgressAnalyticsView sessions={sessions} />}

              {currentTab === 'diet' && (
                <DietTrackerView
                  dietEntries={dietEntries}
                  onAddEntry={handleAddDietEntry}
                  onDeleteEntry={handleDeleteDietEntry}
                />
              )}

              {currentTab === 'notes' && (
                <NotesView notes={notes} onAddNote={handleAddNote} onDeleteNote={handleDeleteNote} />
              )}

              {currentTab === 'safety' && (
                <NotImplementedView
                  title="Safety & SOS"
                  description="Emergency alerting is stored on the server but the patient-facing safety console has not been connected yet."
                  pending={[
                    'Guardian alert history from GET /alerts',
                    'Server-side dispatch (currently a client-side WhatsApp link)',
                    'Missed-routine and pain-spike automation',
                  ]}
                />
              )}
            </>
          )}

          {/* ================================================================ */}
          {/* DOCTOR / PHYSIOTHERAPIST PORTAL VIEWS */}
          {/* ================================================================ */}
          {session.role === 'doctor' && user && (
            <>
              {currentTab === 'doctor_dashboard' && (
                <DoctorDashboard
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  sessions={sessions}
                  patients={patients}
                  onNavigate={(tab) => {
                    if (tab === 'telehealth') setCurrentTab('telehealth');
                    else if (tab === 'reports') setCurrentTab('doctor_reports');
                    else setCurrentTab(tab as AppNavTab);
                  }}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {currentTab === 'doctor_patients' && (
                <DoctorDashboard
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  sessions={sessions}
                  patients={patients}
                  onNavigate={(tab) => setCurrentTab(tab as AppNavTab)}
                  onUpdateAppointmentStatus={handleUpdateAppointment}
                />
              )}

              {currentTab === 'doctor_reports' && (
                <ReportsView
                  user={user}
                  sessions={sessions}
                  appointments={appointments}
                  doctors={doctors}
                  onNavigate={(tab) => setCurrentTab(tab)}
                />
              )}

              {currentTab === 'doctor_progress' && <ProgressAnalyticsView sessions={sessions} />}

              {currentTab === 'doctor_alerts' && (
                <NotImplementedView
                  title="Patient Safety Alerts"
                  description="Alerts are persisted per patient on the server; the clinician alert console is still to be built."
                  pending={[
                    'Aggregate alerts across linked patients (GET /alerts)',
                    'Acknowledge / escalate workflow',
                    'Real-time notification delivery',
                  ]}
                />
              )}

              {(currentTab === 'telehealth' ||
                currentTab === 'doctor_appointments' ||
                currentTab === 'doctor_messages') && (
                <TelehealthView
                  user={user}
                  doctors={doctors}
                  appointments={appointments}
                  messages={messages}
                  userRole={session.role}
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
            <span>Server-verified session ({session.account.email})</span>
            <span>•</span>
            <button
              onClick={() => void handleLogout()}
              className="text-blue-600 hover:text-blue-800 font-medium cursor-pointer"
            >
              Sign out ({session.role.toUpperCase()})
            </button>
          </p>
        </footer>
      </div>

    </div>
  );
}

export default App;
