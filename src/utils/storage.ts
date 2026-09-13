import { Appointment, DietEntry, Doctor, GuardianAlert, Message, MonthlyReport, DailyHistoryEntry, Note, Session, User, UserRole } from '../types';

const STORAGE_KEYS = {
  USER: 'physio_user_profile',
  SESSIONS: 'physio_sessions',
  DIET: 'physio_diet_log',
  NOTES: 'physio_clinical_notes',
  DOCTORS: 'physio_doctors',
  APPOINTMENTS: 'physio_appointments',
  MESSAGES: 'physio_telehealth_messages',
  ALERTS: 'physio_guardian_alerts',
  MONTHLY_REPORTS: 'physio_monthly_reports',
  AUTH_ROLE: 'physio_auth_role',
  ACTIVE_DOCTOR_ID: 'physio_active_doctor_id',
  LAST_RESET_DATE: 'physio_last_reset_date',
  INITIALIZED: 'physio_seeded_v2',
};

export function getTodayDateString(): string {
  const now = new Date();
  return now.toISOString().split('T')[0];
}

// Seed doctors from SQLite database (Dr. Aarav & Dr. Arjun Mehta)
const INITIAL_DOCTORS: Doctor[] = [
  {
    id: 1,
    name: 'Dr. Aarav Patel',
    specialization: 'Orthopedic Rehabilitation',
    experience: 5,
    qualification: 'MBBS, MS (Ortho), DNB',
    availableDays: 'Mon, Wed, Fri',
    timings: '09:00 AM - 02:00 PM',
    about: 'Specialist in post-operative knee recovery, ligament repair rehabilitation, and sports biomechanics.',
    contact: '+91 82934 13240',
    whatsapp: '918293413240',
    email: 'dr.aarav@physioai.health',
    avatarUrl: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=200&auto=format&fit=crop&q=80',
  },
  {
    id: 2,
    name: 'Dr. Arjun Mehta',
    specialization: 'Spine & Neurological Physiotherapy',
    experience: 7,
    qualification: 'MBBS, MPT (Neuro), Certified Ergonomist',
    availableDays: 'Tue, Thu, Sat',
    timings: '10:00 AM - 04:00 PM',
    about: 'Expertise in lumbar disc rehabilitation, cervical spondylosis posture correction, and balance recovery.',
    contact: '+91 82934 13241',
    whatsapp: '918293413241',
    email: 'dr.arjun@physioai.health',
    avatarUrl: 'https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=200&auto=format&fit=crop&q=80',
  },
  {
    id: 3,
    name: 'Dr. Priya Sharma',
    specialization: 'Sports Medicine & Joint Mobility',
    experience: 8,
    qualification: 'BPT, MPT (Sports), CMPT',
    availableDays: 'Mon, Tue, Thu, Fri',
    timings: '01:00 PM - 06:00 PM',
    about: 'Focuses on shoulder rotator cuff strengthening, runners knee, and functional kinematic movement therapy.',
    contact: '+91 98301 44211',
    whatsapp: '919830144211',
    email: 'dr.priya@physioai.health',
    avatarUrl: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=200&auto=format&fit=crop&q=80',
  },
];

// Initial user profile matching SQLite data
const INITIAL_USER: User = {
  id: 1,
  name: 'John Doe',
  age: 28,
  gender: 'Male',
  dob: '1998-05-14',
  email: 'mandalayan1829@gmail.com',
  contactNumber: '+91 82934 13240',
  bloodGroup: 'O+',
  heightCm: 178,
  weightKg: 74,
  occupation: 'Software Engineer (High Desk Sitting)',
  currentProblem: 'Mild left knee stiffness and lower back fatigue after prolonged sitting',
  problemStartDate: '2026-03-10',
  problemCause: 'Workstation ergonomics and occasional running strain',
  previousInjuries: 'Minor right ankle sprain (2024)',
  pastSurgeries: 'None',
  medicalConditions: 'Occasional lumbar tightness',
  painLocation: 'Knee, Lower Back',
  painIntensity: 4,
  painType: 'Dull ache',
  painTriggers: 'Prolonged sitting (>4 hours) and deep flexion',
  painDuration: 'Intermittent across past 3 weeks',
  dailySittingHours: 8,
  activityLevel: 'Moderately active (gym 2x/week)',
  exerciseHabits: 'Light cardio, stretching',
  currentMedications: 'Vitamin D3 & Omega-3 supplements',
  movementRestrictions: 'Avoid rapid twist loads on knee',
  rehabGoals: 'Restore full pain-free squat depth and strengthen gluteal stabilizers',
  allergies: 'None reported',
  precautions: 'Warm up hamstrings and hips thoroughly before squats',
  exerciseLimitations: 'Knee discomfort during excessive valgus angle',
  emergencyContactName: 'Sarah Doe (Spouse)',
  emergencyContactPhone: '+91 98765 43210',
  guardianWhatsapp: '919876543210',
  doctorName: 'Dr. Aarav Patel',
  createdAt: '2026-04-10',
};

const INITIAL_SESSIONS: Session[] = [
  {
    id: 101,
    userId: 1,
    exercise: 'squat',
    exerciseLabel: 'Squat',
    reps: 10,
    targetReps: 10,
    formAccuracy: 94,
    durationSec: 85,
    notes: 'Upright torso and stable pelvic tilt. No knee valgus.',
    date: `${new Date().toISOString().split('T')[0]} 09:30`,
  },
  {
    id: 102,
    userId: 1,
    exercise: 'shoulder_raises',
    exerciseLabel: 'Shoulder Raises',
    reps: 12,
    targetReps: 12,
    formAccuracy: 90,
    durationSec: 90,
    notes: 'Smooth lateral abduction tempo.',
    date: `${new Date().toISOString().split('T')[0]} 09:45`,
  },
  {
    id: 1,
    userId: 1,
    exercise: 'squat',
    exerciseLabel: 'Squat',
    reps: 10,
    targetReps: 10,
    formAccuracy: 92,
    durationSec: 85,
    notes: 'Maintained great chest upright posture. No knee valgus observed.',
    date: '2026-09-12 10:30',
  },
  {
    id: 2,
    userId: 1,
    exercise: 'cat_cow_stretch',
    exerciseLabel: 'Cat-Cow Stretch',
    reps: 10,
    targetReps: 10,
    formAccuracy: 96,
    durationSec: 120,
    notes: 'Fluid spinal flexion and extension with synchronized exhalation.',
    date: '2026-09-12 16:45',
  },
  {
    id: 3,
    userId: 1,
    exercise: 'shoulder_raises',
    exerciseLabel: 'Shoulder Raises',
    reps: 12,
    targetReps: 12,
    formAccuracy: 88,
    durationSec: 92,
    notes: 'Controlled tempo. Slight torso swing on rep 11, quickly corrected.',
    date: '2026-09-11 17:15',
  },
  {
    id: 4,
    userId: 1,
    exercise: 'tree_pose',
    exerciseLabel: 'Tree Pose',
    reps: 4,
    targetReps: 4,
    formAccuracy: 90,
    durationSec: 110,
    notes: 'Sustained 5-second single-leg balances on both left and right sides.',
    date: '2026-09-11 18:20',
  },
  {
    id: 5,
    userId: 1,
    exercise: 'cat_cow_stretch',
    exerciseLabel: 'Cat-Cow Stretch',
    reps: 10,
    targetReps: 10,
    formAccuracy: 95,
    durationSec: 120,
    notes: 'Deep rhythmic breathing coordinated with spine mobilization.',
    date: '2026-09-10 08:45',
  },
  {
    id: 6,
    userId: 1,
    exercise: 'squat',
    exerciseLabel: 'Squat',
    reps: 10,
    targetReps: 10,
    formAccuracy: 91,
    durationSec: 80,
    notes: 'Consistent depth reaching therapeutic 90-degree knee flexion.',
    date: '2026-09-10 11:20',
  },
  {
    id: 7,
    userId: 1,
    exercise: 'tree_pose',
    exerciseLabel: 'Tree Pose',
    reps: 4,
    targetReps: 4,
    formAccuracy: 92,
    durationSec: 115,
    notes: 'Core engaged, solid foot arch engagement.',
    date: '2026-09-08 09:10',
  },
  {
    id: 8,
    userId: 1,
    exercise: 'shoulder_raises',
    exerciseLabel: 'Shoulder Raises',
    reps: 10,
    targetReps: 10,
    formAccuracy: 87,
    durationSec: 80,
    notes: 'Slight deltoid fatigue toward final repetitions.',
    date: '2026-09-08 17:30',
  },
  {
    id: 9,
    userId: 1,
    exercise: 'squat',
    exerciseLabel: 'Squat',
    reps: 10,
    targetReps: 10,
    formAccuracy: 89,
    durationSec: 82,
    notes: 'Good cadence, hamstrings well activated.',
    date: '2026-09-05 10:15',
  },
  {
    id: 10,
    userId: 1,
    exercise: 'cat_cow_stretch',
    exerciseLabel: 'Cat-Cow Stretch',
    reps: 10,
    targetReps: 10,
    formAccuracy: 93,
    durationSec: 110,
    notes: 'Thoracic extension improving significantly.',
    date: '2026-09-05 10:45',
  },
  {
    id: 11,
    userId: 1,
    exercise: 'squat',
    exerciseLabel: 'Squat',
    reps: 10,
    targetReps: 10,
    formAccuracy: 86,
    durationSec: 90,
    notes: 'August baseline calibration session.',
    date: '2026-08-28 11:00',
  },
  {
    id: 12,
    userId: 1,
    exercise: 'shoulder_raises',
    exerciseLabel: 'Shoulder Raises',
    reps: 12,
    targetReps: 12,
    formAccuracy: 84,
    durationSec: 95,
    notes: 'Initial mobility range check.',
    date: '2026-08-24 16:00',
  },
];

const INITIAL_DIET: DietEntry[] = [
  {
    id: 1,
    userId: 1,
    meal: 'Oatmeal with blueberries, whey protein & chia seeds',
    calories: 420,
    protein: 32,
    carbs: 52,
    fats: 9,
    date: new Date().toISOString().split('T')[0],
  },
  {
    id: 2,
    userId: 1,
    meal: 'Grilled chicken breast salad with avocado & olive oil',
    calories: 560,
    protein: 48,
    carbs: 18,
    fats: 22,
    date: new Date().toISOString().split('T')[0],
  },
];

const INITIAL_NOTES: Note[] = [
  {
    id: 1,
    userId: 1,
    noteText: 'Dr. Aarav advised maintaining at least 90-degree knee bend on squats without letting knees track inward.',
    category: 'clinical',
    date: '2026-09-11',
  },
  {
    id: 2,
    userId: 1,
    noteText: 'Morning stiffness has dropped from 4/10 to 2/10 after 4 days of consistent Cat-Cow stretches.',
    category: 'symptom',
    date: '2026-09-12',
  },
];

const INITIAL_APPOINTMENTS: Appointment[] = [
  {
    id: 1,
    userId: 1,
    doctorId: 1,
    doctorName: 'Dr. Aarav Patel',
    specialization: 'Orthopedic Rehabilitation',
    patientName: 'John Doe',
    email: 'mandalayan1829@gmail.com',
    date: '2026-09-18',
    time: '11:00 AM',
    reason: 'Follow-up evaluation on knee joint mobility and squat depth progression.',
    status: 'confirmed',
    adminNote: 'Confirmed by clinic. Please wear athletic shorts for range of motion check.',
    createdAt: '2026-09-12 14:20',
  },
  {
    id: 2,
    userId: 1,
    doctorId: 1,
    doctorName: 'Dr. Aarav Patel',
    specialization: 'Orthopedic Rehabilitation',
    patientName: 'John Doe',
    email: 'mandalayan1829@gmail.com',
    date: getTodayDateString(),
    time: '04:00 PM',
    reason: 'Weekly kinematic posture check-in and bio-feedback alignment review.',
    status: 'ready',
    adminNote: 'Consultation room active. Doctor has prepared kinematic analysis reports.',
    createdAt: `${getTodayDateString()} 08:30`,
  },
  {
    id: 3,
    userId: 1,
    doctorId: 2,
    doctorName: 'Dr. Arjun Mehta',
    specialization: 'Spine & Neurological Physiotherapy',
    patientName: 'John Doe',
    email: 'mandalayan1829@gmail.com',
    date: '2026-09-22',
    time: '02:30 PM',
    reason: 'Ergonomic lumbar spine routine review and desk posture adaptation.',
    status: 'scheduled',
    adminNote: 'Pending physician calendar confirmation.',
    createdAt: `${getTodayDateString()} 10:15`,
  },
];

const INITIAL_MONTHLY_REPORTS: MonthlyReport[] = [
  {
    id: 'report-2026-08',
    monthKey: '2026-08',
    monthName: 'August 2026',
    generatedDate: '2026-08-31 18:00',
    patientId: 1,
    patientName: 'John Doe',
    patientEmail: 'mandalayan1829@gmail.com',
    assignedDoctorName: 'Dr. Aarav Patel',
    assignedDoctorEmail: 'dr.aarav@physioai.health',
    hasUpcomingCheckup: true,
    totalSessions: 14,
    totalReps: 136,
    completedExercises: 3,
    missedSessions: 2,
    avgAccuracy: 86,
    avgScore: 86,
    adherencePercent: 88,
    exerciseBreakdown: [
      { exerciseLabel: 'Squat', sessions: 6, reps: 60, avgAccuracy: 86 },
      { exerciseLabel: 'Shoulder Raises', sessions: 5, reps: 52, avgAccuracy: 85 },
      { exerciseLabel: 'Cat-Cow Stretch', sessions: 3, reps: 24, avgAccuracy: 88 },
    ],
    progressTrend: '+6% improvement in lumbar stabilization compared to July.',
    safetyEventsCount: 0,
    warningsCount: 1,
    emailStatus: 'Sent',
    emailSentDate: '2026-08-31 18:05',
    recipients: ['mandalayan1829@gmail.com', 'dr.aarav@physioai.health'],
  }
];

const INITIAL_MESSAGES: Message[] = [
  {
    id: 1,
    userId: 1,
    doctorId: 1,
    sender: 'user',
    message: 'Hello Dr. Patel, I completed 10 squats today with 92% form score. Should I add weight or stick to bodyweight?',
    timestamp: '2026-09-12 11:05',
  },
  {
    id: 2,
    userId: 1,
    doctorId: 1,
    sender: 'doctor',
    message: 'Great work, John! Stick with bodyweight for another week until your knee stiffness fully resolves. Ensure your heels stay firmly on the ground.',
    timestamp: '2026-09-12 11:42',
  },
];

export function initializeStorage() {
  if (typeof window === 'undefined') return;

  const isInitialized = localStorage.getItem(STORAGE_KEYS.INITIALIZED);
  if (!isInitialized) {
    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(INITIAL_USER));
    localStorage.setItem(STORAGE_KEYS.DOCTORS, JSON.stringify(INITIAL_DOCTORS));
    localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(INITIAL_SESSIONS));
    localStorage.setItem(STORAGE_KEYS.DIET, JSON.stringify(INITIAL_DIET));
    localStorage.setItem(STORAGE_KEYS.NOTES, JSON.stringify(INITIAL_NOTES));
    localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify(INITIAL_APPOINTMENTS));
    localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(INITIAL_MESSAGES));
    localStorage.setItem(STORAGE_KEYS.ALERTS, JSON.stringify([]));
    localStorage.setItem(STORAGE_KEYS.MONTHLY_REPORTS, JSON.stringify(INITIAL_MONTHLY_REPORTS));
    localStorage.setItem(STORAGE_KEYS.LAST_RESET_DATE, getTodayDateString());
    localStorage.setItem(STORAGE_KEYS.INITIALIZED, 'true');
  }

  // Daily reset check: update LAST_RESET_DATE if new day, ensuring daily counters reflect only today
  const lastReset = localStorage.getItem(STORAGE_KEYS.LAST_RESET_DATE);
  const today = getTodayDateString();
  if (lastReset !== today) {
    localStorage.setItem(STORAGE_KEYS.LAST_RESET_DATE, today);
  }
}

// User Profile
export function getUserProfile(): User {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.USER);
  return raw ? JSON.parse(raw) : INITIAL_USER;
}

export function saveUserProfile(user: User) {
  localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
}

// Auth Role
export function getStoredAuthRole(): UserRole | null {
  if (typeof window === 'undefined') return 'patient';
  const role = localStorage.getItem(STORAGE_KEYS.AUTH_ROLE);
  return (role === 'doctor' || role === 'patient') ? (role as UserRole) : 'patient';
}

export function setStoredAuthRole(role: UserRole | null) {
  if (typeof window === 'undefined') return;
  if (role) {
    localStorage.setItem(STORAGE_KEYS.AUTH_ROLE, role);
  } else {
    localStorage.removeItem(STORAGE_KEYS.AUTH_ROLE);
  }
}

export function getActiveDoctorId(): number {
  if (typeof window === 'undefined') return 1;
  const raw = localStorage.getItem(STORAGE_KEYS.ACTIVE_DOCTOR_ID);
  return raw ? parseInt(raw, 10) : 1;
}

export function setActiveDoctorId(id: number) {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEYS.ACTIVE_DOCTOR_ID, id.toString());
}

// Sessions
export function getSessions(): Session[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.SESSIONS);
  return raw ? JSON.parse(raw) : [];
}

export function addSession(session: Omit<Session, 'id'>): Session {
  const sessions = getSessions();
  const newSession: Session = {
    ...session,
    id: Date.now(),
  };
  sessions.unshift(newSession);
  localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(sessions));
  return newSession;
}

// Today's sessions filter (Resets every new day; keeps historical permanent)
export function getTodaySessions(allSessions?: Session[]): Session[] {
  const sessions = allSessions || getSessions();
  const today = getTodayDateString();
  return sessions.filter((s) => s.date.startsWith(today));
}

// Historical daily tracking aggregation
export function getHistoricalDailyEntries(allSessions?: Session[]): DailyHistoryEntry[] {
  const sessions = allSessions || getSessions();
  const groups: { [dateStr: string]: Session[] } = {};

  sessions.forEach((s) => {
    const dateKey = s.date.split(' ')[0] || s.date;
    if (!groups[dateKey]) {
      groups[dateKey] = [];
    }
    groups[dateKey].push(s);
  });

  const sortedDates = Object.keys(groups).sort((a, b) => b.localeCompare(a));

  return sortedDates.map((dateStr) => {
    const daySessions = groups[dateStr];
    const totalReps = daySessions.reduce((acc, s) => acc + s.reps, 0);
    const avgScore = Math.round(daySessions.reduce((acc, s) => acc + s.formAccuracy, 0) / daySessions.length);
    const totalDurationSec = daySessions.reduce((acc, s) => acc + s.durationSec, 0);
    const correctReps = Math.round(totalReps * (avgScore / 100));
    const incorrectReps = totalReps - correctReps;
    const uniqueExercises = Array.from(new Set(daySessions.map((s) => s.exerciseLabel)));

    // Parse friendly display date
    let displayDate = dateStr;
    try {
      const parts = dateStr.split('-');
      if (parts.length === 3) {
        const d = new Date(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]));
        displayDate = d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
      }
    } catch {
      // fallback
    }

    return {
      date: dateStr,
      displayDate,
      sessionsCount: daySessions.length,
      totalReps,
      correctReps,
      incorrectReps,
      avgFormScore: avgScore,
      totalDurationSec,
      exercises: uniqueExercises,
      warningsCount: avgScore < 90 ? 1 : 0,
      safetyEventsCount: 0,
      sessions: daySessions,
    };
  });
}

// Monthly Reports
export function getMonthlyReports(): MonthlyReport[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.MONTHLY_REPORTS);
  return raw ? JSON.parse(raw) : INITIAL_MONTHLY_REPORTS;
}

export function saveMonthlyReport(report: MonthlyReport): MonthlyReport {
  const reports = getMonthlyReports();
  const index = reports.findIndex((r) => r.id === report.id || r.monthKey === report.monthKey);
  if (index >= 0) {
    reports[index] = report;
  } else {
    reports.unshift(report);
  }
  localStorage.setItem(STORAGE_KEYS.MONTHLY_REPORTS, JSON.stringify(reports));
  return report;
}

export function generateMonthlyReport(
  monthKey: string, // e.g. "2026-09"
  user: User,
  sessions: Session[],
  appointments: Appointment[],
  doctors: Doctor[]
): MonthlyReport {
  const monthSessions = sessions.filter((s) => s.date.startsWith(monthKey));
  const totalSessions = monthSessions.length;
  const totalReps = monthSessions.reduce((acc, s) => acc + s.reps, 0);
  const avgAccuracy = totalSessions > 0
    ? Math.round(monthSessions.reduce((acc, s) => acc + s.formAccuracy, 0) / totalSessions)
    : 92;

  // Exercise-wise breakdown
  const exerciseMap: { [label: string]: { sessions: number; reps: number; totalAcc: number } } = {};
  monthSessions.forEach((s) => {
    if (!exerciseMap[s.exerciseLabel]) {
      exerciseMap[s.exerciseLabel] = { sessions: 0, reps: 0, totalAcc: 0 };
    }
    exerciseMap[s.exerciseLabel].sessions += 1;
    exerciseMap[s.exerciseLabel].reps += s.reps;
    exerciseMap[s.exerciseLabel].totalAcc += s.formAccuracy;
  });

  const exerciseBreakdown = Object.keys(exerciseMap).map((label) => ({
    exerciseLabel: label,
    sessions: exerciseMap[label].sessions,
    reps: exerciseMap[label].reps,
    avgAccuracy: Math.round(exerciseMap[label].totalAcc / exerciseMap[label].sessions),
  }));

  // Month name
  const [yearStr, monthStr] = monthKey.split('-');
  const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const monthIndex = parseInt(monthStr, 10) - 1;
  const monthName = `${monthNames[monthIndex] || 'Selected Month'} ${yearStr}`;

  // Check upcoming appointment with assigned doctor
  const upcomingAppointment = appointments.find(
    (a) => (a.status === 'confirmed' || a.status === 'scheduled' || a.status === 'ready' || a.status === 'approved') &&
           a.date >= getTodayDateString()
  );

  const assignedDoc = doctors.find((d) => d.name === user.doctorName) || doctors[0];
  const hasUpcomingCheckup = !!upcomingAppointment;

  const now = new Date();
  const generatedDate = `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`;

  const newReport: MonthlyReport = {
    id: `report-${monthKey}`,
    monthKey,
    monthName,
    generatedDate,
    patientId: user.id,
    patientName: user.name,
    patientEmail: user.email,
    assignedDoctorName: assignedDoc?.name || 'Dr. Aarav Patel',
    assignedDoctorEmail: assignedDoc?.email || 'dr.aarav@physioai.health',
    hasUpcomingCheckup,
    totalSessions,
    totalReps,
    completedExercises: exerciseBreakdown.length,
    missedSessions: totalSessions > 0 ? 1 : 4,
    avgAccuracy,
    avgScore: avgAccuracy,
    adherencePercent: totalSessions >= 10 ? 94 : totalSessions >= 5 ? 85 : 70,
    exerciseBreakdown,
    progressTrend: '+8% improvement in joint kinematic angle stability across rehabilitation workouts.',
    safetyEventsCount: 0,
    warningsCount: 0,
    emailStatus: 'Draft',
    recipients: [user.email],
  };

  return saveMonthlyReport(newReport);
}

export function sendMonthlyReportEmail(reportId: string): MonthlyReport | null {
  const reports = getMonthlyReports();
  const report = reports.find((r) => r.id === reportId);
  if (!report) return null;

  const now = new Date();
  const timeStr = `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`;

  // Rule: Send to patient email. Send to doctor email ONLY when the patient has a scheduled checkup/appointment.
  const recipients = [report.patientEmail];
  if (report.hasUpcomingCheckup && report.assignedDoctorEmail) {
    if (!recipients.includes(report.assignedDoctorEmail)) {
      recipients.push(report.assignedDoctorEmail);
    }
  }

  const updated: MonthlyReport = {
    ...report,
    emailStatus: 'Sent',
    emailSentDate: timeStr,
    recipients,
  };

  saveMonthlyReport(updated);
  return updated;
}

// Diet
export function getDietEntries(): DietEntry[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.DIET);
  return raw ? JSON.parse(raw) : [];
}

export function addDietEntry(entry: Omit<DietEntry, 'id'>): DietEntry {
  const entries = getDietEntries();
  const newEntry: DietEntry = {
    ...entry,
    id: Date.now(),
  };
  entries.unshift(newEntry);
  localStorage.setItem(STORAGE_KEYS.DIET, JSON.stringify(entries));
  return newEntry;
}

export function deleteDietEntry(id: number) {
  const entries = getDietEntries().filter((e) => e.id !== id);
  localStorage.setItem(STORAGE_KEYS.DIET, JSON.stringify(entries));
}

// Notes
export function getNotes(): Note[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.NOTES);
  return raw ? JSON.parse(raw) : [];
}

export function addNote(noteText: string, category: Note['category'] = 'clinical'): Note {
  const notes = getNotes();
  const newNote: Note = {
    id: Date.now(),
    userId: 1,
    noteText,
    category,
    date: new Date().toISOString().split('T')[0],
  };
  notes.unshift(newNote);
  localStorage.setItem(STORAGE_KEYS.NOTES, JSON.stringify(notes));
  return newNote;
}

export function deleteNote(id: number) {
  const notes = getNotes().filter((n) => n.id !== id);
  localStorage.setItem(STORAGE_KEYS.NOTES, JSON.stringify(notes));
}

// Doctors
export function getDoctors(): Doctor[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.DOCTORS);
  return raw ? JSON.parse(raw) : INITIAL_DOCTORS;
}

export function addDoctor(doctor: Omit<Doctor, 'id'>): Doctor {
  const doctors = getDoctors();
  const newDoctor: Doctor = {
    ...doctor,
    id: Date.now(),
  };
  doctors.push(newDoctor);
  localStorage.setItem(STORAGE_KEYS.DOCTORS, JSON.stringify(doctors));
  return newDoctor;
}

// Appointments
export function getAppointments(): Appointment[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.APPOINTMENTS);
  return raw ? JSON.parse(raw) : [];
}

export function bookAppointment(data: Omit<Appointment, 'id' | 'createdAt' | 'status'>): Appointment {
  const appointments = getAppointments();
  const now = new Date();
  const newAppt: Appointment = {
    ...data,
    id: Date.now(),
    status: 'pending',
    createdAt: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
  };
  appointments.unshift(newAppt);
  localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify(appointments));
  return newAppt;
}

export function updateAppointmentStatus(id: number, status: Appointment['status'], adminNote?: string) {
  const appointments = getAppointments().map((a) => {
    if (a.id === id) {
      return { ...a, status, adminNote: adminNote !== undefined ? adminNote : a.adminNote };
    }
    return a;
  });
  localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify(appointments));
}

// Messages
export function getMessages(doctorId?: number): Message[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.MESSAGES);
  const messages: Message[] = raw ? JSON.parse(raw) : [];
  if (doctorId) {
    return messages.filter((m) => m.doctorId === doctorId);
  }
  return messages;
}

export function sendMessage(doctorId: number, sender: 'user' | 'doctor', message: string): Message {
  const messages = getMessages();
  const now = new Date();
  const newMsg: Message = {
    id: Date.now(),
    userId: 1,
    doctorId,
    sender,
    message,
    timestamp: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
  };
  messages.push(newMsg);
  localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(messages));
  return newMsg;
}

// Guardian Alerts
export function getGuardianAlerts(): GuardianAlert[] {
  initializeStorage();
  const raw = localStorage.getItem(STORAGE_KEYS.ALERTS);
  return raw ? JSON.parse(raw) : [];
}

export function logGuardianAlert(alert: Omit<GuardianAlert, 'id' | 'timestamp'>): GuardianAlert {
  const alerts = getGuardianAlerts();
  const now = new Date();
  const newAlert: GuardianAlert = {
    ...alert,
    id: Date.now(),
    timestamp: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
  };
  alerts.unshift(newAlert);
  localStorage.setItem(STORAGE_KEYS.ALERTS, JSON.stringify(alerts));
  return newAlert;
}
