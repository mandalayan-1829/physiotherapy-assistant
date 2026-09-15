/**
 * Reporting helpers.
 *
 * IMPORTANT: this module no longer owns application data. Accounts, medical
 * profiles, exercise sessions, diet records, notes, appointments and messages
 * are all persisted by the FastAPI backend and loaded through `src/services`.
 *
 * What remains here are pure aggregation helpers used by the existing views
 * (they receive sessions as arguments) plus the monthly-report draft store and
 * the client-side guardian alert log, both of which are scheduled to move to
 * the backend with the reports and safety features.
 */

import {
  Appointment,
  DailyHistoryEntry,
  Doctor,
  GuardianAlert,
  MonthlyReport,
  Session,
  User,
} from '../types';

const STORAGE_KEYS = {
  MONTHLY_REPORTS: 'physio_monthly_report_drafts',
  ALERTS: 'physio_guardian_alert_log',
};

export function getTodayDateString(): string {
  const now = new Date();
  return now.toISOString().split('T')[0];
}

// --- Exercise session aggregation (pure) ------------------------------------

/** Sessions recorded today. Historical sessions are never removed. */
export function getTodaySessions(allSessions: Session[]): Session[] {
  const today = getTodayDateString();
  return allSessions.filter((session) => session.date.startsWith(today));
}

/** Group sessions into per-day historical entries, newest first. */
export function getHistoricalDailyEntries(allSessions: Session[]): DailyHistoryEntry[] {
  const groups: { [dateStr: string]: Session[] } = {};

  allSessions.forEach((session) => {
    const dateKey = session.date.split(' ')[0] || session.date;
    if (!groups[dateKey]) {
      groups[dateKey] = [];
    }
    groups[dateKey].push(session);
  });

  const sortedDates = Object.keys(groups).sort((a, b) => b.localeCompare(a));

  return sortedDates.map((dateStr) => {
    const daySessions = groups[dateStr];
    const totalReps = daySessions.reduce((acc, session) => acc + session.reps, 0);
    const avgScore = Math.round(
      daySessions.reduce((acc, session) => acc + session.formAccuracy, 0) / daySessions.length,
    );
    const totalDurationSec = daySessions.reduce((acc, session) => acc + session.durationSec, 0);
    const correctReps = Math.round(totalReps * (avgScore / 100));

    let displayDate = dateStr;
    const parts = dateStr.split('-');
    if (parts.length === 3) {
      const date = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10));
      if (!Number.isNaN(date.getTime())) {
        displayDate = date.toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric',
        });
      }
    }

    return {
      date: dateStr,
      displayDate,
      sessionsCount: daySessions.length,
      totalReps,
      correctReps,
      incorrectReps: totalReps - correctReps,
      avgFormScore: avgScore,
      totalDurationSec,
      exercises: Array.from(new Set(daySessions.map((session) => session.exerciseLabel))),
      warningsCount: avgScore < 90 ? 1 : 0,
      safetyEventsCount: 0,
      sessions: daySessions,
    };
  });
}

// --- Monthly report drafts --------------------------------------------------
// TODO(reports): move report generation and storage to the backend
// (POST /reports/generate already exists). Until then these drafts live in
// browser storage and are derived from server-provided sessions.

const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

export function getMonthlyReports(): MonthlyReport[] {
  if (typeof window === 'undefined') return [];
  const raw = window.localStorage.getItem(STORAGE_KEYS.MONTHLY_REPORTS);
  return raw ? (JSON.parse(raw) as MonthlyReport[]) : [];
}

export function saveMonthlyReport(report: MonthlyReport): MonthlyReport {
  const reports = getMonthlyReports();
  const index = reports.findIndex((r) => r.id === report.id || r.monthKey === report.monthKey);
  if (index >= 0) {
    reports[index] = report;
  } else {
    reports.unshift(report);
  }
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEYS.MONTHLY_REPORTS, JSON.stringify(reports));
  }
  return report;
}

export function generateMonthlyReport(
  monthKey: string,
  user: User,
  sessions: Session[],
  appointments: Appointment[],
  doctors: Doctor[],
): MonthlyReport {
  const monthSessions = sessions.filter((session) => session.date.startsWith(monthKey));
  const totalSessions = monthSessions.length;
  const totalReps = monthSessions.reduce((acc, session) => acc + session.reps, 0);
  const avgAccuracy =
    totalSessions > 0
      ? Math.round(monthSessions.reduce((acc, session) => acc + session.formAccuracy, 0) / totalSessions)
      : 0;

  const exerciseMap: { [label: string]: { sessions: number; reps: number; totalAcc: number } } = {};
  monthSessions.forEach((session) => {
    if (!exerciseMap[session.exerciseLabel]) {
      exerciseMap[session.exerciseLabel] = { sessions: 0, reps: 0, totalAcc: 0 };
    }
    exerciseMap[session.exerciseLabel].sessions += 1;
    exerciseMap[session.exerciseLabel].reps += session.reps;
    exerciseMap[session.exerciseLabel].totalAcc += session.formAccuracy;
  });

  const exerciseBreakdown = Object.keys(exerciseMap).map((label) => ({
    exerciseLabel: label,
    sessions: exerciseMap[label].sessions,
    reps: exerciseMap[label].reps,
    avgAccuracy: Math.round(exerciseMap[label].totalAcc / exerciseMap[label].sessions),
  }));

  const [yearStr, monthStr] = monthKey.split('-');
  const monthName = `${MONTH_NAMES[parseInt(monthStr, 10) - 1] ?? 'Selected Month'} ${yearStr}`;

  const upcomingAppointment = appointments.find(
    (appointment) =>
      (appointment.status === 'confirmed' ||
        appointment.status === 'scheduled' ||
        appointment.status === 'ready' ||
        appointment.status === 'approved') &&
      appointment.date >= getTodayDateString(),
  );

  const assignedDoc = doctors.find((doctor) => doctor.name === user.doctorName) || doctors[0];
  const now = new Date();

  return saveMonthlyReport({
    id: `report-${monthKey}-${user.id}`,
    monthKey,
    monthName,
    generatedDate: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
    patientId: user.id,
    patientName: user.name,
    patientEmail: user.email,
    assignedDoctorName: assignedDoc?.name,
    assignedDoctorEmail: assignedDoc?.email,
    hasUpcomingCheckup: !!upcomingAppointment,
    totalSessions,
    totalReps,
    completedExercises: exerciseBreakdown.length,
    missedSessions: 0,
    avgAccuracy,
    avgScore: avgAccuracy,
    adherencePercent: totalSessions >= 10 ? 94 : totalSessions >= 5 ? 85 : totalSessions > 0 ? 70 : 0,
    exerciseBreakdown,
    progressTrend:
      totalSessions > 0
        ? 'Computed from recorded rehabilitation sessions.'
        : 'No sessions recorded for this month.',
    safetyEventsCount: 0,
    warningsCount: 0,
    emailStatus: 'Draft',
    recipients: [user.email],
  });
}

/** NOTE: no email is actually dispatched yet; this only records the intent. */
export function sendMonthlyReportEmail(reportId: string): MonthlyReport | null {
  const report = getMonthlyReports().find((r) => r.id === reportId);
  if (!report) return null;

  const now = new Date();
  const recipients = [report.patientEmail];
  if (report.hasUpcomingCheckup && report.assignedDoctorEmail) {
    if (!recipients.includes(report.assignedDoctorEmail)) {
      recipients.push(report.assignedDoctorEmail);
    }
  }

  return saveMonthlyReport({
    ...report,
    emailStatus: 'Pending',
    recipients,
    emailSentDate: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
  });
}

// --- Guardian alert log -----------------------------------------------------
// TODO(safety): alerts are also persisted server-side via POST /alerts; the
// patient safety console will read from GET /alerts once it is built.

export function getGuardianAlerts(): GuardianAlert[] {
  if (typeof window === 'undefined') return [];
  const raw = window.localStorage.getItem(STORAGE_KEYS.ALERTS);
  return raw ? (JSON.parse(raw) as GuardianAlert[]) : [];
}

export function logGuardianAlert(
  alert: Omit<GuardianAlert, 'id' | 'timestamp'>,
): GuardianAlert {
  const alerts = getGuardianAlerts();
  const now = new Date();
  const newAlert: GuardianAlert = {
    ...alert,
    id: Date.now(),
    timestamp: `${now.toISOString().split('T')[0]} ${now.toTimeString().slice(0, 5)}`,
  };
  alerts.unshift(newAlert);
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEYS.ALERTS, JSON.stringify(alerts));
  }
  return newAlert;
}
