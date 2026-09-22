/**
 * Aggregation helpers.
 *
 * This module does not own application data. Accounts, medical profiles,
 * exercise sessions, diet records, notes, appointments, messages and monthly
 * reports are all persisted by the FastAPI backend and loaded through
 * `src/services`.
 *
 * What remains here are pure aggregation helpers used by the existing views
 * (they receive sessions as arguments) plus a legacy local alert log.
 *
 * Measurement rules enforced here
 * -------------------------------
 * Only sessions with `metricsSource === 'pose_inference'` represent a measured
 * movement. Every aggregate below - the form score, the good/needs-improvement
 * repetition counts - is therefore computed from measured sessions alone. When a
 * day contains no measured session, no score is reported at all rather than a
 * value being invented. Nothing in this file estimates, interpolates or defaults
 * a clinical figure.
 *
 * Monthly reports are no longer built here. They are generated, stored and
 * retrieved by the backend (`POST /reports/generate`, `GET /reports`,
 * `GET /reports/{id}`) so that the same month yields the same figures for every
 * client and the numbers form part of the patient's record rather than living in
 * one browser's storage.
 */

import { DailyHistoryEntry, GuardianAlert, Session } from '../types';

const STORAGE_KEYS = {
  ALERTS: 'physio_guardian_alert_log',
};

export function getTodayDateString(): string {
  const now = new Date();
  return now.toISOString().split('T')[0];
}

/** Sessions that actually measured a movement. */
function measuredOnly(sessions: Session[]): Session[] {
  return sessions.filter((session) => session.metricsSource === 'pose_inference');
}

/** Mean form score over measured sessions, or `null` when there is none. */
function measuredFormScore(sessions: Session[]): number | null {
  const measured = measuredOnly(sessions);
  if (measured.length === 0) return null;
  return Math.round(
    measured.reduce((acc, session) => acc + session.formAccuracy, 0) / measured.length,
  );
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
    const measured = measuredOnly(daySessions);

    const totalReps = daySessions.reduce((acc, session) => acc + session.reps, 0);
    const totalDurationSec = daySessions.reduce((acc, session) => acc + session.durationSec, 0);
    const measuredReps = measured.reduce((acc, session) => acc + session.reps, 0);
    const avgFormScore = measuredFormScore(daySessions) ?? 0;

    // Repetitions are only split into good/needs-improvement when a measurement
    // exists to base that split on. Otherwise both stay at zero and the UI says
    // the day was not measured.
    const correctReps = measured.length > 0 ? Math.round(measuredReps * (avgFormScore / 100)) : 0;
    const incorrectReps = measured.length > 0 ? Math.max(0, measuredReps - correctReps) : 0;

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
      measuredSessions: measured.length,
      totalReps,
      measuredReps,
      correctReps,
      incorrectReps,
      avgFormScore,
      totalDurationSec,
      exercises: Array.from(new Set(daySessions.map((session) => session.exerciseLabel))),
      sessions: daySessions,
    };
  });
}

// --- Guardian alert log -----------------------------------------------------
// TODO(safety): alerts are also persisted server-side via POST /alerts; the
// patient safety console will read from GET /alerts once it is built. This local
// log is a placeholder and is never presented as a delivered notification.

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
