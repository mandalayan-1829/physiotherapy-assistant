import { useCallback, useEffect, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Heart,
  KeyRound,
  Lock,
  Mail,
  RefreshCw,
  ShieldCheck,
  Stethoscope,
  Timer,
  User as UserIcon,
} from 'lucide-react';
import type { UserRole } from '../types';
import { ApiError } from '../services/api';
import {
  login,
  register,
  requestPasswordReset,
  resetPassword,
  verifyResetCode,
  type AuthSession,
} from '../services/auth';

interface PortalAuthProps {
  onAuthenticated: (session: AuthSession) => void;
}

type AuthMode = 'login' | 'signup' | 'reset';
type ResetStep = 'request' | 'verify' | 'newPassword';

/** Client-side mirror of the backend code lifetime, used only for the label. */
const CODE_TTL_SECONDS = 10 * 60;
const RESEND_COOLDOWN_SECONDS = 60;

function formatDuration(totalSeconds: number): string {
  const safe = Math.max(0, totalSeconds);
  const minutes = Math.floor(safe / 60);
  const seconds = safe % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export function PortalAuth({ onAuthenticated }: PortalAuthProps) {
  // Domain selection: null = selector screen, 'patient' or 'doctor' = credential form.
  const [selectedDomain, setSelectedDomain] = useState<UserRole | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>('login');
  const [resetStep, setResetStep] = useState<ResetStep>('request');

  // Login / registration fields
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [licenseNumber, setLicenseNumber] = useState('');
  const [specialty, setSpecialty] = useState('Orthopedic Rehabilitation');

  // Password-reset fields (never persisted to browser storage)
  const [resetEmail, setResetEmail] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [resetToken, setResetToken] = useState<string | null>(null);

  // Server-driven attempt state
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [showForgotPassword, setShowForgotPassword] = useState(false);

  // Presentational countdowns (the real expiry/cooldown is enforced by the API)
  const [codeSecondsLeft, setCodeSecondsLeft] = useState(0);
  const [resendCooldown, setResendCooldown] = useState(0);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [noticeMessage, setNoticeMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isPatientDomain = selectedDomain === 'patient';

  const resetMessages = useCallback(() => {
    setErrorMessage(null);
    setNoticeMessage(null);
  }, []);

  // Expiry countdown for the verification-code screen.
  useEffect(() => {
    if (resetStep !== 'verify' || codeSecondsLeft <= 0) return;
    const timer = setInterval(() => setCodeSecondsLeft((value) => Math.max(0, value - 1)), 1000);
    return () => clearInterval(timer);
  }, [resetStep, codeSecondsLeft]);

  // Resend cooldown ticker.
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setInterval(() => setResendCooldown((value) => Math.max(0, value - 1)), 1000);
    return () => clearInterval(timer);
  }, [resendCooldown]);

  const selectDomain = (role: UserRole) => {
    setSelectedDomain(role);
    setAuthMode('login');
    setResetStep('request');
    setEmail('');
    setPassword('');
    setFailedAttempts(0);
    setShowForgotPassword(false);
    resetMessages();
  };

  const enterResetFlow = (prefilledEmail: string) => {
    setAuthMode('reset');
    setResetStep('request');
    setResetEmail(prefilledEmail);
    setVerificationCode('');
    setNewPassword('');
    setConfirmPassword('');
    setResetToken(null);
    setCodeSecondsLeft(0);
    setResendCooldown(0);
    resetMessages();
  };

  const backToLogin = (notice?: string) => {
    setAuthMode('login');
    setResetStep('request');
    setVerificationCode('');
    setNewPassword('');
    setConfirmPassword('');
    setResetToken(null);
    setCodeSecondsLeft(0);
    setResendCooldown(0);
    setErrorMessage(null);
    setNoticeMessage(notice ?? null);
    if (notice) setPassword('');
  };

  // --- Sign in / register --------------------------------------------------

  const handleLogin = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedDomain) return;
    resetMessages();
    setIsSubmitting(true);
    try {
      const session = await login(email, password, selectedDomain);
      setFailedAttempts(0);
      setShowForgotPassword(false);
      onAuthenticated(session);
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
        // The backend decides when to surface "Forgot password?"; a frontend
        // counter alone is never trusted.
        setFailedAttempts(error.failedAttempts ?? failedAttempts);
        if (error.showForgotPassword) setShowForgotPassword(true);
      } else {
        setErrorMessage('Unexpected error while contacting the server. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRegister = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedDomain) return;
    resetMessages();
    setIsSubmitting(true);
    try {
      const session = await register({
        fullName,
        email,
        password,
        role: selectedDomain,
        specialization: selectedDomain === 'doctor' ? specialty : undefined,
        licenseNumber: selectedDomain === 'doctor' ? licenseNumber : undefined,
      });
      onAuthenticated(session);
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError ? error.message : 'Could not create the account. Please try again.',
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  // --- Password reset ------------------------------------------------------

  const handleRequestReset = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedDomain) return;
    resetMessages();
    setIsSubmitting(true);
    try {
      const result = await requestPasswordReset(resetEmail.trim(), selectedDomain);
      setResetStep('verify');
      setVerificationCode('');
      setCodeSecondsLeft(CODE_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
      // The message is generic by design: it never confirms the account exists.
      setNoticeMessage(result.message);
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
        if (error.retryAfterSeconds) setResendCooldown(error.retryAfterSeconds);
      } else {
        setErrorMessage('Unexpected error while contacting the server. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResendCode = async () => {
    if (!selectedDomain || resendCooldown > 0) return;
    resetMessages();
    setIsSubmitting(true);
    try {
      const result = await requestPasswordReset(resetEmail.trim(), selectedDomain);
      // A new code is generated server-side and the previous one is invalidated.
      setVerificationCode('');
      setCodeSecondsLeft(CODE_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
      setNoticeMessage(`A new code has been sent. ${result.message}`);
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
        if (error.retryAfterSeconds) setResendCooldown(error.retryAfterSeconds);
      } else {
        setErrorMessage('Could not resend the code. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleVerifyCode = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedDomain) return;
    resetMessages();
    setIsSubmitting(true);
    try {
      const result = await verifyResetCode(resetEmail.trim(), verificationCode.trim(), selectedDomain);
      setResetToken(result.reset_token);
      setResetStep('newPassword');
      setNewPassword('');
      setConfirmPassword('');
    } catch (error) {
      if (error instanceof ApiError) {
        setErrorMessage(error.message);
        if (error.restartRequired) {
          setResetStep('request');
          setVerificationCode('');
          setCodeSecondsLeft(0);
        }
      } else {
        setErrorMessage('Unexpected error while contacting the server. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetPassword = async (event: FormEvent) => {
    event.preventDefault();
    if (!resetToken) return;
    resetMessages();

    // Quick client feedback; the backend performs the authoritative checks.
    if (newPassword !== confirmPassword) {
      setErrorMessage('Passwords do not match.');
      return;
    }
    if (newPassword.length < 8) {
      setErrorMessage('Password must be at least 8 characters long.');
      return;
    }

    setIsSubmitting(true);
    try {
      await resetPassword(resetToken, newPassword, confirmPassword);
      setFailedAttempts(0);
      setShowForgotPassword(false);
      backToLogin('Your password has been updated. Please sign in with your new password.');
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError
          ? error.message
          : 'Could not reset the password. Please start again.',
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const submitHandler =
    authMode === 'login'
      ? handleLogin
      : authMode === 'signup'
      ? handleRegister
      : resetStep === 'request'
      ? handleRequestReset
      : resetStep === 'verify'
      ? handleVerifyCode
      : handleResetPassword;

  const renderField = (
    label: string,
    icon: ReactNode,
    input: ReactNode,
    hint?: string,
  ) => (
    <div>
      <label className="block text-xs font-semibold text-slate-700 mb-1">{label}</label>
      <div className="relative">
        <span className="absolute left-3 top-2.5 text-slate-400">{icon}</span>
        {input}
      </div>
      {hint && <p className="text-[11px] text-slate-500 mt-1">{hint}</p>}
    </div>
  );

  const inputClass =
    'w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white';

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-between text-slate-800">

      {/* Top Clinical Header */}
      <header className="bg-white border-b border-slate-200 px-4 sm:px-8 py-3.5 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-xs">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold text-slate-900 tracking-tight">PhysioAI Clinical Suite</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                Healthcare Platform
              </span>
            </div>
            <p className="text-xs text-slate-500">Kinematic Computer Vision & Physiotherapy Management</p>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>Server-verified credentials</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-8">
        <div className="w-full max-w-4xl">

          {/* STEP 1: PORTAL SELECTION (EXACTLY TWO DOMAINS) */}
          {!selectedDomain ? (
            <div className="space-y-6">
              <div className="text-center max-w-xl mx-auto">
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200 inline-block mb-2">
                  Portal Authentication
                </span>
                <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                  Select Your Clinical Domain
                </h1>
                <p className="text-sm text-slate-600 mt-2">
                  Please choose your dedicated portal to access your personalized rehabilitation plan or clinical patient management tools.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">

                {/* DOMAIN 1: PATIENT PORTAL */}
                <div className="bg-white rounded-xl border-2 border-slate-200 hover:border-blue-500 transition-all shadow-xs p-6 flex flex-col justify-between relative group hover:shadow-md">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-12 h-12 rounded-lg bg-[#F0F7FF] border border-blue-100 flex items-center justify-center text-blue-600 group-hover:scale-105 transition-transform">
                        <Heart className="w-6 h-6" />
                      </div>
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#F0F7FF] text-blue-700 border border-blue-200">
                        Patient Domain
                      </span>
                    </div>

                    <h2 className="text-xl font-bold text-slate-900 tracking-tight">Patient Portal</h2>
                    <p className="text-xs font-medium text-slate-500 mt-0.5">Rehabilitation & Recovery</p>

                    <p className="text-xs sm:text-sm text-slate-600 mt-3 leading-relaxed">
                      For patients undergoing prescribed physical therapy, real-time exercise form correction, daily rehabilitation tracking, and direct consultation with your physiotherapist.
                    </p>

                    <ul className="mt-4 space-y-2 text-xs text-slate-600 border-t border-slate-100 pt-3">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>AI camera-guided exercise form & repetition counter</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Daily recovery progress & reset tracking</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Monthly reports & Telehealth reviews</span>
                      </li>
                    </ul>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-100">
                    <button
                      onClick={() => selectDomain('patient')}
                      className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
                    >
                      <span>Continue to Patient Portal</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* DOMAIN 2: DOCTOR / PHYSIOTHERAPIST PORTAL */}
                <div className="bg-white rounded-xl border-2 border-slate-200 hover:border-emerald-500 transition-all shadow-xs p-6 flex flex-col justify-between relative group hover:shadow-md">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-12 h-12 rounded-lg bg-[#ECFDF3] border border-emerald-100 flex items-center justify-center text-emerald-600 group-hover:scale-105 transition-transform">
                        <Stethoscope className="w-6 h-6" />
                      </div>
                      <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                        Clinical Domain
                      </span>
                    </div>

                    <h2 className="text-xl font-bold text-slate-900 tracking-tight">Doctor & Physiotherapist Portal</h2>
                    <p className="text-xs font-medium text-slate-500 mt-0.5">Clinical Oversight & Consultations</p>

                    <p className="text-xs sm:text-sm text-slate-600 mt-3 leading-relaxed">
                      For licensed physical therapists, orthopedic doctors and clinical specialists to manage patient protocols, review kinematic reports and conduct Telehealth sessions.
                    </p>

                    <ul className="mt-4 space-y-2 text-xs text-slate-600 border-t border-slate-100 pt-3">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Comprehensive patient roster & medical history audits</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Telehealth consultation management & appointment approvals</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Monthly report delivery & safety alert review</span>
                      </li>
                    </ul>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-100">
                    <button
                      onClick={() => selectDomain('doctor')}
                      className="w-full py-2.5 px-4 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
                    >
                      <span>Continue to Clinical Portal</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* STEP 2: CREDENTIALS / PASSWORD RESET */
            <div className="w-full max-w-md mx-auto bg-white rounded-xl border border-slate-200 shadow-xs p-6 sm:p-8">

              <button
                type="button"
                onClick={() => {
                  setSelectedDomain(null);
                  resetMessages();
                  setPassword('');
                  setAuthMode('login');
                  setResetStep('request');
                }}
                className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 mb-4 cursor-pointer"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Back to domain selection</span>
              </button>

              {/* Portal Header */}
              <div className="text-center mb-6">
                <div
                  className="inline-flex p-3 rounded-xl mb-3 border shadow-xs"
                  style={{
                    backgroundColor: isPatientDomain ? '#F0F7FF' : '#ECFDF3',
                    borderColor: isPatientDomain ? '#BFDBFE' : '#A7F3D0',
                    color: isPatientDomain ? '#2563EB' : '#059669',
                  }}
                >
                  {authMode === 'reset' ? (
                    <KeyRound className="w-6 h-6" />
                  ) : isPatientDomain ? (
                    <Heart className="w-6 h-6" />
                  ) : (
                    <Stethoscope className="w-6 h-6" />
                  )}
                </div>

                <div className="flex items-center justify-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                    isPatientDomain
                      ? 'bg-[#F0F7FF] text-blue-700 border-blue-200'
                      : 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]'
                  }`}>
                    {isPatientDomain ? 'Patient Domain' : 'Clinical Specialist Domain'}
                  </span>
                </div>

                <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1.5">
                  {authMode === 'reset'
                    ? resetStep === 'request'
                      ? 'Reset Password'
                      : resetStep === 'verify'
                      ? 'Enter Verification Code'
                      : 'Set New Password'
                    : isPatientDomain
                    ? 'Patient Portal Sign In'
                    : 'Physiotherapist Portal Sign In'}
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  {authMode === 'login' && 'Enter your verified account credentials below.'}
                  {authMode === 'signup' && 'Register your credentials to begin.'}
                  {authMode === 'reset' && resetStep === 'request' &&
                    'Enter your registered email address to receive a verification code.'}
                  {authMode === 'reset' && resetStep === 'verify' &&
                    'Enter the code that was sent to your email address.'}
                  {authMode === 'reset' && resetStep === 'newPassword' &&
                    'Choose a new password for your account.'}
                </p>
              </div>

              {/* Mode Selector Tabs */}
              <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg mb-5 text-xs font-semibold">
                {(['login', 'signup', 'reset'] as AuthMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => {
                      if (mode === 'reset') {
                        enterResetFlow(email);
                      } else {
                        setAuthMode(mode);
                        setResetStep('request');
                        resetMessages();
                      }
                    }}
                    className={`py-1.5 rounded-md transition-colors cursor-pointer ${
                      authMode === mode ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {mode === 'login' ? 'Sign In' : mode === 'signup' ? 'Register' : 'Reset'}
                  </button>
                ))}
              </div>

              {errorMessage && (
                <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 text-red-600 mt-0.5" />
                  <span>{errorMessage}</span>
                </div>
              )}

              {noticeMessage && (
                <div className="mb-4 p-3 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-xs text-[#065F46] flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600 mt-0.5" />
                  <span>{noticeMessage}</span>
                </div>
              )}

              <form onSubmit={submitHandler} className="space-y-4">

                {/* ---------- SIGN IN ---------- */}
                {authMode === 'login' && (
                  <>
                    {renderField(
                      isPatientDomain ? 'Patient Email Address' : 'Doctor / Clinic Email Address',
                      <Mail className="w-4 h-4" />,
                      <input
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        autoComplete="username"
                        className={inputClass}
                      />,
                    )}

                    {renderField(
                      'Password',
                      <Lock className="w-4 h-4" />,
                      <input
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••••••"
                        autoComplete="current-password"
                        className={inputClass}
                      />,
                    )}

                    {/* Offered by the backend after repeated failures. */}
                    {showForgotPassword && (
                      <div className="p-3 rounded-lg bg-[#FFF8E6] border border-amber-200 text-xs text-amber-900 flex flex-col gap-2">
                        <div className="flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
                          <span>
                            {failedAttempts > 0
                              ? `${failedAttempts} unsuccessful sign-in attempts. You can reset your password.`
                              : 'You can reset your password.'}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => enterResetFlow(email)}
                          className="self-start inline-flex items-center gap-1.5 text-xs font-semibold text-blue-700 hover:text-blue-900 cursor-pointer"
                        >
                          <KeyRound className="w-3.5 h-3.5" />
                          <span>Forgot Password?</span>
                        </button>
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors shadow-xs ${
                        isSubmitting
                          ? 'bg-slate-400 cursor-not-allowed'
                          : isPatientDomain
                          ? 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                          : 'bg-slate-900 hover:bg-slate-800 cursor-pointer'
                      }`}
                    >
                      <span>
                        {isSubmitting
                          ? 'Verifying…'
                          : `Sign In to ${isPatientDomain ? 'Patient' : 'Clinical'} Portal`}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}

                {/* ---------- REGISTER ---------- */}
                {authMode === 'signup' && (
                  <>
                    {renderField(
                      'Full Legal Name',
                      <UserIcon className="w-4 h-4" />,
                      <input
                        type="text"
                        required
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        placeholder={isPatientDomain ? 'e.g. John Doe' : 'e.g. Dr. Aarav Patel'}
                        className={inputClass}
                      />,
                    )}

                    {selectedDomain === 'doctor' && (
                      <>
                        <div>
                          <label className="block text-xs font-semibold text-slate-700 mb-1">
                            Medical License / Registration ID
                          </label>
                          <input
                            type="text"
                            value={licenseNumber}
                            onChange={(e) => setLicenseNumber(e.target.value)}
                            placeholder="e.g. MED-PHY-9824-IN"
                            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-semibold text-slate-700 mb-1">
                            Clinical Specialization
                          </label>
                          <select
                            required
                            value={specialty}
                            onChange={(e) => setSpecialty(e.target.value)}
                            className="w-full px-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
                          >
                            <option value="Orthopedic Rehabilitation">Orthopedic Rehabilitation</option>
                            <option value="Spine & Neurological Physiotherapy">Spine & Neurological Physiotherapy</option>
                            <option value="Sports Injury Recovery">Sports Injury Recovery</option>
                            <option value="Geriatric & Post-Op Mobility">Geriatric & Post-Op Mobility</option>
                          </select>
                        </div>
                      </>
                    )}

                    {renderField(
                      isPatientDomain ? 'Patient Email Address' : 'Doctor / Clinic Email Address',
                      <Mail className="w-4 h-4" />,
                      <input
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        autoComplete="username"
                        className={inputClass}
                      />,
                    )}

                    {renderField(
                      'Password',
                      <Lock className="w-4 h-4" />,
                      <input
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••••••"
                        autoComplete="new-password"
                        className={inputClass}
                      />,
                      'At least 8 characters, including a letter and a number.',
                    )}

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors shadow-xs ${
                        isSubmitting
                          ? 'bg-slate-400 cursor-not-allowed'
                          : isPatientDomain
                          ? 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                          : 'bg-slate-900 hover:bg-slate-800 cursor-pointer'
                      }`}
                    >
                      <span>
                        {isSubmitting
                          ? 'Creating account…'
                          : `Create ${isPatientDomain ? 'Patient' : 'Doctor'} Account`}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}

                {/* ---------- RESET: request a code ---------- */}
                {authMode === 'reset' && resetStep === 'request' && (
                  <>
                    {renderField(
                      isPatientDomain ? 'Registered Patient Email' : 'Registered Clinician Email',
                      <Mail className="w-4 h-4" />,
                      <input
                        type="email"
                        required
                        value={resetEmail}
                        onChange={(e) => setResetEmail(e.target.value)}
                        placeholder="you@example.com"
                        autoComplete="username"
                        className={inputClass}
                      />,
                      'A verification code will be sent to this address if it is registered.',
                    )}

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors shadow-xs ${
                        isSubmitting
                          ? 'bg-slate-400 cursor-not-allowed'
                          : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                      }`}
                    >
                      <span>{isSubmitting ? 'Sending…' : 'Send Verification Code'}</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>

                    <button
                      type="button"
                      onClick={() => backToLogin()}
                      className="w-full text-center text-xs text-slate-500 hover:text-slate-800 cursor-pointer"
                    >
                      Back to sign in
                    </button>
                  </>
                )}

                {/* ---------- RESET: enter the code ---------- */}
                {authMode === 'reset' && resetStep === 'verify' && (
                  <>
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">
                        Verification Code
                      </label>
                      <div className="relative">
                        <KeyRound className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                        <input
                          type="text"
                          required
                          inputMode="numeric"
                          autoComplete="one-time-code"
                          maxLength={6}
                          value={verificationCode}
                          onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, ''))}
                          placeholder="6-digit code"
                          className="w-full pl-9 pr-3 py-2 text-sm font-mono tracking-[0.35em] rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                        />
                      </div>
                    </div>

                    {codeSecondsLeft > 0 ? (
                      <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                        <Timer className="w-3.5 h-3.5 text-slate-400" />
                        <span>Code expires in {formatDuration(codeSecondsLeft)}</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 text-[11px] text-amber-700">
                        <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                        <span>This code may have expired. Request a new one.</span>
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={isSubmitting || verificationCode.length < 4}
                      className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors shadow-xs ${
                        isSubmitting || verificationCode.length < 4
                          ? 'bg-slate-400 cursor-not-allowed'
                          : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                      }`}
                    >
                      <span>{isSubmitting ? 'Verifying…' : 'Verify Code'}</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>

                    <div className="flex items-center justify-between gap-2 pt-1">
                      <button
                        type="button"
                        onClick={handleResendCode}
                        disabled={isSubmitting || resendCooldown > 0}
                        className={`inline-flex items-center gap-1.5 text-xs font-medium ${
                          resendCooldown > 0
                            ? 'text-slate-400 cursor-not-allowed'
                            : 'text-blue-600 hover:text-blue-800 cursor-pointer'
                        }`}
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>
                          {resendCooldown > 0
                            ? `Resend Code (${formatDuration(resendCooldown)})`
                            : 'Resend Code'}
                        </span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          setResetStep('request');
                          setVerificationCode('');
                          resetMessages();
                        }}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 cursor-pointer"
                      >
                        <ChevronLeft className="w-3.5 h-3.5" />
                        <span>Back</span>
                      </button>
                    </div>
                  </>
                )}

                {/* ---------- RESET: choose a new password ---------- */}
                {authMode === 'reset' && resetStep === 'newPassword' && (
                  <>
                    {renderField(
                      'New Password',
                      <Lock className="w-4 h-4" />,
                      <input
                        type="password"
                        required
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="••••••••••••"
                        autoComplete="new-password"
                        className={inputClass}
                      />,
                      'At least 8 characters, including a letter and a number.',
                    )}

                    {renderField(
                      'Confirm New Password',
                      <Lock className="w-4 h-4" />,
                      <input
                        type="password"
                        required
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="••••••••••••"
                        autoComplete="new-password"
                        className={inputClass}
                      />,
                    )}

                    <div className="p-2.5 rounded-lg bg-[#FFF8E6] border border-amber-200 text-[11px] text-amber-900 flex items-start gap-2">
                      <AlertCircle className="w-3.5 h-3.5 shrink-0 text-amber-600 mt-0.5" />
                      <span>Changing your password signs you out of every other device.</span>
                    </div>

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors shadow-xs ${
                        isSubmitting
                          ? 'bg-slate-400 cursor-not-allowed'
                          : 'bg-emerald-600 hover:bg-emerald-700 cursor-pointer'
                      }`}
                    >
                      <span>{isSubmitting ? 'Updating…' : 'Update Password'}</span>
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </button>

                    <button
                      type="button"
                      onClick={() => backToLogin()}
                      className="w-full text-center text-xs text-slate-500 hover:text-slate-800 cursor-pointer"
                    >
                      Back to sign in
                    </button>
                  </>
                )}

              </form>

            </div>
          )}

        </div>
      </main>

      {/* Clinical Footer */}
      <footer className="bg-white border-t border-slate-200 px-4 py-3 text-center text-xs text-slate-500">
        <span>PhysioAI Medical Platform • Role-Based Healthcare Architecture • Patient & Clinical Domains</span>
      </footer>

    </div>
  );
}
