import { useState } from 'react';
import { 
  Activity, 
  ArrowRight, 
  CheckCircle2, 
  ChevronRight, 
  Heart, 
  Key, 
  Lock, 
  Mail, 
  ShieldCheck, 
  Stethoscope, 
  User as UserIcon, 
  Users 
} from 'lucide-react';
import { UserRole } from '../types';

interface PortalAuthProps {
  onLogin: (role: UserRole, doctorId?: number) => void;
}

export function PortalAuth({ onLogin }: PortalAuthProps) {
  // Domain selection: null = selector screen, 'patient' or 'doctor' = login/signup forms
  const [selectedDomain, setSelectedDomain] = useState<UserRole | null>(null);
  const [authMode, setAuthMode] = useState<'login' | 'signup' | 'forgot'>('login');
  
  // Form fields
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [licenseNumber, setLicenseNumber] = useState('');
  const [specialty, setSpecialty] = useState('Orthopedic Rehabilitation');
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  // Quick 1-click login presets
  const handleQuickPatientLogin = () => {
    onLogin('patient');
  };

  const handleQuickDoctorLogin = (doctorId: number = 1) => {
    onLogin('doctor', doctorId);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (authMode === 'forgot') {
      setFeedbackMessage(`Password reset link dispatched to ${email || 'your registered email'}.`);
      return;
    }
    if (selectedDomain === 'doctor') {
      onLogin('doctor', 1);
    } else {
      onLogin('patient');
    }
  };

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
            <span>HIPAA-Compliant Architecture</span>
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
                        <span>Automated monthly reports & Telehealth reviews</span>
                      </li>
                    </ul>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-100 space-y-2">
                    <button
                      onClick={() => {
                        setSelectedDomain('patient');
                        setAuthMode('login');
                        setEmail('mandalayan1829@gmail.com');
                      }}
                      className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
                    >
                      <span>Continue to Patient Portal</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                    <button
                      onClick={handleQuickPatientLogin}
                      className="w-full py-2 px-3 rounded-lg bg-[#F0F7FF] hover:bg-blue-100 text-blue-700 text-xs font-medium transition-colors cursor-pointer border border-blue-200 text-center"
                    >
                      1-Click Patient Demo Access (John Doe)
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
                      For licensed physical therapists, orthopedic doctors, and clinical specialists to manage patient protocols, review monthly kinematic reports, and conduct Telehealth sessions.
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
                        <span>Automated monthly report delivery & emergency SOS alerts</span>
                      </li>
                    </ul>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-100 space-y-2">
                    <button
                      onClick={() => {
                        setSelectedDomain('doctor');
                        setAuthMode('login');
                        setEmail('dr.aarav@physioai.health');
                      }}
                      className="w-full py-2.5 px-4 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs sm:text-sm flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
                    >
                      <span>Continue to Clinical Portal</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleQuickDoctorLogin(1)}
                      className="w-full py-2 px-3 rounded-lg bg-[#ECFDF3] hover:bg-emerald-100 text-[#065F46] text-xs font-medium transition-colors cursor-pointer border border-[#A7F3D0] text-center"
                    >
                      1-Click Specialist Demo Access (Dr. Aarav Patel)
                    </button>
                  </div>
                </div>

              </div>

              <div className="bg-slate-100/70 border border-slate-200 rounded-lg p-3 text-center text-xs text-slate-500">
                <span>Secure role separation: Patient data access is protected by HIPAA protocol rules and role-based domain boundaries.</span>
              </div>
            </div>
          ) : (

            /* STEP 2: DEDICATED DOMAIN AUTHENTICATION FORM */
            <div className="max-w-md mx-auto bg-white rounded-xl border border-slate-200 shadow-sm p-6 sm:p-8">
              
              {/* Back to Domain Selector */}
              <button
                onClick={() => {
                  setSelectedDomain(null);
                  setFeedbackMessage(null);
                }}
                className="text-xs text-slate-500 hover:text-slate-800 flex items-center gap-1.5 mb-5 cursor-pointer font-medium"
              >
                <span>← Back to Domain Selection</span>
              </button>

              {/* Portal Header */}
              <div className="text-center mb-6">
                <div className="inline-flex p-3 rounded-xl mb-3 border shadow-xs"
                  style={{
                    backgroundColor: selectedDomain === 'patient' ? '#F0F7FF' : '#ECFDF3',
                    borderColor: selectedDomain === 'patient' ? '#BFDBFE' : '#A7F3D0',
                    color: selectedDomain === 'patient' ? '#2563EB' : '#059669'
                  }}
                >
                  {selectedDomain === 'patient' ? <Heart className="w-6 h-6" /> : <Stethoscope className="w-6 h-6" />}
                </div>

                <div className="flex items-center justify-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                    selectedDomain === 'patient'
                      ? 'bg-[#F0F7FF] text-blue-700 border-blue-200'
                      : 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]'
                  }`}>
                    {selectedDomain === 'patient' ? 'Patient Domain' : 'Clinical Specialist Domain'}
                  </span>
                </div>

                <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1.5">
                  {selectedDomain === 'patient' ? 'Patient Portal Sign In' : 'Physiotherapist Portal Sign In'}
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  {authMode === 'login' && 'Enter your verified account credentials below.'}
                  {authMode === 'signup' && 'Register your new medical credentials to begin.'}
                  {authMode === 'forgot' && 'Reset your password via clinical email link.'}
                </p>
              </div>

              {/* Mode Selector Tabs */}
              <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg mb-5 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => { setAuthMode('login'); setFeedbackMessage(null); }}
                  className={`py-1.5 rounded-md transition-colors cursor-pointer ${
                    authMode === 'login' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  onClick={() => { setAuthMode('signup'); setFeedbackMessage(null); }}
                  className={`py-1.5 rounded-md transition-colors cursor-pointer ${
                    authMode === 'signup' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Register
                </button>
                <button
                  type="button"
                  onClick={() => { setAuthMode('forgot'); setFeedbackMessage(null); }}
                  className={`py-1.5 rounded-md transition-colors cursor-pointer ${
                    authMode === 'forgot' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Reset
                </button>
              </div>

              {feedbackMessage && (
                <div className="mb-4 p-3 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-xs text-[#065F46] flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600" />
                  <span>{feedbackMessage}</span>
                </div>
              )}

              {/* Authentication Form */}
              <form onSubmit={handleSubmit} className="space-y-4">
                {authMode === 'signup' && (
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Full Legal Name</label>
                    <div className="relative">
                      <UserIcon className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="text"
                        required
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        placeholder={selectedDomain === 'patient' ? 'e.g. John Doe' : 'e.g. Dr. Aarav Patel'}
                        className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                      />
                    </div>
                  </div>
                )}

                {authMode === 'signup' && selectedDomain === 'doctor' && (
                  <>
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">Medical License / Registration ID</label>
                      <input
                        type="text"
                        required
                        value={licenseNumber}
                        onChange={(e) => setLicenseNumber(e.target.value)}
                        placeholder="e.g. MED-PHY-9824-IN"
                        className="w-full px-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">Clinical Specialization</label>
                      <select
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

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    {selectedDomain === 'patient' ? 'Patient Email Address' : 'Doctor / Clinic Email Address'}
                  </label>
                  <div className="relative">
                    <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder={selectedDomain === 'patient' ? 'mandalayan1829@gmail.com' : 'dr.aarav@physioai.health'}
                      className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                    />
                  </div>
                </div>

                {authMode !== 'forgot' && (
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Password</label>
                    <div className="relative">
                      <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••••••"
                        className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                      />
                    </div>
                  </div>
                )}

                <button
                  type="submit"
                  className={`w-full py-2.5 px-4 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs ${
                    selectedDomain === 'patient'
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-slate-900 hover:bg-slate-800'
                  }`}
                >
                  {authMode === 'login' && <span>Sign In to {selectedDomain === 'patient' ? 'Patient' : 'Clinical'} Portal</span>}
                  {authMode === 'signup' && <span>Create {selectedDomain === 'patient' ? 'Patient' : 'Doctor'} Account</span>}
                  {authMode === 'forgot' && <span>Send Recovery Email</span>}
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </form>

              {/* One click fast demo trigger */}
              <div className="mt-5 pt-4 border-t border-slate-100 text-center">
                <button
                  type="button"
                  onClick={() => {
                    if (selectedDomain === 'patient') {
                      handleQuickPatientLogin();
                    } else {
                      handleQuickDoctorLogin(1);
                    }
                  }}
                  className="text-xs text-blue-600 hover:text-blue-800 font-medium underline underline-offset-4 cursor-pointer"
                >
                  Instant Access as Verified {selectedDomain === 'patient' ? 'Patient (John Doe)' : 'Specialist (Dr. Aarav Patel)'}
                </button>
              </div>

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
