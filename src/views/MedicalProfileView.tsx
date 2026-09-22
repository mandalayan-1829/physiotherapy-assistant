import { useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Heart, 
  Save, 
  Send, 
  ShieldAlert, 
  User as UserIcon,
  Activity,
  FileText,
  Clock,
  Target,
  Pill,
  Phone
} from 'lucide-react';
import { User } from '../types';
import { logGuardianAlert } from '../utils/storage';

interface MedicalProfileViewProps {
  user: User;
  onSaveProfile: (updated: User) => void;
}

export function MedicalProfileView({ user, onSaveProfile }: MedicalProfileViewProps) {
  const [formData, setFormData] = useState<User>({ ...user });
  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);
  const [testSosSuccess, setTestSosSuccess] = useState<boolean>(false);
  const [testSosError, setTestSosError] = useState<string | null>(null);

  // Accordion open states (one or multiple)
  const [openSection, setOpenSection] = useState<string | null>('basic');

  const toggleSection = (key: string) => {
    setOpenSection((prev) => (prev === key ? null : key));
  };

  const calculateBmi = (heightCm: number, weightKg: number): string => {
    if (!heightCm || !weightKg) return '—';
    const m = heightCm / 100;
    const bmi = weightKg / (m * m);
    return bmi.toFixed(1);
  };

  const currentBmi = calculateBmi(formData.heightCm, formData.weightKg);

  const handleChange = (field: keyof User, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveProfile(formData);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 4000);
  };

  // Opens a pre-filled WhatsApp message to the configured guardian.
  //
  // There is deliberately no fallback phone number: sending a pain report to a
  // hard-coded number would disclose the patient's health data to a stranger.
  // With no number on file the action is refused and the patient is told why.
  const handleTestSos = () => {
    const phone = formData.guardianWhatsapp || formData.emergencyContactPhone;
    const cleanPhone = phone.replace(/[^0-9]/g, '');

    if (!cleanPhone) {
      setTestSosError('No guardian number is saved yet. Add one above first.');
      setTimeout(() => setTestSosError(null), 6000);
      return;
    }

    const message = encodeURIComponent(
      `[PhysioAI Alert] ${formData.name} reported a pain spike (${formData.painIntensity}/10) or requested assistance during rehabilitation routine. Location: Home.`
    );
    const link = `https://api.whatsapp.com/send?phone=${cleanPhone}&text=${message}`;

    logGuardianAlert({
      userId: formData.id,
      alertType: 'emergency_help',
      message: `Patient opened an emergency WhatsApp message for ${formData.emergencyContactName || 'Emergency Contact'}. Delivery depends on the patient pressing send in WhatsApp.`,
      sentTo: phone,
    });

    setTestSosSuccess(true);
    setTimeout(() => setTestSosSuccess(false), 8000);
    window.open(link, '_blank');
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Clinical Health Records
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <UserIcon className="w-5 h-5 text-blue-600" />
              <span>Medical Intake & Biomechanical Profile</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Configure anatomical history, joint constraints, and pain thresholds to calibrate the computer vision pose checks.
            </p>
          </div>

          <div className="flex items-center gap-3 self-start sm:self-auto">
            {savedSuccess && (
              <span className="px-3 py-1.5 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-[#065F46] text-xs font-medium flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Profile Saved</span>
              </span>
            )}
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Save Changes</span>
            </button>
          </div>
        </div>
      </div>

      {/* Expandable Accordion Sections */}
      <form onSubmit={handleSave} className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* 1. Basic Details / Personal Information */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('basic')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <UserIcon className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Basic Details & Anthropometrics
                </h2>
                <span className="text-xs text-slate-500">
                  {formData.name} • {formData.age} yrs • {formData.gender} • BMI {currentBmi}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'basic' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'basic' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'basic' && (
            <div className="pb-6 pt-2 px-2 space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Full Legal Name</label>
                  <input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => handleChange('name', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Email Address</label>
                  <input
                    type="email"
                    required
                    value={formData.email}
                    onChange={(e) => handleChange('email', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Contact Phone</label>
                  <input
                    type="text"
                    value={formData.contactNumber || ''}
                    onChange={(e) => handleChange('contactNumber', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Blood Group</label>
                  <input
                    type="text"
                    value={formData.bloodGroup}
                    onChange={(e) => handleChange('bloodGroup', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Age</label>
                  <input
                    type="number"
                    value={formData.age}
                    onChange={(e) => handleChange('age', parseInt(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Gender</label>
                  <select
                    value={formData.gender}
                    onChange={(e) => handleChange('gender', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Date of Birth</label>
                  <input
                    type="date"
                    value={formData.dob || ''}
                    onChange={(e) => handleChange('dob', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Occupation</label>
                  <input
                    type="text"
                    value={formData.occupation || ''}
                    onChange={(e) => handleChange('occupation', e.target.value)}
                    placeholder="e.g. Software Engineer"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Height (cm)</label>
                  <input
                    type="number"
                    value={formData.heightCm}
                    onChange={(e) => handleChange('heightCm', parseFloat(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Weight (kg)</label>
                  <input
                    type="number"
                    value={formData.weightKg}
                    onChange={(e) => handleChange('weightKg', parseFloat(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="sm:col-span-2 flex items-center gap-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-slate-600">Calculated Body Mass Index:</span>
                  <span className="font-mono font-bold text-slate-900 text-sm">{currentBmi} kg/m²</span>
                  <span className="text-[11px] text-[#065F46] font-medium">Standard Adult Range</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 2. Medical History */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('history')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <FileText className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Medical History & Diagnosis
                </h2>
                <span className="text-xs text-slate-500">
                  {formData.currentProblem || 'No current diagnosis recorded'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'history' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'history' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'history' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="sm:col-span-2">
                  <label className="text-slate-700 font-semibold block mb-1">Current Problem / Primary Complaint</label>
                  <input
                    type="text"
                    value={formData.currentProblem}
                    onChange={(e) => handleChange('currentProblem', e.target.value)}
                    placeholder="e.g. Left knee patellofemoral stiffness"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Onset Date</label>
                  <input
                    type="date"
                    value={formData.problemStartDate || ''}
                    onChange={(e) => handleChange('problemStartDate', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Presumed Cause / Mechanism of Injury</label>
                  <input
                    type="text"
                    value={formData.problemCause || ''}
                    onChange={(e) => handleChange('problemCause', e.target.value)}
                    placeholder="e.g. Sedentary workstation posture and running volume"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Previous Orthopedic Injuries</label>
                  <input
                    type="text"
                    value={formData.previousInjuries || ''}
                    onChange={(e) => handleChange('previousInjuries', e.target.value)}
                    placeholder="e.g. Minor right ankle sprain (2024)"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Past Surgeries / Procedures</label>
                  <input
                    type="text"
                    value={formData.pastSurgeries || ''}
                    onChange={(e) => handleChange('pastSurgeries', e.target.value)}
                    placeholder="None or list dates"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Underlying Medical Conditions</label>
                  <input
                    type="text"
                    value={formData.medicalConditions || ''}
                    onChange={(e) => handleChange('medicalConditions', e.target.value)}
                    placeholder="e.g. Lumbar spine tightness, Hypertension"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Allergies (Medications / Contact)</label>
                  <input
                    type="text"
                    value={formData.allergies || ''}
                    onChange={(e) => handleChange('allergies', e.target.value)}
                    placeholder="e.g. Penicillin, Latex, None reported"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 3. Pain Details */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('pain')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Heart className="w-4 h-4 text-rose-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-rose-600 transition-colors">
                  Pain Details & Symptom Mapping
                </h2>
                <span className="text-xs text-slate-500">
                  Location: {formData.painLocation} • Intensity: {formData.painIntensity}/10 ({formData.painType})
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'pain' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'pain' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'pain' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Anatomical Pain Location(s)</label>
                  <input
                    type="text"
                    value={formData.painLocation}
                    onChange={(e) => handleChange('painLocation', e.target.value)}
                    placeholder="e.g. Left Knee, Lumbar Spine"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">
                    Pain Intensity Score (0-10): <strong className="text-slate-900 font-mono">{formData.painIntensity}</strong>
                  </label>
                  <div className="flex items-center gap-3 mt-1.5">
                    <input
                      type="range"
                      min={0}
                      max={10}
                      value={formData.painIntensity}
                      onChange={(e) => handleChange('painIntensity', parseInt(e.target.value))}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      formData.painIntensity > 6 
                        ? 'bg-rose-50 text-rose-700 border border-rose-200' 
                        : formData.painIntensity > 3
                        ? 'bg-amber-50 text-amber-800 border border-amber-200'
                        : 'bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]'
                    }`}>
                      {formData.painIntensity <= 3 ? 'Mild' : formData.painIntensity <= 6 ? 'Moderate' : 'Severe'}
                    </span>
                  </div>
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Sensation Type</label>
                  <select
                    value={formData.painType}
                    onChange={(e) => handleChange('painType', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="Dull ache">Dull ache</option>
                    <option value="Sharp / stabbing">Sharp / stabbing</option>
                    <option value="Burning sensation">Burning sensation</option>
                    <option value="Throbbing / pulsating">Throbbing / pulsating</option>
                    <option value="Stiffness / tightness">Stiffness / tightness</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Aggravating Triggers</label>
                  <input
                    type="text"
                    value={formData.painTriggers || ''}
                    onChange={(e) => handleChange('painTriggers', e.target.value)}
                    placeholder="e.g. Prolonged sitting (>4 hrs), stairs"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div className="sm:col-span-2">
                  <label className="text-slate-700 font-semibold block mb-1">Chronicity & Duration</label>
                  <input
                    type="text"
                    value={formData.painDuration || ''}
                    onChange={(e) => handleChange('painDuration', e.target.value)}
                    placeholder="e.g. Intermittent across past 3 weeks; worse in mornings"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 4. Lifestyle & Activity */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('lifestyle')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Clock className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Lifestyle & Physical Activity
                </h2>
                <span className="text-xs text-slate-500">
                  Sitting: {formData.dailySittingHours || 8} hrs/day • Activity: {formData.activityLevel || 'Moderate'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'lifestyle' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'lifestyle' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'lifestyle' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Daily Sitting Hours</label>
                  <input
                    type="number"
                    value={formData.dailySittingHours || 8}
                    onChange={(e) => handleChange('dailySittingHours', parseInt(e.target.value) || 0)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">General Activity Level</label>
                  <select
                    value={formData.activityLevel || 'Moderately active'}
                    onChange={(e) => handleChange('activityLevel', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  >
                    <option value="Sedentary (desk work, minimal walks)">Sedentary</option>
                    <option value="Lightly active (daily walks, stretching)">Lightly active</option>
                    <option value="Moderately active (gym or exercise 2-3x/week)">Moderately active</option>
                    <option value="Very active (daily training/sports)">Very active</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Exercise Habits & Routine</label>
                  <input
                    type="text"
                    value={formData.exerciseHabits || ''}
                    onChange={(e) => handleChange('exerciseHabits', e.target.value)}
                    placeholder="e.g. Light cardio, swimming, resistance bands"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 5. Medications & Goals */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('medications')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Pill className="w-4 h-4 text-blue-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">
                  Medications & Rehabilitation Goals
                </h2>
                <span className="text-xs text-slate-500">
                  Goals: {formData.rehabGoals || 'Restore mobility and strength'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'medications' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'medications' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'medications' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Current Medications & Supplements</label>
                  <textarea
                    value={formData.currentMedications}
                    onChange={(e) => handleChange('currentMedications', e.target.value)}
                    placeholder="e.g. Vitamin D3, Omega-3, NSAIDs when needed"
                    rows={2}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Primary Rehabilitation Goals</label>
                  <textarea
                    value={formData.rehabGoals}
                    onChange={(e) => handleChange('rehabGoals', e.target.value)}
                    placeholder="e.g. Restore pain-free squat depth, improve single-leg balance"
                    rows={2}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Safety Precautions</label>
                  <input
                    type="text"
                    value={formData.precautions || ''}
                    onChange={(e) => handleChange('precautions', e.target.value)}
                    placeholder="e.g. Warm up hamstrings thoroughly before squats"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Movement Restrictions</label>
                  <input
                    type="text"
                    value={formData.movementRestrictions}
                    onChange={(e) => handleChange('movementRestrictions', e.target.value)}
                    placeholder="e.g. Avoid rapid twisting under weight load"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 6. Personalized Exercise Plan */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('plan')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Target className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-600 transition-colors">
                  Personalized Exercise Plan & Clinical Assignment
                </h2>
                <span className="text-xs text-slate-500">
                  Attending Physician: {formData.doctorName || 'Dr. Aarav Patel'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'plan' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'plan' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'plan' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Attending Physician / Specialist</label>
                  <input
                    type="text"
                    value={formData.doctorName}
                    onChange={(e) => handleChange('doctorName', e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Clinical Intake Date</label>
                  <input
                    type="text"
                    disabled
                    value={formData.createdAt}
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-500 cursor-not-allowed"
                  />
                </div>

                <div className="sm:col-span-2">
                  <label className="text-slate-700 font-semibold block mb-1">Exercise Limitations / Contraindications</label>
                  <input
                    type="text"
                    value={formData.exerciseLimitations}
                    onChange={(e) => handleChange('exerciseLimitations', e.target.value)}
                    placeholder="e.g. Knee discomfort during excessive valgus angle"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                  <p className="text-[11px] text-slate-500 mt-1">
                    Exercises containing matching kinetic triggers will be flagged with a medical safety warning in the exercise directory.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* 7. Alerts & Emergency Contacts */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            type="button"
            onClick={() => toggleSection('alerts')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <ShieldAlert className="w-4 h-4 text-amber-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-amber-600 transition-colors">
                  Alerts & Emergency Contacts
                </h2>
                <span className="text-xs text-slate-500">
                  {formData.emergencyContactName} ({formData.emergencyContactPhone})
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSection === 'alerts' ? 'Collapse' : 'Edit'}
              </span>
              {openSection === 'alerts' ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSection === 'alerts' && (
            <div className="pb-6 pt-2 px-2 space-y-4 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Emergency Contact Name & Relation</label>
                  <input
                    type="text"
                    value={formData.emergencyContactName}
                    onChange={(e) => handleChange('emergencyContactName', e.target.value)}
                    placeholder="e.g. Sarah Doe (Spouse)"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Contact Phone Number</label>
                  <input
                    type="text"
                    value={formData.emergencyContactPhone}
                    onChange={(e) => handleChange('emergencyContactPhone', e.target.value)}
                    placeholder="+91 98765 43210"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="text-slate-700 font-semibold block mb-1">Guardian WhatsApp Dispatch</label>
                  <input
                    type="text"
                    value={formData.guardianWhatsapp}
                    onChange={(e) => handleChange('guardianWhatsapp', e.target.value)}
                    placeholder="e.g. 919876543210"
                    className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>

              {/* Test SOS dispatch banner */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 mt-2">
                <div>
                  <span className="font-semibold text-slate-900 block">Emergency Contact Check</span>
                  <span className="text-[11px] text-slate-500">
                    Opens a pre-filled WhatsApp message to the number above so you can confirm
                    your guardian can be reached. The application does not send anything itself -
                    the message goes out only when you press send in WhatsApp.
                  </span>
                </div>
                <button
                  type="button"
                  onClick={handleTestSos}
                  className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shrink-0 shadow-xs"
                >
                  <Send className="w-3.5 h-3.5 text-blue-600" />
                  <span>{testSosSuccess ? 'WhatsApp opened - press Send' : 'Open Test Message'}</span>
                </button>
              </div>

              {testSosError && (
                <p className="text-[11px] text-red-700 bg-red-50 border border-red-200 rounded px-2.5 py-1.5 mt-2">
                  {testSosError}
                </p>
              )}
            </div>
          )}
        </div>

      </form>

      {/* Save bar */}
      <div className="pt-2 flex justify-end">
        <button
          type="button"
          onClick={handleSave}
          className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
        >
          <Save className="w-4 h-4" />
          <span>Save Medical Profile</span>
        </button>
      </div>

    </div>
  );
}
