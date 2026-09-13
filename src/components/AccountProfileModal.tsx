import { useState } from 'react';
import { 
  CheckCircle2, 
  Heart, 
  Lock, 
  LogOut, 
  Mail, 
  Phone, 
  Save, 
  Settings, 
  Shield, 
  Stethoscope, 
  User as UserIcon, 
  X 
} from 'lucide-react';
import { Doctor, User, UserRole } from '../types';

interface AccountProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  userRole: UserRole;
  user: User;
  activeDoctor?: Doctor;
  initialTab?: 'general' | 'medical' | 'settings';
  onSaveProfile: (updatedUser: User) => void;
  onSwitchPortal: () => void;
}

export function AccountProfileModal({
  isOpen,
  onClose,
  userRole,
  user,
  activeDoctor,
  initialTab = 'general',
  onSaveProfile,
  onSwitchPortal,
}: AccountProfileModalProps) {
  const [activeTab, setActiveTab] = useState<'general' | 'medical' | 'settings'>(initialTab);
  const [formData, setFormData] = useState<User>({ ...user });
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const isPatient = userRole === 'patient';
  const displayName = isPatient ? formData.name : (activeDoctor?.name || 'Dr. Aarav Patel');
  const displayRole = isPatient ? 'Patient • Rehabilitation Routine' : 'Physiotherapist • Attending Clinician';

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveProfile(formData);
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 1200);
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-2xl w-full overflow-hidden flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150">
        
        {/* Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-11 h-11 rounded-full flex items-center justify-center font-bold text-white text-base ${
              isPatient ? 'bg-blue-600' : 'bg-emerald-600'
            }`}>
              {displayName.charAt(0)}
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">{displayName}</h3>
              <span className="text-xs text-slate-500">{displayRole}</span>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-200 px-5 pt-2 bg-white gap-4 text-xs font-semibold">
          <button
            onClick={() => setActiveTab('general')}
            className={`pb-2.5 border-b-2 transition-colors cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'general'
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <UserIcon className="w-3.5 h-3.5" />
            <span>Profile Overview</span>
          </button>

          <button
            onClick={() => setActiveTab('medical')}
            className={`pb-2.5 border-b-2 transition-colors cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'medical'
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <Heart className="w-3.5 h-3.5" />
            <span>Clinical & Medical Info</span>
          </button>

          <button
            onClick={() => setActiveTab('settings')}
            className={`pb-2.5 border-b-2 transition-colors cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'settings'
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Account & Domain</span>
          </button>
        </div>

        {/* Tab Content */}
        <form onSubmit={handleSave} className="p-5 overflow-y-auto space-y-4 text-xs flex-1">
          
          {savedSuccess && (
            <div className="p-3 rounded-lg bg-[#ECFDF3] border border-[#A7F3D0] text-[#065F46] flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>Profile details saved successfully!</span>
            </div>
          )}

          {activeTab === 'general' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Full Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Email Address</label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Contact Phone</label>
                  <input
                    type="text"
                    value={formData.contactNumber || ''}
                    onChange={(e) => setFormData({ ...formData, contactNumber: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Blood Group</label>
                  <input
                    type="text"
                    value={formData.bloodGroup}
                    onChange={(e) => setFormData({ ...formData, bloodGroup: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Age</label>
                  <input
                    type="number"
                    value={formData.age}
                    onChange={(e) => setFormData({ ...formData, age: parseInt(e.target.value, 10) || 0 })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Gender</label>
                  <select
                    value={formData.gender}
                    onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 bg-white"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-100">
                <h4 className="font-bold text-slate-800 uppercase tracking-wider text-[11px] mb-2">
                  Emergency Contact / Guardian
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">Guardian Name</label>
                    <input
                      type="text"
                      value={formData.emergencyContactName}
                      onChange={(e) => setFormData({ ...formData, emergencyContactName: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">Guardian WhatsApp / Phone</label>
                    <input
                      type="text"
                      value={formData.guardianWhatsapp || formData.emergencyContactPhone}
                      onChange={(e) => setFormData({ ...formData, guardianWhatsapp: e.target.value, emergencyContactPhone: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'medical' && (
            <div className="space-y-3.5">
              <div>
                <label className="block font-semibold text-slate-700 mb-1">Chief Problem / Rehabilitation Goal</label>
                <input
                  type="text"
                  value={formData.currentProblem}
                  onChange={(e) => setFormData({ ...formData, currentProblem: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Pain Rating (0 - 10)</label>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={formData.painIntensity}
                    onChange={(e) => setFormData({ ...formData, painIntensity: parseInt(e.target.value, 10) || 0 })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block font-semibold text-slate-700 mb-1">Pain Type</label>
                  <input
                    type="text"
                    value={formData.painType}
                    onChange={(e) => setFormData({ ...formData, painType: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">Exercise Limitations</label>
                <textarea
                  rows={2}
                  value={formData.exerciseLimitations}
                  onChange={(e) => setFormData({ ...formData, exerciseLimitations: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block font-semibold text-slate-700 mb-1">Precautions & Clinical Advice</label>
                <textarea
                  rows={2}
                  value={formData.precautions || ''}
                  onChange={(e) => setFormData({ ...formData, precautions: e.target.value })}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="space-y-4">
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="font-bold text-slate-900 block">Active Portal Access</span>
                    <span className="text-slate-500 text-xs">
                      Currently operating as: <strong className="text-blue-700">{userRole.toUpperCase()}</strong>
                    </span>
                  </div>
                  <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
                    Active Session
                  </span>
                </div>

                <p className="text-slate-600 text-xs">
                  Need to change role? Switch instantly between the Patient Rehabilitation Portal and the Doctor / Physiotherapist Management Console.
                </p>

                <button
                  type="button"
                  onClick={() => {
                    onClose();
                    onSwitchPortal();
                  }}
                  className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-xs"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Switch Portal Domain</span>
                </button>
              </div>

              <div className="p-4 rounded-xl border border-slate-200 space-y-2 text-xs text-slate-600">
                <span className="font-bold text-slate-800 block">Biomechanical Privacy Notice</span>
                <p>
                  All video tracking calculations run client-side on your local device hardware using MediaPipe kinematics. Video streams are never recorded or stored on external servers.
                </p>
              </div>
            </div>
          )}

          {/* Action Bar */}
          <div className="pt-3 border-t border-slate-200 flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-slate-300 text-slate-700 font-medium hover:bg-slate-50 cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold flex items-center gap-1.5 cursor-pointer shadow-xs"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Save Changes</span>
            </button>
          </div>

        </form>
      </div>
    </div>
  );
}
