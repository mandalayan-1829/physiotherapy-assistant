import { useState } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Bell, 
  Calendar, 
  CheckCircle2, 
  ChevronDown, 
  ChevronUp, 
  ClipboardList, 
  Clock, 
  FileSpreadsheet, 
  FileText, 
  Heart, 
  History, 
  LogOut, 
  Menu, 
  MessageSquare, 
  PieChart, 
  Play, 
  Settings, 
  Shield, 
  Stethoscope, 
  TrendingUp, 
  User as UserIcon, 
  UserCheck, 
  Users, 
  Utensils, 
  Video, 
  X 
} from 'lucide-react';
import { Doctor, User, UserRole } from '../types';

export type PatientNavTab = 
  | 'home' 
  | 'exercises' 
  | 'session' 
  | 'progress' 
  | 'reports' 
  | 'history' 
  | 'diet' 
  | 'notes' 
  | 'telehealth' 
  | 'appointments' 
  | 'safety';

export type DoctorNavTab = 
  | 'doctor_dashboard' 
  | 'doctor_patients' 
  | 'doctor_reports' 
  | 'doctor_progress' 
  | 'doctor_alerts' 
  | 'telehealth' 
  | 'doctor_appointments' 
  | 'doctor_messages';

export type AppNavTab = PatientNavTab | DoctorNavTab;

interface SidebarProps {
  currentTab: AppNavTab;
  onSelectTab: (tab: AppNavTab) => void;
  userRole: UserRole;
  user: User;
  activeDoctor?: Doctor;
  onLogout: () => void;
  onOpenProfileModal?: (section?: 'general' | 'medical' | 'settings') => void;
}

export function Sidebar({
  currentTab,
  onSelectTab,
  userRole,
  user,
  activeDoctor,
  onLogout,
  onOpenProfileModal,
}: SidebarProps) {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);
  const [notificationCount, setNotificationCount] = useState(2);

  // Define Patient Navigation Items
  const patientNavItems: { id: PatientNavTab; label: string; icon: React.ElementType; badge?: string }[] = [
    { id: 'home', label: 'Dashboard', icon: Activity },
    { id: 'exercises', label: 'Exercises', icon: ClipboardList },
    { id: 'session', label: 'Live Session', icon: Play, badge: 'Camera' },
    { id: 'progress', label: 'Progress', icon: TrendingUp },
    { id: 'reports', label: 'Reports', icon: FileSpreadsheet, badge: 'New' },
    { id: 'history', label: 'History', icon: History },
    { id: 'diet', label: 'Diet & Nutrition', icon: Utensils },
    { id: 'notes', label: 'Notes', icon: FileText },
    { id: 'telehealth', label: 'Telehealth', icon: Video, badge: 'Active' },
    { id: 'appointments', label: 'Appointments', icon: Calendar },
    { id: 'safety', label: 'Safety & SOS', icon: Shield },
  ];

  // Define Doctor Navigation Items
  const doctorNavItems: { id: DoctorNavTab; label: string; icon: React.ElementType; badge?: string }[] = [
    { id: 'doctor_dashboard', label: 'Dashboard', icon: Activity },
    { id: 'doctor_patients', label: 'Patients', icon: Users },
    { id: 'doctor_reports', label: 'Patient Reports', icon: FileSpreadsheet },
    { id: 'doctor_progress', label: 'Patient Progress', icon: TrendingUp },
    { id: 'doctor_alerts', label: 'Alerts', icon: AlertTriangle, badge: 'SOS' },
    { id: 'telehealth', label: 'Telehealth', icon: Video, badge: 'Queue' },
    { id: 'doctor_appointments', label: 'Appointments', icon: Calendar },
    { id: 'doctor_messages', label: 'Messages', icon: MessageSquare },
  ];

  const handleNavClick = (tab: AppNavTab) => {
    onSelectTab(tab);
    setIsMobileOpen(false);
  };

  const displayName = userRole === 'patient' 
    ? user.name 
    : (activeDoctor?.name || 'Dr. Aarav Patel');

  const displayRoleLabel = userRole === 'patient'
    ? 'Patient • Rehabilitation'
    : (activeDoctor?.specialization || 'Attending Physiotherapist');

  const displayAvatarLetter = displayName.charAt(0) || 'P';

  return (
    <>
      {/* Mobile Header Bar */}
      <div className="md:hidden sticky top-0 z-40 bg-white border-b border-slate-200 px-4 py-3 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <span className="text-sm font-bold text-slate-900 block leading-tight">PhysioAI</span>
            <span className="text-[10px] text-slate-500 block leading-tight">
              {userRole === 'patient' ? 'Patient Portal' : 'Clinical Portal'}
            </span>
          </div>
        </div>

        <button
          onClick={() => setIsMobileOpen(!isMobileOpen)}
          className="p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 cursor-pointer"
          aria-label="Toggle Navigation Menu"
        >
          {isMobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          onClick={() => setIsMobileOpen(false)}
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 md:hidden"
        />
      )}

      {/* Main Sidebar Container */}
      <aside
        className={`fixed md:sticky top-0 left-0 z-50 h-screen w-64 bg-white border-r border-slate-200 flex flex-col justify-between transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top App Branding */}
        <div>
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-xs">
                <Activity className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-bold text-slate-900 truncate">PhysioAI</span>
                  <span className={`px-1.5 py-0.2 rounded text-[9px] font-semibold border ${
                    userRole === 'patient'
                      ? 'bg-[#F0F7FF] text-blue-700 border-blue-200'
                      : 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]'
                  }`}>
                    {userRole === 'patient' ? 'Patient' : 'Clinical'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 truncate">Kinematic Rehabilitation</p>
              </div>
            </div>

            <button
              onClick={() => setIsMobileOpen(false)}
              className="md:hidden p-1 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-0.5 overflow-y-auto max-h-[calc(100vh-170px)]">
            <div className="px-3 pt-1 pb-1.5 text-[10px] font-mono uppercase tracking-wider text-slate-600 font-semibold">
              {userRole === 'patient' ? 'Rehabilitation Navigation' : 'Clinical Operations'}
            </div>

            {userRole === 'patient' ? (
              patientNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer group ${
                      isActive
                        ? 'bg-[#F0F7FF] text-blue-700 font-semibold border border-blue-200/70 shadow-xs'
                        : 'text-slate-600 hover:bg-slate-100/80 hover:text-slate-900'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <Icon className={`w-4 h-4 shrink-0 transition-colors ${
                        isActive ? 'text-blue-600' : 'text-slate-500 group-hover:text-slate-700'
                      }`} />
                      <span className="truncate">{item.label}</span>
                    </div>
                    {item.badge && (
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase shrink-0 ${
                        item.badge === 'New' 
                          ? 'bg-blue-100 text-blue-800' 
                          : item.badge === 'Camera'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-slate-100 text-slate-600'
                      }`}>
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })
            ) : (
              doctorNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer group ${
                      isActive
                        ? 'bg-[#ECFDF3] text-[#065F46] font-semibold border border-[#A7F3D0] shadow-xs'
                        : 'text-slate-600 hover:bg-slate-100/80 hover:text-slate-900'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <Icon className={`w-4 h-4 shrink-0 transition-colors ${
                        isActive ? 'text-emerald-600' : 'text-slate-500 group-hover:text-slate-700'
                      }`} />
                      <span className="truncate">{item.label}</span>
                    </div>
                    {item.badge && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase bg-amber-100 text-amber-800 border border-amber-200 shrink-0">
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })
            )}
          </nav>
        </div>

        {/* BOTTOM PROFILE SECTION WITH COMPACT DROPDOWN (Requirement 1) */}
        <div className="p-3 border-t border-slate-200 bg-slate-50/50 relative">
          
          {/* Compact Dropdown Menu */}
          {isProfileDropdownOpen && (
            <div className="absolute bottom-full left-3 right-3 mb-2 bg-white rounded-xl border border-slate-200 shadow-lg p-2 z-50 animate-in fade-in slide-in-from-bottom-2 duration-150">
              
              <div className="px-3 py-2 border-b border-slate-100 mb-1">
                <span className="text-[11px] font-semibold text-slate-900 block truncate">{displayName}</span>
                <span className="text-[10px] text-slate-500 block truncate">{displayRoleLabel}</span>
              </div>

              <div className="space-y-0.5 text-xs">
                {userRole === 'patient' ? (
                  <>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        onOpenProfileModal?.('general');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <UserIcon className="w-3.5 h-3.5 text-slate-500" />
                      <span>View Profile</span>
                    </button>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        onOpenProfileModal?.('medical');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <Heart className="w-3.5 h-3.5 text-slate-500" />
                      <span>Medical Profile</span>
                    </button>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        onOpenProfileModal?.('settings');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <Settings className="w-3.5 h-3.5 text-slate-500" />
                      <span>Account Settings</span>
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        onOpenProfileModal?.('general');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <Stethoscope className="w-3.5 h-3.5 text-slate-500" />
                      <span>Doctor Profile</span>
                    </button>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        onOpenProfileModal?.('medical');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <UserCheck className="w-3.5 h-3.5 text-slate-500" />
                      <span>Professional Information</span>
                    </button>
                    <button
                      onClick={() => {
                        setIsProfileDropdownOpen(false);
                        handleNavClick('doctor_appointments');
                      }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                    >
                      <Clock className="w-3.5 h-3.5 text-slate-500" />
                      <span>Availability & Slots</span>
                    </button>
                  </>
                )}

                <button
                  onClick={() => {
                    setIsProfileDropdownOpen(false);
                    setNotificationCount(0);
                  }}
                  className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Bell className="w-3.5 h-3.5 text-slate-500" />
                    <span>Notifications</span>
                  </div>
                  {notificationCount > 0 && (
                    <span className="w-4 h-4 rounded-full bg-blue-600 text-white text-[9px] font-bold flex items-center justify-center">
                      {notificationCount}
                    </span>
                  )}
                </button>

                <div className="pt-1 mt-1 border-t border-slate-100">
                  <button
                    onClick={() => {
                      setIsProfileDropdownOpen(false);
                      onLogout();
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md text-red-600 hover:bg-red-50 transition-colors text-left cursor-pointer font-medium"
                  >
                    <LogOut className="w-3.5 h-3.5 text-red-500" />
                    <span>Switch Domain / Logout</span>
                  </button>
                </div>
              </div>

            </div>
          )}

          {/* Profile Trigger Button (Compact, not a large card) */}
          <button
            onClick={() => setIsProfileDropdownOpen(!isProfileDropdownOpen)}
            className="w-full p-2 rounded-lg bg-white hover:bg-slate-100/80 border border-slate-200 transition-colors flex items-center justify-between cursor-pointer group shadow-xs"
            aria-label="User Account Controls"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0 shadow-xs ${
                userRole === 'patient' ? 'bg-blue-600' : 'bg-emerald-600'
              }`}>
                {displayAvatarLetter}
              </div>
              <div className="text-left min-w-0">
                <span className="text-xs font-semibold text-slate-900 block truncate leading-tight">
                  {displayName}
                </span>
                <span className="text-[10px] text-slate-500 block truncate leading-tight mt-0.5">
                  {userRole === 'patient' ? 'Patient Role' : 'Doctor Role'}
                </span>
              </div>
            </div>

            <div className="text-slate-400 group-hover:text-slate-600 shrink-0 ml-1">
              {isProfileDropdownOpen ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </div>
          </button>

        </div>
      </aside>
    </>
  );
}
