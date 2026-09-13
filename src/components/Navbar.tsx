import { useState, useEffect } from 'react';
import { 
  Activity, 
  Calendar, 
  ChevronRight, 
  ClipboardList, 
  Clock, 
  Dumbbell, 
  HeartPulse, 
  MessageSquare, 
  ShieldAlert, 
  User as UserIcon, 
  Utensils, 
  Volume2, 
  VolumeX,
  Menu,
  X
} from 'lucide-react';
import { soundManager } from '../utils/audio';
import { User } from '../types';

export type NavTab = 
  | 'dashboard'
  | 'exercises'
  | 'tracking'
  | 'telehealth'
  | 'diet'
  | 'progress'
  | 'notes'
  | 'profile'
  | 'admin';

interface NavbarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  user: User;
  unreadCount?: number;
  isAdminLoggedIn?: boolean;
}

export function Navbar({ currentTab, onSelectTab, user, unreadCount = 0, isAdminLoggedIn = false }: NavbarProps) {
  const [timeStr, setTimeStr] = useState<string>('');
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleSound = () => {
    const next = !soundEnabled;
    soundManager.enabled = next;
    setSoundEnabled(next);
  };

  const navItems = [
    { id: 'dashboard' as NavTab, label: 'Dashboard', icon: Activity },
    { id: 'exercises' as NavTab, label: 'Exercises', icon: Dumbbell },
    { id: 'telehealth' as NavTab, label: 'Telehealth', icon: MessageSquare, badge: unreadCount > 0 ? unreadCount : undefined },
    { id: 'diet' as NavTab, label: 'Diet Log', icon: Utensils },
    { id: 'progress' as NavTab, label: 'Analytics', icon: HeartPulse },
    { id: 'notes' as NavTab, label: 'Clinical Notes', icon: ClipboardList },
    { id: 'profile' as NavTab, label: 'Medical Profile', icon: UserIcon },
    { id: 'admin' as NavTab, label: 'Doctor Portal', icon: ShieldAlert, highlight: isAdminLoggedIn },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo & Live Status */}
          <div className="flex items-center gap-3">
            <button 
              onClick={() => onSelectTab('dashboard')} 
              className="flex items-center gap-2.5 group text-left cursor-pointer focus:outline-none"
            >
              <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-sm group-hover:bg-blue-700 transition-colors">
                <HeartPulse className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-lg text-slate-900 tracking-tight">Physio<span className="text-blue-600">AI</span></span>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                    Live
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 -mt-0.5">Clinical Rehabilitation Suite</p>
              </div>
            </button>
          </div>

          {/* Desktop Navigation Tabs */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`relative flex items-center gap-2 px-3 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all cursor-pointer ${
                    isActive
                      ? 'bg-[#F0F7FF] text-blue-700 border border-blue-200 font-semibold shadow-xs'
                      : item.highlight
                      ? 'text-amber-700 hover:text-amber-800 hover:bg-amber-50/80 border border-amber-200'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="w-4 h-4 rounded-full bg-rose-500 text-white text-[10px] font-bold flex items-center justify-center">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right Header Utilities: Sound, Clock, User pill */}
          <div className="flex items-center gap-2.5 sm:gap-3">
            {/* Live Clock */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 font-mono">
              <Clock className="w-3.5 h-3.5 text-blue-600" />
              <span>{timeStr || '12:00:00'}</span>
            </div>

            {/* Audio Synth Toggle */}
            <button
              onClick={toggleSound}
              title={soundEnabled ? 'Mute audio feedback' : 'Unmute audio feedback'}
              className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-900 transition-colors cursor-pointer"
            >
              {soundEnabled ? <Volume2 className="w-4 h-4 text-emerald-600" /> : <VolumeX className="w-4 h-4 text-slate-400" />}
            </button>

            {/* User Profile Pill */}
            <button
              onClick={() => onSelectTab('profile')}
              className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700 transition-colors cursor-pointer"
            >
              <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-xs">
                {user.name ? user.name[0].toUpperCase() : 'J'}
              </div>
              <span className="hidden md:inline max-w-[100px] truncate">{user.name || 'John'}</span>
            </button>

            {/* Mobile Menu Toggle */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg bg-slate-50 border border-slate-200 text-slate-600 hover:text-slate-900"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white border-b border-slate-200 px-4 pt-2 pb-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive 
                    ? 'bg-[#F0F7FF] text-blue-700 border border-blue-200 font-semibold' 
                    : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-400" />
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
}
