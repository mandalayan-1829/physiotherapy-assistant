import { Outlet, useNavigate, useLocation } from "react-router-dom";
import {
  Home,
  Dumbbell,
  BarChart3,
  User,
  LogOut,
  Activity,
} from "lucide-react";
import { logout } from "@/services/userService";
import { useAuth } from "@/hooks/useAuth";

const NAV_ITEMS = [
  { path: "/dashboard", label: "Home", icon: Home },
  { path: "/exercises", label: "Exercises", icon: Dumbbell },
  { path: "/progress", label: "Progress", icon: BarChart3 },
  { path: "/profile", label: "Profile", icon: User },
];

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-surface-50">
      {/* Sidebar */}
      <aside className="w-64 bg-surface-900 text-white flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-surface-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-500 rounded-xl flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg">AI Physio</h1>
              <p className="text-xs text-surface-400">Move Better. Recover Smarter.</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname.startsWith(item.path);
            const Icon = item.icon;
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? "bg-primary-600 text-white"
                    : "text-surface-300 hover:bg-surface-800 hover:text-white"
                }`}
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* User + Logout */}
        <div className="p-4 border-t border-surface-700">
          <div className="flex items-center gap-3 mb-3 px-4">
            <div className="w-8 h-8 bg-primary-700 rounded-full flex items-center justify-center text-sm font-bold">
              {user?.name?.charAt(0) || "U"}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{user?.name || "User"}</p>
              <p className="text-xs text-surface-400 truncate">{user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-surface-300 hover:bg-surface-800 hover:text-white transition-all"
          >
            <LogOut className="w-5 h-5" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
