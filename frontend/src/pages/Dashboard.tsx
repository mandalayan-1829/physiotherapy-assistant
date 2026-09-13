import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Dumbbell, BarChart3, Flame, Target, ArrowRight } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { fetchProgressSummary } from "@/services/progressService";
import StatCard from "@/components/common/StatCard";
import Card from "@/components/common/Card";

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [summary, setSummary] = useState<any>(null);

  useEffect(() => {
    fetchProgressSummary().then(setSummary).catch(() => {});
  }, []);

  const hour = new Date().getHours();
  const greeting =
    hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-10">
        <div>
          <p className="text-primary-500 font-semibold text-sm tracking-wider uppercase mb-1">
            {greeting} 👋
          </p>
          <h1 className="text-3xl font-extrabold text-surface-900">
            Welcome, {user?.name || "User"}
          </h1>
          <p className="text-surface-500 mt-1">Ready for your session today?</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <StatCard label="Total Sessions" value={summary?.total_sessions ?? "—"} icon="📋" color="#6366f1" />
        <StatCard label="Total Reps" value={summary?.total_reps ?? "—"} icon="🏋️" color="#059669" />
        <StatCard label="Avg Form" value={summary?.avg_form_accuracy ? `${summary.avg_form_accuracy}%` : "—"} icon="✅" color="#d97706" />
        <StatCard label="Streak" value={summary?.current_streak ? `${summary.current_streak} days` : "0"} icon="🔥" color="#dc2626" />
      </div>

      {/* Quick Actions */}
      <h2 className="text-lg font-bold text-surface-900 mb-4">Quick Actions</h2>
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <Card onClick={() => navigate("/exercises")} className="text-center">
          <Dumbbell className="w-8 h-8 text-primary-500 mx-auto mb-3" />
          <h3 className="font-semibold text-surface-900 text-sm">Start Exercise</h3>
          <p className="text-xs text-surface-500 mt-1">Begin a rehab session</p>
        </Card>
        <Card onClick={() => navigate("/progress")} className="text-center">
          <BarChart3 className="w-8 h-8 text-emerald-500 mx-auto mb-3" />
          <h3 className="font-semibold text-surface-900 text-sm">View Progress</h3>
          <p className="text-xs text-surface-500 mt-1">Track your improvement</p>
        </Card>
        <Card onClick={() => navigate("/exercises")} className="text-center">
          <Target className="w-8 h-8 text-amber-500 mx-auto mb-3" />
          <h3 className="font-semibold text-surface-900 text-sm">Exercise Library</h3>
          <p className="text-xs text-surface-500 mt-1">Browse all exercises</p>
        </Card>
        <Card onClick={() => navigate("/profile")} className="text-center">
          <Flame className="w-8 h-8 text-red-500 mx-auto mb-3" />
          <h3 className="font-semibold text-surface-900 text-sm">My Profile</h3>
          <p className="text-xs text-surface-500 mt-1">Manage your settings</p>
        </Card>
      </div>

      {/* Exercise timing */}
      {(hour < 5 || hour >= 21) && (
        <div className="bg-red-50 border border-red-200 rounded-2xl p-5 mb-6">
          <p className="text-red-700 font-medium text-sm">
            🚨 <strong>Not recommended!</strong> It is late night / very early morning.
            Please rest and exercise during morning (5–9 AM) or evening (5–8 PM) windows.
          </p>
        </div>
      )}
      {(hour >= 5 && hour < 9) && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-5 mb-6">
          <p className="text-emerald-700 font-medium text-sm">
            ✅ <strong>Perfect time to exercise!</strong> Morning sessions boost metabolism and energy.
          </p>
        </div>
      )}
      {(hour >= 17 && hour < 20) && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-5 mb-6">
          <p className="text-emerald-700 font-medium text-sm">
            ✅ <strong>Perfect time to exercise!</strong> Evening sessions improve muscle recovery.
          </p>
        </div>
      )}

      {/* Recent Activity hint */}
      {summary && summary.total_sessions > 0 && (
        <div className="mt-6">
          <button
            onClick={() => navigate("/progress")}
            className="flex items-center gap-2 text-sm font-medium text-primary-500 hover:text-primary-600 transition-colors"
          >
            View detailed progress
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
