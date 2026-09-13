import { useNavigate } from "react-router-dom";
import { BarChart3 } from "lucide-react";
import { useProgress } from "@/hooks/useProgress";
import StatCard from "@/components/common/StatCard";
import Card from "@/components/common/Card";
import LoadingSpinner from "@/components/common/LoadingSpinner";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

const COLORS = ["#6366f1", "#059669", "#d97706", "#dc2626", "#8b5cf6", "#0891b2"];

export default function Progress() {
  const navigate = useNavigate();
  const { progress, loading, error } = useProgress();

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="p-8 text-red-500">{error}</div>;
  if (!progress) return null;

  const barData = progress.summary.map((s) => ({
    name: s.exercise.replace("_", " "),
    sessions: s.total_sessions,
    reps: s.total_reps,
    form: s.avg_form,
  }));

  const pieData = progress.summary.map((s) => ({
    name: s.exercise.replace("_", " "),
    value: s.total_sessions,
  }));

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-extrabold text-surface-900 mb-1">My Progress</h1>
          <p className="text-surface-500">Track your improvement over time</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <StatCard label="Total Sessions" value={progress.total_sessions} icon="📋" color="#6366f1" />
        <StatCard label="Total Reps" value={progress.total_reps} icon="🏋️" color="#059669" />
        <StatCard label="Avg Form" value={`${progress.avg_form_accuracy}%`} icon="✅" color="#d97706" />
        <StatCard label="Exercises Done" value={progress.exercises_done} icon="🎯" color="#dc2626" />
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6 mb-10">
        <Card>
          <h3 className="font-bold text-surface-900 mb-4">Sessions per Exercise</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={barData}>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="sessions" fill="#6366f1" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="font-bold text-surface-900 mb-4">Exercise Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Exercise Breakdown */}
      <h2 className="text-lg font-bold text-surface-900 mb-4">Exercise Breakdown</h2>
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {progress.summary.map((s) => (
          <Card key={s.exercise}>
            <h4 className="font-bold text-surface-900 mb-2 capitalize">
              {s.exercise.replace(/_/g, " ")}
            </h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-surface-500">Sessions</span>
                <span className="font-medium">{s.total_sessions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-500">Total Reps</span>
                <span className="font-medium">{s.total_reps}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-500">Avg Form</span>
                <span className="font-medium">{Math.round(s.avg_form)}%</span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Recent Sessions */}
      <h2 className="text-lg font-bold text-surface-900 mb-4">Recent Sessions</h2>
      {progress.recent_sessions.length > 0 ? (
        <div className="bg-white rounded-2xl border border-surface-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-50 border-b border-surface-200">
                <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-6 py-3">Date</th>
                <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-6 py-3">Exercise</th>
                <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-6 py-3">Reps</th>
                <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-6 py-3">Form</th>
              </tr>
            </thead>
            <tbody>
              {progress.recent_sessions.map((s) => (
                <tr key={s.id} className="border-b border-surface-100 last:border-0">
                  <td className="px-6 py-3 text-sm text-surface-600">
                    {s.date ? new Date(s.date).toLocaleDateString() : "—"}
                  </td>
                  <td className="px-6 py-3 text-sm font-medium text-surface-900 capitalize">
                    {s.exercise.replace(/_/g, " ")}
                  </td>
                  <td className="px-6 py-3 text-sm text-surface-600">{s.reps}</td>
                  <td className="px-6 py-3 text-sm">
                    <span className={`font-medium ${
                      s.form_accuracy >= 80 ? "text-emerald-600" : s.form_accuracy >= 50 ? "text-amber-600" : "text-red-600"
                    }`}>
                      {s.form_accuracy}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Card className="text-center py-10">
          <p className="text-surface-500">No sessions yet. Start exercising to see your progress!</p>
        </Card>
      )}
    </div>
  );
}
