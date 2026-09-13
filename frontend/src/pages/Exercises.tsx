import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { useExercises } from "@/hooks/useExercises";
import Card from "@/components/common/Card";
import LoadingSpinner from "@/components/common/LoadingSpinner";

export default function Exercises() {
  const navigate = useNavigate();
  const { exercises, loading, error } = useExercises();
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "physio" | "yoga">("all");

  const filtered = exercises.filter((ex) => {
    const matchesSearch = ex.label.toLowerCase().includes(search.toLowerCase()) ||
      ex.target.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === "all" || ex.type === filter;
    return matchesSearch && matchesFilter;
  });

  if (loading) return <LoadingSpinner />;

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-extrabold text-surface-900 mb-2">Exercise Library</h1>
      <p className="text-surface-500 mb-8">Choose an exercise — your AI coach will guide you</p>

      {/* Search + Filter */}
      <div className="flex flex-col sm:flex-row gap-4 mb-8">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400" />
          <input
            type="text"
            placeholder="Search exercises..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-3 rounded-xl border border-surface-200 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>
        <div className="flex bg-surface-100 rounded-xl p-1">
          {(["all", "physio", "yoga"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                filter === f ? "bg-white text-surface-900 shadow-sm" : "text-surface-500"
              }`}
            >
              {f === "all" ? "All" : f === "physio" ? "Physiotherapy" : "Yoga"}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 text-red-600 text-sm px-4 py-3 rounded-xl mb-6">{error}</div>
      )}

      {/* Exercise grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filtered.map((ex) => (
          <Card
            key={ex.id}
            onClick={() => navigate(`/exercises/${ex.id}`)}
            className="text-center group"
          >
            <div className="text-4xl mb-3">{ex.icon}</div>
            <h3 className="font-bold text-surface-900 text-lg mb-1">{ex.label}</h3>
            <p className="text-primary-500 text-xs font-medium mb-2">{ex.target}</p>
            <div className="flex items-center justify-center gap-2">
              <span className={`inline-block px-2.5 py-1 rounded-full text-xs font-medium ${
                ex.type === "yoga"
                  ? "bg-emerald-50 text-emerald-600"
                  : "bg-primary-50 text-primary-600"
              }`}>
                {ex.type === "yoga" ? "🧘 Yoga" : "🏥 Physio"}
              </span>
              <span className="inline-block px-2.5 py-1 rounded-full text-xs font-medium bg-surface-100 text-surface-600">
                {ex.difficulty}
              </span>
            </div>
          </Card>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-16">
          <p className="text-surface-500">No exercises found matching your search.</p>
        </div>
      )}
    </div>
  );
}
