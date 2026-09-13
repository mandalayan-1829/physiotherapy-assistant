interface StatCardProps {
  label: string;
  value: string | number;
  icon?: string;
  color?: string;
}

export default function StatCard({ label, value, icon, color = "#6366f1" }: StatCardProps) {
  return (
    <div className="bg-white rounded-2xl border border-surface-200 p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">
            {label}
          </p>
          <p className="text-2xl font-bold text-surface-900">{value}</p>
        </div>
        {icon && (
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center text-lg"
            style={{ backgroundColor: `${color}15`, color }}
          >
            {icon}
          </div>
        )}
      </div>
      <div
        className="h-1 rounded-full mt-3 opacity-20"
        style={{ backgroundColor: color }}
      />
    </div>
  );
}
