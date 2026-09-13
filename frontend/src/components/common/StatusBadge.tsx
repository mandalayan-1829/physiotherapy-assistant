interface StatusBadgeProps {
  status: "good" | "warning" | "incorrect" | "no_pose";
  className?: string;
}

const STATUS_STYLES = {
  good: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  incorrect: "bg-red-50 text-red-700 border-red-200",
  no_pose: "bg-surface-100 text-surface-500 border-surface-200",
};

const STATUS_LABELS = {
  good: "Good Form",
  warning: "Needs Fix",
  incorrect: "Incorrect Form",
  no_pose: "No Pose Detected",
};

export default function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${STATUS_STYLES[status]} ${className}`}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}
