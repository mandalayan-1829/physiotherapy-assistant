import { AlertCircle, Construction } from 'lucide-react';

interface NotImplementedViewProps {
  title: string;
  description: string;
  pending?: string[];
}

/**
 * Shown instead of a feature that is not built yet.
 *
 * The previous build replaced this area with a hardcoded admin console and
 * demo passcode. Rather than pretend the capability exists, the UI states
 * plainly that it is still to be implemented.
 */
export function NotImplementedView({ title, description, pending = [] }: NotImplementedViewProps) {
  return (
    <div className="max-w-3xl mx-auto space-y-5">
      <div className="pb-5 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-500" />
          <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
            Not implemented yet
          </span>
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
          <Construction className="w-5 h-5 text-amber-600" />
          <span>{title}</span>
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1">{description}</p>
      </div>

      <div className="p-4 rounded-xl bg-[#FFF8E6] border border-amber-200 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 space-y-1.5">
          <p className="font-semibold">
            This screen is intentionally empty rather than showing simulated data.
          </p>
          <p>
            The backend endpoints for this area exist and are covered by tests, but the user
            interface has not been connected yet.
          </p>
          {pending.length > 0 && (
            <ul className="list-disc list-inside space-y-1 pt-1">
              {pending.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
