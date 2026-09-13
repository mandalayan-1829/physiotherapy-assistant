import { useNavigate } from "react-router-dom";
import { Activity, Target, Zap, TrendingUp, ArrowRight } from "lucide-react";

const FEATURES = [
  {
    icon: Activity,
    title: "Real-Time Pose Analysis",
    description: "AI-powered body tracking analyzes your movements frame by frame using advanced computer vision.",
    color: "#6366f1",
  },
  {
    icon: Target,
    title: "Smart Rep Counting",
    description: "Automatic repetition counting with stage detection so you can focus on your form.",
    color: "#059669",
  },
  {
    icon: Zap,
    title: "Form Correction",
    description: "Instant feedback on your posture and movement patterns to prevent injury and maximize results.",
    color: "#d97706",
  },
  {
    icon: TrendingUp,
    title: "Progress Tracking",
    description: "Detailed analytics on your sessions, form scores, and improvement over time.",
    color: "#dc2626",
  },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="flex items-center justify-between px-8 py-5 border-b border-surface-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary-500 rounded-xl flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold text-surface-900">AI Physio</span>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/login")}
            className="px-5 py-2.5 text-sm font-medium text-surface-600 hover:text-surface-900 transition-colors"
          >
            Log In
          </button>
          <button
            onClick={() => navigate("/login?mode=signup")}
            className="px-5 py-2.5 text-sm font-medium bg-primary-500 text-white rounded-xl hover:bg-primary-600 transition-colors"
          >
            Get Started
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-8 py-20 text-center">
        <div className="inline-flex items-center gap-2 bg-primary-50 text-primary-600 px-4 py-2 rounded-full text-sm font-medium mb-8">
          <Activity className="w-4 h-4" />
          AI-Powered Physiotherapy
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold text-surface-900 leading-tight mb-6">
          Your Personal
          <br />
          <span className="text-primary-500">AI Physiotherapy</span> Assistant
        </h1>
        <p className="text-lg text-surface-500 max-w-2xl mx-auto mb-10 leading-relaxed">
          Real-time movement analysis, guided exercises, intelligent form correction,
          and progress tracking — all in one place.
        </p>
        <div className="flex items-center justify-center gap-4">
          <button
            onClick={() => navigate("/login?mode=signup")}
            className="px-8 py-3.5 bg-primary-500 text-white rounded-xl font-semibold text-sm hover:bg-primary-600 transition-all shadow-lg shadow-primary-200 flex items-center gap-2"
          >
            Start Therapy
            <ArrowRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => navigate("/login")}
            className="px-8 py-3.5 bg-surface-100 text-surface-700 rounded-xl font-semibold text-sm hover:bg-surface-200 transition-all"
          >
            Explore Exercises
          </button>
        </div>
      </section>

      {/* Tagline */}
      <section className="text-center pb-16">
        <p className="text-surface-400 text-sm font-medium tracking-widest uppercase">
          Move Better. Recover Smarter.
        </p>
      </section>

      {/* Features */}
      <section className="bg-surface-50 border-t border-surface-100">
        <div className="max-w-6xl mx-auto px-8 py-20">
          <h2 className="text-3xl font-bold text-surface-900 text-center mb-4">
            Built for Better Recovery
          </h2>
          <p className="text-surface-500 text-center mb-14 max-w-xl mx-auto">
            Combining cutting-edge AI with physiotherapy expertise to help you recover faster and move better.
          </p>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className="bg-white rounded-2xl p-6 border border-surface-200 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                    style={{ backgroundColor: `${feature.color}10` }}
                  >
                    <Icon className="w-6 h-6" style={{ color: feature.color }} />
                  </div>
                  <h3 className="font-semibold text-surface-900 mb-2">{feature.title}</h3>
                  <p className="text-sm text-surface-500 leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-surface-900 text-surface-400 text-center py-8">
        <div className="flex items-center justify-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-primary-400" />
          <span className="font-semibold text-white">AI Physio</span>
        </div>
        <p className="text-sm">Move Better. Recover Smarter.</p>
      </footer>
    </div>
  );
}
