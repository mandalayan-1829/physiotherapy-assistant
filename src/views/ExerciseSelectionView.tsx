import { useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  ChevronDown, 
  ChevronRight, 
  Dumbbell, 
  Play, 
  Search, 
  Sliders, 
  Video, 
  X,
  Minus,
  Plus,
  Info
} from 'lucide-react';
import { Exercise, ExerciseType, User } from '../types';
import { EXERCISES, isExerciseSafe } from '../data/exercises';
import { ExerciseIllustrationPlaceholder } from '../components/ExerciseIllustrationPlaceholder';

interface ExerciseSelectionViewProps {
  user: User;
  onSelectExercise: (exercise: Exercise, targetReps?: number) => void;
}

export function ExerciseSelectionView({ user, onSelectExercise }: ExerciseSelectionViewProps) {
  const [selectedFilter, setSelectedFilter] = useState<'all' | ExerciseType>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // Single active accordion row
  const [openExerciseId, setOpenExerciseId] = useState<string | null>('squat');

  // Per-exercise target reps override state
  const [repsMap, setRepsMap] = useState<Record<string, number>>({});

  // Video guide modal
  const [videoModalExercise, setVideoModalExercise] = useState<Exercise | null>(null);

  const allExercises = Object.values(EXERCISES);

  const filteredExercises = allExercises.filter((ex) => {
    if (selectedFilter !== 'all' && ex.type !== selectedFilter) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        ex.label.toLowerCase().includes(q) ||
        ex.target.toLowerCase().includes(q) ||
        ex.primaryJoint.toLowerCase().includes(q) ||
        ex.description.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const toggleAccordion = (id: string) => {
    setOpenExerciseId((prev) => (prev === id ? null : id));
  };

  const getTargetReps = (ex: Exercise): number => {
    return repsMap[ex.id] !== undefined ? repsMap[ex.id] : ex.defaultTargetReps;
  };

  const adjustTargetReps = (exId: string, delta: number, defaultReps: number) => {
    const current = repsMap[exId] !== undefined ? repsMap[exId] : defaultReps;
    const next = Math.max(1, Math.min(50, current + delta));
    setRepsMap((prev) => ({ ...prev, [exId]: next }));
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header & Controls */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Exercise Protocols
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <Dumbbell className="w-5 h-5 text-blue-600" />
              <span>Exercise Catalogue & Biomechanics</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Select an exercise below to review motion guidelines, target reps, and begin live AI pose tracking.
            </p>
          </div>

          {/* Search bar */}
          <div className="relative w-full md:w-72">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search exercise, joint, or goal..."
              className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-xs"
            />
          </div>
        </div>

        {/* Category Filters */}
        <div className="flex items-center gap-2 mt-4 pt-3 border-t border-slate-200">
          <button
            onClick={() => setSelectedFilter('all')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
              selectedFilter === 'all'
                ? 'bg-blue-600 text-white font-semibold shadow-xs'
                : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            All Movements ({allExercises.length})
          </button>
          <button
            onClick={() => setSelectedFilter('physio')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
              selectedFilter === 'physio'
                ? 'bg-blue-600 text-white font-semibold shadow-xs'
                : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            Physiotherapy (7)
          </button>
          <button
            onClick={() => setSelectedFilter('yoga')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
              selectedFilter === 'yoga'
                ? 'bg-blue-600 text-white font-semibold shadow-xs'
                : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            Yoga & Balance (3)
          </button>
        </div>
      </div>

      {/* Full-Width Vertical Accordion (Zero Cards) */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        {filteredExercises.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500">
            No movements matched your query. Clear search to see all exercises.
          </div>
        ) : (
          filteredExercises.map((ex) => {
            const isOpen = openExerciseId === ex.id;
            const safety = isExerciseSafe(ex.id, user);
            const currentTargetReps = getTargetReps(ex);

            return (
              <div key={ex.id} className="transition-colors">
                
                {/* Accordion Row Header */}
                <button
                  onClick={() => toggleAccordion(ex.id)}
                  aria-expanded={isOpen}
                  className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-[#F0F7FF]/50 transition-colors cursor-pointer select-none rounded-lg"
                >
                  <div className="flex-1 min-w-0 pr-4">
                    <div className="flex items-center gap-2.5">
                      <span className="text-base font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
                        {ex.label}
                      </span>
                      {!safety.safe && (
                        <span 
                          className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#FFF8E6] text-amber-800 border border-amber-200 flex items-center gap-1"
                          title={safety.warning}
                        >
                          <AlertTriangle className="w-3 h-3 text-amber-600 shrink-0" />
                          <span>Safety Notice</span>
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {ex.target} • {ex.primaryJoint}
                    </p>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span className="px-2.5 py-0.5 rounded text-[10px] font-mono uppercase font-bold tracking-wider bg-slate-100 text-slate-600 border border-slate-200">
                      {ex.type}
                    </span>
                    <div className="w-6 h-6 rounded flex items-center justify-center text-slate-400 group-hover:text-slate-600 transition-transform">
                      {isOpen ? (
                        <ChevronDown className="w-4 h-4 text-blue-600" />
                      ) : (
                        <ChevronRight className="w-4 h-4" />
                      )}
                    </div>
                  </div>
                </button>

                {/* Expanded Accordion Body */}
                {isOpen && (
                  <div className="pb-6 pt-2 px-2 space-y-4">
                    
                    {/* [ LINE VECTOR IMAGE AREA ] */}
                    <div className="w-full max-w-2xl mx-auto">
                      <ExerciseIllustrationPlaceholder
                        exerciseId={ex.id}
                        exerciseLabel={ex.label}
                        aspectRatio="video"
                      />
                    </div>

                    {/* Metadata & Details Grid */}
                    <div className="max-w-2xl mx-auto space-y-3 pt-1 text-xs">
                      
                      {/* Target & Difficulty Row */}
                      <div className="flex flex-wrap items-center justify-between gap-3 py-2 border-b border-slate-200">
                        <div>
                          <span className="text-slate-500">Target Repetitions: </span>
                          <span className="text-slate-900 font-mono font-bold">{currentTargetReps} reps</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Difficulty: </span>
                          <span className="text-emerald-700 font-semibold">
                            {ex.defaultTargetReps > 10 ? 'Moderate' : 'Therapeutic / Beginner'}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">Target Kinematic Angle: </span>
                          <span className="text-slate-800 font-mono">{ex.idealAngleRange}</span>
                        </div>
                      </div>

                      {/* Clinical Description */}
                      <p className="text-slate-600 leading-relaxed">
                        {ex.description}
                      </p>

                      {/* Real-time checks */}
                      <div className="py-2">
                        <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-500 block mb-1.5">
                          Computer Vision Pose Checks:
                        </span>
                        <div className="flex flex-wrap gap-2">
                          {ex.formChecks.map((chk, i) => (
                            <span 
                              key={i} 
                              className="px-2.5 py-1 rounded bg-slate-50 border border-slate-200 text-slate-700 text-[11px] flex items-center gap-1.5"
                            >
                              <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                              <span>{chk}</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Safety Information if applicable */}
                      {!safety.safe && (
                        <div className="p-3 rounded-lg bg-[#FFF8E6] border border-amber-200 text-amber-900 flex items-start gap-2.5">
                          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                          <div>
                            <span className="font-bold text-amber-800 block">Medical Safety Advisory</span>
                            <span className="text-[11px] text-amber-900/90">{safety.warning}</span>
                          </div>
                        </div>
                      )}

                      {/* Repetition Adjuster & Actions */}
                      <div className="pt-3 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        
                        {/* Target reps stepper */}
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500 font-medium">Adjust Target:</span>
                          <div className="flex items-center rounded-lg bg-white border border-slate-200 p-0.5 shadow-2xs">
                            <button
                              onClick={() => adjustTargetReps(ex.id, -1, ex.defaultTargetReps)}
                              className="w-8 h-8 rounded flex items-center justify-center text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer"
                              title="Decrease reps"
                            >
                              <Minus className="w-3.5 h-3.5" />
                            </button>
                            <span className="w-10 text-center font-mono font-bold text-slate-900 text-xs">
                              {currentTargetReps}
                            </span>
                            <button
                              onClick={() => adjustTargetReps(ex.id, 1, ex.defaultTargetReps)}
                              className="w-8 h-8 rounded flex items-center justify-center text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer"
                              title="Increase reps"
                            >
                              <Plus className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <span className="text-[11px] text-slate-500">reps</span>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex items-center gap-2.5">
                          {ex.videoId && (
                            <button
                              onClick={() => setVideoModalExercise(ex)}
                              className="px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 hover:text-slate-900 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
                            >
                              <Video className="w-3.5 h-3.5 text-blue-600" />
                              <span>Clinical Video</span>
                            </button>
                          )}

                          <button
                            onClick={() => onSelectExercise(ex, currentTargetReps)}
                            className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
                          >
                            <Play className="w-3.5 h-3.5 fill-white" />
                            <span>Start Tracking</span>
                          </button>
                        </div>

                      </div>

                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Video Modal Guide */}
      {videoModalExercise && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-xl w-full max-w-2xl overflow-hidden shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <Video className="w-4 h-4 text-blue-600" />
                <h3 className="text-sm font-bold text-slate-900">Clinical Tutorial: {videoModalExercise.label}</h3>
              </div>
              <button
                onClick={() => setVideoModalExercise(null)}
                className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="aspect-video bg-black">
              <iframe
                src={`https://www.youtube-nocookie.com/embed/${videoModalExercise.videoId}?autoplay=1&rel=0`}
                title={videoModalExercise.label}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full border-0"
              />
            </div>

            <div className="p-4 flex items-center justify-between bg-slate-50 border-t border-slate-200">
              <span className="text-xs text-slate-600">{videoModalExercise.tip}</span>
              <button
                onClick={() => {
                  const target = videoModalExercise;
                  setVideoModalExercise(null);
                  onSelectExercise(target, getTargetReps(target));
                }}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
              >
                <Play className="w-3.5 h-3.5 fill-white" />
                <span>Begin Tracking Now</span>
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
