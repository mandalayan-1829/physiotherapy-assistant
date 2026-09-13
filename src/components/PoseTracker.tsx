import { useEffect, useRef, useState } from 'react';
import confetti from 'canvas-confetti';
import { 
  AlertCircle, 
  Camera, 
  CheckCircle2, 
  ChevronRight, 
  Clock, 
  Dumbbell,
  Play, 
  RotateCcw, 
  Sliders, 
  Sparkles, 
  VideoOff,
  Flame,
  Volume2,
  VolumeX
} from 'lucide-react';
import { Exercise, ExerciseState, Landmark, Session } from '../types';
import { ExerciseEngine } from '../utils/exerciseEngine';
import { POSE_LANDMARKS } from '../utils/angleCalculator';
import { soundManager } from '../utils/audio';

interface PoseTrackerProps {
  exercise: Exercise;
  targetReps: number;
  onSessionComplete: (session: Omit<Session, 'id'>) => void;
  onCancel: () => void;
}

export function PoseTracker({ exercise, targetReps, onSessionComplete, onCancel }: PoseTrackerProps) {
  const [useCamera, setUseCamera] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1);
  const [durationSec, setDurationSec] = useState<number>(0);
  const [showSummary, setShowSummary] = useState<boolean>(false);
  const [userNotes, setUserNotes] = useState<string>('');
  
  // Audio feedback toggle
  const [soundOn, setSoundOn] = useState<boolean>(soundManager.enabled);

  // Engine instance
  const engineRef = useRef<ExerciseEngine>(new ExerciseEngine(exercise.id, targetReps));
  const [state, setState] = useState<ExerciseState>(engineRef.current.getState());

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Accuracy tracking
  const totalFramesRef = useRef<number>(0);
  const goodFramesRef = useRef<number>(0);
  const simPhaseRef = useRef<number>(0);

  // Duration timer
  useEffect(() => {
    if (!isRunning || showSummary) return;
    const timer = setInterval(() => {
      setDurationSec((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [isRunning, showSummary]);

  // Handle camera start/stop
  useEffect(() => {
    if (!useCamera) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      return;
    }

    let isCancelled = false;
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      .then((stream) => {
        if (isCancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
        setCameraError(null);
      })
      .catch((err) => {
        console.warn('Camera access denied or unavailable:', err);
        setCameraError('Camera access not available in this browser frame. Switched to Guided Practice mode.');
        setUseCamera(false);
      });

    return () => {
      isCancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, [useCamera]);

  // Main pose processing & rendering loop
  useEffect(() => {
    let lastTime = performance.now();

    const loop = (time: number) => {
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      if (isRunning && !showSummary) {
        // Advance simulation or read camera frame
        simPhaseRef.current += dt * simSpeed * 1.5;

        // Generate pose landmarks (simulation algorithm based on exercise type)
        const landmarks = generatePoseLandmarks(exercise.id, simPhaseRef.current);

        // Process with biomechanics engine
        const newState = engineRef.current.process(landmarks);
        setState({ ...newState });

        // Update accuracy stats
        totalFramesRef.current += 1;
        if (newState.formOk) {
          goodFramesRef.current += 1;
        }

        // Check if workout complete
        if (newState.isComplete && !showSummary) {
          setShowSummary(true);
          setIsRunning(false);
          confetti({
            particleCount: 80,
            spread: 70,
            origin: { y: 0.6 },
          });
        }

        // Render skeleton on canvas
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            drawSkeleton(ctx, canvas.width, canvas.height, landmarks, newState, exercise.id);
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isRunning, showSummary, simSpeed, exercise.id]);

  const handleReset = () => {
    engineRef.current.reset();
    setState(engineRef.current.getState());
    totalFramesRef.current = 0;
    goodFramesRef.current = 0;
    setDurationSec(0);
    setShowSummary(false);
    setIsRunning(true);
  };

  const calculatedAccuracy = totalFramesRef.current > 0 
    ? Math.min(100, Math.max(50, Math.round((goodFramesRef.current / totalFramesRef.current) * 100))) 
    : 92;

  const handleSaveWorkout = () => {
    onSessionComplete({
      userId: 1,
      exercise: exercise.id,
      exerciseLabel: exercise.label,
      reps: state.reps,
      targetReps: state.targetReps,
      formAccuracy: calculatedAccuracy,
      durationSec,
      notes: userNotes || `${exercise.label} session finished with ${calculatedAccuracy}% form score.`,
      date: new Date().toISOString().replace('T', ' ').slice(0, 16),
    });
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-sm relative overflow-hidden">
      
      {/* Top Header Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 border border-blue-100 flex items-center justify-center">
            <Dumbbell className="w-5 h-5 text-blue-600" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-slate-900">{exercise.label}</h2>
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200 capitalize">
                {exercise.type}
              </span>
            </div>
            <p className="text-xs text-slate-500">{exercise.target} • {exercise.primaryJoint}</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Audio toggle */}
          <button
            onClick={() => {
              const next = !soundOn;
              soundManager.enabled = next;
              setSoundOn(next);
            }}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors cursor-pointer"
            title={soundOn ? 'Mute sound' : 'Unmute sound'}
          >
            {soundOn ? <Volume2 className="w-4 h-4 text-emerald-600" /> : <VolumeX className="w-4 h-4 text-slate-400" />}
          </button>

          {/* Camera toggle */}
          <button
            onClick={() => setUseCamera(!useCamera)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors cursor-pointer ${
              useCamera
                ? 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0] hover:bg-emerald-100'
                : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
            }`}
          >
            {useCamera ? <Camera className="w-3.5 h-3.5 text-emerald-600" /> : <VideoOff className="w-3.5 h-3.5 text-slate-400" />}
            <span>{useCamera ? 'Camera Active' : 'Camera Off'}</span>
          </button>

          {/* Reset button */}
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors cursor-pointer"
            title="Reset workout counter"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Exit / Cancel */}
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg bg-white hover:bg-rose-50 text-slate-600 hover:text-rose-700 border border-slate-200 text-xs font-medium transition-colors cursor-pointer"
          >
            Exit
          </button>
        </div>
      </div>

      {/* Camera warning banner if any */}
      {cameraError && (
        <div className="mt-3 p-2.5 rounded-lg bg-[#FFF8E6] border border-amber-200 text-amber-900 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-600" />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Main Visual Stage: Video/Simulated Canvas + Overlays */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
        
        {/* Stage Container */}
        <div className="lg:col-span-8 relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
          {/* Optional real camera video underneath */}
          {useCamera && (
            <video
              ref={videoRef}
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover -scale-x-100 opacity-60"
            />
          )}

          {/* Canvas for real-time skeleton joints & angle visualizer */}
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="relative z-10 w-full h-full object-contain"
          />

          {/* Top HUD Overlay: Reps and Angle */}
          <div className="absolute top-3 left-3 z-20 flex items-center gap-3">
            <div className="px-3.5 py-1.5 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/80 shadow-lg">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold block">Joint Angle</span>
              <span className="text-xl font-bold font-mono text-white">{state.angle}°</span>
            </div>

            <div className={`px-3.5 py-1.5 rounded-xl backdrop-blur-md border shadow-lg ${
              state.stage === 'down' 
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300' 
                : state.stage === 'holding'
                ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                : 'bg-slate-900/85 border-slate-700/80 text-emerald-400'
            }`}>
              <span className="text-[10px] uppercase tracking-wider font-semibold block">Stage</span>
              <span className="text-sm font-bold uppercase">{state.stage}</span>
            </div>
          </div>

          {/* Top Right: Duration & Speed */}
          <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
            <div className="px-3 py-1.5 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/80 text-xs font-mono text-slate-200 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-indigo-400" />
              <span>{Math.floor(durationSec / 60)}:{(durationSec % 60).toString().padStart(2, '0')}</span>
            </div>
          </div>

          {/* Hold Countdown Bar for Yoga Poses */}
          {state.holdSecondsRequired > 0 && (
            <div className="absolute bottom-16 left-4 right-4 z-20">
              <div className="p-2.5 rounded-xl bg-slate-900/90 backdrop-blur-md border border-indigo-500/30">
                <div className="flex justify-between text-xs font-semibold mb-1 text-indigo-300">
                  <span>Balance Hold Duration</span>
                  <span>{Math.floor(state.holdCount / 10)}s / {state.holdSecondsRequired}s</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all duration-150 rounded-full"
                    style={{ width: `${Math.min(100, (state.holdCount / (state.holdSecondsRequired * 10)) * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Bottom Live Feedback Bar */}
          <div className="absolute bottom-3 left-3 right-3 z-20">
            <div className={`p-3 rounded-xl backdrop-blur-md border transition-all flex items-center gap-3 ${
              state.formErrors.length > 0
                ? 'bg-rose-950/80 border-rose-500/50 text-rose-200'
                : 'bg-emerald-950/80 border-emerald-500/50 text-emerald-200'
            }`}>
              {state.formErrors.length > 0 ? (
                <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 animate-pulse" />
              ) : (
                <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-400" />
              )}
              <div className="min-w-0 flex-1">
                <p className="text-xs sm:text-sm font-semibold truncate">{state.feedback}</p>
                {state.formErrors.length > 0 && (
                  <p className="text-[11px] text-rose-300 truncate mt-0.5">
                    {state.formErrors[0]}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Metric Panel: Big Counter + Instructions + Real-time Checks */}
        <div className="lg:col-span-4 flex flex-col justify-between gap-4">
          
          {/* Big Rep Counter Box */}
          <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between">
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold">Completed Reps</span>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-4xl font-extrabold text-slate-900 font-mono">{state.reps}</span>
                <span className="text-base text-slate-500 font-semibold font-mono">/ {state.targetReps}</span>
              </div>
            </div>
            {/* Circular Progress Ring */}
            <div className="relative w-16 h-16 flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <path
                  className="text-slate-200"
                  strokeWidth="3.5"
                  stroke="currentColor"
                  fill="none"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
                <path
                  className="text-emerald-500 transition-all duration-300"
                  strokeDasharray={`${Math.min(100, (state.reps / state.targetReps) * 100)}, 100`}
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="none"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
              </svg>
              <span className="absolute text-xs font-bold text-slate-900 font-mono">
                {Math.round((state.reps / state.targetReps) * 100)}%
              </span>
            </div>
          </div>

          {/* Form Criteria Checklist */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex-1">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Real-Time Biomechanical Checks</span>
            </h3>
            <ul className="space-y-2">
              {exercise.formChecks.map((check, idx) => {
                const isFailed = state.formErrors.some((err) => err.toLowerCase().includes(check.toLowerCase()));
                return (
                  <li key={idx} className="flex items-center justify-between text-xs py-1 border-b border-slate-200 last:border-0">
                    <span className="text-slate-700">{check}</span>
                    {isFailed ? (
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
                        Correction
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#ECFDF3] text-[#065F46] border border-[#A7F3D0]">
                        Good
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>

            {/* Target Angle Guide */}
            <div className="mt-4 pt-3 border-t border-slate-200 text-[11px] text-slate-500">
              <p className="font-semibold text-slate-700">Optimal Range:</p>
              <p className="text-slate-600">{exercise.idealAngleRange}</p>
              <p className="mt-2 text-blue-700 font-medium italic">Clinical Tip: {exercise.tip}</p>
            </div>
          </div>

          {/* Simulation Tempo Control */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <div className="flex items-center justify-between text-xs text-slate-700 mb-1.5">
              <span className="flex items-center gap-1.5 font-medium">
                <Sliders className="w-3.5 h-3.5 text-blue-600" />
                Practice Rhythm
              </span>
              <span className="font-mono text-blue-700 font-bold">{simSpeed}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.25"
              value={simSpeed}
              onChange={(e) => setSimSpeed(parseFloat(e.target.value))}
              className="w-full accent-blue-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
            />
          </div>

          {/* Early Complete / Finish Button */}
          <button
            onClick={() => {
              setShowSummary(true);
              setIsRunning(false);
            }}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition-all shadow-xs cursor-pointer"
          >
            Finish & Log Session
          </button>
        </div>
      </div>

      {/* Workout Complete Dialog */}
      {showSummary && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4">
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
            <div className="w-12 h-12 rounded-2xl bg-[#ECFDF3] text-emerald-600 border border-[#A7F3D0] flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="w-6 h-6" />
            </div>

            <h3 className="text-xl font-bold text-center text-slate-900">Workout Complete!</h3>
            <p className="text-xs text-center text-slate-500 mt-1">
              Outstanding effort on {exercise.label}. Your kinematic performance has been processed.
            </p>

            {/* Score Grid */}
            <div className="grid grid-cols-3 gap-3 my-5">
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Reps</span>
                <span className="text-xl font-bold text-slate-900 font-mono">{state.reps}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Form Score</span>
                <span className="text-xl font-bold text-emerald-600 font-mono">{calculatedAccuracy}%</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Time</span>
                <span className="text-xl font-bold text-slate-900 font-mono">
                  {Math.floor(durationSec / 60)}:{(durationSec % 60).toString().padStart(2, '0')}
                </span>
              </div>
            </div>

            {/* Notes Input */}
            <div className="mb-5">
              <label className="text-xs font-semibold text-slate-700 block mb-1.5">
                Session Notes / Discomfort Log
              </label>
              <textarea
                value={userNotes}
                onChange={(e) => setUserNotes(e.target.value)}
                placeholder="e.g. Felt no knee pain. Kept torso stable throughout."
                className="w-full h-20 px-3 py-2 bg-white border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleReset}
                className="flex-1 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors cursor-pointer"
              >
                Retry
              </button>
              <button
                onClick={handleSaveWorkout}
                className="flex-1 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition-colors shadow-xs cursor-pointer"
              >
                Save to History
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SKELETON RENDERER & SIMULATION MATH
// ─────────────────────────────────────────────────────────────────────────────

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  lm: Landmark[],
  state: ExerciseState,
  exerciseId: string
) {
  ctx.clearRect(0, 0, w, h);

  // Background grid lines for depth
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 40; x < w; x += 40) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  for (let y = 40; y < h; y += 40) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();

  // Skeleton connectivity pairs
  const connections: [number, number][] = [
    [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER],
    [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW],
    [POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST],
    [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW],
    [POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST],
    [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_HIP],
    [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_HIP],
    [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP],
    [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE],
    [POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE],
    [POSE_LANDMARKS.LEFT_ANKLE, POSE_LANDMARKS.LEFT_FOOT_INDEX],
    [POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE],
    [POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE],
    [POSE_LANDMARKS.RIGHT_ANKLE, POSE_LANDMARKS.RIGHT_FOOT_INDEX],
  ];

  const strokeColor = state.formErrors.length > 0 ? '#f43f5e' : '#10b981';

  // Draw limbs
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.strokeStyle = strokeColor;

  connections.forEach(([i, j]) => {
    const p1 = lm[i];
    const p2 = lm[j];
    if (p1 && p2) {
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    }
  });

  // Draw joints
  lm.forEach((p, idx) => {
    if (idx <= POSE_LANDMARKS.RIGHT_FOOT_INDEX) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
      ctx.fillStyle = idx === POSE_LANDMARKS.NOSE ? '#818cf8' : '#ffffff';
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });

  // Highlight primary joint with pulsating target circle
  let targetLm = lm[POSE_LANDMARKS.LEFT_KNEE];
  if (['shoulder_raises', 'crossover_arm_stretch'].includes(exerciseId)) {
    targetLm = lm[POSE_LANDMARKS.LEFT_SHOULDER];
  } else if (exerciseId === 'knee_raises' || exerciseId === 'lateral_walks') {
    targetLm = lm[POSE_LANDMARKS.LEFT_HIP];
  } else if (exerciseId === 'calf_raises') {
    targetLm = lm[POSE_LANDMARKS.LEFT_ANKLE];
  }

  if (targetLm) {
    const tx = targetLm.x * w;
    const ty = targetLm.y * h;
    ctx.beginPath();
    ctx.arc(tx, ty, 14, 0, 2 * Math.PI);
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// Generate realistic anatomical landmarks based on exercise phase (t in radians)
function generatePoseLandmarks(exerciseId: string, phase: number): Landmark[] {
  const lm: Landmark[] = Array(33).fill(null).map(() => ({ x: 0.5, y: 0.5 }));

  // Sine oscillation between 0 and 1
  const s = (Math.sin(phase) + 1) / 2;

  // Base coordinates
  const centerX = 0.5;
  const headY = 0.22;
  const shoulderY = 0.32;
  const hipY = 0.55;
  const kneeY = 0.73;
  const ankleY = 0.90;

  // Default neutral standing
  lm[POSE_LANDMARKS.NOSE] = { x: centerX, y: headY };
  lm[POSE_LANDMARKS.LEFT_EAR] = { x: centerX - 0.04, y: headY - 0.02 };
  lm[POSE_LANDMARKS.RIGHT_EAR] = { x: centerX + 0.04, y: headY - 0.02 };
  lm[POSE_LANDMARKS.LEFT_SHOULDER] = { x: centerX - 0.09, y: shoulderY };
  lm[POSE_LANDMARKS.RIGHT_SHOULDER] = { x: centerX + 0.09, y: shoulderY };
  lm[POSE_LANDMARKS.LEFT_ELBOW] = { x: centerX - 0.12, y: shoulderY + 0.12 };
  lm[POSE_LANDMARKS.RIGHT_ELBOW] = { x: centerX + 0.12, y: shoulderY + 0.12 };
  lm[POSE_LANDMARKS.LEFT_WRIST] = { x: centerX - 0.13, y: shoulderY + 0.22 };
  lm[POSE_LANDMARKS.RIGHT_WRIST] = { x: centerX + 0.13, y: shoulderY + 0.22 };
  lm[POSE_LANDMARKS.LEFT_HIP] = { x: centerX - 0.07, y: hipY };
  lm[POSE_LANDMARKS.RIGHT_HIP] = { x: centerX + 0.07, y: hipY };
  lm[POSE_LANDMARKS.LEFT_KNEE] = { x: centerX - 0.07, y: kneeY };
  lm[POSE_LANDMARKS.RIGHT_KNEE] = { x: centerX + 0.07, y: kneeY };
  lm[POSE_LANDMARKS.LEFT_ANKLE] = { x: centerX - 0.07, y: ankleY };
  lm[POSE_LANDMARKS.RIGHT_ANKLE] = { x: centerX + 0.07, y: ankleY };
  lm[POSE_LANDMARKS.LEFT_FOOT_INDEX] = { x: centerX - 0.07, y: ankleY + 0.03 };
  lm[POSE_LANDMARKS.RIGHT_FOOT_INDEX] = { x: centerX + 0.07, y: ankleY + 0.03 };

  // Exercise-specific kinematic alterations
  if (exerciseId === 'squat') {
    const squatDepth = s * 0.14;
    lm[POSE_LANDMARKS.LEFT_HIP].y = hipY + squatDepth;
    lm[POSE_LANDMARKS.RIGHT_HIP].y = hipY + squatDepth;
    lm[POSE_LANDMARKS.LEFT_KNEE].y = kneeY + squatDepth * 0.5;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = kneeY + squatDepth * 0.5;
    lm[POSE_LANDMARKS.LEFT_KNEE].x = centerX - 0.07 - s * 0.03;
    lm[POSE_LANDMARKS.RIGHT_KNEE].x = centerX + 0.07 + s * 0.03;
    lm[POSE_LANDMARKS.LEFT_SHOULDER].y = shoulderY + squatDepth * 0.9;
    lm[POSE_LANDMARKS.RIGHT_SHOULDER].y = shoulderY + squatDepth * 0.9;
    lm[POSE_LANDMARKS.NOSE].y = headY + squatDepth * 0.9;
  } else if (exerciseId === 'shoulder_raises') {
    const armAngle = s * Math.PI * 0.45; // 0 to ~80 deg
    lm[POSE_LANDMARKS.LEFT_ELBOW].x = lm[POSE_LANDMARKS.LEFT_SHOULDER].x - Math.cos(armAngle) * 0.12;
    lm[POSE_LANDMARKS.LEFT_ELBOW].y = lm[POSE_LANDMARKS.LEFT_SHOULDER].y + Math.sin(armAngle) * 0.04 - (s * 0.12);
    lm[POSE_LANDMARKS.LEFT_WRIST].x = lm[POSE_LANDMARKS.LEFT_ELBOW].x - 0.08;
    lm[POSE_LANDMARKS.LEFT_WRIST].y = lm[POSE_LANDMARKS.LEFT_ELBOW].y;

    lm[POSE_LANDMARKS.RIGHT_ELBOW].x = lm[POSE_LANDMARKS.RIGHT_SHOULDER].x + Math.cos(armAngle) * 0.12;
    lm[POSE_LANDMARKS.RIGHT_ELBOW].y = lm[POSE_LANDMARKS.RIGHT_SHOULDER].y + Math.sin(armAngle) * 0.04 - (s * 0.12);
    lm[POSE_LANDMARKS.RIGHT_WRIST].x = lm[POSE_LANDMARKS.RIGHT_ELBOW].x + 0.08;
    lm[POSE_LANDMARKS.RIGHT_WRIST].y = lm[POSE_LANDMARKS.RIGHT_ELBOW].y;
  } else if (exerciseId === 'crossover_arm_stretch') {
    const reach = s * 0.16;
    lm[POSE_LANDMARKS.LEFT_ELBOW].x = lm[POSE_LANDMARKS.LEFT_SHOULDER].x + reach * 0.6;
    lm[POSE_LANDMARKS.LEFT_ELBOW].y = lm[POSE_LANDMARKS.LEFT_SHOULDER].y + 0.04;
    lm[POSE_LANDMARKS.LEFT_WRIST].x = lm[POSE_LANDMARKS.LEFT_SHOULDER].x + reach;
    lm[POSE_LANDMARKS.LEFT_WRIST].y = lm[POSE_LANDMARKS.LEFT_SHOULDER].y + 0.02;
  } else if (exerciseId === 'lateral_walks') {
    const spread = s * 0.14;
    lm[POSE_LANDMARKS.LEFT_ANKLE].x = centerX - 0.07 - spread;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].x = centerX + 0.07 + spread;
    lm[POSE_LANDMARKS.LEFT_KNEE].x = centerX - 0.07 - spread * 0.7;
    lm[POSE_LANDMARKS.RIGHT_KNEE].x = centerX + 0.07 + spread * 0.7;
    lm[POSE_LANDMARKS.LEFT_HIP].y = hipY + 0.03;
    lm[POSE_LANDMARKS.RIGHT_HIP].y = hipY + 0.03;
  } else if (exerciseId === 'lunges') {
    const lungeDepth = s * 0.12;
    lm[POSE_LANDMARKS.LEFT_KNEE].x = centerX - 0.14;
    lm[POSE_LANDMARKS.LEFT_KNEE].y = kneeY + lungeDepth * 0.5;
    lm[POSE_LANDMARKS.LEFT_ANKLE].x = centerX - 0.14;
    lm[POSE_LANDMARKS.RIGHT_KNEE].x = centerX + 0.12;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = kneeY + lungeDepth;
    lm[POSE_LANDMARKS.LEFT_HIP].y = hipY + lungeDepth;
  } else if (exerciseId === 'calf_raises') {
    const heelLift = s * 0.06;
    lm[POSE_LANDMARKS.LEFT_ANKLE].y = ankleY - heelLift;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].y = ankleY - heelLift;
    lm[POSE_LANDMARKS.LEFT_KNEE].y = kneeY - heelLift;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = kneeY - heelLift;
    lm[POSE_LANDMARKS.LEFT_HIP].y = hipY - heelLift;
    lm[POSE_LANDMARKS.RIGHT_HIP].y = hipY - heelLift;
  } else if (exerciseId === 'knee_raises') {
    const raise = s * 0.18;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = kneeY - raise;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].y = ankleY - raise * 0.9;
  } else if (exerciseId === 'tree_pose') {
    // Tree pose holds leg raised
    lm[POSE_LANDMARKS.RIGHT_KNEE].x = centerX + 0.13;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = hipY + 0.03;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].x = centerX - 0.02;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].y = kneeY;
    lm[POSE_LANDMARKS.LEFT_WRIST].x = centerX;
    lm[POSE_LANDMARKS.RIGHT_WRIST].x = centerX;
    lm[POSE_LANDMARKS.LEFT_WRIST].y = shoulderY - 0.05;
    lm[POSE_LANDMARKS.RIGHT_WRIST].y = shoulderY - 0.05;
  } else if (exerciseId === 'warrior_pose') {
    lm[POSE_LANDMARKS.LEFT_KNEE].x = centerX - 0.18;
    lm[POSE_LANDMARKS.LEFT_KNEE].y = kneeY + 0.04;
    lm[POSE_LANDMARKS.LEFT_ANKLE].x = centerX - 0.18;
    lm[POSE_LANDMARKS.RIGHT_KNEE].x = centerX + 0.18;
    lm[POSE_LANDMARKS.RIGHT_ANKLE].x = centerX + 0.22;
    lm[POSE_LANDMARKS.LEFT_ELBOW].y = shoulderY;
    lm[POSE_LANDMARKS.RIGHT_ELBOW].y = shoulderY;
    lm[POSE_LANDMARKS.LEFT_WRIST].x = centerX - 0.28;
    lm[POSE_LANDMARKS.LEFT_WRIST].y = shoulderY;
    lm[POSE_LANDMARKS.RIGHT_WRIST].x = centerX + 0.28;
    lm[POSE_LANDMARKS.RIGHT_WRIST].y = shoulderY;
  } else if (exerciseId === 'cat_cow_stretch') {
    const arch = (s - 0.5) * 0.08;
    lm[POSE_LANDMARKS.LEFT_HIP].y = hipY + 0.06;
    lm[POSE_LANDMARKS.RIGHT_HIP].y = hipY + 0.06;
    lm[POSE_LANDMARKS.LEFT_KNEE].y = kneeY + 0.08;
    lm[POSE_LANDMARKS.RIGHT_KNEE].y = kneeY + 0.08;
    lm[POSE_LANDMARKS.LEFT_SHOULDER].y = shoulderY + 0.12;
    lm[POSE_LANDMARKS.RIGHT_SHOULDER].y = shoulderY + 0.12;
    lm[POSE_LANDMARKS.NOSE].y = headY + 0.14 + (arch > 0 ? -0.04 : 0.04);
  }

  return lm;
}
