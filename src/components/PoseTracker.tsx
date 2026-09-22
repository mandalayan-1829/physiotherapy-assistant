import { useCallback, useEffect, useRef, useState } from 'react';
import confetti from 'canvas-confetti';
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  Clock,
  Dumbbell,
  Hand,
  Info,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  VideoOff,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { Exercise, ExerciseState, Landmark, MetricsSource, Session } from '../types';
import { ExerciseEngine } from '../utils/exerciseEngine';
import { POSE_LANDMARKS } from '../utils/angleCalculator';
import { soundManager } from '../utils/audio';
import { PoseInferenceError, createPoseEstimator } from '../utils/poseEstimator';

interface PoseTrackerProps {
  exercise: Exercise;
  targetReps: number;
  onSessionComplete: (session: Omit<Session, 'id'>) => void;
  onCancel: () => void;
}

/**
 * How the session is being recorded.
 *
 *  - `camera`  real pose inference runs on this device and produces a form score
 *  - `guided`  a self-paced timer with manually logged repetitions; no camera
 *              analysis happens, so no form score is claimed
 *
 * The two are never blended: metrics from a guided session are stored as
 * `manual` and are excluded from clinical aggregates by the backend.
 */
type TrackerMode = 'idle' | 'camera' | 'guided';

type InferenceStatus = 'idle' | 'loading' | 'ready' | 'error';

export function PoseTracker({ exercise, targetReps, onSessionComplete, onCancel }: PoseTrackerProps) {
  const [mode, setMode] = useState<TrackerMode>('idle');
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus>('idle');
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  /** True while the most recent frame contained a usable, complete pose. */
  const [poseVisible, setPoseVisible] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [durationSec, setDurationSec] = useState<number>(0);
  const [showSummary, setShowSummary] = useState<boolean>(false);
  const [userNotes, setUserNotes] = useState<string>('');
  const [manualReps, setManualReps] = useState<number>(0);
  const [soundOn, setSoundOn] = useState<boolean>(soundManager.enabled);

  // Engine instance
  const engineRef = useRef<ExerciseEngine>(new ExerciseEngine(exercise.id, targetReps));
  const [state, setState] = useState<ExerciseState>(engineRef.current.getState());

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const estimatorRef = useRef<ReturnType<typeof createPoseEstimator> | null>(null);

  // Accuracy tracking. Only frames that carried a usable pose are counted, so the
  // score is a share of *measured* frames rather than of wall-clock time.
  const measuredFramesRef = useRef<number>(0);
  const goodFramesRef = useRef<number>(0);

  const isCamera = mode === 'camera';
  const isGuided = mode === 'guided';

  // Duration timer
  useEffect(() => {
    if (mode === 'idle' || !isRunning || showSummary) return;
    const timer = setInterval(() => setDurationSec((prev) => prev + 1), 1000);
    return () => clearInterval(timer);
  }, [mode, isRunning, showSummary]);

  // --- Camera + model lifecycle ---------------------------------------------
  useEffect(() => {
    if (mode !== 'camera') {
      // Release the camera the moment we leave camera mode.
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      return;
    }

    let cancelled = false;
    setInferenceStatus('loading');
    setInferenceError(null);

    const start = async () => {
      // 1. Camera first, so the patient sees themselves immediately.
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('getUserMedia is not available in this browser.');
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setCameraError(null);
      } catch {
        if (cancelled) return;
        setCameraError(
          'Camera access was denied or is unavailable. Use Guided practice, or allow camera access and retry.',
        );
        setInferenceStatus('error');
        return;
      }

      // 2. Then the pose model. Failures are surfaced, never silently replaced
      //    with synthetic landmarks.
      try {
        if (!estimatorRef.current) {
          estimatorRef.current = createPoseEstimator();
        }
        await estimatorRef.current.initialise();
        if (cancelled) return;
        setInferenceStatus('ready');
      } catch (error) {
        if (cancelled) return;
        setInferenceStatus('error');
        setInferenceError(
          error instanceof PoseInferenceError
            ? error.message
            : 'Pose analysis could not be started on this device.',
        );
      }
    };

    void start();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [mode]);

  // Release the model when the component goes away.
  useEffect(() => {
    return () => {
      estimatorRef.current?.close();
      estimatorRef.current = null;
    };
  }, []);

  // --- Main analysis + rendering loop ---------------------------------------
  //
  // The latest engine state is mirrored in a ref so the animation loop does not
  // have to be torn down and rebuilt on every frame.
  const stateRef = useRef<ExerciseState>(state);

  useEffect(() => {
    if (mode === 'idle') return;

    const loop = (time: number) => {
      /** Anything the model detected, drawn on the canvas for feedback. */
      let drawnLandmarks: Landmark[] | null = null;
      /** Only a complete pose, handed to the engine as a measurement. */
      let measuredLandmarks: Landmark[] | null = null;

      if (isRunning && !showSummary) {
        if (isCamera && inferenceStatus === 'ready' && estimatorRef.current && videoRef.current) {
          try {
            const frame = estimatorRef.current.detect(videoRef.current, time);
            if (frame) {
              drawnLandmarks = frame.landmarks;
              if (frame.usable) {
                measuredLandmarks = frame.landmarks;
                measuredFramesRef.current += 1;
              }
            }
          } catch (error) {
            // An inference failure mid-session is surfaced immediately rather
            // than being papered over with a stand-in pose.
            setInferenceStatus('error');
            setInferenceError(
              error instanceof PoseInferenceError
                ? error.message
                : 'Pose analysis stopped unexpectedly.',
            );
          }
        }

        setPoseVisible(measuredLandmarks !== null);

        // A frame without a complete pose is passed through as `null`, which the
        // engine treats as "no measurement" - it cannot produce a rep.
        const newState = engineRef.current.process(measuredLandmarks);
        stateRef.current = newState;
        setState({ ...newState });
        if (measuredLandmarks && newState.formOk) {
          goodFramesRef.current += 1;
        }

        if (newState.isComplete && !showSummary) {
          setShowSummary(true);
          setIsRunning(false);
          confetti({ particleCount: 80, spread: 70, origin: { y: 0.6 } });
        }
      }

      // Render whatever the model actually found. With nothing found, the stage
      // shows a waiting message instead of an invented skeleton.
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          drawStage(
            ctx,
            canvas.width,
            canvas.height,
            drawnLandmarks,
            stateRef.current,
            exercise.id,
          );
        }
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [mode, isRunning, showSummary, inferenceStatus, isCamera, exercise.id]);

  const handleReset = useCallback(() => {
    engineRef.current.reset();
    setState(engineRef.current.getState());
    measuredFramesRef.current = 0;
    goodFramesRef.current = 0;
    setManualReps(0);
    setDurationSec(0);
    setShowSummary(false);
    setIsRunning(true);
  }, []);

  const handleSwitchExercise = useCallback(
    (next: TrackerMode) => {
      setMode(next);
      setCameraError(null);
      setInferenceError(null);
      setInferenceStatus(next === 'camera' ? 'loading' : 'idle');
      setPoseVisible(false);
      handleReset();
    },
    [handleReset],
  );

  const handleLogManualRep = () => {
    setManualReps((previous) => previous + 1);
    soundManager.playRepSuccess();
  };

  /**
   * The form score, or `null` when nothing was actually measured.
   *
   * A number is only reported when real pose inference produced usable frames.
   * There is no default or fallback value.
   */
  const formAccuracy: number | null =
    isCamera && measuredFramesRef.current > 0
      ? Math.round((goodFramesRef.current / measuredFramesRef.current) * 100)
      : null;

  const repsRecorded = isGuided ? manualReps : state.reps;
  const totalReps = isGuided ? Math.max(manualReps, targetReps) : state.targetReps;
  const metricsSource: MetricsSource =
    isCamera && measuredFramesRef.current > 0 ? 'pose_inference' : 'manual';
  const wasMeasured = metricsSource === 'pose_inference';

  const handleSaveWorkout = () => {
    const accuracy = formAccuracy ?? 0;
    const notes =
      userNotes ||
      (wasMeasured
        ? `${exercise.label}: ${repsRecorded} reps, ${accuracy}% form score from on-device pose analysis.`
        : `${exercise.label}: ${repsRecorded} reps recorded manually. No camera analysis was performed.`);

    onSessionComplete({
      userId: 1,
      exercise: exercise.id,
      exerciseLabel: exercise.label,
      reps: repsRecorded,
      targetReps: isGuided ? repsRecorded : state.targetReps,
      formAccuracy: accuracy,
      durationSec,
      notes,
      metricsSource,
      date: new Date().toISOString().replace('T', ' ').slice(0, 16),
    });
  };

  // --- Idle: choose how to record the session -------------------------------
  if (mode === 'idle') {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 shadow-sm">
        <div className="flex items-center gap-3 pb-5 border-b border-slate-200">
          <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 border border-blue-100 flex items-center justify-center">
            <Dumbbell className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">{exercise.label}</h2>
            <p className="text-xs text-slate-500">
              {exercise.target} • {exercise.primaryJoint} • target {targetReps} reps
            </p>
          </div>
        </div>

        <div className="mt-5 p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 flex items-start gap-2.5">
          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-slate-800">Choose how this session is recorded</p>
            <p className="mt-1 leading-relaxed">
              Pose analysis runs entirely on this device using MediaPipe. Camera frames are
              processed in the browser and are never uploaded or stored. Only the finished
              session totals are saved to your record.
            </p>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => handleSwitchExercise('camera')}
            className="text-left p-5 rounded-xl border-2 border-slate-200 hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
          >
            <div className="w-11 h-11 rounded-lg bg-[#F0F7FF] border border-blue-100 flex items-center justify-center text-blue-600 mb-3">
              <Camera className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900">Camera analysis</h3>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              Real pose tracking with automatic repetition counting and live form feedback.
              Produces a form score measured from your movement.
            </p>
          </button>

          <button
            onClick={() => handleSwitchExercise('guided')}
            className="text-left p-5 rounded-xl border-2 border-slate-200 hover:border-emerald-500 hover:shadow-md transition-all cursor-pointer"
          >
            <div className="w-11 h-11 rounded-lg bg-[#ECFDF3] border border-emerald-100 flex items-center justify-center text-emerald-600 mb-3">
              <Hand className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900">Guided practice (no camera)</h3>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              A timer and manual repetition counter. No pose analysis is performed, so no form
              score is produced or recorded.
            </p>
          </button>
        </div>

        <button
          onClick={onCancel}
          className="mt-5 px-3.5 py-2 rounded-lg bg-white hover:bg-slate-50 text-slate-600 border border-slate-200 text-xs font-medium transition-colors cursor-pointer"
        >
          Back to exercises
        </button>
      </div>
    );
  }

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
              <span
                className={`px-2 py-0.5 rounded text-xs font-semibold border ${
                  isCamera
                    ? 'bg-[#ECFDF3] text-[#065F46] border-[#A7F3D0]'
                    : 'bg-[#FFF8E6] text-amber-900 border-amber-200'
                }`}
              >
                {isCamera ? 'Camera analysis' : 'Guided practice · no analysis'}
              </span>
            </div>
            <p className="text-xs text-slate-500">{exercise.target} • {exercise.primaryJoint}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isCamera && (
            <button
              onClick={() => handleSwitchExercise('guided')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-white text-slate-700 border-slate-200 hover:bg-slate-50 transition-colors cursor-pointer"
            >
              <Hand className="w-3.5 h-3.5 text-slate-400" />
              <span>Switch to guided</span>
            </button>
          )}
          {isGuided && (
            <button
              onClick={() => handleSwitchExercise('camera')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-white text-slate-700 border-slate-200 hover:bg-slate-50 transition-colors cursor-pointer"
            >
              <Camera className="w-3.5 h-3.5 text-slate-400" />
              <span>Enable camera analysis</span>
            </button>
          )}

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

          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors cursor-pointer"
            title="Reset session counter"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg bg-white hover:bg-rose-50 text-slate-600 hover:text-rose-700 border border-slate-200 text-xs font-medium transition-colors cursor-pointer"
          >
            Exit
          </button>
        </div>
      </div>

      {/* Status banners */}
      {cameraError && (
        <div className="mt-3 p-2.5 rounded-lg bg-[#FFF8E6] border border-amber-200 text-amber-900 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-600" />
          <span>{cameraError}</span>
        </div>
      )}

      {inferenceError && (
        <div className="mt-3 p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
          <span>{inferenceError}</span>
        </div>
      )}

      {isCamera && inferenceStatus === 'loading' && !cameraError && (
        <div className="mt-3 p-2.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-xs flex items-center gap-2">
          <Loader2 className="w-4 h-4 shrink-0 text-blue-600 animate-spin" />
          <span>Starting on-device pose analysis…</span>
        </div>
      )}

      {isCamera && inferenceStatus === 'ready' && !poseVisible && !showSummary && (
        <div className="mt-3 p-2.5 rounded-lg bg-[#FFF8E6] border border-amber-200 text-amber-900 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-600" />
          <span>
            No complete body pose detected — step back so your whole body is in frame.
            Repetitions are only counted while your pose is tracked.
          </span>
        </div>
      )}

      {isGuided && (
        <div className="mt-3 p-2.5 rounded-lg bg-[#FFF8E6] border border-amber-200 text-amber-900 text-xs flex items-center gap-2">
          <Info className="w-4 h-4 shrink-0 text-amber-600" />
          <span>
            Guided practice performs <strong>no camera analysis</strong>. Repetitions are logged by
            you, no form score is produced, and this session is stored as manually recorded.
          </span>
        </div>
      )}

      {/* Main Visual Stage */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-5">

        <div className="lg:col-span-8 relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
          {isCamera && (
            <video
              ref={videoRef}
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover -scale-x-100 opacity-60"
            />
          )}

          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="relative z-10 w-full h-full object-contain"
          />

          {/* Guided mode replaces the stage with a manual counter */}
          {isGuided && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 text-center px-6">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                Manual repetition counter
              </p>
              <div className="text-6xl font-extrabold font-mono text-white">{manualReps}</div>
              <button
                onClick={handleLogManualRep}
                className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition-colors cursor-pointer"
              >
                Log one repetition
              </button>
              <p className="text-[11px] text-slate-400 max-w-sm">
                No pose analysis is running in this mode.
              </p>
            </div>
          )}

          {isCamera && (
            <>
              <div className="absolute top-3 left-3 z-20 flex items-center gap-3">
                <div className="px-3.5 py-1.5 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/80 shadow-lg">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold block">
                    Joint Angle
                  </span>
                  <span className="text-xl font-bold font-mono text-white">
                    {poseVisible ? `${state.angle}°` : '--'}
                  </span>
                </div>

                <div className={`px-3.5 py-1.5 rounded-xl backdrop-blur-md border shadow-lg ${
                  state.stage === 'down'
                    ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                    : state.stage === 'holding'
                    ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                    : 'bg-slate-900/85 border-slate-700/80 text-emerald-400'
                }`}>
                  <span className="text-[10px] uppercase tracking-wider font-semibold block">Stage</span>
                  <span className="text-sm font-bold uppercase">{poseVisible ? state.stage : '--'}</span>
                </div>
              </div>

              <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
                <div className="px-3 py-1.5 rounded-xl bg-slate-900/85 backdrop-blur-md border border-slate-700/80 text-xs font-mono text-slate-200 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-indigo-400" />
                  <span>{Math.floor(durationSec / 60)}:{(durationSec % 60).toString().padStart(2, '0')}</span>
                </div>
              </div>

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

              <div className="absolute bottom-3 left-3 right-3 z-20">
                <div className={`p-3 rounded-xl backdrop-blur-md border transition-all flex items-center gap-3 ${
                  !poseVisible
                    ? 'bg-slate-900/85 border-slate-700/80 text-slate-200'
                    : state.formErrors.length > 0
                    ? 'bg-rose-950/80 border-rose-500/50 text-rose-200'
                    : 'bg-emerald-950/80 border-emerald-500/50 text-emerald-200'
                }`}>
                  {!poseVisible ? (
                    <AlertCircle className="w-5 h-5 shrink-0 text-slate-400" />
                  ) : state.formErrors.length > 0 ? (
                    <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 animate-pulse" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-400" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="text-xs sm:text-sm font-semibold truncate">
                      {poseVisible ? state.feedback : 'Waiting for a full-body pose…'}
                    </p>
                    {poseVisible && state.formErrors.length > 0 && (
                      <p className="text-[11px] text-rose-300 truncate mt-0.5">{state.formErrors[0]}</p>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Right Metric Panel */}
        <div className="lg:col-span-4 flex flex-col justify-between gap-4">

          <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between">
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
                {isGuided ? 'Logged Reps' : 'Completed Reps'}
              </span>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-4xl font-extrabold text-slate-900 font-mono">{repsRecorded}</span>
                <span className="text-base text-slate-500 font-semibold font-mono">/ {targetReps}</span>
              </div>
            </div>
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
                  strokeDasharray={`${Math.min(100, (repsRecorded / Math.max(1, totalReps)) * 100)}, 100`}
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="none"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
              </svg>
              <span className="absolute text-xs font-bold text-slate-900 font-mono">
                {Math.round((repsRecorded / Math.max(1, totalReps)) * 100)}%
              </span>
            </div>
          </div>

          {/* Form score — only shown when something was actually measured */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold block">
              Form score
            </span>
            {formAccuracy === null ? (
              <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">
                Not measured.{' '}
                {isGuided
                  ? 'Guided practice does not analyse your movement.'
                  : 'No complete pose has been tracked yet.'}
              </p>
            ) : (
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-2xl font-bold font-mono text-emerald-600">{formAccuracy}%</span>
                <span className="text-[11px] text-slate-500">
                  from {measuredFramesRef.current} analysed frames
                </span>
              </div>
            )}
          </div>

          {/* Form criteria */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex-1">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Biomechanical Checks</span>
            </h3>

            {!isCamera ? (
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Not assessed — these checks require camera analysis. Refer to the descriptions below
                while you practise.
              </p>
            ) : (
              <ul className="space-y-2">
                {exercise.formChecks.map((check, idx) => {
                  const isFailed = state.formErrors.some((err) =>
                    err.toLowerCase().includes(check.toLowerCase()),
                  );
                  return (
                    <li
                      key={idx}
                      className="flex items-center justify-between text-xs py-1 border-b border-slate-200 last:border-0"
                    >
                      <span className="text-slate-700">{check}</span>
                      {!poseVisible ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-500 border border-slate-200">
                          Not assessed
                        </span>
                      ) : isFailed ? (
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
            )}

            <div className="mt-4 pt-3 border-t border-slate-200 text-[11px] text-slate-500">
              <p className="font-semibold text-slate-700">Optimal Range:</p>
              <p className="text-slate-600">{exercise.idealAngleRange}</p>
              <p className="mt-2 text-blue-700 font-medium italic">Clinical Tip: {exercise.tip}</p>
            </div>
          </div>

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

      {/* Summary dialog */}
      {showSummary && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4">
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-2xl">
            <div className="w-12 h-12 rounded-2xl bg-[#ECFDF3] text-emerald-600 border border-[#A7F3D0] flex items-center justify-center mx-auto mb-4">
              <CheckCircle2 className="w-6 h-6" />
            </div>

            <h3 className="text-xl font-bold text-center text-slate-900">Session complete</h3>
            <p className="text-xs text-center text-slate-500 mt-1">
              {wasMeasured
                ? 'Your movement was analysed on this device.'
                : 'No movement analysis was performed for this session.'}
            </p>

            <div className="grid grid-cols-3 gap-3 my-5">
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Reps</span>
                <span className="text-xl font-bold text-slate-900 font-mono">{repsRecorded}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Form Score</span>
                <span className={`font-mono ${formAccuracy === null ? 'text-sm text-slate-500' : 'text-xl font-bold text-emerald-600'}`}>
                  {formAccuracy === null ? 'Not measured' : `${formAccuracy}%`}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
                <span className="text-[10px] text-slate-500 font-semibold uppercase block">Time</span>
                <span className="text-xl font-bold text-slate-900 font-mono">
                  {Math.floor(durationSec / 60)}:{(durationSec % 60).toString().padStart(2, '0')}
                </span>
              </div>
            </div>

            <div className={`mb-4 p-2.5 rounded-lg text-[11px] flex items-start gap-2 ${
              wasMeasured
                ? 'bg-[#ECFDF3] border border-[#A7F3D0] text-[#065F46]'
                : 'bg-[#FFF8E6] border border-amber-200 text-amber-900'
            }`}>
              {wasMeasured ? (
                <>
                  <ShieldCheck className="w-3.5 h-3.5 shrink-0 mt-0.5 text-emerald-600" />
                  <span>
                    Recorded as a camera-measured session ({measuredFramesRef.current} analysed
                    frames). It is included in your rehabilitation analytics.
                  </span>
                </>
              ) : (
                <>
                  <Info className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-600" />
                  <span>
                    Recorded as a manually logged session. It is kept in your history but is
                    excluded from form-score analytics, because no measurement was taken.
                  </span>
                </>
              )}
            </div>

            <div className="mb-5">
              <label className="text-xs font-semibold text-slate-700 block mb-1.5">
                Session Notes / Discomfort Log
              </label>
              <textarea
                value={userNotes}
                onChange={(event) => setUserNotes(event.target.value)}
                placeholder="e.g. Felt no knee pain. Kept torso stable throughout."
                className="w-full h-20 px-3 py-2 bg-white border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
              />
            </div>

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
// SKELETON RENDERER
// Draws the landmarks returned by real pose inference. With no landmarks it
// renders only the reference grid — it never invents a skeleton.
// ─────────────────────────────────────────────────────────────────────────────

function drawStage(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  lm: Landmark[] | null,
  state: ExerciseState,
  exerciseId: string,
) {
  ctx.clearRect(0, 0, w, h);

  // Background grid for depth reference
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

  if (!lm || lm.length === 0) {
    ctx.fillStyle = '#475569';
    ctx.font = '600 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for a full-body pose…', w / 2, h / 2);
    ctx.textAlign = 'start';
    return;
  }

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

  let targetLm = lm[POSE_LANDMARKS.LEFT_KNEE];
  if (['shoulder_raises', 'crossover_arm_stretch'].includes(exerciseId)) {
    targetLm = lm[POSE_LANDMARKS.LEFT_SHOULDER];
  } else if (exerciseId === 'knee_raises' || exerciseId === 'lateral_walks') {
    targetLm = lm[POSE_LANDMARKS.LEFT_HIP];
  } else if (exerciseId === 'calf_raises') {
    targetLm = lm[POSE_LANDMARKS.LEFT_ANKLE];
  }

  if (targetLm) {
    ctx.beginPath();
    ctx.arc(targetLm.x * w, targetLm.y * h, 14, 0, 2 * Math.PI);
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
