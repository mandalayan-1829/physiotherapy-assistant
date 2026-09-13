import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import {
  Camera,
  CameraOff,
  RotateCcw,
  Pause,
  Play,
  Square,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import { useSession } from "@/hooks/useSession";
import { fetchExercise } from "@/services/exerciseService";
import { createSession, endSession } from "@/services/sessionService";
import type { Exercise } from "@/types/exercise";
import StatusBadge from "@/components/common/StatusBadge";

export default function LiveSession() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const targetReps = (location.state as any)?.targetReps || 10;

  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);

  const {
    poseState,
    isConnected,
    startSession,
    sendFrame,
    endSession: endWsSession,
    resetSession,
    formAccuracy,
    elapsedSeconds,
  } = useSession();

  useEffect(() => {
    if (id) {
      fetchExercise(id).then(setExercise).catch(() => navigate("/exercises"));
    }
  }, [id]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const startCamera = useCallback(async () => {
    setSessionError(null);

    // Step 1: Request camera access
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
    } catch (err) {
      console.error("Camera access denied:", err);
      setSessionError(
        "Camera access was denied. Please allow camera permissions in your browser settings and try again."
      );
      return;
    }

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }

    // Step 2: Create backend session (may fail — camera still works)
    try {
      if (id) {
        const session = await createSession(id, targetReps);
        setSessionId(session.id);
      }
    } catch (err) {
      console.error("Failed to create session on server:", err);
      setSessionError(
        "Could not connect to the server to save session data. Camera will work but progress won't be saved."
      );
    }

    // Step 3: Start WebSocket session for real-time analysis
    startSession(id || "unknown", targetReps);
    setCameraActive(true);

    // Start frame capture loop
    const captureFrame = () => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
          sendFrame(dataUrl);
        }
      }
      animFrameRef.current = requestAnimationFrame(captureFrame);
    };

    // Start capturing after a short delay to let video load
    setTimeout(captureFrame, 500);
  }, [id, targetReps, sendFrame, startSession]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    setCameraActive(false);
  }, []);

  const handleEndSession = useCallback(async () => {
    endWsSession();
    stopCamera();
    if (sessionId) {
      await endSession(sessionId, poseState?.rep_count || 0, formAccuracy, elapsedSeconds);
    }
    navigate(`/exercises/${id}/results`, {
      state: {
        reps: poseState?.rep_count || 0,
        targetReps,
        formAccuracy,
        duration: elapsedSeconds,
        exercise: exercise,
      },
    });
  }, [sessionId, poseState, formAccuracy, elapsedSeconds, targetReps, exercise, id, navigate, endWsSession, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const formStatus = poseState?.form_status || "no_pose";
  const progress = poseState ? Math.min(poseState.rep_count / targetReps, 1) : 0;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Session Error Banner */}
      {sessionError && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-2xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
          <p className="text-sm text-red-700">{sessionError}</p>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{exercise?.icon}</span>
          <div>
            <h1 className="text-xl font-bold text-surface-900">{exercise?.label} — Live Session</h1>
            <p className="text-sm text-surface-500">Target: {targetReps} reps</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={resetSession}
            className="px-4 py-2 text-sm font-medium border border-surface-200 rounded-xl hover:bg-surface-50 transition-colors flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          <button
            onClick={() => navigate("/exercises")}
            className="px-4 py-2 text-sm font-medium border border-surface-200 rounded-xl hover:bg-surface-50 transition-colors"
          >
            Switch Exercise
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_340px] gap-6">
        {/* Camera Area */}
        <div className="relative">
          <div className="bg-surface-900 rounded-2xl overflow-hidden aspect-video relative">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              playsInline
              muted
              style={{ display: cameraActive ? "block" : "none" }}
            />
            <canvas ref={canvasRef} className="hidden" />

            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
                <Camera className="w-16 h-16 mb-4 text-surface-500" />
                <p className="text-surface-400 mb-2">Camera preview will appear here</p>
                <p className="text-surface-600 text-sm">Step back so your full body is visible</p>
              </div>
            )}

            {/* Alarm overlay */}
            {poseState?.alarm_active && (
              <div className="absolute inset-0 border-4 border-red-500 rounded-2xl animate-pulse pointer-events-none" />
            )}

            {/* Form status overlay */}
            {cameraActive && (
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
                <div
                  className={`px-3 py-1.5 rounded-lg text-sm font-bold ${
                    formStatus === "good"
                      ? "bg-emerald-500 text-white"
                      : formStatus === "warning"
                      ? "bg-amber-500 text-white"
                      : formStatus === "incorrect"
                      ? "bg-red-500 text-white"
                      : "bg-surface-700 text-surface-300"
                  }`}
                >
                  {formStatus === "good" && "✅ GOOD FORM"}
                  {formStatus === "warning" && "⚠️ FIX POSTURE"}
                  {formStatus === "incorrect" && "❌ INCORRECT"}
                  {formStatus === "no_pose" && "👀 NO POSE"}
                </div>
              </div>
            )}
          </div>

          {/* Camera controls */}
          <div className="flex items-center justify-center gap-4 mt-4">
            {!cameraActive ? (
              <button
                onClick={startCamera}
                className="px-8 py-3 bg-primary-500 text-white rounded-xl font-semibold text-sm hover:bg-primary-600 transition-all shadow-lg shadow-primary-200 flex items-center gap-2"
              >
                <Camera className="w-5 h-5" />
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={handleEndSession}
                  className="px-6 py-3 bg-red-500 text-white rounded-xl font-semibold text-sm hover:bg-red-600 transition-all flex items-center gap-2"
                >
                  <Square className="w-4 h-4" />
                  End Session
                </button>
              </>
            )}
          </div>
        </div>

        {/* Stats Panel */}
        <div className="space-y-4">
          {/* Reps */}
          <div className="bg-white rounded-2xl border border-surface-200 p-5">
            <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Reps</p>
            <p className="text-3xl font-extrabold text-surface-900">
              {poseState?.rep_count ?? 0}
              <span className="text-lg text-surface-400 font-normal ml-1">/ {targetReps}</span>
            </p>
            <div className="w-full bg-surface-100 rounded-full h-2 mt-3">
              <div
                className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          {/* Angle + Stage */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white rounded-2xl border border-surface-200 p-4">
              <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Angle</p>
              <p className="text-2xl font-bold text-surface-900">{poseState?.angle ?? 0}°</p>
            </div>
            <div className="bg-white rounded-2xl border border-surface-200 p-4">
              <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Stage</p>
              <p className="text-2xl font-bold text-surface-900 uppercase">{poseState?.stage ?? "—"}</p>
            </div>
          </div>

          {/* Time + Form */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white rounded-2xl border border-surface-200 p-4">
              <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Time</p>
              <p className="text-xl font-bold text-surface-900">{formatTime(elapsedSeconds)}</p>
            </div>
            <div className="bg-white rounded-2xl border border-surface-200 p-4">
              <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">Form</p>
              <StatusBadge status={formStatus} />
            </div>
          </div>

          {/* Feedback */}
          <div className="bg-white rounded-2xl border border-surface-200 p-5">
            <p className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-2">AI Feedback</p>
            <p className="text-sm text-surface-700 leading-relaxed">
              {poseState?.feedback || "Start the camera to receive real-time feedback."}
            </p>
          </div>

          {/* Form Errors */}
          {poseState?.form_errors && poseState.form_errors.length > 0 && (
            <div className="bg-red-50 rounded-2xl border border-red-200 p-5">
              <p className="text-xs font-semibold text-red-600 uppercase tracking-wider mb-2">Posture Issues</p>
              <ul className="space-y-1">
                {poseState.form_errors.map((err, i) => (
                  <li key={i} className="text-sm text-red-600">{err}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
