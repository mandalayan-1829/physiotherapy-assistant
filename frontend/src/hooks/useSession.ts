import { useState, useCallback, useRef, useEffect } from "react";
import type { PoseState } from "@/types/exercise";
import { getWsUrl } from "@/services/api";
import { isMockMode } from "@/services/api";

interface UseSessionReturn {
  poseState: PoseState | null;
  isConnected: boolean;
  startSession: (exercise: string, targetReps: number) => void;
  sendFrame: (base64Frame: string) => void;
  endSession: () => void;
  resetSession: () => void;
  formAccuracy: number;
  elapsedSeconds: number;
}

const MOCK_POSE: PoseState = {
  rep_count: 0,
  angle: 0,
  stage: "up",
  feedback: "Get ready to start!",
  form_status: "no_pose",
  form_errors: [],
  form_ok: true,
  exercise: "squat",
  hold_count: 0,
  alarm_active: false,
};

export function useSession(): UseSessionReturn {
  const [poseState, setPoseState] = useState<PoseState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [formAccuracy, setFormAccuracy] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>("");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startSession = useCallback((exercise: string, targetReps: number) => {
    if (isMockMode()) {
      sessionIdRef.current = "mock-session";
      setIsConnected(true);
      setPoseState({ ...MOCK_POSE, exercise });
      setElapsedSeconds(0);
      intervalRef.current = setInterval(() => {
        setElapsedSeconds((s) => s + 1);
      }, 1000);
      return;
    }

    const sessionId = `session-${Date.now()}`;
    sessionIdRef.current = sessionId;
    const ws = new WebSocket(getWsUrl(sessionId));

    ws.onopen = () => {
      setIsConnected(true);
      setElapsedSeconds(0);
      intervalRef.current = setInterval(() => {
        setElapsedSeconds((s) => s + 1);
      }, 1000);
      ws.send(JSON.stringify({ type: "start", exercise, target_reps: targetReps }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "analysis") {
        setPoseState(msg as PoseState);
      } else if (msg.type === "completed") {
        setFormAccuracy(msg.form_accuracy);
        setPoseState((prev) => (prev ? { ...prev, rep_count: msg.rep_count } : prev));
      } else if (msg.type === "ended") {
        setFormAccuracy(msg.form_accuracy);
        setIsConnected(false);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };

    wsRef.current = ws;
  }, []);

  const sendFrame = useCallback((base64Frame: string) => {
    if (isMockMode()) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "frame", data: base64Frame }));
    }
  }, []);

  const endSession = useCallback(() => {
    if (isMockMode()) {
      setIsConnected(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end" }));
    }
    setIsConnected(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  }, []);

  const resetSession = useCallback(() => {
    setPoseState(null);
    setFormAccuracy(0);
    setElapsedSeconds(0);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return {
    poseState,
    isConnected,
    startSession,
    sendFrame,
    endSession,
    resetSession,
    formAccuracy,
    elapsedSeconds,
  };
}
