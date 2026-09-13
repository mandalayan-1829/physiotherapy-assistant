"""
WebSocket route for real-time physiotherapy session.
Receives video frames, processes them with AI, and sends back analysis.
"""

from __future__ import annotations
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Lazy import — pose_service requires mediapipe which may not be installed.
# Imported inside the handler when a WebSocket connection actually arrives.

router = APIRouter(tags=["websocket"])

# Active session managers keyed by session_id
_active_sessions: dict[str, SessionManager] = {}


@router.websocket("/ws/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pose analysis.

    Client sends:
        - {"type": "start", "exercise": "squat", "target_reps": 10}
        - {"type": "frame", "data": "<base64 jpeg>"}
        - {"type": "reset"}
        - {"type": "end"}

    Server sends:
        - {"type": "started", "session_id": "..."}
        - {"type": "analysis", ...pose state...}
        - {"type": "reset_ack"}
        - {"type": "ended", "form_accuracy": 85}
    """
    await websocket.accept()

    manager = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "start":
                exercise = msg.get("exercise", "squat")
                target = msg.get("target_reps", 10)

                from backend.services.pose_service import SessionManager, decode_frame_from_base64
                manager = SessionManager(exercise)
                manager.target_reps = target
                _active_sessions[session_id] = manager

                await websocket.send_json({
                    "type": "started",
                    "session_id": session_id,
                    "exercise": exercise,
                    "target_reps": target,
                })

            elif msg_type == "frame" and manager:
                frame_data = msg.get("data", "")
                from backend.services.pose_service import decode_frame_from_base64
                frame = decode_frame_from_base64(frame_data)

                if frame is not None:
                    # Process in a thread to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, manager.process_frame, frame)

                    await websocket.send_json({
                        "type": "analysis",
                        **result,
                    })

                    # Check if target reps reached
                    if result["rep_count"] >= manager.target_reps:
                        form_accuracy = manager.get_form_accuracy()
                        await websocket.send_json({
                            "type": "completed",
                            "rep_count": result["rep_count"],
                            "form_accuracy": form_accuracy,
                            "exercise": manager.exercise,
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to decode frame",
                    })

            elif msg_type == "reset" and manager:
                manager.reset()
                await websocket.send_json({"type": "reset_ack"})

            elif msg_type == "end" and manager:
                form_accuracy = manager.get_form_accuracy()
                await websocket.send_json({
                    "type": "ended",
                    "form_accuracy": form_accuracy,
                    "rep_count": manager.prev_reps,
                })
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
    finally:
        _active_sessions.pop(session_id, None)
