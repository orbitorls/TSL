from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tsl_translate.inference import process_rgb_frame
from tsl_translate.tracks import TRACKS
from translate_api.deps import get_store
from translate_api.store import SessionStore

router = APIRouter()


def _decode_jpeg_frame(data: str | bytes) -> np.ndarray:
    if isinstance(data, str):
        raw = base64.b64decode(data.split(",", 1)[-1])
    else:
        raw = data
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("ถอดรหัสภาพ JPEG ไม่ได้")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _frame_to_message(result, transcript_text: str, hist: list[float]) -> dict[str, Any]:
    return {
        "type": "prediction",
        "label": result.label,
        "confidence": result.confidence,
        "committed_label": result.committed_label,
        "topk": [{"label": e.label, "p": e.p} for e in result.topk],
        "topk_text": result.topk_text,
        "status": result.status,
        "buffering": result.buffering,
        "fps": result.fps,
        "transcript": transcript_text,
        "confidence_hist": hist[-100:],
    }


@router.websocket("/ws/session/{session_id}")
async def ws_inference(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    store = get_store()
    session = store.get(session_id)
    if session is None:
        await websocket.send_json({"type": "error", "message": "ไม่พบ session"})
        await websocket.close()
        return

    track = TRACKS[session.track_key]
    session.streaming = True

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON ไม่ถูกต้อง"})
                continue

            msg_type = msg.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "stop":
                break

            if msg_type != "frame":
                continue

            if session.loaded is None or session.service is None:
                await websocket.send_json({"type": "error", "message": "ยังไม่ได้โหลดโมเดล"})
                continue

            try:
                rgb = _decode_jpeg_frame(msg.get("jpeg_base64", ""))
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            mp_runtime = store.ensure_mp(session)
            result = process_rgb_frame(
                track,
                session.loaded,
                session.service,
                mp_runtime,
                rgb,
                session.settings,
            )

            session.confidence_hist.append(result.confidence)
            if len(session.confidence_hist) > 100:
                session.confidence_hist = session.confidence_hist[-100:]

            transcript_text = session.transcript.text if session.transcript else ""
            should_update_transcript = (
                session.transcript is not None
                and result.committed_label not in (None, "", "?", "Unknown / รอท่าชัดเจน")
            )
            if should_update_transcript:
                update = session.transcript.update(
                    result.committed_label, result.confidence, session.settings.threshold
                )
                if update is not None:
                    transcript_text = update.text
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": update.text,
                            "last_token": update.last_token,
                        }
                    )

            await websocket.send_json(
                _frame_to_message(result, transcript_text, session.confidence_hist)
            )
    except WebSocketDisconnect:
        pass
    finally:
        session.streaming = False
