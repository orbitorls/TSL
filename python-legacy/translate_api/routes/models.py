from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from tsl_translate.loader import load_artifacts
from tsl_translate.registry import ModelRegistry
from tsl_translate.session import PredictService
from tsl_translate.tracks import TRACKS
from tsl_translate.transcript import TranscriptEngine
from translate_api.deps import get_repo_root, get_store
from translate_api.store import SessionStore

router = APIRouter(prefix="/api", tags=["models"])


class TrackOut(BaseModel):
    key: str
    title: str


class ArtifactOut(BaseModel):
    name: str
    model: str
    labels: str
    scaler: str
    manifest: str | None


class CreateSessionIn(BaseModel):
    track: str = Field(..., description="fingerspelling or tsl51")


class CreateSessionOut(BaseModel):
    session_id: str
    track: str


class LoadArtifactIn(BaseModel):
    artifact_name: str | None = None
    model_path: str | None = None
    labels_path: str | None = None
    scaler_path: str | None = None
    manifest_path: str | None = None


class LoadSessionOut(BaseModel):
    ok: bool
    backend: str
    num_classes: int
    load_time_ms: float
    model_path: str


class SettingsIn(BaseModel):
    threshold: float | None = None
    alpha: float | None = None
    top_k: int | None = None
    motion_min: float | None = None
    min_sign_frames: int | None = None
    sign_end_frames: int | None = None
    min_confidence_margin: float | None = None
    commit_on_preview: bool | None = None


class TranscriptActionIn(BaseModel):
    action: str  # space | backspace | clear


def _registry(root: Path) -> ModelRegistry:
    return ModelRegistry(root)


@router.get("/tracks", response_model=list[TrackOut])
def list_tracks() -> list[TrackOut]:
    return [TrackOut(key=t.key, title=t.title) for t in TRACKS.values()]


@router.get("/artifacts", response_model=list[ArtifactOut])
def list_artifacts(track: str, root: Path = Depends(get_repo_root)) -> list[ArtifactOut]:
    if track not in TRACKS:
        raise HTTPException(400, detail=f"ไม่รู้จักแทร็ก: {track}")
    reg = _registry(root)
    candidates = reg.discover(TRACKS[track])
    return [
        ArtifactOut(
            name=c.name,
            model=str(c.model),
            labels=str(c.labels),
            scaler=str(c.scaler),
            manifest=str(c.manifest) if c.manifest else None,
        )
        for c in candidates
    ]


@router.post("/session", response_model=CreateSessionOut)
def create_session(
    body: CreateSessionIn, store: SessionStore = Depends(get_store)
) -> CreateSessionOut:
    if body.track not in TRACKS:
        raise HTTPException(400, detail=f"ไม่รู้จักแทร็ก: {body.track}")
    session = store.create(body.track)
    return CreateSessionOut(session_id=session.id, track=body.track)


@router.post("/session/{session_id}/load", response_model=LoadSessionOut)
def load_session_model(
    session_id: str,
    body: LoadArtifactIn,
    store: SessionStore = Depends(get_store),
    root: Path = Depends(get_repo_root),
) -> LoadSessionOut:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, detail="ไม่พบ session")

    track = TRACKS[session.track_key]
    reg = _registry(root)

    if body.model_path and body.labels_path and body.scaler_path:
        model_path = Path(body.model_path)
        labels_path = Path(body.labels_path)
        scaler_path = Path(body.scaler_path)
        manifest_path = Path(body.manifest_path) if body.manifest_path else None
    else:
        if not body.artifact_name:
            raise HTTPException(400, detail="ระบุ artifact_name หรือ path ครบชุด")
        candidates = reg.discover(track)
        match = next((c for c in candidates if c.name == body.artifact_name), None)
        if match is None:
            raise HTTPException(404, detail="ไม่พบชุดไฟล์ที่เลือก")
        model_path, labels_path, scaler_path, manifest_path = (
            match.model,
            match.labels,
            match.scaler,
            match.manifest,
        )

    try:
        loaded = load_artifacts(track, model_path, labels_path, scaler_path, manifest_path)
    except Exception as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    session.loaded = loaded
    session.service = PredictService(track, alpha=session.settings.alpha)
    session.transcript = TranscriptEngine(session.track_key)
    session.confidence_hist = []
    return LoadSessionOut(
        ok=True,
        backend=loaded.backend,
        num_classes=len(loaded.labels),
        load_time_ms=loaded.load_time_ms,
        model_path=str(loaded.model_path),
    )


@router.patch("/session/{session_id}/settings")
def patch_settings(
    session_id: str, body: SettingsIn, store: SessionStore = Depends(get_store)
) -> dict[str, Any]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, detail="ไม่พบ session")
    if body.threshold is not None:
        session.settings.threshold = body.threshold
    if body.alpha is not None:
        session.settings.alpha = body.alpha
        if session.service:
            session.service.set_alpha(body.alpha)
    if body.top_k is not None:
        session.settings.top_k = body.top_k
    if body.motion_min is not None:
        session.settings.motion_min = body.motion_min
    if body.min_sign_frames is not None:
        session.settings.min_sign_frames = body.min_sign_frames
    if body.sign_end_frames is not None:
        session.settings.sign_end_frames = body.sign_end_frames
    if body.min_confidence_margin is not None:
        session.settings.min_confidence_margin = body.min_confidence_margin
    if body.commit_on_preview is not None:
        session.settings.commit_on_preview = body.commit_on_preview
    return {"ok": True, "settings": session.settings.__dict__}


@router.post("/session/{session_id}/transcript")
def transcript_action(
    session_id: str, body: TranscriptActionIn, store: SessionStore = Depends(get_store)
) -> dict[str, str]:
    session = store.get(session_id)
    if session is None or session.transcript is None:
        raise HTTPException(404, detail="ไม่พบ session")
    if body.action == "space":
        session.transcript.add_space()
    elif body.action == "backspace":
        session.transcript.backspace()
    elif body.action == "clear":
        session.transcript.clear()
    else:
        raise HTTPException(400, detail="action ไม่ถูกต้อง")
    return {"text": session.transcript.text}


@router.delete("/session/{session_id}")
def delete_session(session_id: str, store: SessionStore = Depends(get_store)) -> dict[str, bool]:
    store.delete(session_id)
    return {"ok": True}
