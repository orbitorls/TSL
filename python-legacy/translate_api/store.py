from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from tsl_translate.inference import InferenceSettings
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService
from tsl_translate.transcript import TranscriptEngine


@dataclass
class TranslateSession:
    id: str
    track_key: str
    loaded: LoadedModel | None = None
    service: PredictService | None = None
    mp: MediaPipeRuntime | None = None
    settings: InferenceSettings = field(default_factory=InferenceSettings)
    transcript: TranscriptEngine | None = None
    confidence_hist: list[float] = field(default_factory=list)
    streaming: bool = False
    send_landmarks: bool = True


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, TranslateSession] = {}

    def create(self, track_key: str) -> TranslateSession:
        session_id = uuid.uuid4().hex
        if track_key == "fingerspelling":
            settings = InferenceSettings(
                threshold=0.70,
                alpha=0.40,
                min_confidence_margin=0.10,
                prediction_stable_frames=2,
                transcript_stable_frames=2,
                transcript_debounce_s=0.2,
            )
        elif track_key == "fingerspelling_dynamic":
            settings = InferenceSettings(
                threshold=0.65,
                alpha=0.40,
                min_confidence_margin=0.10,
                min_sign_frames=12,
                sign_end_frames=5,
                commit_on_preview=True,
                prefer_seq_buf_on_commit=True,
                transcript_stable_frames=1,
                transcript_debounce_s=0.2,
            )
        else:
            settings = InferenceSettings(
                threshold=0.62,
                min_sign_frames=4,
                sign_end_frames=3,
                min_confidence_margin=0.10,
                commit_on_preview=True,
                prefer_seq_buf_on_commit=True,
                transcript_stable_frames=1,
                transcript_debounce_s=0.2,
            )
        if track_key == "fingerspelling":
            transcript = TranscriptEngine(
                track_key,
                debounce_s=settings.transcript_debounce_s,
                stable_frames=settings.transcript_stable_frames,
            )
        else:
            transcript = TranscriptEngine(
                track_key,
                debounce_s=settings.transcript_debounce_s,
                stable_frames=settings.transcript_stable_frames,
            )
        session = TranslateSession(
            id=session_id,
            track_key=track_key,
            settings=settings,
            transcript=transcript,
            send_landmarks=True,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> TranslateSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session and session.mp is not None:
            session.mp.close()

    def ensure_mp(self, session: TranslateSession) -> MediaPipeRuntime:
        if session.mp is None:
            session.mp = MediaPipeRuntime()
        return session.mp
