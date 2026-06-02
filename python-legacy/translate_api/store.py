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


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, TranslateSession] = {}

    def create(self, track_key: str) -> TranslateSession:
        session_id = uuid.uuid4().hex
        default_threshold = 0.7 if track_key == "fingerspelling" else 0.55
        session = TranslateSession(
            id=session_id,
            track_key=track_key,
            settings=InferenceSettings(threshold=default_threshold),
            transcript=TranscriptEngine(track_key),
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
