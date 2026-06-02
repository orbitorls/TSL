from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TranscriptUpdate:
    text: str
    last_token: str


class TranscriptEngine:
    """Accumulate confirmed Thai tokens from stable predictions."""

    def __init__(
        self,
        track_key: str,
        *,
        debounce_s: float = 0.4,
        stable_frames: int = 2,
    ) -> None:
        self.track_key = track_key
        self.debounce_s = debounce_s
        self.stable_frames = max(1, stable_frames)
        self._text = ""
        self._pending_label: str | None = None
        self._pending_since: float = 0.0
        self._stable_count = 0
        self._last_committed: str | None = None

    @property
    def text(self) -> str:
        return self._text

    def clear(self) -> None:
        self._text = ""
        self._pending_label = None
        self._stable_count = 0
        self._last_committed = None

    def add_space(self) -> None:
        if self._text and not self._text.endswith(" "):
            self._text += " "

    def backspace(self) -> None:
        if self._text:
            self._text = self._text[:-1]

    def update(self, label: str, confidence: float, threshold: float) -> TranscriptUpdate | None:
        """Return update when a new token is committed to the transcript."""
        if not label or label == "?":
            self._pending_label = None
            self._stable_count = 0
            return None
        if confidence < threshold:
            self._pending_label = None
            self._stable_count = 0
            return None

        now = time.monotonic()
        if label != self._pending_label:
            self._pending_label = label
            self._pending_since = now
            self._stable_count = 1
            return None

        self._stable_count += 1
        if self.track_key == "fingerspelling":
            if now - self._pending_since < self.debounce_s:
                return None
        elif self._stable_count < self.stable_frames:
            return None

        if label == self._last_committed:
            return None

        self._last_committed = label
        if self.track_key == "fingerspelling":
            self._text += label
        else:
            if self._text and not self._text.endswith(" "):
                self._text += " "
            self._text += label

        self._pending_label = None
        self._stable_count = 0
        return TranscriptUpdate(text=self._text, last_token=label)
