from __future__ import annotations

import tsl_translate.transcript as transcript_module
from tsl_translate.transcript import TranscriptEngine


def test_tsl51_transcript_commits_predicted_label_without_ready_status_delay(monkeypatch) -> None:
    engine = TranscriptEngine("tsl51")

    monkeypatch.setattr(transcript_module.time, "monotonic", lambda: 1.0)
    first = engine.update("สวัสดี", 0.9, 0.55)
    assert first is None

    monkeypatch.setattr(transcript_module.time, "monotonic", lambda: 1.1)
    second = engine.update("สวัสดี", 0.9, 0.55)

    assert second is not None
    assert second.last_token == "สวัสดี"
    assert second.text == "สวัสดี"
