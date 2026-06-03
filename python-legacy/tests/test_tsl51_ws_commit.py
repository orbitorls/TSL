from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

import tsl_translate.transcript as transcript_module
from tsl_translate.transcript import TranscriptEngine
from translate_api.ws import inference as ws_module


class DummyResult:
    def __init__(self, status: str, label: str, confidence: float) -> None:
        self.status = status
        self.label = label
        self.confidence = confidence
        self.committed_label = label if status == "predicted" else None
        self.topk = []
        self.topk_text = ""
        self.buffering = None
        self.fps = 15.0


def test_tsl51_predicted_status_is_eligible_for_transcript_commit(monkeypatch) -> None:
    transcript = TranscriptEngine("tsl51")
    result = DummyResult(status="predicted", label="สวัสดี", confidence=0.9)

    monkeypatch.setattr(transcript_module.time, "monotonic", lambda: 1.0)
    update = transcript.update(result.label, result.confidence, 0.55)
    assert update is None

    monkeypatch.setattr(transcript_module.time, "monotonic", lambda: 1.1)
    update = transcript.update(result.label, result.confidence, 0.55)
    assert update is not None

    msg = ws_module._frame_to_message(result, update.text, [result.confidence])
    assert msg["label"] == "สวัสดี"
    assert msg["transcript"] == "สวัสดี"
