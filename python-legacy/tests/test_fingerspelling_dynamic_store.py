from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_API = PROJECT_ROOT / "translate_api"
LEGACY_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from translate_api.store import SessionStore  # noqa: E402


def test_fingerspelling_dynamic_session_defaults() -> None:
    store = SessionStore()
    session = store.create("fingerspelling_dynamic")

    assert session.track_key == "fingerspelling_dynamic"
    assert session.settings.threshold == 0.65
    assert session.settings.min_sign_frames == 12
    assert session.settings.sign_end_frames == 5
    assert session.settings.min_confidence_margin == 0.10
    assert session.settings.commit_on_preview is True
    assert session.settings.prefer_seq_buf_on_commit is True
    assert session.settings.transcript_stable_frames == 1
    assert session.settings.transcript_debounce_s == 0.2

    assert session.transcript is not None
    assert session.transcript.stable_frames == 1
    assert session.transcript.debounce_s == 0.2
