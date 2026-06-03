from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "translate_api"
LEGACY_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from translate_api.store import SessionStore  # noqa: E402


def test_tsl51_session_defaults_use_accuracy_profile() -> None:
    store = SessionStore()
    session = store.create("tsl51")

    assert session.settings.threshold == 0.65
    assert session.settings.min_sign_frames == 10
    assert session.settings.sign_end_frames == 8
    assert session.settings.min_confidence_margin == 0.12
    assert session.settings.commit_on_preview is False
