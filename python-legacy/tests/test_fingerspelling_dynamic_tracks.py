from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsl_translate.tracks import TRACKS  # noqa: E402


def test_fingerspelling_dynamic_track_spec() -> None:
    track = TRACKS.get("fingerspelling_dynamic")
    assert track is not None, "TRACKS must contain 'fingerspelling_dynamic'"

    assert track.key == "fingerspelling_dynamic"
    assert track.expected_seq_len == 30
    assert track.expected_feature_dim == 63
    assert track.default_model == "fs_dynamic_model.keras"
    assert track.default_labels == "fs_dynamic_labels.json"
    assert track.default_scaler == "fs_dynamic_scaler.pkl"
    assert track.manifest == "fs_dynamic_model_manifest.json"
    assert track.runtime_track == "thai_fingerspelling_dynamic"
    assert track.title == "Fingerspelling Dynamic"
