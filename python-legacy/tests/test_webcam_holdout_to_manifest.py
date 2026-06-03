from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "webcam_holdout_to_manifest.py"
spec = importlib.util.spec_from_file_location("webcam_holdout_to_manifest", SCRIPT)
whm = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["webcam_holdout_to_manifest"] = whm
spec.loader.exec_module(whm)


def test_holdout_split_maps_to_external_test() -> None:
    row = whm._row_to_manifest(
        {
            "id": "clip1",
            "path": "v.mp4",
            "expected": "กิน_var_1",
            "start_s": "0.0",
            "end_s": "5.0",
            "split": "holdout",
        },
        split_override=None,
    )
    assert row["split"] == "external_test"
    assert row["track"] == "tsl51"
    assert row["label"] == "กิน_var_1"
    assert row["quality_status"] == "reviewed"


def test_train_split_override() -> None:
    row = whm._row_to_manifest(
        {"id": "c", "path": "v.mp4", "expected": "ข้าว_var_1", "start_s": "0", "end_s": "5", "split": "holdout"},
        split_override="train",
    )
    assert row["split"] == "train"
