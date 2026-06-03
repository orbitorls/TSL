from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "split_reviewed_external_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("split_reviewed_external_manifest_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(video_id: str, label: str, *, quality_status: str = "reviewed") -> dict[str, str]:
    return {
        "video_id": video_id,
        "path": f"videos/{video_id}.mp4",
        "track": "tsl51",
        "label": label,
        "start_s": "0",
        "end_s": "1",
        "source_url": "",
        "split": "train",
        "license_note": "",
        "quality_status": quality_status,
    }


def test_split_reviewed_rows_reports_insufficient_holdout() -> None:
    module = load_module()
    rows = [row("v1", "a"), row("v2", "b")]

    split_rows, report = module.split_reviewed_rows(
        rows,
        min_holdout_rows=3,
        min_holdout_labels=3,
        min_val_rows=1,
    )

    assert len(split_rows) == 2
    assert report["ready_for_external_holdout"] is False
    assert report["splits"]["external_test"]["rows"] == 2
    assert "not enough reviewed" in report["reason"]


def test_split_reviewed_rows_creates_train_val_external_test_without_group_leakage() -> None:
    module = load_module()
    rows = [row(f"v{idx}", f"label-{idx}") for idx in range(8)]

    split_rows, report = module.split_reviewed_rows(
        rows,
        min_holdout_rows=3,
        min_holdout_labels=3,
        min_val_rows=2,
    )

    groups_by_split: dict[str, set[str]] = {}
    for item in split_rows:
        groups_by_split.setdefault(item["split"], set()).add(module.row_group_id(item))
    all_groups = [group for groups in groups_by_split.values() for group in groups]

    assert report["ready_for_external_holdout"] is True
    assert len(all_groups) == len(set(all_groups))
    assert report["splits"]["external_test"]["rows"] >= 3
    assert report["splits"]["val"]["rows"] >= 2
    assert report["splits"]["train"]["rows"] > 0
