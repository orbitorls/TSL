from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "export_reviewed_external_assets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_reviewed_external_assets_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(video_id: str, split: str, label: str = "label") -> dict[str, str]:
    return {
        "video_id": video_id,
        "path": f"videos/{video_id}.mp4",
        "track": "tsl51",
        "label": label,
        "start_s": "0",
        "end_s": "1",
        "source_url": "",
        "split": split,
        "license_note": "",
        "quality_status": "reviewed",
    }


def test_export_assets_writes_split_manifests_and_eval_samples(tmp_path: Path) -> None:
    module = load_module()
    rows = [
        row("train-1", "train", "a"),
        row("val-1", "val", "b"),
        row("test-1", "external_test", "c"),
    ]
    fieldnames = list(rows[0])

    summary = module.export_assets(rows, fieldnames, tmp_path, Path("D:/repo"))

    assert summary["ready_for_training"] is True
    assert summary["ready_for_eval"] is True
    assert Path(summary["files"]["train_manifest"]).exists()
    samples = Path(summary["files"]["external_test_samples"]).read_text(encoding="utf-8")
    assert "expected" in samples
    assert "D:\\repo\\videos\\test-1.mp4" in samples or "D:/repo/videos/test-1.mp4" in samples


def test_export_assets_reports_not_ready_without_train_val(tmp_path: Path) -> None:
    module = load_module()
    rows = [row("test-1", "external_test", "c")]

    summary = module.export_assets(rows, list(rows[0]), tmp_path, Path("D:/repo"))

    assert summary["ready_for_training"] is False
    assert summary["ready_for_eval"] is True
