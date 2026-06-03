from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "apply_review_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("apply_review_queue_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(**overrides: str) -> dict[str, str]:
    item = {
        "video_id": "v1",
        "path": "videos/v1.mp4",
        "track": "tsl51",
        "label": "old",
        "reviewed_label": "",
        "start_s": "0",
        "end_s": "1",
        "source_url": "",
        "split": "train",
        "license_note": "",
        "quality_status": "prelabel",
        "recommended_split": "external_test",
        "review_decision": "approved",
    }
    item.update(overrides)
    return item


def test_apply_decisions_approves_rows_with_recommended_split() -> None:
    module = load_module()

    reviewed, summary = module.apply_decisions([row(reviewed_label="new")])

    assert reviewed[0]["label"] == "new"
    assert reviewed[0]["split"] == "external_test"
    assert reviewed[0]["quality_status"] == "reviewed"
    assert summary["approved_rows"] == 1
    assert summary["output_splits"]["external_test"]["rows"] == 1


def test_apply_decisions_rejects_missing_decisions() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="missing review_decision"):
        module.apply_decisions([row(review_decision="")])


def test_apply_decisions_can_skip_missing_decisions() -> None:
    module = load_module()

    reviewed, summary = module.apply_decisions(
        [row(review_decision=""), row(video_id="v2")],
        allow_missing_decisions=True,
    )

    assert len(reviewed) == 1
    assert summary["approved_rows"] == 1


def test_apply_decisions_counts_rejected_rows() -> None:
    module = load_module()

    reviewed, summary = module.apply_decisions([row(review_decision="rejected")])

    assert reviewed == []
    assert summary["rejected_rows"] == 1


def test_merge_reviewed_rows_keeps_base_and_adds_approved() -> None:
    module = load_module()
    base = [row(video_id="base", quality_status="reviewed", split="external_test")]
    approved = [row(video_id="new", quality_status="reviewed", split="val")]

    merged = module.merge_reviewed_rows(base, approved)

    assert {item["video_id"] for item in merged} == {"base", "new"}


def test_merge_reviewed_rows_replaces_duplicate_with_approved() -> None:
    module = load_module()
    base = [row(video_id="same", label="old", quality_status="reviewed")]
    approved = [row(video_id="same", label="new", quality_status="reviewed")]

    merged = module.merge_reviewed_rows(base, approved)

    assert len(merged) == 1
    assert merged[0]["label"] == "new"


def test_split_summary_counts_rows_labels_and_groups() -> None:
    module = load_module()
    rows = [
        row(video_id="a", label="x", split="external_test"),
        row(video_id="b", label="x", split="external_test"),
        row(video_id="c", label="y", split="train"),
    ]

    assert module.split_summary(rows, "external_test") == {
        "rows": 2,
        "labels": 1,
        "groups": 2,
    }


def test_write_manifest_deduplicates_fieldnames(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "manifest.csv"

    module.write_manifest(out, [row()], ["video_id", "path", "video_id", "label"])

    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",").count("video_id") == 1
