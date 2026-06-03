from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_review_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("validate_review_queue_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(**overrides: str) -> dict[str, str]:
    item = {
        "video_id": "v1",
        "path": "videos/v1.mp4",
        "label": "กรุงเทพ_var_1",
        "reviewed_label": "",
        "start_s": "0",
        "end_s": "1",
        "split": "train",
        "recommended_split": "train",
        "review_decision": "approved",
    }
    item.update(overrides)
    return item


def test_load_allowed_labels_accepts_indexed_label_map(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"0": "a", "1": "b"}), encoding="utf-8")

    assert module.load_allowed_labels(labels) == {"a", "b"}


def test_validate_rows_accepts_ready_reviewed_queue() -> None:
    module = load_module()
    rows = [
        row(video_id="train1", recommended_split="train"),
        row(video_id="val1", recommended_split="val"),
        row(video_id="test1", recommended_split="external_test"),
    ]

    summary = module.validate_rows(
        rows,
        allowed_labels={"กรุงเทพ_var_1"},
        skip_path_check=True,
    )

    assert summary["ready"] is True
    assert summary["approved_rows"] == 3
    assert summary["splits"]["train"]["groups"] == 1


def test_validate_rows_rejects_missing_decisions_by_default() -> None:
    module = load_module()

    summary = module.validate_rows(
        [row(review_decision="")],
        allowed_labels={"กรุงเทพ_var_1"},
        skip_path_check=True,
    )

    assert summary["ready"] is False
    assert any("missing review_decision" in error for error in summary["errors"])


def test_validate_rows_allows_pending_rows_but_keeps_split_minimums() -> None:
    module = load_module()

    summary = module.validate_rows(
        [row(review_decision="")],
        allowed_labels={"กรุงเทพ_var_1"},
        allow_missing_decisions=True,
        skip_path_check=True,
    )

    assert summary["ready"] is False
    assert summary["missing_decisions"] == 1
    assert summary["warnings"]
    assert any("train has 0 approved rows" in error for error in summary["errors"])


def test_validate_rows_rejects_unknown_label_and_bad_decision() -> None:
    module = load_module()

    summary = module.validate_rows(
        [
            row(video_id="bad-label", reviewed_label="นอกชุด"),
            row(video_id="bad-decision", review_decision="maybe"),
        ],
        allowed_labels={"กรุงเทพ_var_1"},
        skip_path_check=True,
    )

    assert summary["ready"] is False
    assert any("not in labels" in error for error in summary["errors"])
    assert any("unsupported review_decision" in error for error in summary["errors"])


def test_validate_rows_rejects_group_leakage_across_splits() -> None:
    module = load_module()

    summary = module.validate_rows(
        [
            row(video_id="same", recommended_split="train"),
            row(video_id="same", start_s="2", end_s="3", recommended_split="val"),
            row(video_id="test", recommended_split="external_test"),
        ],
        allowed_labels={"กรุงเทพ_var_1"},
        skip_path_check=True,
    )

    assert summary["ready"] is False
    assert any("appears in both" in error for error in summary["errors"])


def test_validate_rows_checks_approved_paths(tmp_path: Path) -> None:
    module = load_module()

    summary = module.validate_rows(
        [
            row(video_id="train", path="missing.mp4", recommended_split="train"),
            row(video_id="val", path="missing2.mp4", recommended_split="val"),
            row(video_id="test", path="missing3.mp4", recommended_split="external_test"),
        ],
        allowed_labels={"กรุงเทพ_var_1"},
        repo_root=tmp_path,
    )

    assert summary["ready"] is False
    assert any("path does not exist" in error for error in summary["errors"])
