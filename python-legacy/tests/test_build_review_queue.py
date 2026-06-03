from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_review_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_review_queue_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(video_id: str, label: str, status: str = "prelabel") -> dict[str, str]:
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
        "quality_status": status,
    }


def test_review_queue_prioritizes_unique_groups_and_splits() -> None:
    module = load_module()
    rows = [row("reviewed-1", "known", "reviewed")]
    rows.extend(row(f"new-{idx}", f"label-{idx}") for idx in range(12))

    queue, summary = module.build_review_queue(
        rows,
        holdout_target_rows=3,
        holdout_target_labels=3,
        val_target_rows=2,
        train_target_rows=2,
    )

    assert summary["queue_rows"] == 6
    assert summary["queued_by_recommended_split"]["external_test"]["rows"] == 2
    assert summary["queued_by_recommended_split"]["val"]["rows"] == 2
    assert summary["queued_by_recommended_split"]["train"]["rows"] == 2
    assert len({module.group_id(item) for item in queue}) == len(queue)
    assert all("review_decision" in item for item in queue)
    assert all("reviewed_label" in item for item in queue)


def test_review_queue_avoids_already_reviewed_groups() -> None:
    module = load_module()
    rows = [
        row("same-video", "a", "reviewed"),
        row("same-video", "b", "prelabel"),
        row("new-video", "c", "prelabel"),
    ]

    queue, _summary = module.build_review_queue(
        rows,
        holdout_target_rows=2,
        holdout_target_labels=2,
        val_target_rows=0,
        train_target_rows=0,
    )

    assert [item["video_id"] for item in queue] == ["new-video"]


def test_review_queue_continues_until_label_target_is_met() -> None:
    module = load_module()
    rows = [
        row("new-a1", "a"),
        row("new-a2", "a"),
        row("new-b1", "b"),
        row("new-c1", "c"),
    ]

    queue, summary = module.build_review_queue(
        rows,
        holdout_target_rows=2,
        holdout_target_labels=3,
        val_target_rows=0,
        train_target_rows=0,
    )

    holdout = [item for item in queue if item["recommended_split"] == "external_test"]
    assert len(holdout) == 3
    assert len({item["label"] for item in holdout}) == 3
    assert summary["queued_by_recommended_split"]["external_test"]["labels"] == 3
