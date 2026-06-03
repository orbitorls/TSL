from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_tsl51_goal_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("report_tsl51_goal_readiness_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(
    accuracy: float,
    *,
    classes: int = 51,
    external_augmented: bool = False,
    external_val_samples: int = 0,
) -> dict[str, object]:
    return {
        "artifact_dir": "artifact",
        "class_count": classes,
        "internal_accuracy": accuracy,
        "external_augmented": external_augmented,
        "external_val_samples": external_val_samples,
    }


def report(accuracy: float, *, samples_file: str = "holdout.csv") -> dict[str, object]:
    return {
        "summary": "summary.json",
        "samples_file": samples_file,
        "total_samples": 10,
        "top1_accuracy": accuracy,
        "accepted_as_holdout": "train" not in samples_file.lower(),
    }


def test_goal_readiness_requires_internal_and_external_holdout_pass() -> None:
    module = load_module()

    readiness = module.assess([candidate(0.95)], [report(0.91)])

    assert readiness["ready"] is True
    assert readiness["internal_pass"] is True
    assert readiness["external_holdout_pass"] is True


def test_goal_readiness_rejects_internal_only_pass() -> None:
    module = load_module()

    split_summary = {"ready_for_external_holdout": False, "reviewed_rows": 5}
    queue_summary = {"ready_to_review": True, "queue_rows": 15}
    validate_summary = {"ready": False, "approved_rows": 0}
    apply_summary = {"approved_rows": 0, "ready_for_splitter": False}
    assets_summary = {"ready_for_training": False, "ready_for_eval": True}
    strategy_summary = {"all_strategies_failed": True}
    readiness = module.assess(
        [candidate(0.95)],
        [],
        reviewed_split_summary=split_summary,
        review_queue_summary=queue_summary,
        review_validate_summary=validate_summary,
        review_apply_summary=apply_summary,
        reviewed_assets_summary=assets_summary,
        strategy_summary=strategy_summary,
    )

    assert readiness["ready"] is False
    assert readiness["internal_pass"] is True
    assert readiness["external_holdout_pass"] is False
    assert readiness["decision"] == "not_ready_without_external_holdout_pass"
    assert readiness["reviewed_split_summary"] == split_summary
    assert readiness["review_queue_summary"] == queue_summary
    assert readiness["review_validate_summary"] == validate_summary
    assert readiness["review_apply_summary"] == apply_summary
    assert readiness["reviewed_assets_summary"] == assets_summary
    assert readiness["strategy_summary"] == strategy_summary


def test_goal_readiness_rejects_wrong_class_count() -> None:
    module = load_module()

    readiness = module.assess([candidate(0.99, classes=47)], [report(0.99)])

    assert readiness["ready"] is False
    assert readiness["internal_pass"] is False
    assert readiness["decision"] == "not_ready_without_internal_pass"


def test_goal_readiness_rejects_external_augmented_without_external_val() -> None:
    module = load_module()

    readiness = module.assess(
        [candidate(0.99, external_augmented=True, external_val_samples=0)],
        [report(0.99)],
    )

    assert readiness["internal_pass"] is False
    assert readiness["ready"] is False


def test_external_report_marks_training_samples_unaccepted(tmp_path: Path) -> None:
    module = load_module()
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"samples_file": "work/reviewed_seed_train_manifest.csv", "total_samples": 6, "top1_accuracy": 1.0}',
        encoding="utf-8",
    )

    loaded = module.load_external_report(summary)

    assert loaded["accepted_as_holdout"] is False
    assert loaded["rejection_reason"] == "external report was not marked as trusted holdout"


def test_external_report_accepts_only_explicit_trusted_holdout(tmp_path: Path) -> None:
    module = load_module()
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"samples_file": "work/reviewed_holdout.csv", "total_samples": 10, "top1_accuracy": 0.91}',
        encoding="utf-8",
    )

    loaded = module.load_external_report(summary, trusted_holdout=True)

    assert loaded["accepted_as_holdout"] is True
    assert loaded["rejection_reason"] is None
