from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TSL51_CLASSES = 51
TARGET_ACCURACY = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess whether available TSL51 artifacts satisfy the >90% accuracy goal."
    )
    parser.add_argument(
        "--artifact-dir",
        action="append",
        type=Path,
        default=[],
        help="TSL51 artifact directory containing tsl51_model_manifest.json.",
    )
    parser.add_argument(
        "--external-report",
        action="append",
        type=Path,
        default=[],
        help="External evaluation summary.json to list as supporting evidence only.",
    )
    parser.add_argument(
        "--trusted-holdout-report",
        action="append",
        type=Path,
        default=[],
        help="External summary.json whose samples are confirmed not to have been used for training or tuning.",
    )
    parser.add_argument("--out", type=Path, default=Path("reports") / "tsl51_goal_readiness.json")
    parser.add_argument(
        "--reviewed-split-summary",
        type=Path,
        default=None,
        help="Optional split summary from split_reviewed_external_manifest.py.",
    )
    parser.add_argument(
        "--review-queue-summary",
        type=Path,
        default=None,
        help="Optional review queue summary from build_review_queue.py.",
    )
    parser.add_argument(
        "--review-apply-summary",
        type=Path,
        default=None,
        help="Optional apply summary from apply_review_queue.py.",
    )
    parser.add_argument(
        "--review-validate-summary",
        type=Path,
        default=None,
        help="Optional validation summary from validate_review_queue.py.",
    )
    parser.add_argument(
        "--reviewed-assets-summary",
        type=Path,
        default=None,
        help="Optional asset summary from export_reviewed_external_assets.py.",
    )
    parser.add_argument(
        "--strategy-summary",
        type=Path,
        default=None,
        help="Optional multi-strategy external eval summary.",
    )
    parser.add_argument("--target", type=float, default=TARGET_ACCURACY)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_artifact_candidate(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "tsl51_model_manifest.json"
    labels_path = artifact_dir / "tsl51_labels.json"
    manifest = read_json(manifest_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else []
    label_count = len(labels) if isinstance(labels, list) else len(labels.keys()) if isinstance(labels, dict) else 0
    manifest_class_count = int(manifest.get("num_classes") or 0)
    class_count = manifest_class_count or label_count
    accuracy = manifest.get("test_accuracy")
    return {
        "artifact_dir": str(artifact_dir),
        "manifest": str(manifest_path),
        "class_count": class_count,
        "label_count": label_count,
        "internal_accuracy": float(accuracy) if accuracy is not None else None,
        "internal_accuracy_source": "tsl51_model_manifest.test_accuracy",
        "external_augmented": bool(manifest.get("external_augmented")),
        "external_val_samples": int(manifest.get("external_val_samples") or 0),
    }


def load_external_report(summary_path: Path, *, trusted_holdout: bool = False) -> dict[str, Any]:
    summary = read_json(summary_path)
    samples_file = str(summary.get("samples_file") or "")
    accuracy = summary.get("top1_accuracy")
    return {
        "summary": str(summary_path),
        "artifact_dir": str(summary.get("artifact_dir") or ""),
        "samples_file": samples_file,
        "total_samples": int(summary.get("total_samples") or 0),
        "top1_accuracy": float(accuracy) if accuracy is not None else None,
        "top3_accuracy": float(summary["top3_accuracy"]) if summary.get("top3_accuracy") is not None else None,
        "accepted_as_holdout": trusted_holdout,
        "rejection_reason": None if trusted_holdout else "external report was not marked as trusted holdout",
    }


def assess(
    artifacts: list[dict[str, Any]],
    external_reports: list[dict[str, Any]],
    *,
    target: float = TARGET_ACCURACY,
    reviewed_split_summary: dict[str, Any] | None = None,
    review_queue_summary: dict[str, Any] | None = None,
    review_validate_summary: dict[str, Any] | None = None,
    review_apply_summary: dict[str, Any] | None = None,
    reviewed_assets_summary: dict[str, Any] | None = None,
    strategy_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    class_valid = [item for item in artifacts if item["class_count"] == EXPECTED_TSL51_CLASSES]
    internally_eligible = [
        item
        for item in class_valid
        if not item["external_augmented"] or item["external_val_samples"] > 0
    ]
    internal_pass = [
        item
        for item in internally_eligible
        if item["internal_accuracy"] is not None and item["internal_accuracy"] > target
    ]
    accepted_external = [
        item
        for item in external_reports
        if item["accepted_as_holdout"] and item["top1_accuracy"] is not None and item["total_samples"] > 0
    ]
    external_pass = [item for item in accepted_external if item["top1_accuracy"] > target]
    best_internal = max(internal_pass, key=lambda item: item["internal_accuracy"], default=None)
    best_external = max(accepted_external, key=lambda item: item["top1_accuracy"], default=None)

    return {
        "target_accuracy": target,
        "expected_classes": EXPECTED_TSL51_CLASSES,
        "internal_pass": bool(internal_pass),
        "external_holdout_pass": bool(external_pass),
        "ready": bool(internal_pass and external_pass),
        "best_internal_candidate": best_internal,
        "best_external_holdout": best_external,
        "internal_candidate_rule": (
            "51 classes and internal accuracy above target; external-augmented artifacts "
            "must include external_val_samples > 0 to be eligible"
        ),
        "artifacts": artifacts,
        "external_reports": external_reports,
        "reviewed_split_summary": reviewed_split_summary,
        "review_queue_summary": review_queue_summary,
        "review_validate_summary": review_validate_summary,
        "review_apply_summary": review_apply_summary,
        "reviewed_assets_summary": reviewed_assets_summary,
        "strategy_summary": strategy_summary,
        "decision": (
            "ready"
            if internal_pass and external_pass
            else "not_ready_without_external_holdout_pass"
            if internal_pass
            else "not_ready_without_internal_pass"
        ),
    }


def main() -> None:
    args = parse_args()
    artifacts = [load_artifact_candidate(path) for path in args.artifact_dir]
    external_reports = [load_external_report(path) for path in args.external_report]
    external_reports.extend(
        load_external_report(path, trusted_holdout=True) for path in args.trusted_holdout_report
    )
    reviewed_split_summary = (
        read_json(args.reviewed_split_summary) if args.reviewed_split_summary is not None else None
    )
    review_queue_summary = (
        read_json(args.review_queue_summary) if args.review_queue_summary is not None else None
    )
    review_apply_summary = (
        read_json(args.review_apply_summary) if args.review_apply_summary is not None else None
    )
    review_validate_summary = (
        read_json(args.review_validate_summary) if args.review_validate_summary is not None else None
    )
    reviewed_assets_summary = (
        read_json(args.reviewed_assets_summary) if args.reviewed_assets_summary is not None else None
    )
    strategy_summary = read_json(args.strategy_summary) if args.strategy_summary is not None else None
    report = assess(
        artifacts,
        external_reports,
        target=args.target,
        reviewed_split_summary=reviewed_split_summary,
        review_queue_summary=review_queue_summary,
        review_validate_summary=review_validate_summary,
        review_apply_summary=review_apply_summary,
        reviewed_assets_summary=reviewed_assets_summary,
        strategy_summary=strategy_summary,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"ready={report['ready']} internal_pass={report['internal_pass']} "
        f"external_holdout_pass={report['external_holdout_pass']} decision={report['decision']}"
    )


if __name__ == "__main__":
    main()
