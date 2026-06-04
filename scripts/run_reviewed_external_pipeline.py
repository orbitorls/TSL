from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reviewed external TSL51 apply/split/export/train/eval pipeline."
    )
    parser.add_argument("--queue", default=Path("work") / "tsl51_review_queue.csv", type=Path)
    parser.add_argument("--base-manifest", default=Path("data") / "external_tsl51" / "manifest_all.csv", type=Path)
    parser.add_argument("--base-cache", required=True, type=Path)
    parser.add_argument("--base-artifact-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--work-dir", default=Path("work") / "reviewed_external_pipeline", type=Path)
    parser.add_argument("--reports-dir", default=Path("reports") / "reviewed_external_pipeline", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--allow-missing-decisions", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def run(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    return subprocess.run(cmd, check=check)


def read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    root = Path(".")

    reviewed_manifest = args.work_dir / "reviewed_from_queue_manifest.csv"
    validate_summary = args.reports_dir / "review_validate_summary.json"
    apply_summary = args.reports_dir / "review_apply_summary.json"
    split_manifest = args.work_dir / "reviewed_split_manifest.csv"
    split_summary = args.reports_dir / "reviewed_split_summary.json"
    assets_dir = args.work_dir / "assets"
    assets_summary = args.reports_dir / "reviewed_assets_summary.json"
    train_cache = args.work_dir / "reviewed_train_cache.npz"
    val_cache = args.work_dir / "reviewed_val_cache.npz"
    train_out = args.work_dir / "train_out"
    eval_out = args.reports_dir / "eval"

    validate_cmd = [
        args.python,
        "-B",
        "scripts/validate_review_queue.py",
        "--queue",
        str(args.queue),
        "--labels",
        str(args.labels),
        "--summary",
        str(validate_summary),
        "--repo-root",
        str(root),
    ]
    if args.allow_missing_decisions:
        validate_cmd.append("--allow-missing-decisions")
    run(validate_cmd, dry_run=args.dry_run, check=False)

    summary: dict[str, Any] = {
        "validate_summary": str(validate_summary),
        "apply_summary": str(apply_summary),
        "split_summary": str(split_summary),
        "assets_summary": str(assets_summary),
        "trained": False,
        "evaluated": False,
    }
    if args.dry_run:
        summary["status"] = "dry_run"
        return summary

    validation = read_json(validate_summary)
    summary["validation"] = validation
    if not validation.get("ready"):
        summary["status"] = "blocked_review_queue_not_ready"
        summary["blocker"] = "review queue must have approved, label-valid, leakage-safe train/val/external_test rows"
        return summary

    apply_cmd = [
        args.python,
        "-B",
        "scripts/apply_review_queue.py",
        "--queue",
        str(args.queue),
        "--base-manifest",
        str(args.base_manifest),
        "--out",
        str(reviewed_manifest),
        "--summary",
        str(apply_summary),
    ]
    if args.allow_missing_decisions:
        apply_cmd.append("--allow-missing-decisions")
    run(apply_cmd, dry_run=args.dry_run)

    split_cmd = [
        args.python,
        "-B",
        "scripts/split_reviewed_external_manifest.py",
        "--manifest",
        str(reviewed_manifest),
        "--out",
        str(split_manifest),
        "--report",
        str(split_summary),
    ]
    run(split_cmd, dry_run=args.dry_run)

    export_cmd = [
        args.python,
        "-B",
        "scripts/export_reviewed_external_assets.py",
        "--manifest",
        str(split_manifest),
        "--out-dir",
        str(assets_dir),
        "--summary",
        str(assets_summary),
        "--repo-root",
        str(root),
    ]
    run(export_cmd, dry_run=args.dry_run)

    assets = read_json(assets_summary)
    summary["assets"] = assets
    if not assets.get("ready_for_training"):
        summary["status"] = "blocked_missing_reviewed_train_val"
        summary["blocker"] = "reviewed train and val splits are required before fine-tuning"
        return summary

    for split_name, out_path in (("train", train_cache), ("val", val_cache)):
        manifest = assets_dir / f"{split_name}_manifest.csv"
        build_cmd = [
            args.python,
            "-B",
            "scripts/build_external_dataset.py",
            "--manifest",
            str(manifest),
            "--track",
            "tsl51",
            "--labels",
            str(args.labels),
            "--out",
            str(out_path),
            "--require-reviewed",
            "--cache-split",
            split_name,
            "--manifest-base-dir",
            str(root),
        ]
        run(build_cmd)

    train_cmd = [
        args.python,
        "-B",
        "scripts/train_external_augmented.py",
        "--track",
        "tsl51",
        "--base-cache",
        str(args.base_cache),
        "--external-cache",
        str(train_cache),
        "--external-cache",
        str(val_cache),
        "--out-dir",
        str(train_out),
        "--base-artifact-dir",
        str(args.base_artifact_dir),
        "--base-labels",
        str(args.labels),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--require-external-val",
    ]
    run(train_cmd)
    summary["trained"] = True

    eval_cmd = [
        args.python,
        "-B",
        "scripts/evaluate_tsl51_video.py",
        "--samples",
        str(assets_dir / "external_test_samples.csv"),
        "--artifact-dir",
        str(train_out / "artifacts" / "tsl51"),
        "--out-dir",
        str(eval_out),
    ]
    run(eval_cmd)
    summary["evaluated"] = True
    summary["eval_summary"] = str(eval_out / "summary.json")
    summary["status"] = "complete"
    return summary


def main() -> None:
    args = parse_args()
    summary = pipeline(args)
    summary_path = args.reports_dir / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
