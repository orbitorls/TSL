#!/usr/bin/env python3
"""Fine-tune TSL51 v3 with webcam train clips when holdout Top-1 is below target."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / ".tools"
    / "tsl51_experiments"
    / "full51_v3_external_weighted"
    / "artifacts"
    / "tsl51"
)
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "webcam_train" / "webcam_train.csv"
DEFAULT_HOLDOUT_CSV = REPO_ROOT / "data" / "webcam_holdout" / "webcam_holdout.csv"
DEFAULT_BASE_CACHE = (
    REPO_ROOT
    / ".tools"
    / "train_runs_25690531-175818"
    / "tsl51_e40_b32_s12000"
    / "work"
    / "features"
    / "tsl51_features.npz"
)
VENV_PYTHON = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--holdout-csv", type=Path, default=DEFAULT_HOLDOUT_CSV)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "work" / "webcam_finetune")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".tools" / "tsl51_experiments" / "full51_v4_webcam_seed" / "artifacts" / "tsl51")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--external-weight", type=float, default=50.0)
    parser.add_argument("--target-top1", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=None)
    return parser.parse_args()


def _python(args: argparse.Namespace) -> str:
    if args.python:
        return args.python
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    python = _python(args)
    labels = args.artifact_dir / "tsl51_labels.json"
    if not labels.exists():
        print(f"[ERROR] labels not found: {labels}")
        return 1
    if not args.train_csv.exists():
        print(
            f"[ERROR] train CSV not found: {args.train_csv}\n"
            "Record train clips first:\n"
            "  python scripts/record_webcam_holdout.py "
            f"--out-dir {args.train_csv.parent} --split train --signs 0:15"
        )
        return 1
    if not args.base_cache.exists():
        print(
            f"[ERROR] base NPZ not found: {args.base_cache}\n"
            "Pass --base-cache to an existing tsl51_features.npz from train_local_all.py"
        )
        return 1

    args.work_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = args.work_dir / "webcam_train_manifest.csv"
    train_npz = args.work_dir / "webcam_train_cache.npz"
    eval_out = args.work_dir / "holdout_eval"

    _run(
        [
            python,
            str(REPO_ROOT / "scripts" / "webcam_holdout_to_manifest.py"),
            "--samples",
            str(args.train_csv),
            "--out",
            str(train_manifest),
            "--split",
            "train",
        ],
        dry_run=args.dry_run,
    )
    _run(
        [
            python,
            str(REPO_ROOT / "scripts" / "build_external_dataset.py"),
            "--manifest",
            str(train_manifest),
            "--track",
            "tsl51",
            "--labels",
            str(labels),
            "--out",
            str(train_npz),
            "--cache-split",
            "train",
            "--target-fps",
            "15",
        ],
        dry_run=args.dry_run,
    )
    _run(
        [
            python,
            str(REPO_ROOT / "scripts" / "train_external_augmented.py"),
            "--track",
            "tsl51",
            "--base-cache",
            str(args.base_cache),
            "--external-cache",
            str(train_npz),
            "--external-cache-split",
            "train",
            "--base-artifact-dir",
            str(args.artifact_dir),
            "--out-dir",
            str(args.out_dir),
            "--epochs",
            str(args.epochs),
            "--external-sample-weight",
            str(args.external_weight),
            "--no-freeze-base",
            "--learning-rate",
            "5e-5",
            "--augment-external",
            "--augment-copies",
            "5",
        ],
        dry_run=args.dry_run,
    )

    if args.holdout_csv.exists():
        _run(
            [
                python,
                str(REPO_ROOT / "scripts" / "evaluate_tsl51_video.py"),
                "--samples",
                str(args.holdout_csv),
                "--artifact-dir",
                str(args.out_dir),
                "--out-dir",
                str(eval_out),
                "--strategy",
                "uniform",
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            summary_path = eval_out / "summary.json"
            if summary_path.exists():
                top1 = float(json.loads(summary_path.read_text(encoding="utf-8")).get("top1_accuracy") or 0.0)
                status = "pass" if top1 >= args.target_top1 else "needs_more_data_or_epochs"
                report = {
                    "artifact_dir": str(args.out_dir),
                    "holdout_top1": top1,
                    "target_top1": args.target_top1,
                    "status": status,
                    "eval_summary": str(summary_path),
                }
                report_path = args.work_dir / "finetune_report.json"
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0 if top1 >= args.target_top1 else 1

    print("[OK] fine-tune steps completed (no holdout eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
