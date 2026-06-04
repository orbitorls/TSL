#!/usr/bin/env python3
"""Evaluate TSL51 on webcam holdout clips and suggest live threshold/margin tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "webcam_holdout" / "webcam_holdout.csv"
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / ".tools"
    / "tsl51_experiments"
    / "full51_v3_external_weighted"
    / "artifacts"
    / "tsl51"
)
DEFAULT_OUT = REPO_ROOT / "reports" / "webcam_holdout_eval"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_tsl51_video.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strategy", default="uniform", choices=("uniform", "first", "last", "sliding"))
    parser.add_argument("--target-fps", type=float, default=15.0)
    return parser.parse_args()


def _suggest_tuning(summary: dict, predictions_csv: Path) -> dict[str, object]:
    top1 = float(summary.get("top1_accuracy") or 0.0)
    suggestions: dict[str, object] = {
        "top1_accuracy": top1,
        "recommended_preset": "balanced",
        "recommended_threshold": 0.62,
        "recommended_min_confidence_margin": 0.10,
        "recommended_commit_on_preview": True,
        "recommended_min_sign_frames": 4,
        "recommended_sign_end_frames": 3,
        "recommended_transcript_stable_frames": 1,
        "latency_benchmark_script": "scripts/benchmark_tsl51_live_presets.py",
        "notes": [],
    }

    if top1 >= 0.7:
        suggestions["notes"].append(
            "Runtime accuracy sufficient; use balanced preset (default) for multi-word sentences."
        )
    elif top1 >= 0.5:
        suggestions["recommended_threshold"] = 0.7
        suggestions["recommended_min_confidence_margin"] = 0.15
        suggestions["notes"].append("Raise threshold/margin slightly; re-record holdout with clearer sign boundaries.")
    else:
        suggestions["recommended_threshold"] = 0.75
        suggestions["recommended_min_confidence_margin"] = 0.18
        suggestions["notes"].append("Consider Phase 3 retrain after fixing holdout recording quality.")

    if not predictions_csv.exists():
        return suggestions

    wrong = Counter()
    import csv

    with predictions_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected = row.get("expected") or row.get("label") or ""
            predicted = row.get("predicted") or row.get("top1") or ""
            if expected and predicted and expected != predicted:
                wrong[(expected, predicted)] += 1

    if wrong:
        suggestions["top_confusions"] = [
            {"expected": exp, "predicted": pred, "count": count}
            for (exp, pred), count in wrong.most_common(5)
        ]
    return suggestions


def main() -> int:
    args = parse_args()
    if not args.samples.exists():
        print(
            f"[INFO] Holdout not found: {args.samples}\n"
            "Record clips first:\n"
            "  python scripts/record_webcam_holdout.py "
            f"--out-dir {args.samples.parent} "
            f"--labels {args.artifact_dir / 'tsl51_labels.json'}"
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--samples",
        str(args.samples),
        "--artifact-dir",
        str(args.artifact_dir),
        "--out-dir",
        str(args.out_dir),
        "--strategy",
        args.strategy,
        "--target-fps",
        str(args.target_fps),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    summary_path = args.out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tuning = _suggest_tuning(summary, args.out_dir / "predictions.csv")
    tuning_path = args.out_dir / "tuning_suggestions.json"
    tuning_path.write_text(json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"summary": summary, "tuning": tuning}, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {tuning_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
