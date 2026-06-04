#!/usr/bin/env python3
"""Run webcam holdout eval when recorded; otherwise report domain-proxy findings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT = REPO_ROOT / "data" / "webcam_holdout" / "webcam_holdout.csv"
HOLDOUT_EVAL = REPO_ROOT / "scripts" / "evaluate_webcam_holdout.py"
DIAGNOSIS = REPO_ROOT / "reports" / "live_vs_clip_diagnosis"
THSL_PROXY = REPO_ROOT / "work" / "thsl_crossval" / "thsl_samples.csv"
OUT = REPO_ROOT / "reports" / "webcam_holdout_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-csv", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"holdout_csv": str(args.holdout_csv)}

    reviewed_summary = DIAGNOSIS / "reviewed_external_postfix" / "summary.json"
    if not reviewed_summary.exists():
        reviewed_summary = DIAGNOSIS / "reviewed_external" / "summary.json"
    thsl_summary = DIAGNOSIS / "thsl_failed_10" / "summary.json"

    if reviewed_summary.exists():
        report["reviewed_diagnosis"] = json.loads(reviewed_summary.read_text(encoding="utf-8"))
    if thsl_summary.exists():
        report["thsl_failed_diagnosis"] = json.loads(thsl_summary.read_text(encoding="utf-8"))

    if args.holdout_csv.exists():
        cmd = [
            sys.executable,
            str(HOLDOUT_EVAL),
            "--samples",
            str(args.holdout_csv),
            "--out-dir",
            str(args.out_dir / "holdout_eval"),
        ]
        subprocess.run(cmd, check=False)
        holdout_summary = args.out_dir / "holdout_eval" / "summary.json"
        if holdout_summary.exists():
            report["holdout_eval"] = json.loads(holdout_summary.read_text(encoding="utf-8"))
            report["mode"] = "real_webcam_holdout"
        else:
            report["mode"] = "holdout_csv_present_eval_failed"
    else:
        report["mode"] = "proxy_only"
        report["note"] = (
            "No webcam holdout recorded. Run: "
            "python scripts/record_webcam_holdout.py --out-dir data/webcam_holdout"
        )
        if THSL_PROXY.exists():
            proxy_out = args.out_dir / "thsl_domain_proxy"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_tsl51_video.py"),
                "--samples",
                str(THSL_PROXY),
                "--artifact-dir",
                str(
                    REPO_ROOT
                    / ".tools"
                    / "tsl51_experiments"
                    / "full51_v3_external_weighted"
                    / "artifacts"
                    / "tsl51"
                ),
                "--out-dir",
                str(proxy_out),
                "--strategy",
                "uniform",
            ]
            subprocess.run(cmd, check=False)
            proxy_summary = proxy_out / "summary.json"
            if proxy_summary.exists():
                report["thsl_domain_proxy"] = json.loads(proxy_summary.read_text(encoding="utf-8"))

    reviewed = report.get("reviewed_diagnosis") or {}
    thsl = report.get("thsl_failed_diagnosis") or report.get("thsl_domain_proxy") or {}
    pipeline_gap = reviewed.get("pipeline_gap")
    offline_thsl = thsl.get("top1_accuracy") or thsl.get("baseline_offline_top1")

    conclusions: list[str] = []
    if pipeline_gap is not None and pipeline_gap >= 0.2:
        conclusions.append(
            "Live pipeline mismatch was significant on training-domain clips; "
            "seq_buf commit + lower min_sign_frames applied."
        )
    if offline_thsl is not None and float(offline_thsl) < 0.5:
        conclusions.append(
            "Model domain shift is primary for th-sl / webcam-like input (~9% offline). "
            "Runtime tuning alone cannot fix; Phase 3 retrain recommended after holdout."
        )
    if args.holdout_csv.exists() and report.get("holdout_eval"):
        top1 = float(report["holdout_eval"].get("top1_accuracy") or 0.0)
        if top1 >= 0.7:
            conclusions.append("Webcam holdout meets 70% target; focus on live UX tuning.")
        else:
            conclusions.append(f"Webcam holdout Top-1 {top1:.1%} below 70%; retrain with holdout data.")

    report["conclusions"] = conclusions
    out_path = args.out_dir / "summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
