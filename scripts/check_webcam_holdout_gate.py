#!/usr/bin/env python3
"""Gate Phase 3 retrain: exit 0 when webcam holdout Top-1 meets target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "reports" / "webcam_holdout_eval" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-top1", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print(f"[SKIP] No holdout summary at {args.summary}; record webcam holdout first.")
        return 2

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    top1 = float(summary.get("top1_accuracy") or 0.0)
    passed = top1 >= args.target_top1
    status = "pass" if passed else "needs_retrain"
    print(json.dumps({"top1_accuracy": top1, "target": args.target_top1, "status": status}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
