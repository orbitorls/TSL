from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiple TSL51 external eval strategy reports.")
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def summarize(paths: list[Path]) -> dict[str, Any]:
    reports = []
    for path in paths:
        item = read_json(path)
        reports.append(
            {
                "summary": str(path),
                "strategy": item.get("strategy"),
                "samples_file": item.get("samples_file"),
                "artifact_dir": item.get("artifact_dir"),
                "total_samples": int(item.get("total_samples") or 0),
                "detected_samples": int(item.get("detected_samples") or 0),
                "top1_accuracy": float(item.get("top1_accuracy") or 0.0),
                "top3_accuracy": float(item.get("top3_accuracy") or 0.0),
                "predictions_csv": item.get("predictions_csv"),
            }
        )
    best_top1 = max(reports, key=lambda item: item["top1_accuracy"], default=None)
    best_top3 = max(reports, key=lambda item: item["top3_accuracy"], default=None)
    return {
        "reports": reports,
        "best_top1": best_top1,
        "best_top3": best_top3,
        "all_strategies_failed": bool(reports) and all(item["top1_accuracy"] == 0.0 for item in reports),
    }


def main() -> None:
    args = parse_args()
    report = summarize(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
