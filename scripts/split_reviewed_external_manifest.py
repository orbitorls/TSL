from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/val/external_test splits from reviewed external manifest rows."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--track", default="tsl51")
    parser.add_argument("--min-holdout-rows", type=int, default=10)
    parser.add_argument("--min-holdout-labels", type=int, default=5)
    parser.add_argument("--min-val-rows", type=int, default=5)
    return parser.parse_args()


def row_group_id(row: dict[str, str]) -> str:
    for key in ("session_id", "video_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return (row.get("path") or "").strip()


def read_reviewed_rows(path: Path, track: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = []
        for row in reader:
            if (row.get("track") or "").strip() != track:
                continue
            if (row.get("quality_status") or "").strip() != "reviewed":
                continue
            rows.append(dict(row))
    return rows


def split_reviewed_rows(
    rows: list[dict[str, str]],
    *,
    min_holdout_rows: int,
    min_holdout_labels: int,
    min_val_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row_group_id(row)].append(row)

    ordered_groups = sorted(
        groups.values(),
        key=lambda items: (len({row["label"] for row in items}), len(items), row_group_id(items[0])),
        reverse=True,
    )
    split_rows: list[dict[str, str]] = []
    holdout: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    train: list[dict[str, str]] = []

    for group in ordered_groups:
        target = train
        if len(holdout) < min_holdout_rows or len({row["label"] for row in holdout}) < min_holdout_labels:
            target = holdout
        elif len(val) < min_val_rows:
            target = val
        target.extend(group)

    for split_name, items in (("train", train), ("val", val), ("external_test", holdout)):
        for row in items:
            split_row = dict(row)
            split_row["split"] = split_name
            split_rows.append(split_row)

    report = {
        "reviewed_rows": len(rows),
        "reviewed_groups": len(groups),
        "reviewed_labels": len({row["label"] for row in rows}),
        "splits": {
            "train": _split_summary(train),
            "val": _split_summary(val),
            "external_test": _split_summary(holdout),
        },
        "ready_for_external_holdout": (
            len(holdout) >= min_holdout_rows
            and len({row["label"] for row in holdout}) >= min_holdout_labels
            and len(val) >= min_val_rows
            and len(train) > 0
        ),
    }
    if not report["ready_for_external_holdout"]:
        report["reason"] = (
            "not enough reviewed rows/groups to create train, val, and external_test splits "
            "without video/session leakage"
        )
    return split_rows, report


def _split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "groups": len({row_group_id(row) for row in rows}),
        "labels": len({row["label"] for row in rows}),
    }


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reviewed = read_reviewed_rows(args.manifest, args.track)
    split_rows, report = split_reviewed_rows(
        reviewed,
        min_holdout_rows=args.min_holdout_rows,
        min_holdout_labels=args.min_holdout_labels,
        min_val_rows=args.min_val_rows,
    )
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
    if "split" not in fieldnames:
        fieldnames.append("split")
    write_manifest(args.out, split_rows, fieldnames)
    report_path = args.report or args.out.with_suffix(".summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
