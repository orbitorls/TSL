from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prioritize prelabel manifest rows for human review toward a trusted TSL51 holdout."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--track", default="tsl51")
    parser.add_argument("--holdout-target-rows", type=int, default=10)
    parser.add_argument("--holdout-target-labels", type=int, default=5)
    parser.add_argument("--val-target-rows", type=int, default=5)
    parser.add_argument("--train-target-rows", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path, track: str) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = [dict(row) for row in reader if (row.get("track") or "").strip() == track]
        return rows, list(reader.fieldnames)


def group_id(row: dict[str, str]) -> str:
    return (row.get("session_id") or row.get("video_id") or row.get("path") or "").strip()


def build_review_queue(
    rows: list[dict[str, str]],
    *,
    holdout_target_rows: int,
    holdout_target_labels: int,
    val_target_rows: int,
    train_target_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    reviewed = [row for row in rows if row.get("quality_status") == "reviewed"]
    prelabel = [row for row in rows if row.get("quality_status") == "prelabel"]
    used_groups = {group_id(row) for row in reviewed}

    selected: list[dict[str, str]] = []
    selected_groups: set[str] = set()
    phase_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    holdout_labels = {row["label"] for row in reviewed}

    def phase_needs_more(phase: str, target_rows: int, label_target: int | None = None) -> bool:
        if len(phase_rows[phase]) < target_rows:
            return True
        if label_target is None:
            return False
        current_labels = len({item["label"] for item in phase_rows[phase]})
        return current_labels < label_target

    def add_for_phase(phase: str, target_rows: int, label_target: int | None = None) -> None:
        while phase_needs_more(phase, target_rows, label_target):
            candidates = sorted(
                prelabel,
                key=lambda row: (
                    group_id(row) in used_groups or group_id(row) in selected_groups,
                    row["label"] in holdout_labels if phase == "external_test" else False,
                    row["label"],
                    group_id(row),
                ),
            )
            row = next(
                (
                    candidate
                    for candidate in candidates
                    if group_id(candidate) not in used_groups
                    and group_id(candidate) not in selected_groups
                ),
                None,
            )
            if row is None:
                break
            gid = group_id(row)
            queued = dict(row)
            queued["recommended_split"] = phase
            queued["review_priority"] = str(len(selected) + 1)
            queued["review_action"] = "confirm_label_or_reject"
            queued["review_decision"] = ""
            queued["reviewed_label"] = ""
            selected.append(queued)
            phase_rows[phase].append(queued)
            selected_groups.add(gid)
            if phase == "external_test":
                holdout_labels.add(row["label"])

    reviewed_holdout_rows = len(reviewed)
    reviewed_holdout_labels = len({row["label"] for row in reviewed})
    add_for_phase(
        "external_test",
        max(0, holdout_target_rows - reviewed_holdout_rows),
        max(0, holdout_target_labels - reviewed_holdout_labels),
    )
    add_for_phase("val", val_target_rows)
    add_for_phase("train", train_target_rows)

    summary = {
        "reviewed_rows": len(reviewed),
        "prelabel_rows": len(prelabel),
        "queue_rows": len(selected),
        "queue_groups": len(selected_groups),
        "queue_labels": len({row["label"] for row in selected}),
        "targets": {
            "holdout_rows": holdout_target_rows,
            "holdout_labels": holdout_target_labels,
            "val_rows": val_target_rows,
            "train_rows": train_target_rows,
        },
        "queued_by_recommended_split": {
            split: {
                "rows": len(items),
                "groups": len({group_id(row) for row in items}),
                "labels": len({row["label"] for row in items}),
            }
            for split, items in sorted(phase_rows.items())
        },
        "ready_to_review": bool(selected),
    }
    return selected, summary


def write_queue(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    extra = ["recommended_split", "review_priority", "review_action", "review_decision", "reviewed_label"]
    output_fields = [field for field in fieldnames if field in rows[0]] if rows else list(fieldnames)
    for field in extra:
        if field not in output_fields:
            output_fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_rows(args.manifest, args.track)
    queue, summary = build_review_queue(
        rows,
        holdout_target_rows=args.holdout_target_rows,
        holdout_target_labels=args.holdout_target_labels,
        val_target_rows=args.val_target_rows,
        train_target_rows=args.train_target_rows,
    )
    write_queue(args.out, queue, fieldnames)
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
