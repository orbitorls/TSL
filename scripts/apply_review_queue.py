from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


APPROVED = "approved"
REJECTED = "rejected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply human review decisions from a review queue into a reviewed manifest."
    )
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=None,
        help="Optional manifest whose existing reviewed rows should be included in the output.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--allow-missing-decisions", action="store_true", default=False)
    return parser.parse_args()


def read_queue(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def read_base_reviewed(path: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    if path is None:
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = [
            dict(row)
            for row in reader
            if (row.get("quality_status") or "").strip() == "reviewed"
        ]
        return rows, list(reader.fieldnames)


def merge_reviewed_rows(
    base_rows: list[dict[str, str]],
    approved_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in base_rows:
        merged[row_key(row)] = dict(row)
    for row in approved_rows:
        merged[row_key(row)] = dict(row)
    return list(merged.values())


def apply_decisions(
    rows: list[dict[str, str]],
    *,
    allow_missing_decisions: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    approved_rows: list[dict[str, str]] = []
    rejected_rows = 0
    missing_decisions: list[str] = []

    for row in rows:
        decision = (row.get("review_decision") or "").strip().lower()
        if not decision:
            if allow_missing_decisions:
                continue
            missing_decisions.append(row_id(row))
            continue
        if decision not in {APPROVED, REJECTED}:
            raise ValueError(f"{row_id(row)} has unsupported review_decision {decision!r}")
        if decision == REJECTED:
            rejected_rows += 1
            continue

        reviewed = dict(row)
        reviewed["label"] = (reviewed.get("reviewed_label") or reviewed.get("label") or "").strip()
        if not reviewed["label"]:
            raise ValueError(f"{row_id(row)} is approved but has no label")
        reviewed["split"] = (reviewed.get("recommended_split") or reviewed.get("split") or "").strip()
        if reviewed["split"] not in {"train", "val", "external_test"}:
            raise ValueError(f"{row_id(row)} has unsupported recommended_split {reviewed['split']!r}")
        reviewed["quality_status"] = "reviewed"
        approved_rows.append(reviewed)

    if missing_decisions:
        raise ValueError(
            "missing review_decision for " + ", ".join(missing_decisions[:10])
        )

    summary = {
        "input_rows": len(rows),
        "approved_rows": len(approved_rows),
        "rejected_rows": rejected_rows,
        "output_splits": {
            split: split_summary(approved_rows, split)
            for split in ("train", "val", "external_test")
        },
        "ready_for_splitter": bool(approved_rows),
    }
    return approved_rows, summary


def group_id(row: dict[str, str]) -> str:
    return (row.get("session_id") or row.get("video_id") or row.get("path") or "").strip()


def split_summary(rows: list[dict[str, str]], split: str) -> dict[str, int]:
    matching = [row for row in rows if row.get("split") == split]
    return {
        "rows": len(matching),
        "labels": len({row["label"] for row in matching}),
        "groups": len({group_id(row) for row in matching}),
    }


def row_id(row: dict[str, str]) -> str:
    return f"{row.get('video_id') or row.get('path')}:{row.get('start_s')}-{row.get('end_s')}"


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        (row.get("video_id") or "").strip(),
        (row.get("path") or "").strip(),
        (row.get("start_s") or "").strip(),
        (row.get("end_s") or "").strip(),
    )


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    unique_fieldnames = list(dict.fromkeys(fieldnames))
    passthrough = [
        field
        for field in unique_fieldnames
        if field not in {"recommended_split", "review_priority", "review_action", "review_decision", "reviewed_label"}
    ]
    for field in ("reviewer_id", "reviewed_at", "review_notes"):
        if field not in passthrough:
            passthrough.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=passthrough, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_queue(args.queue)
    base_rows, base_fieldnames = read_base_reviewed(args.base_manifest)
    reviewed, summary = apply_decisions(rows, allow_missing_decisions=args.allow_missing_decisions)
    merged = merge_reviewed_rows(base_rows, reviewed)
    summary["base_reviewed_rows"] = len(base_rows)
    summary["output_reviewed_rows"] = len(merged)
    summary["merged_output_splits"] = {
        split: split_summary(merged, split)
        for split in ("train", "val", "external_test")
    }
    write_manifest(args.out, merged, [*base_fieldnames, *fieldnames])
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
