from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALID_DECISIONS = {"approved", "rejected"}
REQUIRED_SPLITS = ("train", "val", "external_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate human review decisions before building trusted external TSL51 assets."
    )
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--allow-missing-decisions", action="store_true", default=False)
    parser.add_argument("--skip-path-check", action="store_true", default=False)
    parser.add_argument("--min-approved-per-split", type=int, default=1)
    parser.add_argument("--min-groups-per-split", type=int, default=1)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return [dict(row) for row in reader]


def load_allowed_labels(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(value) for value in raw.values()}
    if isinstance(raw, list):
        return {str(value) for value in raw}
    raise ValueError(f"{path} must contain a JSON object or list of labels")


def row_id(row: dict[str, str]) -> str:
    return f"{row.get('video_id') or row.get('path')}:{row.get('start_s')}-{row.get('end_s')}"


def group_id(row: dict[str, str]) -> str:
    return (row.get("session_id") or row.get("video_id") or row.get("path") or "").strip()


def approved_label(row: dict[str, str]) -> str:
    return (row.get("reviewed_label") or row.get("label") or "").strip()


def approved_split(row: dict[str, str]) -> str:
    return (row.get("recommended_split") or row.get("split") or "").strip()


def split_summary(rows: list[dict[str, str]], split: str) -> dict[str, int]:
    matching = [row for row in rows if approved_split(row) == split]
    return {
        "rows": len(matching),
        "labels": len({approved_label(row) for row in matching}),
        "groups": len({group_id(row) for row in matching}),
    }


def validate_rows(
    rows: list[dict[str, str]],
    *,
    allowed_labels: set[str],
    repo_root: Path = Path("."),
    allow_missing_decisions: bool = False,
    skip_path_check: bool = False,
    min_approved_per_split: int = 1,
    min_groups_per_split: int = 1,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    approved_rows: list[dict[str, str]] = []
    rejected_rows = 0
    missing_decisions = 0
    group_to_split: dict[str, str] = {}

    for row in rows:
        rid = row_id(row)
        decision = (row.get("review_decision") or "").strip().lower()
        if not decision:
            missing_decisions += 1
            if not allow_missing_decisions:
                errors.append(f"{rid} missing review_decision")
            continue
        if decision not in VALID_DECISIONS:
            errors.append(f"{rid} has unsupported review_decision {decision!r}")
            continue
        if decision == "rejected":
            rejected_rows += 1
            continue

        label = approved_label(row)
        split = approved_split(row)
        group = group_id(row)
        if not label:
            errors.append(f"{rid} approved without label")
        elif label not in allowed_labels:
            errors.append(f"{rid} approved label {label!r} is not in labels")
        if split not in REQUIRED_SPLITS:
            errors.append(f"{rid} has unsupported recommended_split {split!r}")
        if not group:
            errors.append(f"{rid} has no leakage group id")
        elif group in group_to_split and group_to_split[group] != split:
            errors.append(
                f"{rid} group {group!r} appears in both {group_to_split[group]!r} and {split!r}"
            )
        else:
            group_to_split[group] = split
        path = (row.get("path") or "").strip()
        if not skip_path_check:
            if not path:
                errors.append(f"{rid} approved without path")
            else:
                candidate = Path(path)
                if not candidate.is_absolute():
                    candidate = repo_root / candidate
                if not candidate.exists():
                    errors.append(f"{rid} path does not exist: {path}")
        approved_rows.append(row)

    split_summaries = {
        split: split_summary(approved_rows, split) for split in REQUIRED_SPLITS
    }
    for split, stats in split_summaries.items():
        if stats["rows"] < min_approved_per_split:
            errors.append(
                f"{split} has {stats['rows']} approved rows; need at least {min_approved_per_split}"
            )
        if stats["groups"] < min_groups_per_split:
            errors.append(
                f"{split} has {stats['groups']} approved groups; need at least {min_groups_per_split}"
            )
    if allow_missing_decisions and missing_decisions:
        warnings.append(f"{missing_decisions} rows have no review_decision and were ignored")

    return {
        "input_rows": len(rows),
        "approved_rows": len(approved_rows),
        "rejected_rows": rejected_rows,
        "missing_decisions": missing_decisions,
        "splits": split_summaries,
        "errors": errors,
        "warnings": warnings,
        "ready": not errors,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.queue)
    summary = validate_rows(
        rows,
        allowed_labels=load_allowed_labels(args.labels),
        repo_root=args.repo_root,
        allow_missing_decisions=args.allow_missing_decisions,
        skip_path_check=args.skip_path_check,
        min_approved_per_split=args.min_approved_per_split,
        min_groups_per_split=args.min_groups_per_split,
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
