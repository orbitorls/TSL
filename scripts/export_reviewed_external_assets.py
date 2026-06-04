from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "external_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed split manifest assets for cache building and TSL51 video evaluation."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = [dict(row) for row in reader if (row.get("quality_status") or "") == "reviewed"]
        return rows, list(reader.fieldnames)


def rows_for_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if (row.get("split") or "") == split]


def group_id(row: dict[str, str]) -> str:
    return (row.get("session_id") or row.get("video_id") or row.get("path") or "").strip()


def split_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "labels": len({row["label"] for row in rows}),
        "groups": len({group_id(row) for row in rows}),
    }


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eval_sample_row(row: dict[str, str], repo_root: Path) -> dict[str, str]:
    path = Path(row["path"])
    if not path.is_absolute():
        path = repo_root / path
    return {
        "id": f"{row['video_id']}_{row['label']}",
        "path": str(path),
        "expected": row["label"],
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "split": row["split"],
        "source": "reviewed_external_manifest",
    }


def write_eval_samples(path: Path, rows: list[dict[str, str]], repo_root: Path) -> None:
    fieldnames = ["id", "path", "expected", "start_s", "end_s", "split", "source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_sample_row(row, repo_root) for row in rows)


def export_assets(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    out_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"splits": {}, "files": {}}
    for split in SPLITS:
        split_rows = rows_for_split(rows, split)
        manifest_path = out_dir / f"{split}_manifest.csv"
        write_manifest(manifest_path, split_rows, fieldnames)
        summary["splits"][split] = split_summary(split_rows)
        summary["files"][f"{split}_manifest"] = str(manifest_path)
        if split == "external_test":
            samples_path = out_dir / "external_test_samples.csv"
            write_eval_samples(samples_path, split_rows, repo_root)
            summary["files"]["external_test_samples"] = str(samples_path)
    summary["ready_for_training"] = (
        summary["splits"]["train"]["rows"] > 0 and summary["splits"]["val"]["rows"] > 0
    )
    summary["ready_for_eval"] = summary["splits"]["external_test"]["rows"] > 0
    return summary


def main() -> None:
    args = parse_args()
    rows, fieldnames = read_manifest(args.manifest)
    summary = export_assets(rows, fieldnames, args.out_dir, args.repo_root.resolve())
    summary_path = args.summary or args.out_dir / "reviewed_external_assets_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
