from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report TSL51 metadata and artifact label coverage.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_artifact_labels(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[str(i)]) for i in range(len(raw))]
    raise ValueError(f"{path} must contain a label list or numeric-string label map")


def build_report(metadata_path: Path, labels_path: Path) -> dict[str, Any]:
    total_rows = 0
    null_rows = 0
    row_counts: Counter[str] = Counter()
    video_ids: dict[str, set[str]] = defaultdict(set)

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            sign_id = (row.get("sign_id") or "").strip()
            if not sign_id:
                continue
            if sign_id == "null_act":
                null_rows += 1
                continue
            row_counts[sign_id] += 1
            video_id = (row.get("original_video_id") or row.get("video_id") or "").strip()
            if video_id:
                video_ids[sign_id].add(video_id)

    artifact_labels = load_artifact_labels(labels_path)
    metadata_labels = set(row_counts)
    artifact_label_set = set(artifact_labels)
    per_class = {
        label: {
            "metadata_rows": row_counts[label],
            "unique_video_ids": len(video_ids[label]),
            "in_artifact": label in artifact_label_set,
        }
        for label in sorted(metadata_labels)
    }

    return {
        "metadata": str(metadata_path),
        "labels": str(labels_path),
        "total_rows": total_rows,
        "null_rows": null_rows,
        "non_null_rows": sum(row_counts.values()),
        "non_null_class_count": len(metadata_labels),
        "artifact_class_count": len(artifact_labels),
        "missing_from_artifact": sorted(metadata_labels - artifact_label_set),
        "extra_in_artifact": sorted(artifact_label_set - metadata_labels),
        "artifact_labels": artifact_labels,
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.metadata, args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"metadata_classes={report['non_null_class_count']} "
        f"artifact_classes={report['artifact_class_count']} "
        f"missing={len(report['missing_from_artifact'])}"
    )


if __name__ == "__main__":
    main()
