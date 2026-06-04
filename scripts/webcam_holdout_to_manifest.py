#!/usr/bin/env python3
"""Convert record_webcam_holdout CSV to build_external_dataset manifest format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))

from src.external_dataset import MANIFEST_FIELDS  # noqa: E402

SPLIT_MAP = {
    "holdout": "external_test",
    "external_test": "external_test",
    "train": "train",
    "val": "val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path, help="webcam_holdout.csv or webcam_train.csv")
    parser.add_argument("--out", required=True, type=Path, help="Output manifest CSV")
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "external_test", "holdout"),
        help="Override split for all rows (default: map from CSV split column)",
    )
    return parser.parse_args()


def _row_to_manifest(raw: dict[str, str], *, split_override: str | None) -> dict[str, str]:
    clip_id = raw.get("id") or raw.get("video_id") or ""
    path = raw.get("path") or ""
    label = raw.get("expected") or raw.get("label") or ""
    start_s = raw.get("start_s") or "0.0"
    end_s = raw.get("end_s") or ""
    if not end_s:
        raise ValueError(f"row {clip_id}: missing end_s")

    src_split = (raw.get("split") or "holdout").strip()
    split = SPLIT_MAP.get(split_override or src_split, SPLIT_MAP.get(src_split, "external_test"))

    return {
        "video_id": clip_id,
        "path": path,
        "track": "tsl51",
        "label": label,
        "start_s": start_s,
        "end_s": end_s,
        "source_url": f"file:///{Path(path).as_posix()}",
        "split": split,
        "license_note": "own webcam recording for TSL51 adaptation",
        "quality_status": "reviewed",
        "source_type": "own_recording",
        "rights_status": "own_internal_consent",
        "rights_evidence": raw.get("source") or "webcam_holdout",
        "signer_id": raw.get("signer_id") or "local_user",
        "session_id": raw.get("session_id") or "webcam_session",
    }


def main() -> int:
    args = parse_args()
    if not args.samples.exists():
        print(f"[ERROR] samples not found: {args.samples}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with args.samples.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(_row_to_manifest(raw, split_override=args.split))

    if not rows:
        print(f"[ERROR] no rows in {args.samples}", file=sys.stderr)
        return 1

    fieldnames = list(MANIFEST_FIELDS) + ["source_type", "rights_status", "rights_evidence", "signer_id", "session_id"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
