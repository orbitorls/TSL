from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))

from src.external_dataset import (  # noqa: E402
    FINGERSPELLING_FEATURE_DIM,
    TSL51_FEATURE_DIM,
    TSL51_SEQ_LEN,
    ManifestRow,
    build_fingerspelling_cache,
    build_tsl51_cache,
    load_label_set,
    load_manifest,
)
from src.keypoints import extract_and_normalize  # noqa: E402
from src.sequence_keypoints import extract_holistic_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NPZ feature caches from reviewed external video manifests."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=("fingerspelling", "tsl51"))
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--require-reviewed", action="store_true", default=False)
    parser.add_argument("--min-detected-frames", type=int, default=12)
    parser.add_argument("--split", action="append", default=None)
    return parser.parse_args()


def ordered_labels(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[str(i)]) for i in range(len(raw))]
    raise ValueError(f"{path} must contain a label list or dict")


def iter_times(start_s: float, end_s: float, target_fps: float):
    step = 1.0 / max(target_fps, 0.1)
    t = start_s
    while t <= end_s:
        yield t
        t += step


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def extract_fs_sample(row: ManifestRow, target_fps: float, hands) -> tuple[list[np.ndarray], dict[str, object]]:
    cap = open_video(row.path)
    decoded = 0
    features: list[np.ndarray] = []
    try:
        for time_s in iter_times(row.start_s, row.end_s, target_fps):
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            decoded += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            feat = extract_and_normalize(results)
            if feat is not None:
                features.append(feat.astype(np.float32))
    finally:
        cap.release()
    return features, {
        "video_id": row.video_id,
        "label": row.label,
        "split": row.split,
        "decoded_frames": decoded,
        "detected_frames": len(features),
        "accepted": bool(features),
    }


def sequence_from_features(features: list[np.ndarray], seq_len: int = TSL51_SEQ_LEN) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[0] >= seq_len:
        idx = np.linspace(0, arr.shape[0] - 1, seq_len).round().astype(int)
        return arr[idx].astype(np.float32)
    out = np.zeros((seq_len, TSL51_FEATURE_DIM), dtype=np.float32)
    if arr.shape[0] > 0:
        out[seq_len - arr.shape[0] :] = arr
    return out


def extract_tsl51_sample(row: ManifestRow, target_fps: float, holistic) -> tuple[np.ndarray | None, dict[str, object]]:
    cap = open_video(row.path)
    decoded = 0
    features: list[np.ndarray] = []
    try:
        for time_s in iter_times(row.start_s, row.end_s, target_fps):
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            decoded += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            feat = extract_holistic_frame(results)
            if feat is not None:
                features.append(feat.astype(np.float32))
    finally:
        cap.release()
    return (sequence_from_features(features) if features else None), {
        "video_id": row.video_id,
        "label": row.label,
        "split": row.split,
        "decoded_frames": decoded,
        "detected_frames": len(features),
        "accepted": bool(features),
    }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    label_list = ordered_labels(args.labels)
    known_labels = load_label_set(args.labels)
    rows = load_manifest(
        args.manifest,
        known_labels,
        allowed_tracks={args.track},
        require_reviewed=args.require_reviewed,
    )
    if args.split:
        allowed_splits = set(args.split)
        rows = [row for row in rows if row.split in allowed_splits]

    report_rows: list[dict[str, object]] = []
    if args.track == "fingerspelling":
        samples: list[tuple[str, np.ndarray]] = []
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.45,
        ) as hands:
            for row in rows:
                features, report = extract_fs_sample(row, args.target_fps, hands)
                report["accepted"] = len(features) >= 1
                report_rows.append(report)
                samples.extend((row.label, feat) for feat in features)
        cache = build_fingerspelling_cache(samples, label_list)
    else:
        samples = []
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            for row in rows:
                sequence, report = extract_tsl51_sample(row, args.target_fps, holistic)
                report["accepted"] = (
                    sequence is not None
                    and int(report["detected_frames"]) >= args.min_detected_frames
                )
                report_rows.append(report)
                if report["accepted"] and sequence is not None:
                    samples.append((row.label, sequence))
        cache = build_tsl51_cache(samples, label_list)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **cache)
    report_path = args.report or args.out.with_suffix(".report.csv")
    write_report(report_path, report_rows)
    summary = {
        "track": args.track,
        "manifest": str(args.manifest),
        "output": str(args.out),
        "report": str(report_path),
        "manifest_rows": len(rows),
        "accepted_rows": sum(1 for row in report_rows if row["accepted"]),
        "samples": int(cache["X"].shape[0]),
        "shape": list(cache["X"].shape),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
