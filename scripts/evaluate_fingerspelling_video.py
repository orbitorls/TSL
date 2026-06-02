from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))

from src.keypoints import extract_and_normalize  # noqa: E402
from src.external_benchmark import summarize_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Thai fingerspelling predictions on labeled video timestamps."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--samples",
        required=True,
        type=Path,
        help="CSV with expected,time_s columns. One row per labeled sample.",
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "reports" / "external_fingerspelling_eval", type=Path)
    parser.add_argument("--offset", action="append", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in raw.items()}


def read_samples(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["expected"], float(row["time_s"])))
    return rows


def main() -> None:
    args = parse_args()
    if args.offset is None:
        args.offset = [0.0]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.artifact_dir / "model.keras")
    scaler = joblib.load(args.artifact_dir / "scaler.pkl")
    labels = load_labels(args.artifact_dir / "labels.json")
    samples = read_samples(args.samples)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    rows: list[dict[str, object]] = []
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
    ) as hands:
        for expected, center_s in samples:
            for offset_s in args.offset:
                time_s = max(0.0, center_s + offset_s)
                cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    rows.append(
                        {
                            "expected": expected,
                            "time_s": round(time_s, 3),
                            "detected_hand": False,
                            "predicted": "",
                            "confidence": "",
                            "top_k": "",
                            "correct": False,
                        }
                    )
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                feat = extract_and_normalize(results)
                if feat is None:
                    rows.append(
                        {
                            "expected": expected,
                            "time_s": round(time_s, 3),
                            "detected_hand": False,
                            "predicted": "",
                            "confidence": "",
                            "top_k": "",
                            "correct": False,
                        }
                    )
                    continue

                feat_scaled = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
                probs = model.predict(feat_scaled, verbose=0)[0]
                order = np.argsort(probs)[::-1]
                pred_idx = int(order[0])
                top_idxs = order[: max(1, int(args.top_k))]
                predicted = labels[str(pred_idx)]
                topk = [
                    f"{labels[str(int(idx))]}:{float(probs[int(idx)]):.4f}"
                    for idx in top_idxs
                ]
                rows.append(
                    {
                        "expected": expected,
                        "time_s": round(time_s, 3),
                        "detected_hand": True,
                        "predicted": predicted,
                        "confidence": round(float(probs[pred_idx]), 6),
                        "top_k": " | ".join(topk),
                        "correct": predicted == expected,
                    }
                )

    cap.release()

    bench = summarize_predictions(rows, detected_key="detected_hand")
    summary = {
        "video": str(args.video),
        "artifact_dir": str(args.artifact_dir),
        "samples_file": str(args.samples),
        "labeled_samples": len(samples),
        "frames_per_sample": len(args.offset),
        "total_frames": len(rows),
        "hand_detected_frames": bench["detected_samples"],
        "rejected_frames": bench["rejected_samples"],
        "hand_detection_rate": bench["detection_rate"],
        "top1_correct": bench["top1_correct"],
        "top3_correct": bench["top3_correct"],
        "top1_accuracy_all": bench["top1_accuracy"],
        "top3_accuracy_all": bench["top3_accuracy"],
        "top1_accuracy_on_detected": (
            bench["top1_correct"] / bench["detected_samples"]
            if bench["detected_samples"]
            else 0.0
        ),
        "confusion_pairs": bench["confusion_pairs"],
    }

    predictions_path = args.out_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
