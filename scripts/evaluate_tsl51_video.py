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

from src.sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT, extract_holistic_frame, resample_frames  # noqa: E402
from src.external_benchmark import summarize_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TSL-51 word-sign predictions on labeled video clips."
    )
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "reports" / "external_tsl51_eval", type=Path)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--strategy", choices=("uniform", "first", "last", "sliding"), default="uniform")
    parser.add_argument("--window-frames", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--stride-frames", type=int, default=15)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in raw.items()}


def read_samples(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sequence_from_features(features: list[np.ndarray], seq_len: int, strategy: str) -> np.ndarray:
    if not features:
        raise ValueError("no holistic hand frames detected")
    arr = np.asarray(features, dtype=np.float32)
    if len(arr) >= seq_len:
        if strategy == "first":
            return arr[:seq_len]
        if strategy == "last":
            return arr[-seq_len:]
        # "uniform" and any unrecognised strategy: delegate to canonical function
        return resample_frames(arr, seq_len)
    # T < seq_len: delegate to canonical function (leading zero-pad)
    return resample_frames(arr, seq_len)




def sliding_sequences(
    features: list[np.ndarray],
    seq_len: int,
    window_frames: int,
    stride_frames: int,
) -> list[np.ndarray]:
    if not features:
        raise ValueError("no holistic hand frames detected")
    arr = np.asarray(features, dtype=np.float32)
    if len(arr) <= seq_len:
        return [sequence_from_features(features, seq_len, "uniform")]
    window = max(seq_len, int(window_frames))
    stride = max(1, int(stride_frames))
    sequences: list[np.ndarray] = []
    for start in range(0, max(1, len(arr) - window + 1), stride):
        chunk = arr[start : start + window]
        sequences.append(sequence_from_features(list(chunk), seq_len, "uniform"))
    last_chunk = arr[-window:]
    last_seq = sequence_from_features(list(last_chunk), seq_len, "uniform")
    if not sequences or not np.array_equal(sequences[-1], last_seq):
        sequences.append(last_seq)
    return sequences


def predict_sequences(model, scaler, sequences: list[np.ndarray], seq_len: int) -> tuple[np.ndarray, int]:
    batch = np.asarray(sequences, dtype=np.float32)
    batch_scaled = scaler.transform(batch.reshape(-1, FEATURE_DIM)).reshape(
        len(batch), seq_len, FEATURE_DIM
    ).astype(np.float32)
    probs = model.predict(batch_scaled, verbose=0)
    return np.max(probs, axis=0), len(batch)

def extract_segment_features(
    video_path: Path,
    start_s: float | None,
    end_s: float | None,
    target_fps: float,
    holistic,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / source_fps if source_fps else 0.0
    start = max(0.0, float(start_s or 0.0))
    end = float(end_s) if end_s is not None else duration_s
    if end <= start:
        end = duration_s

    step_s = 1.0 / max(target_fps, 0.1)
    features: list[np.ndarray] = []
    decoded = 0
    detected = 0
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True
        feat = extract_holistic_frame(results)
        if feat is not None:
            detected += 1
            features.append(feat)
        t += step_s

    cap.release()
    return features, {
        "source_fps": round(source_fps, 3),
        "duration_s": round(duration_s, 3),
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "decoded_frames": decoded,
        "hand_detected_frames": detected,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.artifact_dir / "tsl51_labels.json")
    scaler = joblib.load(args.artifact_dir / "tsl51_scaler.pkl")
    model = tf.keras.models.load_model(args.artifact_dir / "tsl51_model.keras")

    rows: list[dict[str, object]] = []
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic:
        for sample in read_samples(args.samples):
            video_path = Path(sample["path"])
            start_s = float(sample["start_s"]) if sample.get("start_s") else None
            end_s = float(sample["end_s"]) if sample.get("end_s") else None
            expected = sample["expected"]
            try:
                features, meta = extract_segment_features(
                    video_path, start_s, end_s, args.target_fps, holistic
                )
                if args.strategy == "sliding":
                    sequences = sliding_sequences(
                        features,
                        args.seq_len,
                        args.window_frames,
                        args.stride_frames,
                    )
                    probs, window_count = predict_sequences(model, scaler, sequences, args.seq_len)
                else:
                    seq = sequence_from_features(features, args.seq_len, args.strategy)
                    seq_scaled = scaler.transform(seq.reshape(-1, FEATURE_DIM)).reshape(
                        1, args.seq_len, FEATURE_DIM
                    ).astype(np.float32)
                    probs = model.predict(seq_scaled, verbose=0)[0]
                    window_count = 1
                order = np.argsort(probs)[::-1][: args.top_k]
                predicted = labels[str(int(order[0]))]
                topk = [
                    f"{labels[str(int(idx))]}:{float(probs[int(idx)]):.4f}"
                    for idx in order
                ]
                rows.append(
                    {
                        **sample,
                        **meta,
                        "used_frames": len(features),
                        "window_count": window_count,
                        "predicted": predicted,
                        "confidence": round(float(probs[int(order[0])]), 6),
                        "top_k": " | ".join(topk),
                        "correct": predicted == expected,
                        "detected_hand": bool(len(features) > 0),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **sample,
                        "source_fps": "",
                        "duration_s": "",
                        "start_s": sample.get("start_s", ""),
                        "end_s": sample.get("end_s", ""),
                        "decoded_frames": 0,
                        "hand_detected_frames": 0,
                        "used_frames": 0,
                        "window_count": 0,
                        "predicted": "",
                        "confidence": "",
                        "top_k": "",
                        "correct": False,
                        "detected_hand": False,
                        "error": str(exc),
                    }
                )

    output_csv = args.out_dir / "predictions.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bench = summarize_predictions(rows, detected_key="detected_hand")
    summary = {
        "samples_file": str(args.samples),
        "artifact_dir": str(args.artifact_dir),
        "strategy": args.strategy,
        "target_fps": args.target_fps,
        "total_samples": len(rows),
        "detected_samples": bench["detected_samples"],
        "rejected_samples": bench["rejected_samples"],
        "detection_rate": bench["detection_rate"],
        "top1_correct": bench["top1_correct"],
        "top3_correct": bench["top3_correct"],
        "top1_accuracy": bench["top1_accuracy"],
        "top3_accuracy": bench["top3_accuracy"],
        "confusion_pairs": bench["confusion_pairs"],
        "predictions_csv": str(output_csv),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
