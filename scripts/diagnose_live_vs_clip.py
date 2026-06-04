#!/usr/bin/env python3
"""Compare offline clip eval vs live process_rgb_frame replay on the same videos.

Outputs per-sample offline/live labels and an aggregate summary with optional A/B
variants (jpeg, flip, stretch, min_sign_frames, rolling segment mode).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))
sys.path.insert(0, str(PY_LEGACY / "src"))

from sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT, extract_holistic_frame, resample_frames  # noqa: E402
from tsl_translate.inference import InferenceSettings, process_rgb_frame  # noqa: E402
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService  # noqa: E402
from tsl_translate.tracks import TRACKS  # noqa: E402

DEFAULT_ARTIFACT = (
    REPO_ROOT
    / ".tools"
    / "tsl51_experiments"
    / "full51_v3_external_weighted"
    / "artifacts"
    / "tsl51"
)
DEFAULT_SAMPLES = REPO_ROOT / "work" / "reviewed_external_assets" / "external_test_samples.csv"
DEFAULT_OUT = REPO_ROOT / "reports" / "live_vs_clip_diagnosis"


@dataclass(frozen=True)
class ReplayVariant:
    name: str
    jpeg_quality: float | None = None  # None = raw RGB
    jpeg_width: int = 640
    flip: bool = False
    stretch_live: bool = False
    segment_mode: str = "motion"
    min_sign_frames: int | None = None
    sign_end_frames: int | None = None
    threshold: float | None = None
    min_confidence_margin: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument(
        "--ab",
        choices=("baseline", "all"),
        default="baseline",
        help="baseline = store defaults only; all = run A/B variants",
    )
    parser.add_argument("--tail-still-frames", type=int, default=12)
    parser.add_argument("--lead-idle-frames", type=int, default=5)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in raw.items()}


def read_samples(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def store_default_settings() -> InferenceSettings:
    return InferenceSettings(
        threshold=0.65,
        min_sign_frames=6,
        sign_end_frames=5,
        min_confidence_margin=0.12,
        commit_on_preview=False,
        stretch_live=False,
        segment_mode="motion",
        prefer_seq_buf_on_commit=True,
    )


def variants_for_ab(mode: str) -> list[ReplayVariant]:
    base = ReplayVariant(name="baseline")
    if mode != "all":
        return [base]
    return [
        base,
        ReplayVariant(name="jpeg_q65_640", jpeg_quality=0.65, jpeg_width=640),
        ReplayVariant(name="jpeg_q85_960", jpeg_quality=0.85, jpeg_width=960),
        ReplayVariant(name="flip", flip=True),
        ReplayVariant(name="stretch_live", stretch_live=True),
        ReplayVariant(name="rolling_segment", segment_mode="rolling"),
        ReplayVariant(name="min_sign_6", min_sign_frames=6, sign_end_frames=5),
        ReplayVariant(name="lower_gates", threshold=0.55, min_confidence_margin=0.08),
    ]


def apply_variant(settings: InferenceSettings, variant: ReplayVariant) -> InferenceSettings:
    updates: dict[str, Any] = {}
    if variant.min_sign_frames is not None:
        updates["min_sign_frames"] = variant.min_sign_frames
    if variant.sign_end_frames is not None:
        updates["sign_end_frames"] = variant.sign_end_frames
    if variant.threshold is not None:
        updates["threshold"] = variant.threshold
    if variant.min_confidence_margin is not None:
        updates["min_confidence_margin"] = variant.min_confidence_margin
    updates["stretch_live"] = variant.stretch_live
    updates["segment_mode"] = variant.segment_mode
    return replace(settings, **updates)


def load_rgb_frames(
    video_path: Path,
    start_s: float | None,
    end_s: float | None,
    target_fps: float,
) -> list[np.ndarray]:
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
    frames: list[np.ndarray] = []
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        t += step_s
    cap.release()
    return frames


def simulate_web_jpeg(rgb: np.ndarray, width: int, quality: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    target_h = max(1, int(round(h * (width / max(w, 1)))))
    resized = cv2.resize(rgb, (width, target_h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)])
    if not ok:
        return rgb
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def preprocess_rgb(rgb: np.ndarray, variant: ReplayVariant) -> np.ndarray:
    out = rgb
    if variant.flip:
        out = cv2.flip(out, 1)
    if variant.jpeg_quality is not None:
        out = simulate_web_jpeg(out, variant.jpeg_width, variant.jpeg_quality)
    return out


def offline_predict(
    features: list[np.ndarray],
    model,
    scaler,
    labels: dict[str, str],
    seq_len: int,
) -> tuple[str, float]:
    if not features:
        raise ValueError("no hand frames")
    arr = np.asarray(features, dtype=np.float32)
    seq = resample_frames(arr, seq_len)
    seq_scaled = scaler.transform(seq.reshape(-1, FEATURE_DIM)).reshape(1, seq_len, FEATURE_DIM)
    probs = model.predict(seq_scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[str(idx)], float(probs[idx])


def extract_features_from_rgb_frames(
    rgb_frames: list[np.ndarray],
    holistic,
    variant: ReplayVariant,
) -> list[np.ndarray]:
    features: list[np.ndarray] = []
    for rgb in rgb_frames:
        processed = preprocess_rgb(rgb, variant)
        processed.flags.writeable = False
        results = holistic.process(processed)
        processed.flags.writeable = True
        feat = extract_holistic_frame(results)
        if feat is not None:
            features.append(feat)
    return features


def live_replay(
    rgb_frames: list[np.ndarray],
    *,
    track,
    loaded: LoadedModel,
    mp_runtime: MediaPipeRuntime,
    settings: InferenceSettings,
    variant: ReplayVariant,
    tail_still_frames: int,
    lead_idle_frames: int,
) -> dict[str, Any]:
    service = PredictService(track, alpha=settings.alpha)
    service.reset()

    committed: str | None = None
    last_label = ""
    last_conf = 0.0
    last_status = "ready"
    max_sign_frames = 0
    sign_end_frame_idx: int | None = None
    frame_idx = 0

    monotonic_t = time.monotonic()
    time_step = 1.0 / 15.0

    def feed(rgb: np.ndarray, idx: int) -> None:
        nonlocal committed, last_label, last_conf, last_status, max_sign_frames, sign_end_frame_idx, monotonic_t
        monotonic_t += time_step
        processed = preprocess_rgb(rgb, variant)
        # Patch monotonic for preview_interval logic — use real time progression
        result = process_rgb_frame(track, loaded, service, mp_runtime, processed, settings)
        last_label = result.label
        last_conf = result.confidence
        last_status = result.status
        max_sign_frames = max(max_sign_frames, len(service.sign_frames))
        if result.committed_label:
            committed = result.committed_label
            if sign_end_frame_idx is None:
                sign_end_frame_idx = idx

    # Lead-in: repeat first frame to simulate idle before sign
    if rgb_frames and lead_idle_frames > 0:
        idle = rgb_frames[0]
        for i in range(lead_idle_frames):
            feed(idle, frame_idx)
            frame_idx += 1

    for rgb in rgb_frames:
        feed(rgb, frame_idx)
        frame_idx += 1

    # Tail: hold last frame still to trigger sign_end (low motion)
    if rgb_frames and tail_still_frames > 0:
        still = rgb_frames[-1]
        for i in range(tail_still_frames):
            feed(still, frame_idx)
            frame_idx += 1

    preview_label = last_label
    if preview_label.startswith("Unknown") or preview_label.startswith("กำลัง"):
        preview_label = ""

    return {
        "live_committed_label": committed or "",
        "live_preview_label": preview_label,
        "live_last_label": last_label,
        "live_confidence": round(last_conf, 6),
        "live_status": last_status,
        "n_sign_frames_max": max_sign_frames,
        "sign_end_frame_idx": sign_end_frame_idx,
        "total_frames_fed": frame_idx,
    }


def summarize_rows(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    detected = [r for r in rows if r.get("detected_hand")]
    offline_ok = sum(1 for r in detected if r.get("offline_correct"))
    live_ok = sum(1 for r in detected if r.get("live_correct"))
    agree = sum(
        1
        for r in detected
        if r.get("offline_predicted") and r.get("live_committed_label")
        and r["offline_predicted"] == r["live_committed_label"]
    )
    n = len(detected) or 1
    return {
        "variant": variant_name,
        "total_samples": len(rows),
        "detected_samples": len(detected),
        "offline_top1_accuracy": round(offline_ok / n, 4),
        "live_committed_top1_accuracy": round(live_ok / n, 4),
        "offline_live_agreement": round(agree / n, 4),
        "live_no_commit": sum(1 for r in detected if not r.get("live_committed_label")),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.artifact_dir / "tsl51_labels.json")
    scaler = joblib.load(args.artifact_dir / "tsl51_scaler.pkl")
    model = tf.keras.models.load_model(args.artifact_dir / "tsl51_model.keras")

    track = TRACKS["tsl51"]
    loaded = LoadedModel(
        predictor=model,
        labels=labels,
        scaler=scaler,
        backend="keras",
        model_path=args.artifact_dir / "tsl51_model.keras",
        labels_path=args.artifact_dir / "tsl51_labels.json",
        scaler_path=args.artifact_dir / "tsl51_scaler.pkl",
        load_time_ms=0.0,
    )

    samples = read_samples(args.samples)
    if args.limit > 0:
        samples = samples[: args.limit]

    base_settings = store_default_settings()
    variants = variants_for_ab(args.ab)

    all_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []

    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        mp_runtime = MediaPipeRuntime()

        try:
            for variant in variants:
                settings = apply_variant(base_settings, variant)
                variant_rows: list[dict[str, Any]] = []

                for sample in samples:
                    video_path = Path(sample["path"])
                    expected = sample["expected"]
                    start_s = float(sample["start_s"]) if sample.get("start_s") else None
                    end_s = float(sample["end_s"]) if sample.get("end_s") else None
                    row: dict[str, Any] = {
                        **sample,
                        "variant": variant.name,
                        "expected": expected,
                        "error": "",
                    }
                    try:
                        rgb_frames = load_rgb_frames(video_path, start_s, end_s, args.target_fps)
                        features = extract_features_from_rgb_frames(rgb_frames, holistic, variant)
                        offline_pred, offline_conf = offline_predict(
                            features, model, scaler, labels, SEQ_LEN_DEFAULT
                        )
                        live_meta = live_replay(
                            rgb_frames,
                            track=track,
                            loaded=loaded,
                            mp_runtime=mp_runtime,
                            settings=settings,
                            variant=variant,
                            tail_still_frames=args.tail_still_frames,
                            lead_idle_frames=args.lead_idle_frames,
                        )
                        row.update(
                            {
                                "decoded_frames": len(rgb_frames),
                                "hand_detected_frames": len(features),
                                "detected_hand": bool(features),
                                "offline_predicted": offline_pred,
                                "offline_confidence": round(offline_conf, 6),
                                "offline_correct": offline_pred == expected,
                                **live_meta,
                                "live_correct": live_meta["live_committed_label"] == expected,
                                "labels_match": offline_pred == live_meta["live_committed_label"],
                            }
                        )
                    except Exception as exc:
                        row.update(
                            {
                                "detected_hand": False,
                                "offline_predicted": "",
                                "offline_confidence": "",
                                "offline_correct": False,
                                "live_committed_label": "",
                                "live_correct": False,
                                "labels_match": False,
                                "error": str(exc),
                            }
                        )
                    variant_rows.append(row)
                    all_rows.append(row)

                variant_summaries.append(summarize_rows(variant_rows, variant.name))
        finally:
            mp_runtime.close()

    out_csv = args.out_dir / "predictions.csv"
    if all_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    # Rank A/B variants by live accuracy on detected samples
    ab_ranking = sorted(
        variant_summaries,
        key=lambda s: (s["live_committed_top1_accuracy"], s["offline_live_agreement"]),
        reverse=True,
    )

    baseline = next((s for s in variant_summaries if s["variant"] == "baseline"), variant_summaries[0])
    gap = baseline["offline_top1_accuracy"] - baseline["live_committed_top1_accuracy"]

    diagnosis = {
        "samples_file": str(args.samples),
        "artifact_dir": str(args.artifact_dir),
        "target_fps": args.target_fps,
        "baseline_offline_top1": baseline["offline_top1_accuracy"],
        "baseline_live_top1": baseline["live_committed_top1_accuracy"],
        "baseline_offline_live_agreement": baseline["offline_live_agreement"],
        "pipeline_gap": round(gap, 4),
        "pipeline_gap_is_primary": gap >= 0.2 and baseline["offline_top1_accuracy"] >= 0.8,
        "model_domain_is_primary": baseline["offline_top1_accuracy"] < 0.5,
        "variant_summaries": variant_summaries,
        "ab_ranking": ab_ranking,
        "recommended_variant": ab_ranking[0]["variant"] if ab_ranking else "baseline",
        "predictions_csv": str(out_csv),
        "settings_baseline": asdict(store_default_settings()),
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
