#!/usr/bin/env python3
"""
Export golden test vectors from the Python feature contract for Rust parity tests.

Run from repo root:
  python scripts/export_golden.py

Writes JSON under tests/golden/ with f32 tolerance 1e-4 (see manifest.json).
Optionally exports TFLite inference samples when model artifacts exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "python-legacy"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
TOLERANCE = 1e-4

if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from src.keypoints import (  # noqa: E402
    FEATURE_SIZE,
    NUM_LANDMARKS,
    extract_and_normalize,
    extract_hand_landmarks,
    normalize_landmarks,
)
from src.sequence_keypoints import (  # noqa: E402
    FEATURE_DIM,
    LEFT_SHOULDER_IDX,
    NUM_HAND_LANDMARKS,
    RIGHT_SHOULDER_IDX,
    SEQ_LEN_DEFAULT,
    SequenceBuffer,
    csv_to_sequence,
    extract_holistic_frame,
    pad_truncate_sequence,
    read_landmark_csv,
    tsl51_csv_column_names,
)


def _vec(arr: np.ndarray | None) -> list[float] | None:
    if arr is None:
        return None
    return [float(x) for x in arr.flatten().tolist()]


def _matrix(arr: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in arr.tolist()]


def make_hand(coords_21x3: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            for c in coords_21x3
        ]
    )


def make_handedness(score: float) -> SimpleNamespace:
    return SimpleNamespace(classification=[SimpleNamespace(score=float(score))])


def make_landmark_list(coords_by_index: dict[int, tuple[float, float, float]]) -> SimpleNamespace:
    max_idx = max(coords_by_index.keys()) if coords_by_index else -1
    landmarks = []
    for i in range(max_idx + 1):
        x, y, z = coords_by_index.get(i, (0.0, 0.0, 0.0))
        landmarks.append(SimpleNamespace(x=float(x), y=float(y), z=float(z)))
    return SimpleNamespace(landmark=landmarks)


def make_holistic_results(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        pose_landmarks=kwargs.get("pose"),
        face_landmarks=kwargs.get("face"),
        left_hand_landmarks=kwargs.get("left_hand"),
        right_hand_landmarks=kwargs.get("right_hand"),
    )


def export_keypoints(rng: np.random.Generator) -> dict:
    valid = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    valid[9] = valid[0] + np.array([0.3, 0.2, 0.1], dtype=np.float32)
    valid_flat = valid.flatten()

    coords_lo = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    coords_hi = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    multi_results = SimpleNamespace(
        multi_hand_landmarks=[make_hand(coords_lo), make_hand(coords_hi)],
        multi_handedness=[make_handedness(0.4), make_handedness(0.9)],
    )

    e2e_coords = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    e2e_coords[9] = e2e_coords[0] + np.array([0.25, 0.15, 0.05], dtype=np.float32)
    e2e_results = SimpleNamespace(
        multi_hand_landmarks=[make_hand(e2e_coords)],
        multi_handedness=[make_handedness(0.99)],
    )

    collapsed = np.ones(FEATURE_SIZE, dtype=np.float32) * 0.5

    return {
        "feature_size": FEATURE_SIZE,
        "valid_landmarks_flat": _vec(valid_flat),
        "normalize_output": _vec(normalize_landmarks(valid_flat)),
        "normalize_degenerate": None,
        "multi_hand_coords_lo": _vec(coords_lo.flatten()),
        "multi_hand_coords_hi": _vec(coords_hi.flatten()),
        "extract_highest_confidence": _vec(extract_hand_landmarks(multi_results)),
        "e2e_coords": _vec(e2e_coords.flatten()),
        "extract_and_normalize": _vec(extract_and_normalize(e2e_results)),
        "collapsed_input": _vec(collapsed),
    }


def _make_tsl51_csv_rows(num_frames: int = 2) -> str:
    cols = ["frame", "t_ms"] + tsl51_csv_column_names()
    lines = [",".join(cols)]
    for frame_idx in range(num_frames):
        values = [str(frame_idx), str(frame_idx * 33)]
        for prefix in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"):
            if prefix == "l_shoulder":
                values.extend(["0.3", "0.4", "0.0"])
            elif prefix == "r_shoulder":
                values.extend(["0.7", "0.4", "0.0"])
            else:
                values.extend(["0.5", "0.4", "0.0"])
        for _ in range(6):
            values.extend(["0.5", "0.4", "0.0"])
        for _ in range(NUM_HAND_LANDMARKS * 2):
            values.extend(["0.5", "0.4", "0.0"])
        lines.append(",".join(values))
    return "\n".join(lines)


def export_sequence(rng: np.random.Generator) -> dict:
    seq_len = 5
    buf = SequenceBuffer(seq_len)
    frame = np.arange(FEATURE_DIM, dtype=np.float32)
    buf.push(frame)
    padded_5 = buf.get_padded()

    buf60 = SequenceBuffer(SEQ_LEN_DEFAULT)
    frames = [
        np.full(FEATURE_DIM, 1.0, dtype=np.float32),
        np.full(FEATURE_DIM, 2.0, dtype=np.float32),
        np.full(FEATURE_DIM, 3.0, dtype=np.float32),
    ]
    for f in frames:
        buf60.push(f)
    padded_60_3frames = buf60.get_padded()

    coords = rng.uniform(0.0, 1.0, size=(NUM_HAND_LANDMARKS, 3)).astype(np.float32)
    holistic_one_hand = make_holistic_results(
        pose=make_landmark_list({11: (0.3, 0.4, 0.0), 12: (0.7, 0.4, 0.0)}),
        left_hand=make_hand(coords),
        right_hand=None,
    )

    anchor = (0.5, 0.4, 0.0)
    pose = make_landmark_list(
        {
            LEFT_SHOULDER_IDX: (0.3, 0.4, 0.0),
            RIGHT_SHOULDER_IDX: (0.7, 0.4, 0.0),
            13: anchor,
        }
    )
    hand = make_hand(np.full((NUM_HAND_LANDMARKS, 3), anchor, dtype=np.float32))
    shoulder_test = make_holistic_results(pose=pose, left_hand=hand, right_hand=None)

    csv_text = _make_tsl51_csv_rows(2)
    csv_arr = read_landmark_csv(csv_text)
    csv_seq = csv_to_sequence(csv_text, seq_len=4)

    return {
        "feature_dim": FEATURE_DIM,
        "seq_len_default": SEQ_LEN_DEFAULT,
        "holistic_one_hand_coords": _vec(coords.flatten()),
        "sequence_buffer_padded_5x162": _matrix(padded_5),
        "sequence_buffer_padded_60x162_3frames": _matrix(padded_60_3frames),
        "holistic_one_hand": _vec(extract_holistic_frame(holistic_one_hand)),
        "shoulder_anchor_frame": _vec(extract_holistic_frame(shoulder_test)),
        "csv_read_2_frames": _matrix(csv_arr),
        "csv_to_sequence_4": _matrix(csv_seq),
        "pad_truncate_short": _matrix(pad_truncate_sequence(csv_arr, seq_len=4)),
    }


def export_artifacts_contract() -> dict:
    """Fixed labels + scaler JSON contract for Rust unit tests."""
    labels_list = ["ก", "ข"]
    mean = [0.0] * FEATURE_SIZE
    scale = [1.0] * FEATURE_SIZE
    sample = np.linspace(0.0, 1.0, FEATURE_SIZE, dtype=np.float32)
    scaler = {"mean": mean, "scale": scale, "n_features": FEATURE_SIZE}
    return {
        "labels_list": labels_list,
        "labels_dict": {"0": labels_list[0], "1": labels_list[1]},
        "scaler_json": scaler,
        "scaler_transform_input": _vec(sample),
        "scaler_transform_output": _vec((sample - np.asarray(mean)) / np.asarray(scale)),
    }


def export_tflite_optional() -> dict | None:
    """Export scaler + softmax samples when artifacts exist in repo root."""
    scaler_p = PROJECT_ROOT / "scaler.pkl"
    model_p = PROJECT_ROOT / "model.tflite"
    labels_p = PROJECT_ROOT / "labels.json"
    legacy_scaler = LEGACY_ROOT / "scaler.pkl"
    legacy_model = LEGACY_ROOT / "model.tflite"
    legacy_labels = LEGACY_ROOT / "labels.json"
    for base in (PROJECT_ROOT, LEGACY_ROOT):
        if (base / "scaler.pkl").exists():
            scaler_p = base / "scaler.pkl"
            model_p = base / "model.tflite"
            labels_p = base / "labels.json"
            break

    if not (scaler_p.exists() and model_p.exists() and labels_p.exists()):
        return None

    try:
        import joblib
    except ImportError:
        return {"error": "joblib not installed"}

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    raw[9] = raw[0] + np.array([0.2, 0.15, 0.1], dtype=np.float32)
    feat = normalize_landmarks(raw.flatten())
    if feat is None:
        return None

    scaler = joblib.load(str(scaler_p))
    scaled = scaler.transform(feat.reshape(1, -1)).astype(np.float32)

    try:
        from src.webcam_runtime import Predictor, _load_tflite_interpreter

        predictor = Predictor("tflite", _load_tflite_interpreter(model_p))
        probs = predictor.predict(scaled)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "scaled_input": _matrix(scaled)}

    return {
        "scaled_input": _matrix(scaled),
        "softmax_output": _vec(probs),
    }


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    manifest = {
        "tolerance_f32": TOLERANCE,
        "rng_seed": 42,
        "source": "scripts/export_golden.py",
    }

    keypoints = export_keypoints(rng)
    sequence = export_sequence(rng)
    artifacts = export_artifacts_contract()
    tflite = export_tflite_optional()

    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (GOLDEN_DIR / "keypoints.json").write_text(
        json.dumps(keypoints, indent=2), encoding="utf-8"
    )
    (GOLDEN_DIR / "sequence_keypoints.json").write_text(
        json.dumps(sequence, indent=2), encoding="utf-8"
    )
    (GOLDEN_DIR / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
    if tflite is not None:
        (GOLDEN_DIR / "tflite_fingerspelling.json").write_text(
            json.dumps(tflite, indent=2), encoding="utf-8"
        )
        manifest["has_tflite"] = True
    else:
        manifest["has_tflite"] = False

    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote golden vectors to {GOLDEN_DIR}")
    print(f"  tolerance: {TOLERANCE}")
    print(f"  tflite sample: {manifest['has_tflite']}")


if __name__ == "__main__":
    main()
