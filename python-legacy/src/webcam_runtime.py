"""Shared utilities for webcam demos.

This module centralises:
  - model loading (Keras or TFLite),
  - EMA smoothing,
  - Thai font loading and overlay rendering,
  - frame read retries,
  - common artifact validation.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TextIO

import joblib
import numpy as np

from src.artifacts import ensure_required_files, load_labels


def _format_shape(shape: tuple[Any, ...]) -> str:
    return "(" + ", ".join(str(int(v)) if isinstance(v, (int, np.integer)) else str(v) for v in shape) + ")"


def _validate_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be known to validate artifacts")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got boolean: {value}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


class Predictor:
    """TFLite / Keras predictor wrapper with simple shape validation."""

    def __init__(self, kind: str, handle: Any):
        if kind not in ("tflite", "keras"):
            raise ValueError(f"Unknown predictor kind: {kind!r}")
        self.kind = kind
        self._handle = handle
        if kind == "tflite":
            handle.allocate_tensors()
            self._input_details = handle.get_input_details()
            self._output_details = handle.get_output_details()
            self._input_dtype = self._input_details[0]["dtype"]
        else:
            self._input_details = None
            self._output_details = None
            self._input_dtype = np.float32

    @property
    def input_shape(self) -> tuple[Any, ...]:
        if self.kind == "tflite":
            return tuple(self._input_details[0]["shape"])
        raw = self._handle.input_shape
        if isinstance(raw, tuple):
            return tuple(raw)
        return tuple(raw)

    @property
    def output_shape(self) -> tuple[Any, ...]:
        if self.kind == "tflite":
            return tuple(self._output_details[0]["shape"])
        raw = self._handle.output_shape
        if isinstance(raw, tuple):
            return tuple(raw)
        return tuple(raw)

    @property
    def num_classes(self) -> int:
        return _validate_int(self.output_shape[-1], "output feature dimension")

    def validate_output(self, expected_classes: int) -> None:
        if expected_classes <= 0:
            raise ValueError(f"expected_classes must be > 0, got {expected_classes}")
        if self.num_classes != expected_classes:
            raise ValueError(
                f"Model outputs {self.num_classes} classes but labels file has "
                f"{expected_classes} entries"
            )

    def validate_input(self, expected_feature_dim: int, expected_sequence_len: int | None) -> None:
        if expected_feature_dim <= 0:
            raise ValueError(
                f"expected_feature_dim must be > 0, got {expected_feature_dim}"
            )
        shape = self.input_shape
        if len(shape) not in (2, 3):
            raise ValueError(f"Unsupported model input rank {len(shape)}: {_format_shape(shape)}")
        input_dim = _validate_int(shape[-1], "input feature dimension")
        if input_dim != expected_feature_dim:
            raise ValueError(
                f"Model input feature dim {input_dim} does not match expected "
                f"{expected_feature_dim}"
            )
        if expected_sequence_len is not None and len(shape) == 3:
            seq_len = shape[1]
            if seq_len is not None and _validate_int(seq_len, "input sequence length") != expected_sequence_len:
                raise ValueError(
                    f"Model sequence length {seq_len} does not match expected "
                    f"{expected_sequence_len}"
                )

    def predict(self, feat_1xN_or_1xTxD: np.ndarray) -> np.ndarray:
        """Return 1-D probability vector (shape (num_classes,))."""
        if self.kind == "tflite":
            x = np.ascontiguousarray(feat_1xN_or_1xTxD, dtype=self._input_dtype)
            self._handle.set_tensor(self._input_details[0]["index"], x)
            self._handle.invoke()
            probs = self._handle.get_tensor(self._output_details[0]["index"])
            return np.asarray(probs[0], dtype=np.float32)
        x = np.asarray(feat_1xN_or_1xTxD, dtype=np.float32)
        probs = self._handle(x, training=False).numpy()
        return np.asarray(probs[0], dtype=np.float32)


class EMABuffer:
    """Exponential moving average over probability vectors."""

    def __init__(self, alpha: float):
        self._alpha = float(alpha)
        self._smoothed: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        self._smoothed = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if probs.ndim != 1:
            raise ValueError(f"Expected 1-D probability vector, got shape {probs.shape}")
        probs = probs.astype(np.float32, copy=False)
        if self._smoothed is None or self._smoothed.shape != probs.shape:
            self._smoothed = probs.copy()
        else:
            self._smoothed = self._alpha * probs + (1.0 - self._alpha) * self._smoothed
        return self._smoothed


def _load_tflite_interpreter(tflite_path: Path) -> Any:
    """Prefer tflite_runtime; fall back to tensorflow.lite.Interpreter."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        print("  Backend: tflite_runtime")
        return Interpreter(model_path=str(tflite_path))
    except ImportError:
        import tensorflow as tf  # type: ignore

        print("  Backend: tensorflow.lite")
        return tf.lite.Interpreter(model_path=str(tflite_path))


def _load_keras_model(model_path: Path) -> Any:
    import tensorflow as tf  # type: ignore

    print("  Backend: tensorflow.keras")
    return tf.keras.models.load_model(str(model_path))


def load_demo_artifacts(
    model_path: str,
    labels_path: str,
    scaler_path: str,
    use_tflite: bool,
    expected_feature_dim: int,
    expected_sequence_len: int | None = None,
    track: str = "demo",
    manifest_path: str | Path | None = None,
) -> tuple[Predictor, dict[str, str], Any]:
    """Load predictor, labels, and scaler with defensive validation."""
    model_p = Path(model_path)
    labels_p = Path(labels_path)
    scaler_p = Path(scaler_path)
    tflite_p = model_p.with_suffix(".tflite")

    want_tflite = bool(use_tflite or tflite_p.exists())
    required = {
        "labels": labels_p,
        "scaler": scaler_p,
        "model": tflite_p if want_tflite else model_p,
    }
    ensure_required_files(
        required,
        (
            f"Train for '{track}' in Colab and place artifacts in this directory: "
            f"{model_p.name}, {model_p.with_suffix('.tflite').name}, "
            f"{labels_p.name}, {scaler_p.name}"
        ),
    )

    if want_tflite:
        predictor = Predictor("tflite", _load_tflite_interpreter(tflite_p))
    else:
        predictor = Predictor("keras", _load_keras_model(model_p))

    labels = load_labels(labels_p)
    predictor.validate_output(len(labels))
    predictor.validate_input(expected_feature_dim, expected_sequence_len)

    scaler = joblib.load(str(scaler_p))
    if hasattr(scaler, "n_features_in_"):
        feature_count = int(getattr(scaler, "n_features_in_"))
        if feature_count != expected_feature_dim:
            raise ValueError(
                f"Scaler expects {feature_count} features, but {track} expects "
                f"{expected_feature_dim}"
            )

    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(manifest, Mapping) and manifest.get("track") and manifest["track"] != track:
                raise ValueError(
                    f"Manifest track '{manifest['track']}' does not match loaded script track '{track}'"
                )

    return predictor, labels, scaler


def _load_font_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "assets" / "NotoSansThai-Regular.ttf",
        project_root / "assets" / "Sarabun-Regular.ttf",
        project_root / "assets" / "THSarabunNew.ttf",
    ]


def load_thai_font(project_root: Path, size: int, candidates: list[Path] | None = None):
    """Try bundled Thai fonts first and gracefully fallback to PIL default."""
    from PIL import ImageFont

    font_candidates = candidates or _load_font_candidates(project_root)
    for path in font_candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass
    print(
        f"[WARN] No Thai TTF found in {[str(p) for p in font_candidates]}. "
        "Download NotoSansThai-Regular.ttf from Google Fonts → assets/"
    )
    return ImageFont.load_default()


def draw_thai_text(
    frame_bgr: np.ndarray, text: str, pos: tuple, font, color=(0, 255, 0)
) -> np.ndarray:
    """Overlay Thai text on an OpenCV BGR frame using PIL."""
    import cv2

    from PIL import Image, ImageDraw

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def format_topk(
    smoothed: np.ndarray, labels: Mapping[str, str], k: int
) -> tuple[str, list[tuple[str, float]]]:
    """Return (pretty_text, [(label, prob), ...]) for top-k entries."""
    if smoothed.ndim != 1:
        raise ValueError(f"Expected 1-D probability vector, got {smoothed.shape}")
    if k < 1:
        return "", []
    k = min(int(k), smoothed.shape[0])
    idxs = np.argsort(smoothed)[::-1][:k]
    pairs: list[tuple[str, float]] = []
    parts: list[str] = []
    for idx in idxs:
        idx_int = int(idx)
        label = labels.get(str(idx_int), "?")
        prob = float(smoothed[idx_int])
        pairs.append((label, prob))
        parts.append(f"{label} {prob * 100:.0f}%")
    return " · ".join(parts), pairs


def read_frame_with_retry(cap: Any, retries: int = 3, sleep_s: float = 0.05) -> tuple[bool, np.ndarray | None]:
    """Read one frame with tiny retries to absorb transient camera hiccups."""
    for _ in range(max(1, retries)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        if sleep_s > 0:
            import time
            time.sleep(sleep_s)
    return False, None


def open_csv_writer(path: str, header: list[str]) -> tuple[TextIO, Any]:
    """Create UTF-8 CSV writer with ensured parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(header)
    csv_file.flush()
    return csv_file, writer


def iso_timestamp_ms() -> str:
    return datetime.now().isoformat(timespec="milliseconds")
