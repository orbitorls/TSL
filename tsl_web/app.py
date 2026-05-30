"""
Thai Sign Language (TSL-51) Real-Time Web Translator
====================================================
Flask web app that loads a trained GRU model and accepts
pre-extracted landmark vectors from the browser for inference.

Architecture:
  Browser  --(MediaPipe Holistic)--> 162-dim landmarks
         --(JSON POST)--> Flask /predict --> PyTorch GRU --> predictions
         --(JSON)--> Browser display

Usage:
    python app.py
    # Then open http://localhost:5000
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, make_response, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from src.core.models import MODEL_REGISTRY, GRUModel

# Make project modules resolvable when running from repo root.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _resolve_default_model_path() -> Path:
    candidate_paths = [
        Path(
            PROJECT_DIR
            / "tsl51_results_20260518_164101"
            / "models"
            / "tsl51_gru_20260518_162535.pt"
        ),
        Path(
            PROJECT_DIR
            / "artifacts"
            / "runs"
            / "tsl51_results_20260518_164101"
            / "models"
            / "tsl51_gru_20260518_162535.pt"
        ),
        Path(PROJECT_DIR / "models" / "tsl51_gru_best.pt"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path

    candidates = sorted(
        [
            *Path(PROJECT_DIR / "models").glob("*.pt"),
            *Path(PROJECT_DIR / "results").glob("**/*.pt"),
            *Path(PROJECT_DIR / "artifacts" / "runs").glob("**/*.pt"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return candidate_paths[0]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.environ.get("TSL_WEB_MODEL_PATH", str(_resolve_default_model_path())))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

MODEL_DEVICE = torch.device(
    os.environ.get(
        "TSL_WEB_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
)
MODEL_DEVICE = torch.device(str(MODEL_DEVICE))

# Global model state
model: torch.nn.Module | None = None
idx_to_label: dict[int, str] = {}
mean: np.ndarray = np.array([])
std: np.ndarray = np.array([])
norm_std: np.ndarray = np.array([1.0], dtype=np.float32)
expected_input_dim = 162
sequence_mode = False

# Rolling buffer for sequence inference
TARGET_FRAMES = 30
INFERENCE_EVERY_N = 5
MAX_LANDMARK_BATCH = 120
MAX_ACTIVE_SESSIONS = 128
_state_lock = threading.Lock()
_session_buffers: dict[str, deque[np.ndarray]] = {}
_session_cached_predictions: dict[str, dict] = {}
_session_last_active: dict[str, float] = {}
_model_init_lock = threading.Lock()
_model_load_error: str | None = None
_model_warmup_done = False
DEFAULT_PREDICTION: dict = {
    "top1": {"label": "", "confidence": 0.0},
    "top3": [],
    "buffer_filled": 0.0,
}
LANDMARK_BUFFER: deque[np.ndarray] = deque(maxlen=TARGET_FRAMES)
cached_prediction: dict = dict(DEFAULT_PREDICTION)


def load_model_checkpoint(path: str | Path) -> None:
    """Load model, normalisation stats, and label map from checkpoint."""
    global expected_input_dim, idx_to_label, mean, model, sequence_mode, std, norm_std, TARGET_FRAMES, INFERENCE_EVERY_N, _model_warmup_done

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"[INFO] Loading model from: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    _model_warmup_done = False

    model_name = (
        str(checkpoint.get("model", config.get("model", "gru")))
        .strip()
        .lower()
    )
    if model_name == "":
        model_name = "gru"
    if model_name == "mlpmodel":
        model_name = "mlp"

    if "classes" in checkpoint:
        labels = [str(x) for x in checkpoint["classes"]]
        idx_to_label = dict(enumerate(labels))
    else:
        raise KeyError("Checkpoint missing 'classes'")

    mean_raw = checkpoint["mean"] if "mean" in checkpoint else checkpoint.get("normalization_mean")
    std_raw = checkpoint["std"] if "std" in checkpoint else checkpoint.get("normalization_std")
    if mean_raw is None or std_raw is None:
        raise KeyError("Checkpoint missing mean/std")
    mean = np.asarray(mean_raw, dtype=np.float32)
    std = np.asarray(std_raw, dtype=np.float32)
    norm_std = np.where(std == 0, 1.0, std).astype(np.float32) + 1e-8

    input_dim = checkpoint.get("input_dim", len(mean))
    num_classes = checkpoint.get("num_classes", len(labels))
    if input_dim != len(mean) or input_dim != len(std):
        raise ValueError(
            f"Checkpoint input_dim/mean/std mismatch: {input_dim}, {len(mean)}, {len(std)}"
        )
    expected_input_dim = int(input_dim)
    target_frames = checkpoint.get("target_frames", config.get("target_frames"))
    if target_frames:
        try:
            TARGET_FRAMES = max(1, int(target_frames))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid target_frames in checkpoint: {target_frames}")

    inference_every_n = checkpoint.get("inference_every_n", config.get("inference_every_n"))
    if inference_every_n:
        try:
            INFERENCE_EVERY_N = max(1, int(inference_every_n))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid inference_every_n in checkpoint: {inference_every_n}")

    if "seq_mode" in checkpoint:
        sequence_mode = bool(checkpoint.get("seq_mode"))
    elif "sequence_mode" in checkpoint:
        sequence_mode = bool(checkpoint.get("sequence_mode"))
    else:
        # For non-sequence architectures such as MLP, default to single-frame inference.
        sequence_mode = bool(config.get("seq_mode", model_name != "mlp"))
    global LANDMARK_BUFFER
    LANDMARK_BUFFER = deque(maxlen=TARGET_FRAMES)
    _session_buffers.clear()
    _session_cached_predictions.clear()
    _session_last_active.clear()

    model_class = MODEL_REGISTRY.get(model_name, GRUModel)
    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get("hidden", config.get("hidden_dim", 256)),
        num_layers=config.get("layers", config.get("num_layers", 3)),
        dropout=config.get("dropout", 0.3),
    )

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing model weights")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Some older checkpoints are saved with MLP weights while model is set to GRU.
        if model_class is not GRUModel:
            raise

        fallback_model = MODEL_REGISTRY.get("mlp", GRUModel)(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=config.get("hidden", config.get("hidden_dim", 256)),
            num_layers=config.get("layers", config.get("num_layers", 3)),
            dropout=config.get("dropout", 0.3),
        )
        fallback_model.load_state_dict(state_dict)
        model = fallback_model
    model = model.to(MODEL_DEVICE)
    model.eval()

    acc = checkpoint.get("accuracy", "?")
    print(f"[INFO] Model: {model.__class__.__name__} | Classes: {num_classes} | Accuracy: {acc}")
    print(f"[INFO] Input dim: {input_dim}")
    print(f"[INFO] Sequence mode: {sequence_mode}")


def ensure_model_loaded() -> None:
    global _model_load_error, model
    if _model_load_error is not None:
        raise RuntimeError(_model_load_error)
    if model is not None:
        return
    with _model_init_lock:
        if model is not None:
            return
        if _model_load_error is not None:
            raise RuntimeError(_model_load_error)
        try:
            load_model_checkpoint(MODEL_PATH)
            _model_load_error = None
            _model_warmup()
        except Exception as err:
            _model_load_error = str(err)
            raise


def _model_warmup() -> None:
    global _model_warmup_done, model
    if _model_warmup_done or model is None:
        return

    if sequence_mode:
        sample = np.zeros((TARGET_FRAMES, expected_input_dim), dtype=np.float32)
        sample = _normalise_for_model(sample)
        tensor = torch.from_numpy(sample).to(MODEL_DEVICE).unsqueeze(0)
    else:
        sample = np.zeros(expected_input_dim, dtype=np.float32)
        sample = _normalise_for_model(sample)
        tensor = torch.from_numpy(sample).to(MODEL_DEVICE).unsqueeze(0)

    with torch.inference_mode():
        _ = model(tensor)

    _model_warmup_done = True


def _empty_prediction(fill: float = 0.0) -> dict:
    result = {
        "top1": {"label": "", "confidence": 0.0},
        "top3": [],
        "buffer_filled": fill,
    }
    return result


def _normalise_for_model(seq: np.ndarray) -> np.ndarray:
    if mean.size == 0 or norm_std.size == 0:
        return seq
    if seq.ndim < 1:
        return seq
    if mean.shape[0] != seq.shape[-1]:
        return seq

    return (seq - mean) / norm_std


def predict(landmarks: list | np.ndarray, *, already_normalized: bool = False) -> dict:
    """Run inference on a sequence of 162-dim landmark vectors.

    Args:
        landmarks: sequence of 162-dim vectors (list of lists or ndarrays).
    """
    global model, mean, std, idx_to_label

    if model is None:
        raise RuntimeError("Model not loaded")
    if landmarks is None or len(landmarks) == 0:
        return _empty_prediction()

    seq = landmarks if isinstance(landmarks, np.ndarray) else np.asarray(landmarks, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != expected_input_dim:
        raise ValueError(f"Expected landmark sequence with shape (n, {expected_input_dim})")

    if sequence_mode:
        n = len(seq)
        if n != TARGET_FRAMES:
            indices = np.linspace(0, n - 1, TARGET_FRAMES).astype(int)
            seq = seq[indices]
        seq_norm = seq if already_normalized else _normalise_for_model(seq)
        tensor = torch.from_numpy(seq_norm).unsqueeze(0)
    else:
        frame = seq.mean(axis=0)
        frame_norm = frame if already_normalized else _normalise_for_model(frame)
        tensor = torch.from_numpy(frame_norm).unsqueeze(0)

    with torch.inference_mode():
        tensor = tensor.to(MODEL_DEVICE)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].to("cpu")

    top_k = min(3, probs.shape[-1])
    top3_idx = torch.topk(probs, k=top_k).indices.tolist()
    top3 = [
        {"label": idx_to_label.get(i, f"cls_{i}"), "confidence": float(probs[i])}
        for i in top3_idx
    ]

    return {
        "top1": top3[0],
        "top3": top3,
        "buffer_filled": len(landmarks) / TARGET_FRAMES,
    }


def _get_session_id() -> tuple[str, bool]:
    session_id = request.headers.get("X-TSL-Session")
    created = False
    if not session_id:
        session_id = request.cookies.get("tsl_session")
    if not session_id:
        session_id = uuid.uuid4().hex
        created = True
    return session_id, created


def _evict_old_sessions() -> None:
    now = time.time()
    for session_id, ts in list(_session_last_active.items()):
        if now - ts > 60 * 60 * 24:
            _session_buffers.pop(session_id, None)
            _session_cached_predictions.pop(session_id, None)
            _session_last_active.pop(session_id, None)

    if len(_session_buffers) <= MAX_ACTIVE_SESSIONS:
        return

    items = sorted(_session_last_active.items(), key=lambda item: item[1], reverse=True)
    keep = items[:MAX_ACTIVE_SESSIONS]
    keep_set = {session_id for session_id, _ in keep}

    for session_id in list(_session_buffers.keys()):
        if session_id not in keep_set:
            _session_buffers.pop(session_id, None)
            _session_cached_predictions.pop(session_id, None)
            _session_last_active.pop(session_id, None)


def _validate_landmarks_payload(data: dict | None) -> tuple[np.ndarray | None, str | None]:
    if data is None or "landmarks" not in data:
        return None, "No landmark data"

    landmarks = data["landmarks"]
    if not isinstance(landmarks, list) or not landmarks:
        return None, f"Invalid landmark data (expected {expected_input_dim}-dim)"

    first = landmarks[0]
    is_nested = isinstance(first, (list, tuple))

    if is_nested:
        if len(landmarks) > MAX_LANDMARK_BATCH:
            return None, "Too many landmark frames"
        try:
            arr = np.asarray(landmarks, dtype=np.float32)
        except (TypeError, ValueError):
            return None, "Invalid landmark data (must be numeric)"

    if not is_nested:
        if len(landmarks) != expected_input_dim:
            return None, f"Invalid landmark data (expected {expected_input_dim}-dim)"
        try:
            arr = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)
        except (TypeError, ValueError):
            return None, "Invalid landmark data (must be numeric)"

    if arr.ndim == 1:
        if arr.shape[0] == expected_input_dim:
            arr = arr.reshape(1, -1)
        else:
            return None, f"Invalid landmark data (expected {expected_input_dim}-dim)"
    elif arr.ndim != 2:
        return None, f"Invalid landmark data (expected {expected_input_dim}-dim)"
    if arr.shape[1] != expected_input_dim:
        return None, f"Invalid landmark data (expected {expected_input_dim}-dim)"
    if not np.isfinite(arr).all():
        return None, "Invalid landmark data (must be finite numeric values)"

    return arr, None


def _model_health_payload() -> dict[str, object]:
    path = Path(MODEL_PATH)
    return {
        "loaded": model is not None,
        "model_loaded": model is not None,
        "path": str(path),
        "path_exists": path.exists(),
        "load_error": _model_load_error,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/model/<path:filename>")
def serve_model(filename: str):
    models_dir = Path(__file__).resolve().parent / "models"
    return send_from_directory(str(models_dir), filename)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            **_model_health_payload(),
            "sessions": len(_session_buffers),
            "buffer_limit": TARGET_FRAMES,
            "input_dim": expected_input_dim,
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(error):  # noqa: ARG001
    return jsonify({"error": "Request payload too large"}), 413


@app.route("/predict", methods=["POST"])
def predict_route():
    global LANDMARK_BUFFER, cached_prediction

    try:
        ensure_model_loaded()
    except Exception as err:
        return jsonify({"error": f"Model loading failed: {err}"}), 503

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    landmarks, error = _validate_landmarks_payload(data)
    if error:
        status = 400
        return jsonify({"error": error}), status

    session_id, created_session = _get_session_id()
    landmarks_norm: np.ndarray
    seq_for_infer = None
    fill = 0.0

    try:
        landmarks_norm = _normalise_for_model(landmarks)
    except Exception as err:
        return jsonify({"error": f"Normalization failed: {err}"}), 422

    with _state_lock:
        buffer = _session_buffers.setdefault(session_id, deque(maxlen=TARGET_FRAMES))
        cached = _session_cached_predictions.setdefault(session_id, dict(DEFAULT_PREDICTION))
        _session_last_active[session_id] = time.time()
        _evict_old_sessions()

        # `landmarks` is already validated to `np.ndarray` so we can append rows directly.
        buffer.extend(landmarks_norm)

        LANDMARK_BUFFER = buffer
        fill = min(1.0, len(buffer) / TARGET_FRAMES)
        if sequence_mode:
            should_infer = len(buffer) >= TARGET_FRAMES and (len(buffer) % INFERENCE_EVERY_N == 0)
        else:
            min_infer_frames = max(1, min(INFERENCE_EVERY_N, TARGET_FRAMES))
            should_infer = len(buffer) >= min_infer_frames and (len(buffer) % INFERENCE_EVERY_N == 0)

        if should_infer:
            seq_for_infer = np.array(buffer, dtype=np.float32)

    result = None
    if seq_for_infer is not None:
        try:
            try:
                result = predict(seq_for_infer, True)
            except TypeError:
                result = predict(seq_for_infer)
        except (RuntimeError, ValueError) as err:
            return jsonify({"error": str(err)}), 422

    with _state_lock:
        if result is not None:
            _session_cached_predictions[session_id] = result
            cached_prediction = result
        else:
            result = dict(cached)
            result["buffer_filled"] = fill

    response = make_response(jsonify(result))
    if created_session:
        response.set_cookie("tsl_session", session_id, httponly=True, samesite="Lax")
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_model_loaded()
    print("[INFO] Starting web server at http://localhost:5000")
    print("[INFO] Open this URL in your browser and allow camera access.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
