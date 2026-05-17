"""Training utilities: array coercion, JSON serialization, parameter estimation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert value to numpy array with conservative fallback."""
    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return np.array(value, dtype=dtype)


def _coerce_dataset_arrays(X: Any, y: Any, classes: Any):
    """Normalize dataset containers to numpy arrays and derive input_dim."""
    X_np = _to_numpy_array(X, dtype=np.float32)
    y_np = _to_numpy_array(y)
    classes_np = _to_numpy_array(classes)
    input_dim = int(X_np.shape[-1]) if getattr(X_np, "ndim", 1) >= 2 else 0
    return X_np, y_np, classes_np, input_dim


def _serialize_for_json(obj: Any) -> Any:
    """Convert common training outputs to JSON-serializable values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    if hasattr(obj, "__dict__"):
        try:
            return {k: _serialize_for_json(v) for k, v in obj.__dict__.items()}
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        logger.exception("Failed to serialize object of type %s", type(obj))
        return None


def estimate_params(
    model_type: str, hidden_dim: int, num_layers: int, input_dim: int, num_classes: int
) -> int:
    """Estimate number of parameters in the model."""
    if model_type in ["gru", "mopgru"]:
        return hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2
    elif model_type == "hybrid":
        gru_params = hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2
        trans_params = 8 * hidden_dim * hidden_dim * 2 + hidden_dim * 4
        return gru_params + trans_params
    elif model_type == "ctc":
        return (
            hidden_dim * hidden_dim * 4 * 3 * num_layers
            + hidden_dim * input_dim * 2
            + hidden_dim * num_classes
        )
    else:
        return (
            hidden_dim * hidden_dim * (num_layers - 1)
            + hidden_dim * input_dim
            + hidden_dim * num_classes
        )
