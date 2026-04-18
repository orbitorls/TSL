"""
Small dataset utilities: safe_mean and validation helpers.
"""
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def safe_mean(series: Any, default: float = 0.0) -> float:
    """Return a safe float mean for a pandas Series / list-like / numpy array.

    If the input is missing, empty, or raises during mean(), return the default.
    Always cast result to float.
    """
    try:
        if series is None:
            return float(default)
        # numpy arrays have .size
        if hasattr(series, 'size') and getattr(series, 'size') == 0:
            return float(default)
        # pandas Series: mean() returns a scalar
        result = None
        if hasattr(series, 'mean'):
            result = series.mean()
        else:
            # try numpy mean
            result = float(np.mean(series))

        if result is None:
            return float(default)
        return float(result)
    except Exception:
        logger.debug("safe_mean failed for series: %r", type(series), exc_info=True)
        return float(default)


def validate_feature_vector(vec, expected_length=162) -> bool:
    try:
        arr = np.asarray(vec)
        return arr.size == expected_length
    except Exception:
        logger.debug("Feature vector validation failed", exc_info=True)
        return False


def safe_std(series: Any, default: float = 0.0) -> float:
    """Return a safe float std deviation for a pandas Series / list-like / numpy array.

    If the input is missing, empty, or raises during computation, return the default.
    Always cast result to float.
    """
    try:
        if series is None:
            return float(default)
        if hasattr(series, 'size') and getattr(series, 'size') == 0:
            return float(default)
        if hasattr(series, 'std'):
            result = series.std()
        else:
            result = float(np.std(series))
        if result is None:
            return float(default)
        return float(result)
    except Exception:
        logger.debug("safe_std failed for series: %r", type(series), exc_info=True)
        return float(default)
