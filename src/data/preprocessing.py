"""Redirect to core.features for backward compatibility.

This module is deprecated. Use src.core instead.
"""

import numpy as np
from src.core.features import FEATURE_LEVELS as FEATURE_DIMS, extract_features
from src.core.normalizer import Normalizer

# Re-export with old names for backward compatibility
def extract_features_from_landmarks(lm_df, feature_level='basic'):
    """Deprecated: Use src.core.extract_features instead."""
    return extract_features(lm_df, feature_level)


def normalize_landmarks(features, mean=None, std=None):
    """Deprecated: Use src.core.Normalizer instead."""
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1, std)
    normalized = (features - mean) / std
    return normalized, (mean, std)


__all__ = ["FEATURE_DIMS", "extract_features", "extract_features_from_landmarks", "normalize_landmarks"]