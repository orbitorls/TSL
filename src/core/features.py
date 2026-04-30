"""Core feature extraction for TSL-51.

Single source of truth for landmark → feature vector conversion.
"""

import numpy as np

FEATURE_LEVELS = {
    'basic': 162,      # Hand (63+63) + Pose (36)
    'enhanced': 249,   # + geometric features
    'finger': 258,     # + finger joints
    'full': 1596,     # + face (478*3)
    'face': 1434,      # Face only
}

_POSE_BASES = [
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'lbrow_outer', 'lbrow_inner',
    'rbrow_inner', 'rbrow_outer', 'mouth_right', 'mouth_left',
]


def safe_mean(series, default: float = 0.0) -> float:
    """Safe mean - import from utils for consistency."""
    from utils.dataset_utils import safe_mean as _safe_mean
    return _safe_mean(series, default)


def extract_features(lm_df, feature_level: str = 'basic') -> np.ndarray:
    """Extract mean-aggregated features from landmark DataFrame.

    Args:
        lm_df: DataFrame with landmark columns (lh_x0, rh_x0, etc.)
        feature_level: One of 'basic', 'enhanced', 'finger', 'full', 'face'

    Returns:
        numpy array of shape (feature_dim,)
    """
    features = []

    # Left hand (21 * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)

    # Right hand (21 * 3 = 63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)

    # Pose (12 * 3 = 36)
    for base in _POSE_BASES:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)

    feature_dim = FEATURE_LEVELS.get(feature_level, 162)
    return np.array(features[:feature_dim], dtype=np.float32)