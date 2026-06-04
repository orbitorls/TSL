"""Core feature extraction for TSL-51.

Single source of truth for landmark → feature vector conversion.
"""

import numpy as np

FEATURE_LEVELS = {
    "basic": 162,  # Hand (63+63) + Pose (36)
    "enhanced": 249,  # + geometric features
    "finger": 252,  # + finger joints (162 + 90)
    "full": 1596,  # + face (478*3)
    "face": 1434,  # Face only
}

_POSE_BASES = [
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "lbrow_outer",
    "lbrow_inner",
    "rbrow_inner",
    "rbrow_outer",
    "mouth_right",
    "mouth_left",
]

_BASIC_KEYS = []
for i in range(21):
    for c in ["x", "y", "z"]:
        _BASIC_KEYS.append(f"lh_{c}{i}")
for i in range(21):
    for c in ["x", "y", "z"]:
        _BASIC_KEYS.append(f"rh_{c}{i}")
for base in _POSE_BASES:
    for c in ["x", "y", "z"]:
        _BASIC_KEYS.append(f"{base}_{c}")


def extract_features(lm_df, feature_level: str = "basic") -> np.ndarray:
    """Extract mean-aggregated features from landmark DataFrame.

    Args:
        lm_df: DataFrame with landmark columns (lh_x0, rh_x0, etc.)
        feature_level: One of 'basic', 'enhanced', 'finger', 'full', 'face'

    Returns:
        numpy array of shape (feature_dim,)
    """
    feature_dim = FEATURE_LEVELS.get(feature_level, 162)
    keys = _BASIC_KEYS[:feature_dim]

    # Fast path vectorized mean calculation using pandas
    cols_to_use = lm_df.columns.intersection(keys)
    means = lm_df[cols_to_use].mean().fillna(0.0).to_dict()

    features = [float(means.get(k, 0.0)) for k in keys]

    if len(features) < feature_dim:
        features.extend([0.0] * (feature_dim - len(features)))

    return np.array(features, dtype=np.float32)
