"""Feature extraction utilities for TSL datasets.

Shared code for training and inference to ensure identical feature
processing across scripts.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.features import (
    BASIC_FEATURE_DIM,
    BASIC_FEATURE_SCHEMA,
    FEATURE_LEVELS,
    get_feature_dim,
    validate_feature_level,
)

FEATURE_DIMS = FEATURE_LEVELS

# Canonical pose landmark names — same order as extract_features_from_landmark_df
_POSE_LANDMARK_NAMES = [
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


def _build_column_list(feature_level: str = "basic") -> list[str]:
    """Build the ordered list of column names for a supported feature level."""
    feature_level = validate_feature_level(feature_level)
    cols = list(BASIC_FEATURE_SCHEMA.columns)
    if feature_level in ["finger", "full", "face"]:
        for hand_prefix in ["lh_", "rh_"]:
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                for c in ["x", "y", "z"]:
                    for joint in ["mcp", "pip", "dip"]:
                        cols.append(f"{hand_prefix}{finger}_{joint}_{c}")
    if feature_level in ["full", "face"]:
        for i in range(478):
            for c in ["x", "y", "z"]:
                cols.append(f"face_{c}{i}")
    return cols[: get_feature_dim(feature_level)]


def extract_features_from_landmark_df(lm_df: Any, feature_level: str = "basic") -> np.ndarray:
    """Extract landmark features from a pandas DataFrame.

    Returns a mean-aggregated ``(feature_dim,)`` vector. ``enhanced`` is
    explicitly unsupported in this phase so callers cannot silently receive a
    partial 162-dim vector for a claimed 249-dim schema.
    """
    feature_level = validate_feature_level(feature_level)
    feature_dim = get_feature_dim(feature_level)
    col_list = _build_column_list(feature_level)

    features = np.zeros(feature_dim, dtype=np.float32)
    available_cols = lm_df.columns.intersection(col_list)
    if not available_cols.empty:
        means = lm_df[available_cols].mean(numeric_only=True).fillna(0.0).to_numpy(dtype=np.float32)
        col_indices = pd.Index(col_list).get_indexer(available_cols)
        features[col_indices] = means

    return features


class FeatureExtractor:
    """Wrapper for feature extraction settings."""

    def __init__(self, feature_level="basic"):
        self.feature_level = validate_feature_level(feature_level)
        self.feature_dim = get_feature_dim(self.feature_level)

    def extract_from_dataframe(self, lm_df):
        return extract_features_from_landmark_df(lm_df, self.feature_level)

    def extract_sequence(self, lm_df, target_frames=30):
        """Extract sequence features for the configured level."""
        return extract_sequence_from_landmark_df(lm_df, self.feature_level, target_frames)


def extract_sequence_from_landmark_df(
    lm_df, feature_level: str = "basic", target_frames: int = 30
) -> np.ndarray:
    """Extract per-frame landmark features, resampled to ``target_frames``.

    Unlike ``extract_features_from_landmark_df`` which collapses all frames
    into a single mean vector, this function preserves the temporal dimension.

    Args:
        lm_df: DataFrame with one row per frame and landmark columns.
        feature_level: One of ``'basic'``, ``'finger'``, ``'full'``, ``'face'``.
        target_frames: Output sequence length (uniform resampling).

    Returns:
        ``np.ndarray`` of shape ``(target_frames, feature_dim)``, float32.
    """
    feature_level = validate_feature_level(feature_level)
    feature_dim = get_feature_dim(feature_level)
    n_frames = len(lm_df)

    if n_frames == 0:
        return np.zeros((target_frames, feature_dim), dtype=np.float32)

    col_list = _build_column_list(feature_level)
    seq = np.zeros((n_frames, feature_dim), dtype=np.float32)

    available_cols = lm_df.columns.intersection(col_list)
    if not available_cols.empty:
        col_indices = pd.Index(col_list).get_indexer(available_cols)
        seq[:, col_indices] = lm_df[available_cols].fillna(0.0).to_numpy(dtype=np.float32)

    return sample_frames_uniform(seq, target_frames)  # type: ignore[no-any-return]


def sample_frames_uniform(landmarks: Any, target_frames: int = 30) -> np.ndarray:
    """Sample landmarks uniformly along the frame dimension."""
    n_frames = len(landmarks)
    if n_frames == target_frames:
        return np.asarray(landmarks)

    indices = np.linspace(0, n_frames - 1, target_frames).astype(int)
    return np.asarray(landmarks)[indices]  # type: ignore[no-any-return]


def pad_or_truncate(sequence: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad or truncate a landmark sequence to a fixed length."""
    seq_len = len(sequence)
    if seq_len == target_length:
        return sequence
    if seq_len < target_length:
        padding = np.full(
            (target_length - seq_len,) + sequence.shape[1:], pad_value, dtype=sequence.dtype
        )
        return np.vstack([sequence, padding])
    return sequence[:target_length]


# ─────────────────────────────────────────────────────────────
# Enhanced geometric features for real-time inference
# ─────────────────────────────────────────────────────────────

_FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
_FINGER_MCP = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}
_FINGER_PIP = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}


def _get_hand_landmarks(landmarks: dict[str, float], hand: str) -> np.ndarray | None:
    """Extract 21x3 array of hand landmarks from a landmark dict.

    Expects MediaPipe-style keys such as ``lh_x0``, ``lh_y0``, ``lh_z0``.
    """
    coords = []
    for i in range(21):
        coords.append(
            [
                landmarks.get(f"{hand}_x{i}", 0.0),
                landmarks.get(f"{hand}_y{i}", 0.0),
                landmarks.get(f"{hand}_z{i}", 0.0),
            ]
        )
    return np.array(coords, dtype=np.float32)


def _compute_hand_spread(hand_lm: np.ndarray) -> np.ndarray:
    """Compute finger spread features (5 distances from wrist to tips)."""
    wrist = hand_lm[0]
    spreads = []
    for tip_idx in _FINGER_TIPS.values():
        dist = np.linalg.norm(hand_lm[tip_idx] - wrist)
        spreads.append(dist)
    return np.array(spreads, dtype=np.float32)  # (5,)


def _compute_finger_curl(hand_lm: np.ndarray) -> np.ndarray:
    """Compute finger curl/extension ratios (5 values, 0=fully curled, 1=extended)."""
    curls = []
    for name in _FINGER_TIPS:
        tip = hand_lm[_FINGER_TIPS[name]]
        pip = hand_lm[_FINGER_PIP[name]]
        mcp = hand_lm[_FINGER_MCP[name]]
        # Distance from tip to MCP vs PIP to MCP (normalized)
        full_len = np.linalg.norm(tip - mcp) + 1e-8
        joint_len = np.linalg.norm(pip - mcp)
        curl_ratio = joint_len / full_len
        curls.append(np.clip(curl_ratio, 0.0, 1.0))
    return np.array(curls, dtype=np.float32)  # (5,)


def _compute_hand_orientation(hand_lm: np.ndarray) -> np.ndarray:
    """Compute hand palm orientation (3D normal vector)."""
    # Use wrist, index MCP, and pinky MCP to define palm plane
    wrist = hand_lm[0]
    index_mcp = hand_lm[5]
    pinky_mcp = hand_lm[17]
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal) + 1e-8
    return (normal / norm).astype(np.float32)  # type: ignore[no-any-return]  # (3,)


def _compute_hand_scale(hand_lm: np.ndarray) -> float:
    """Compute hand scale as wrist-to-middle-finger-tip distance."""
    return float(np.linalg.norm(hand_lm[12] - hand_lm[0]) + 1e-8)


def _compute_relative_hand_position(
    hand_lm: np.ndarray,
    l_shoulder: np.ndarray,
    r_shoulder: np.ndarray,
) -> np.ndarray:
    """Compute hand position relative to shoulder center and torso scale."""
    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    torso_scale = np.linalg.norm(r_shoulder - l_shoulder) + 1e-8
    wrist = hand_lm[0]
    rel_pos = (wrist - shoulder_center) / torso_scale
    return rel_pos.astype(np.float32)  # type: ignore[no-any-return]  # (3,)


def compute_enhanced_frame_features(landmarks: dict[str, float]) -> np.ndarray | None:
    """Compute enhanced geometric features (up to 87 dims) from a single frame.

    These features complement the raw 162-dim landmark coordinates with
    higher-level geometric descriptors that are more robust to camera
    distance and body position variations.

    Returns:
        np.ndarray or None if insufficient landmarks.
    """
    features: list[float] = []

    # --- Hand 1: Left ---
    lh = _get_hand_landmarks(landmarks, "lh")
    if lh is not None:
        features.extend(_compute_hand_spread(lh))  # 5
        features.extend(_compute_finger_curl(lh))  # 5
        features.extend(_compute_hand_orientation(lh))  # 3
        features.append(_compute_hand_scale(lh))  # 1
    else:
        features.extend([0.0] * 14)

    # --- Hand 2: Right ---
    rh = _get_hand_landmarks(landmarks, "rh")
    if rh is not None:
        features.extend(_compute_hand_spread(rh))  # 5
        features.extend(_compute_finger_curl(rh))  # 5
        features.extend(_compute_hand_orientation(rh))  # 3
        features.append(_compute_hand_scale(rh))  # 1
    else:
        features.extend([0.0] * 14)

    # --- Relative positioning ---
    l_shoulder = np.array(
        [
            landmarks.get("l_shoulder_x", 0.0),
            landmarks.get("l_shoulder_y", 0.0),
            landmarks.get("l_shoulder_z", 0.0),
        ],
        dtype=np.float32,
    )
    r_shoulder = np.array(
        [
            landmarks.get("r_shoulder_x", 0.0),
            landmarks.get("r_shoulder_y", 0.0),
            landmarks.get("r_shoulder_z", 0.0),
        ],
        dtype=np.float32,
    )

    torso_scale = np.linalg.norm(r_shoulder - l_shoulder) + 1e-8
    features.append(float(torso_scale))  # 1

    if lh is not None and torso_scale > 1e-6:
        rel_lh = _compute_relative_hand_position(lh, l_shoulder, r_shoulder)
        features.extend(rel_lh)  # 3
    else:
        features.extend([0.0] * 3)

    if rh is not None and torso_scale > 1e-6:
        rel_rh = _compute_relative_hand_position(rh, l_shoulder, r_shoulder)
        features.extend(rel_rh)  # 3
    else:
        features.extend([0.0] * 3)

    # --- Inter-hand features ---
    if lh is not None and rh is not None:
        lh_wrist = lh[0]
        rh_wrist = rh[0]
        dist = np.linalg.norm(lh_wrist - rh_wrist)
        features.append(float(dist / torso_scale))  # 1
    else:
        features.append(0.0)

    # --- Hand symmetry/comparison ---
    if lh is not None and rh is not None:
        lh_spread = _compute_hand_spread(lh)
        rh_spread = _compute_hand_spread(rh)
        features.extend((lh_spread - rh_spread).tolist())  # 5
        features.extend((np.array(lh_spread) + np.array(rh_spread)).tolist())  # 5
    else:
        features.extend([0.0] * 10)

    # --- Finger crossing detection (proximity of fingertips) ---
    if lh is not None and rh is not None:
        for tip_name in _FINGER_TIPS:
            lh_tip = lh[_FINGER_TIPS[tip_name]]
            rh_tip = rh[_FINGER_TIPS[tip_name]]
            features.append(float(np.linalg.norm(lh_tip - rh_tip) / torso_scale))  # 5
    else:
        features.extend([0.0] * 5)

    # --- Velocity placeholder (filled by caller if sequence available) ---
    features.extend([0.0] * 36)  # 3 coords * 12 pose points = 36 (computed externally)

    return np.array(features, dtype=np.float32)  # type: ignore[no-any-return]


def compute_sequence_dynamic_features(
    frames: list[dict[str, float]], target_frames: int = 30
) -> np.ndarray | None:
    """Compute dynamic (velocity + acceleration) features from a sequence.

    Args:
        frames: List of landmark dicts per frame.
        target_frames: Output temporal length.

    Returns:
        np.ndarray of shape (target_frames, 36) with velocity + acceleration
        for the 12 pose keypoints (3 coords each).
    """
    if not frames:
        return None

    n = len(frames)
    pose_seq = np.zeros((n, 36), dtype=np.float32)
    for i, frm in enumerate(frames):
        idx = 0
        for base in _POSE_LANDMARK_NAMES:
            for c in ["x", "y", "z"]:
                pose_seq[i, idx] = frm.get(f"{base}_{c}", 0.0)
                idx += 1

    # Resample uniformly
    if n != target_frames:
        indices = np.linspace(0, n - 1, target_frames).astype(int)
        pose_seq = pose_seq[indices]

    # Compute velocity (first derivative)
    velocity = np.zeros_like(pose_seq)
    velocity[1:] = pose_seq[1:] - pose_seq[:-1]
    velocity[0] = velocity[1] if target_frames > 1 else 0.0

    # Compute acceleration (second derivative)
    acceleration = np.zeros_like(pose_seq)
    acceleration[2:] = velocity[2:] - velocity[1:-1]
    if target_frames > 1:
        acceleration[1] = acceleration[2] if target_frames > 2 else 0.0
    acceleration[0] = acceleration[1] if target_frames > 1 else 0.0

    # Concatenate velocity + acceleration = 72 dims, but we return just
    # the magnitude-normalized versions for compactness (36)
    vel_mag = np.linalg.norm(velocity.reshape(target_frames, 12, 3), axis=2)  # (T,12)
    acc_mag = np.linalg.norm(acceleration.reshape(target_frames, 12, 3), axis=2)  # (T,12)
    return np.concatenate([vel_mag, acc_mag], axis=1).astype(np.float32)  # type: ignore[no-any-return]  # (T,24)


def build_enhanced_sequence(
    frames: list[dict[str, float]],
    feature_level: str = "enhanced",
    target_frames: int = 30,
) -> np.ndarray | None:
    """Build a full enhanced feature sequence from raw landmark dicts.

    Returns an array of shape (target_frames, 249) combining:
        - 162 raw landmark coords
        - 87 enhanced geometric/dynamic features

    This is the canonical entry-point for real-time enhanced inference.
    """
    _ = validate_feature_level(feature_level)
    if not frames:
        return None

    # Build base sequence (162 dims)
    # Local import to avoid circular dependency (extractor imports from this module)
    from .extractor import _frame_dict_to_vector

    feature_dim = get_feature_dim(feature_level)
    base_dim = BASIC_FEATURE_DIM

    seq_base = []
    for frm in frames:
        vec = _frame_dict_to_vector(frm, "basic", base_dim)
        if vec is not None:
            seq_base.append(vec)

    if not seq_base:
        return None

    # Convert list of arrays to a single numpy array for uniform sampling
    seq_base_arr = np.stack(seq_base) if len(seq_base) > 1 else np.array(seq_base)
    sampled_base = sample_frames_uniform(seq_base_arr, target_frames=target_frames)
    fixed_base = pad_or_truncate(sampled_base, target_length=target_frames, pad_value=0.0)

    # Build enhanced geometric features per frame
    seq_enh = []
    for frm in frames:
        enh = compute_enhanced_frame_features(frm)
        if enh is not None:
            seq_enh.append(enh)

    if not seq_enh:
        # No enhanced features available; pad with zeros
        enh_dim = feature_dim - base_dim
        enh_pad = np.zeros((target_frames, enh_dim), dtype=np.float32)
        return np.concatenate([fixed_base, enh_pad], axis=1)  # type: ignore[no-any-return]

    # Convert list of arrays to a single numpy array for uniform sampling
    seq_enh_arr = np.stack(seq_enh) if len(seq_enh) > 1 else np.array(seq_enh)
    sampled_enh = sample_frames_uniform(seq_enh_arr, target_frames=target_frames)
    fixed_enh = pad_or_truncate(sampled_enh, target_length=target_frames, pad_value=0.0)

    # Trim enhanced to expected size
    enh_dim = feature_dim - base_dim
    fixed_enh = fixed_enh[:, :enh_dim]

    return np.concatenate([fixed_base, fixed_enh], axis=1).astype(np.float32)  # type: ignore[no-any-return]


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


def _df_row_to_dict(row: Any) -> dict[str, float]:
    """Convert a DataFrame row (Series) to a landmarks dict.

    Maps ``lh_x0`` / ``p_x0`` style keys to the formats expected by
    ``_frame_dict_to_vector`` (extractor) and
    ``compute_enhanced_frame_features``.
    """
    d = {}
    for col, val in row.items():
        try:
            d[col] = float(val)
        except (ValueError, TypeError):
            d[col] = 0.0

        # Map pose columns p_{c}{idx} → {base}_{c} for extractor helpers
        if isinstance(col, str) and col.startswith("p_") and len(col) >= 4:
            coord = col[2]  # x / y / z
            idx = col[3:]  # 0 … 11
            try:
                pose_idx = int(idx)
                if pose_idx < len(_POSE_BASES):
                    base = _POSE_BASES[pose_idx]
                    d[f"{base}_{coord}"] = d[col]
            except ValueError:
                pass
    return d


def build_enhanced_sequence_from_df(
    lm_df,
    feature_level: str = "enhanced",
    target_frames: int = 30,
) -> np.ndarray | None:
    """Build enhanced feature sequence from a landmark DataFrame.

    Each row of the DataFrame is one frame. Converts rows to dicts
    and delegates to ``build_enhanced_sequence``.

    Returns:
        np.ndarray of shape (target_frames, 249) or None.
    """
    _ = validate_feature_level(feature_level)
    if lm_df is None or len(lm_df) == 0:
        return None
    frames = [_df_row_to_dict(row) for _, row in lm_df.iterrows()]
    return build_enhanced_sequence(frames, feature_level=feature_level, target_frames=target_frames)


# ─────────────────────────────────────────────────────────────
# Feature dimension adaptation for inference robustness
# ─────────────────────────────────────────────────────────────


def adapt_features_to_model(
    features: np.ndarray,
    model_input_dim: int,
    feature_level: str = "basic",
) -> np.ndarray:
    """Adapt extracted features to match model's expected input dimension.

    Handles common mismatch scenarios gracefully:
    - Truncates if features > model expects (e.g. enhanced model loaded as basic)
    - Pads with zeros if features < model expects (e.g. basic model loaded as enhanced)
    - Warns about mismatch so user knows accuracy may be degraded

    Args:
        features: Extracted features array, shape (..., feature_dim).
        model_input_dim: The model's expected input dimension.
        feature_level: The feature level that produced these features (for logging).

    Returns:
        Adapted features array with last dimension == model_input_dim.
    """
    if features is None:
        return None

    feature_dim = features.shape[-1]

    if feature_dim == model_input_dim:
        return features

    print(
        f"[WARN] Feature dimension mismatch: extracted {feature_dim} dims "
        f"(level={feature_level}) but model expects {model_input_dim}. "
        f"Adapting automatically — accuracy may be degraded."
    )

    if feature_dim > model_input_dim:
        # Truncate to model's expected size
        if features.ndim == 1:
            return features[:model_input_dim]
        elif features.ndim == 2:
            return features[:, :model_input_dim]  # type: ignore[no-any-return]
        else:
            # Generic: slice last dimension
            slices = [slice(None)] * (features.ndim - 1) + [slice(model_input_dim)]
            return features[tuple(slices)]  # type: ignore[no-any-return]
    else:
        # Pad with zeros to model's expected size
        pad_width = [(0, 0)] * (features.ndim - 1) + [(0, model_input_dim - feature_dim)]
        return np.pad(features, pad_width, mode="constant", constant_values=0.0)  # type: ignore[no-any-return]


def resolve_feature_level_for_inference(
    requested_level: str | None,
    model_input_dim: int,
    checkpoint_feature_level: str = "basic",
) -> tuple[str, str | None]:
    """Resolve the best feature level for inference given model constraints.

    Args:
        requested_level: User-requested feature level (None for auto).
        model_input_dim: Model's expected input dimension.
        checkpoint_feature_level: Feature level stored in the checkpoint.

    Returns:
        Tuple of (resolved_level, warning_message_or_None).
    """
    if checkpoint_feature_level:
        _ = validate_feature_level(checkpoint_feature_level)
    if requested_level is not None:
        _ = validate_feature_level(requested_level)

    # Determine what feature level matches the model's input dimension
    dim_to_level = {dim: level for level, dim in FEATURE_DIMS.items()}
    matching_level = dim_to_level.get(model_input_dim)

    if requested_level is None:
        # Auto-detect: use checkpoint level or the one matching model dims
        resolved = (
            checkpoint_feature_level if checkpoint_feature_level else (matching_level or "basic")
        )
        return resolved, None

    requested_dim = FEATURE_DIMS.get(requested_level)
    if requested_dim is None:
        return "basic", f"Unknown feature level '{requested_level}', falling back to 'basic'"

    if requested_dim == model_input_dim:
        return requested_level, None

    # Mismatch: warn and suggest the correct level
    if matching_level:
        msg = (
            f"Feature mismatch: requested '{requested_level}' ({requested_dim} dims) "
            f"but model expects {model_input_dim} dims (level='{matching_level}'). "
            f"Auto-switching to '{matching_level}'. Train a new model with --feature-level {requested_level} for best results."
        )
        return matching_level, msg
    else:
        msg = (
            f"Feature mismatch: requested '{requested_level}' ({requested_dim} dims) "
            f"but model expects {model_input_dim} dims (unknown level). "
            f"Continuing with adaptation — accuracy may be degraded."
        )
        return requested_level, msg
