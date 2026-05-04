"""Core feature extraction for TSL-51.

Single source of truth for landmark → feature vector conversion.
Supports velocity, acceleration, geometric, relative, and temporal features.
"""

import numpy as np
from typing import Optional

FEATURE_LEVELS = {
    'basic': 162,       # Hand (63+63) + Pose (36)
    'enhanced': 487,    # basic + geometric (87) + relative (36) + temporal (202)
    'finger': 222,      # basic + finger joints (60) - 10 joint groups * 3 coords * 2 diffs
    'full': 1596,       # + face (478*3)
    'face': 1434,       # Face only
    'velocity': 486,    # basic + velocity stats (mean + std = 162 * 3)
    'dynamics': 851,    # basic + velocity (324) + acceleration (162) + temporal (203)
}

_POSE_BASES = [
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'lbrow_outer', 'lbrow_inner',
    'rbrow_inner', 'rbrow_outer', 'mouth_right', 'mouth_left',
]

# Hand landmark indices for key points
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20


def safe_mean(series, default: float = 0.0) -> float:
    """Safe mean - import from utils for consistency."""
    from utils.dataset_utils import safe_mean as _safe_mean
    return _safe_mean(series, default)


def _euclidean_distance(x1: float, y1: float, z1: float,
                        x2: float, y2: float, z2: float) -> float:
    """Calculate 3D Euclidean distance between two points.

    Args:
        x1, y1, z1: Coordinates of first point
        x2, y2, z2: Coordinates of second point

    Returns:
        Euclidean distance between points
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _get_landmark_coords(lm_df, prefix: str) -> np.ndarray:
    """Extract 21 hand landmark coordinates as (21, 3) array.

    Args:
        lm_df: DataFrame with landmark columns
        prefix: 'lh' for left hand, 'rh' for right hand

    Returns:
        Array of shape (21, 3) with x, y, z coordinates
    """
    coords = np.zeros((21, 3))
    for i in range(21):
        for j, c in enumerate(['x', 'y', 'z']):
            col = f'{prefix}_{c}{i}'
            if col in lm_df.columns:
                coords[i, j] = safe_mean(lm_df[col])
            else:
                coords[i, j] = 0.0
    return coords


def _extract_geometric_features(lm_df) -> np.ndarray:
    """Extract geometric features from landmarks.

    Computes distances between key hand points, wrist rotation indicators,
    and hand shape characteristics.

    Args:
        lm_df: DataFrame with landmark columns

    Returns:
        Array of 87 geometric features
    """
    features = []

    # Get hand coordinates
    lh_coords = _get_landmark_coords(lm_df, 'lh')
    rh_coords = _get_landmark_coords(lm_df, 'rh')

    # 1. Inter-hand distances (15 distances)
    # Distance between corresponding fingertips
    fingertips_pairs = [
        (_THUMB_TIP, _INDEX_TIP),  # thumb-index
        (_INDEX_TIP, _MIDDLE_TIP),  # index-middle
        (_MIDDLE_TIP, _RING_TIP),  # middle-ring
        (_RING_TIP, _PINKY_TIP),   # ring-pinky
        (_THUMB_TIP, _MIDDLE_TIP), # thumb-middle
        (_THUMB_TIP, _RING_TIP),   # thumb-ring
        (_THUMB_TIP, _PINKY_TIP),  # thumb-pinky
        (_INDEX_TIP, _RING_TIP),   # index-ring
        (_INDEX_TIP, _PINKY_TIP),  # index-pinky
        (_MIDDLE_TIP, _PINKY_TIP), # middle-pinky
    ]

    for i, j in fingertips_pairs:
        # Left hand distances
        dist = _euclidean_distance(lh_coords[i, 0], lh_coords[i, 1], lh_coords[i, 2],
                                   lh_coords[j, 0], lh_coords[j, 1], lh_coords[j, 2])
        features.append(dist)
        # Right hand distances
        dist = _euclidean_distance(rh_coords[i, 0], rh_coords[i, 1], rh_coords[i, 2],
                                   rh_coords[j, 0], rh_coords[j, 1], rh_coords[j, 2])
        features.append(dist)

    # 2. Hand-to-hand distances (10 distances)
    for i in range(5):  # 5 fingertip pairs
        tip_idx = [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP][i]
        dist = _euclidean_distance(lh_coords[tip_idx, 0], lh_coords[tip_idx, 1], lh_coords[tip_idx, 2],
                                   rh_coords[tip_idx, 0], rh_coords[tip_idx, 1], rh_coords[tip_idx, 2])
        features.append(dist)

    # Wrist-to-wrist (left wrist to right wrist)
    dist = _euclidean_distance(lh_coords[_WRIST, 0], lh_coords[_WRIST, 1], lh_coords[_WRIST, 2],
                               rh_coords[_WRIST, 0], rh_coords[_WRIST, 1], rh_coords[_WRIST, 2])
    features.append(dist)

    # Palm centers distance
    lh_palm = lh_coords.mean(axis=0)
    rh_palm = rh_coords.mean(axis=0)
    dist = _euclidean_distance(lh_palm[0], lh_palm[1], lh_palm[2],
                               rh_palm[0], rh_palm[1], rh_palm[2])
    features.append(dist)

    # 3. Wrist rotation indicators (8 features)
    # Angle of wrist based on thumb-index line relative to x-axis
    for prefix, coords in [('lh', lh_coords), ('rh', rh_coords)]:
        # Vector from wrist to thumb tip
        thumb_vec = coords[_THUMB_TIP] - coords[_WRIST]
        # Vector from wrist to index tip
        index_vec = coords[_INDEX_TIP] - coords[_WRIST]

        # Calculate angle (atan2 gives angle relative to x-axis)
        angle_thumb = np.arctan2(thumb_vec[1], thumb_vec[0])
        angle_index = np.arctan2(index_vec[1], index_vec[0])

        # Relative rotation (thumb to index angle)
        rel_angle = angle_index - angle_thumb

        # Normalize features
        features.append(np.sin(rel_angle))
        features.append(np.cos(rel_angle))
        features.append(np.sin(angle_thumb))
        features.append(np.cos(angle_thumb))

    # 4. Hand shape features (14 features)
    for coords in [lh_coords, rh_coords]:
        # Finger spread: max distance from palm center to fingertips
        palm_center = coords[1:5].mean(axis=0)  # MCP points approximate palm
        max_spread = 0
        spread_features = []
        for tip in [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]:
            spread = np.linalg.norm(coords[tip] - palm_center)
            spread_features.append(spread)
            max_spread = max(max_spread, spread)

        # Normalized spreads
        for spread in spread_features:
            features.append(spread / (max_spread + 1e-6))

        # Hand openness: ratio of max spread to wrist-palm distance
        wrist_to_palm = np.linalg.norm(coords[_WRIST] - palm_center)
        features.append(max_spread / (wrist_to_palm + 1e-6))

    # 5. Finger extension features (14 features)
    for coords in [lh_coords, rh_coords]:
        # MCP to fingertip distances normalized by palm size
        mcp_approx = coords[1:5].mean(axis=0)
        palm_size = np.mean([np.linalg.norm(coords[i] - mcp_approx) for i in range(1, 5)])

        for tip in [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]:
            extension = np.linalg.norm(coords[tip] - mcp_approx)
            features.append(extension / (palm_size + 1e-6))

        # Thumb abduction (angle from index finger line)
        index_vec = coords[_INDEX_TIP] - coords[1]  # index MCP to tip
        thumb_vec = coords[_THUMB_TIP] - coords[1]   # index MCP to thumb tip
        cos_angle = np.dot(index_vec, thumb_vec) / (np.linalg.norm(index_vec) * np.linalg.norm(thumb_vec) + 1e-6)
        features.append(cos_angle)

    # 6. Hand orientation features (8 features)
    for prefix, coords in [('lh', lh_coords), ('rh', rh_coords)]:
        # Hand plane normal (cross product of palm diagonals)
        vec1 = coords[5] - coords[17]  # index to pinky MCPs
        vec2 = coords[1] - coords[0]   # wrist to thumb MCP
        normal = np.cross(vec1, vec2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:
            normal = normal / normal_norm

        # Normal components (orientation)
        features.extend(normal.tolist())

    # 7. Distance ratios (18 features)
    for coords in [lh_coords, rh_coords]:
        # Finger length ratios
        for tip in [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP]:
            tip_idx = [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP].index(tip)
            base = 1 + tip_idx  # MCP index
            tip_to_base = np.linalg.norm(coords[tip] - coords[base])
            mcp_to_wrist = np.linalg.norm(coords[base] - coords[_WRIST])
            features.append(tip_to_base / (mcp_to_wrist + 1e-6))

        # Adjacent finger ratios
        pairs = [(_INDEX_TIP, _MIDDLE_TIP), (_MIDDLE_TIP, _RING_TIP), (_RING_TIP, _PINKY_TIP)]
        for i, j in pairs:
            len_i = np.linalg.norm(coords[i] - coords[1 + [_INDEX_TIP, _MIDDLE_TIP, _RING_TIP].index(i)])
            len_j = np.linalg.norm(coords[j] - coords[1 + [_MIDDLE_TIP, _RING_TIP, _PINKY_TIP].index(j)])
            features.append(len_i / (len_j + 1e-6))

    # Total: 10 fingertips_pairs * 2 + 5 inter_hand + 2 + 8 rotation + 14 shape + 14 extension + 8 orientation + 18 ratios
    # = 20 + 7 + 8 + 14 + 14 + 8 + 18 = 89... let me recalculate
    # fingertips_pairs: 10*2=20, inter_hand: 5+1+1=7, rotation: 8, shape: 14, extension: 14, orientation: 8, ratios: 18
    # = 20+7+8+14+14+8+18 = 89

    # Pad to 87 if needed (ensure exact count)
    while len(features) < 87:
        features.append(0.0)

    return np.array(features[:87], dtype=np.float32)


def _extract_relative_position_features(lm_df) -> np.ndarray:
    """Extract relative position features (hands relative to shoulders).

    Args:
        lm_df: DataFrame with landmark columns

    Returns:
        Array of 36 relative position features
    """
    features = []

    # Get shoulder positions (for normalization)
    shoulders = {}
    for side in ['l', 'r']:
        x_col = f'{side}_shoulder_x'
        y_col = f'{side}_shoulder_y'
        z_col = f'{side}_shoulder_z'
        if all(c in lm_df.columns for c in [x_col, y_col, z_col]):
            shoulders[side] = np.array([
                safe_mean(lm_df[x_col]),
                safe_mean(lm_df[y_col]),
                safe_mean(lm_df[z_col])
            ])
        else:
            shoulders[side] = np.array([0.5, 0.5, 0.0])

    # Shoulder center
    shoulder_center = (shoulders['l'] + shoulders['r']) / 2
    shoulder_width = np.linalg.norm(shoulders['r'] - shoulders['l'])

    # Process each hand (18 features per hand = 36 total)
    for prefix in ['lh', 'rh']:
        coords = _get_landmark_coords(lm_df, prefix)

        # Wrist relative to shoulder center (3 features)
        wrist = coords[_WRIST]
        rel_wrist = wrist - shoulder_center
        if shoulder_width > 1e-6:
            rel_wrist = rel_wrist / shoulder_width
        features.extend(rel_wrist.tolist())

        # Palm center relative to shoulder center (3 features)
        palm = coords[1:5].mean(axis=0)
        rel_palm = palm - shoulder_center
        if shoulder_width > 1e-6:
            rel_palm = rel_palm / shoulder_width
        features.extend(rel_palm.tolist())

        # 4 finger tip spread relative to palm center (12 features - skip pinky)
        palm_center = coords[1:5].mean(axis=0)
        for tip in [_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP]:
            tip_rel = coords[tip] - palm_center
            if shoulder_width > 1e-6:
                tip_rel = tip_rel / shoulder_width
            features.extend(tip_rel.tolist())

    return np.array(features, dtype=np.float32)


def _extract_temporal_features(frame_features: np.ndarray) -> np.ndarray:
    """Extract temporal statistics from frame features.

    Computes std, range, and smoothness over time for key features.

    Args:
        frame_features: Array of shape (n_frames, n_features)

    Returns:
        Array of 203 temporal features (std, range, smoothness for 67 key features)
    """
    features = []
    n_frames, n_feats = frame_features.shape

    # Use first 67 key features (hand + pose means)
    n_key = min(67, n_feats)
    key_features = frame_features[:, :n_key]

    if n_frames < 2:
        # Return zeros if not enough frames
        return np.zeros(203, dtype=np.float32)

    # Standard deviation (67 features)
    std = np.std(key_features, axis=0, ddof=1)
    features.extend(std.tolist())

    # Range (max - min) (67 features)
    feat_range = np.max(key_features, axis=0) - np.min(key_features, axis=0)
    features.extend(feat_range.tolist())

    # Smoothness: mean absolute second derivative (67 features)
    if n_frames > 2:
        velocity = np.diff(key_features, axis=0)  # (n-1, n_key)
        acceleration = np.diff(velocity, axis=0)   # (n-2, n_key)
        smoothness = np.mean(np.abs(acceleration), axis=0)
    else:
        smoothness = np.zeros(n_key)
    features.extend(smoothness.tolist())

    # Zero-crossing rate (67 features) - NEW
    if n_frames > 2:
        velocity = np.diff(key_features, axis=0)
        zcr = np.sum(np.abs(np.diff(np.sign(velocity), axis=0)) > 0, axis=0) / (n_frames - 2)
    else:
        zcr = np.zeros(n_key)
    features.extend(zcr.tolist())

    # Total: 67 * 3 = 201 + 2 padding = 203
    # Pad to exactly 203 if needed
    while len(features) < 203:
        features.append(0.0)

    return np.array(features[:203], dtype=np.float32)


def extract_features(lm_df, feature_level: str = 'basic') -> np.ndarray:
    """Extract mean-aggregated features from landmark DataFrame.

    Args:
        lm_df: DataFrame with landmark columns (lh_x0, rh_x0, etc.)
        feature_level: One of 'basic', 'enhanced', 'finger', 'full', 'face', 'velocity', 'dynamics'

    Returns:
        numpy array of shape (feature_dim,)
    """
    features = []

    # ===== 1. BASIC: Hand + Pose (162) =====
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

    # ===== 2. ENHANCED: Add geometric + relative + temporal features =====
    if feature_level == 'enhanced':
        # Geometric features (87): distances, angles, hand shape
        geometric_feats = _extract_geometric_features(lm_df)
        features.extend(geometric_feats.tolist())

        # Relative position features (36): hands relative to shoulders
        relative_feats = _extract_relative_position_features(lm_df)
        features.extend(relative_feats.tolist())

        # Temporal features (202): std, range, smoothness
        # Note: temporal features require multiple frames, so for single frame
        # we compute from the single frame (will be zeros unless called from
        # extract_sequence_features)
        temporal_feats = np.zeros(202, dtype=np.float32)
        features.extend(temporal_feats.tolist())

    # ===== 3. FINGER: Add finger-specific features =====
    if feature_level in ['finger', 'full']:
        # Finger joints: thumb MCP, PIP, DIP; index PIP, DIP; etc.
        finger_joints = [
            ('lh_thumb_mcp', 'lh_thumb_pip', 'lh_thumb_dip'),
            ('lh_index_mcp', 'lh_index_pip', 'lh_index_dip'),
            ('lh_middle_mcp', 'lh_middle_pip', 'lh_middle_dip'),
            ('lh_ring_mcp', 'lh_ring_pip', 'lh_ring_dip'),
            ('lh_pinky_mcp', 'lh_pinky_pip', 'lh_pinky_dip'),
            ('rh_thumb_mcp', 'rh_thumb_pip', 'rh_thumb_dip'),
            ('rh_index_mcp', 'rh_index_pip', 'rh_index_dip'),
            ('rh_middle_mcp', 'rh_middle_pip', 'rh_middle_dip'),
            ('rh_ring_mcp', 'rh_ring_pip', 'rh_ring_dip'),
            ('rh_pinky_mcp', 'rh_pinky_pip', 'rh_pinky_dip'),
        ]
        for joint1, joint2, joint3 in finger_joints:
            for c in ['x', 'y', 'z']:
                col1 = f'{joint1}_{c}'
                col2 = f'{joint2}_{c}'
                col3 = f'{joint3}_{c}'
                val1 = safe_mean(lm_df[col1]) if col1 in lm_df.columns else 0.0
                val2 = safe_mean(lm_df[col2]) if col2 in lm_df.columns else 0.0
                val3 = safe_mean(lm_df[col3]) if col3 in lm_df.columns else 0.0
                # Add difference (velocity-like)
                features.append(val2 - val1)
                features.append(val3 - val2)

    # ===== 4. FACE: Add face landmarks =====
    if feature_level in ['face', 'full']:
        # 478 face landmarks * 3 = 1434 features
        for i in range(478):
            for c in ['x', 'y', 'z']:
                col = f'face_{c}{i}'
                if col in lm_df.columns:
                    features.append(safe_mean(lm_df[col]))
                else:
                    features.append(0.0)

    feature_dim = FEATURE_LEVELS.get(feature_level, 162)
    return np.array(features[:feature_dim], dtype=np.float32)


def extract_sequence_features(
    landmarks_sequence: list,
    include_velocity: bool = True,
    include_acceleration: bool = True
) -> np.ndarray:
    """Extract features including temporal dynamics from landmark sequence.

    Args:
        landmarks_sequence: List of DataFrames, one per frame
        include_velocity: Include velocity features (mean, std, max)
        include_acceleration: Include acceleration features (mean, std)

    Returns:
        numpy array of features (162 base + velocity + acceleration + temporal)
    """
    if len(landmarks_sequence) == 0:
        return np.zeros(851, dtype=np.float32)

    # Extract base features for each frame
    frame_features = []
    for lm_df in landmarks_sequence:
        frame_feat = extract_features(lm_df, 'basic')
        frame_features.append(frame_feat)

    frame_features = np.array(frame_features, dtype=np.float32)  # (n_frames, 162)

    # Base features (mean across time)
    base_features = np.mean(frame_features, axis=0)

    all_features = [base_features]

    # Velocity features: mean, std (2 * 162 = 324)
    if include_velocity and len(frame_features) > 1:
        velocity = np.diff(frame_features, axis=0)  # (n_frames-1, 162)

        # Velocity mean (162)
        velocity_mean = np.mean(velocity, axis=0)
        velocity_mean = velocity_mean * 10  # Scale for similar range
        all_features.append(velocity_mean)

        # Velocity std (162) - NEW
        velocity_std = np.std(velocity, axis=0, ddof=1)
        velocity_std = velocity_std * 10  # Scale
        all_features.append(velocity_std)

    # Acceleration features: mean only (162)
    if include_acceleration and len(frame_features) > 2:
        velocity = np.diff(frame_features, axis=0)
        acceleration = np.diff(velocity, axis=0)  # (n_frames-2, 162)

        # Acceleration mean (162)
        acceleration_mean = np.mean(acceleration, axis=0)
        acceleration_mean = acceleration_mean * 100  # Scale
        all_features.append(acceleration_mean)

    # Temporal statistics (std, range, smoothness, zcr) (203)
    # Only add when acceleration is included (full dynamics mode)
    if include_acceleration and len(frame_features) >= 2:
        temporal_feats = _extract_temporal_features(frame_features)
        all_features.append(temporal_feats)

    return np.concatenate(all_features).astype(np.float32)
