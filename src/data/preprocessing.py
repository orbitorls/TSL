# src/data/preprocessing.py - Feature extraction and normalization
import numpy as np


# Feature dimension mapping
FEATURE_DIMS = {
    'basic': 162,      # Hand + Pose (63 + 63 + 36)
    'finger': 258,     # + finger details
    'full': 1596,      # + face
    'face': 1434,      # Face only
}


def extract_features_from_landmarks(lm_df, feature_level='basic'):
    """Extract features from MediaPipe landmark DataFrame.
    
    Args:
        lm_df: DataFrame with landmark columns (lh_x0, rh_x0, etc.)
        feature_level: 'basic', 'finger', 'full', or 'face'
        
    Returns:
        numpy array of features (feature_dim,)
    """
    features = []
    
    from utils.dataset_utils import safe_mean
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
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            if col in lm_df.columns:
                features.append(safe_mean(lm_df[col]))
            else:
                features.append(0.0)
    
    # ===== 2. FINGER: Additional finger details =====
    if feature_level in ['finger', 'full', 'face']:
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for hand_prefix in ['lh_', 'rh_']:
            for finger in finger_names:
                for c in ['x', 'y', 'z']:
                    for joint in ['mcp', 'pip', 'dip']:
                        col = f'{hand_prefix}{finger}_{joint}_{c}'
                        if col in lm_df.columns:
                            from utils.dataset_utils import safe_mean
                            features.append(safe_mean(lm_df[col]))
                        else:
                            features.append(0.0)
    
    # ===== 3. FACE: 478 Facial Landmarks (1434 features) =====
    if feature_level in ['face', 'full']:
        for i in range(478):
            for c in ['x', 'y', 'z']:
                col = f'face_{c}{i}'
                if col in lm_df.columns:
                    from utils.dataset_utils import safe_mean
                    features.append(safe_mean(lm_df[col]))
                else:
                    features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def normalize_landmarks(features, mean=None, std=None):
    """Normalize features using z-score normalization.
    
    Args:
        features: Raw feature array
        mean: Pre-computed mean (if None, computed from features)
        std: Pre-computed std (if None, computed from features)
        
    Returns:
        Normalized features, (mean, std) tuple
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, (mean, std)


def extract_hand_features(landmarks):
    """Extract simplified hand features only.
    
    Args:
        landmarks: Dict of landmark values
        
    Returns:
        Feature array (162,)
    """
    features = []
    
    # Left hand (63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            features.append(landmarks.get(f'lh_{c}{i}', 0.0))
    
    # Right hand (63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            features.append(landmarks.get(f'rh_{c}{i}', 0.0))
    
    # Pose (36)
    pose_keys = [
        'l_shoulder_x', 'l_shoulder_y', 'l_shoulder_z',
        'r_shoulder_x', 'r_shoulder_y', 'r_shoulder_z',
        'l_elbow_x', 'l_elbow_y', 'l_elbow_z',
        'r_elbow_x', 'r_elbow_y', 'r_elbow_z',
        'l_wrist_x', 'l_wrist_y', 'l_wrist_z',
        'r_wrist_x', 'r_wrist_y', 'r_wrist_z',
        'lbrow_outer_x', 'lbrow_outer_y', 'lbrow_outer_z',
        'lbrow_inner_x', 'lbrow_inner_y', 'lbrow_inner_z',
        'rbrow_inner_x', 'rbrow_inner_y', 'rbrow_inner_z',
        'rbrow_outer_x', 'rbrow_outer_y', 'rbrow_outer_z',
        'mouth_right_x', 'mouth_right_y', 'mouth_right_z',
        'mouth_left_x', 'mouth_left_y', 'mouth_left_z',
    ]
    for key in pose_keys:
        features.append(landmarks.get(key, 0.0))
    
    return np.array(features, dtype=np.float32)
