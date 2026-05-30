"""Data augmentation for TSL-51 training."""

import numpy as np

from src.core.features import BASIC_FEATURE_DIM, HAND_FEATURE_DIM


def _validate_basic_feature_array(X: np.ndarray) -> None:
    """Ensure augmentation operates on the canonical 162-dim basic schema."""
    if X.ndim not in (2, 3):
        raise ValueError(
            "augment_data expects basic 162-dim feature arrays shaped "
            "(n_samples, 162) or (n_samples, T, 162)."
        )
    if X.shape[-1] != BASIC_FEATURE_DIM:
        raise ValueError(
            f"augment_data only supports the basic {BASIC_FEATURE_DIM}-dim feature schema; "
            f"got last dimension {X.shape[-1]}."
        )


def augment_data(X, y, _classes, augmentation_factor=2, noise_level=0.01, scale_range=(0.95, 1.05)):
    """Augment basic 162-dim data by applying transformations.

    Args:
        X: Feature array — either (n_samples, n_features) or (n_samples, T, n_features)
        y: Label array
        classes: Class names
        augmentation_factor: How many augmented copies per sample
        noise_level: Standard deviation of Gaussian noise
        scale_range: Tuple of (min, max) scale factors

    Returns:
        X_aug, y_aug (augmented data)
    """
    _validate_basic_feature_array(X)
    print(f"Applying data augmentation ({augmentation_factor}x)...")

    seq_mode = X.ndim == 3  # (N, T, feature_dim)

    X_list = [X]  # Original data
    y_list = [y]

    n_samples = len(X)

    for aug_idx in range(augmentation_factor):
        X_aug = np.zeros_like(X)

        for i in range(n_samples):
            features = X[i].copy()

            aug_type = np.random.choice(["noise", "scale", "noise_scale", "flip"])

            if aug_type == "noise":
                noise = np.random.normal(0, noise_level, features.shape)
                X_aug[i] = features + noise

            elif aug_type == "scale":
                scale = np.random.uniform(scale_range[0], scale_range[1])
                X_aug[i] = features * scale

            elif aug_type == "noise_scale":
                scale = np.random.uniform(scale_range[0], scale_range[1])
                noise = np.random.normal(0, noise_level, features.shape)
                X_aug[i] = features * scale + noise

            else:  # flip - mirror left/right hand in the canonical basic schema
                if seq_mode:
                    left_hand = features[:, 0:HAND_FEATURE_DIM].copy()
                    right_hand = features[:, HAND_FEATURE_DIM : HAND_FEATURE_DIM * 2].copy()
                    X_aug[i] = np.concatenate(
                        [right_hand, left_hand, features[:, HAND_FEATURE_DIM * 2 :]], axis=1
                    )
                else:
                    left_hand = features[0:HAND_FEATURE_DIM].copy()
                    right_hand = features[HAND_FEATURE_DIM : HAND_FEATURE_DIM * 2].copy()
                    X_aug[i] = np.concatenate(
                        [right_hand, left_hand, features[HAND_FEATURE_DIM * 2 :]]
                    )

        X_list.append(X_aug)
        y_list.append(y)

        if (aug_idx + 1) % 5 == 0:
            print(f"  Augmented {aug_idx + 1}/{augmentation_factor}")

    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)

    print(f"Augmented: {len(X)} -> {len(X_final)} samples")
    return X_final, y_final
