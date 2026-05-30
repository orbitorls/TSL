"""Data augmentation for TSL-51 training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.core.features import BASIC_FEATURE_DIM, HAND_FEATURE_DIM
from src.train.splits import get_manifest_split_rows


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


def _normalise_sample_ids(sample_ids: Sequence[object]) -> list[str]:
    normalised = [str(sample_id) for sample_id in sample_ids]
    if len(normalised) != len(set(normalised)):
        raise ValueError("sample_ids must be unique when building split datasets")
    return normalised


def _select_split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    split_rows: Sequence[Mapping[str, Any]],
    index_by_id: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    split_ids = []
    indices = []
    for row in split_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("Manifest row is missing sample_id")
        if sample_id not in index_by_id:
            raise ValueError(f"Manifest sample_id {sample_id!r} not found in provided sample_ids")
        split_ids.append(sample_id)
        indices.append(index_by_id[sample_id])
    return X[indices], y[indices], split_ids


def _build_augmented_sample_ids(sample_ids: Sequence[str], augmentation_factor: int) -> list[str]:
    generated = []
    for aug_idx in range(augmentation_factor):
        generated.extend(f"{sample_id}__aug_{aug_idx + 1}" for sample_id in sample_ids)
    return list(sample_ids) + generated


def augment_train_split_only(
    X,
    y,
    sample_ids: Sequence[object],
    manifest: Mapping[str, Any],
    classes,
    augmentation_factor=0,
    noise_level=0.01,
    scale_range=(0.95, 1.05),
):
    """Materialize split datasets and apply augmentation to train only.

    The manifest is the source of truth for split membership. Validation and
    test are always returned original-only and in manifest order.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    sample_id_list = _normalise_sample_ids(sample_ids)

    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must contain the same number of samples")
    if len(X_arr) != len(sample_id_list):
        raise ValueError("sample_ids length must match X and y")

    index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_id_list)}
    split_data = {}
    for split in ("train", "val", "test"):
        split_rows = get_manifest_split_rows(manifest, split)
        split_X, split_y, split_sample_ids = _select_split_arrays(
            X_arr,
            y_arr,
            split_rows,
            index_by_id,
        )

        if split == "train" and augmentation_factor > 0:
            split_X, split_y = augment_data(
                split_X,
                split_y,
                classes,
                augmentation_factor=augmentation_factor,
                noise_level=noise_level,
                scale_range=scale_range,
            )
            split_sample_ids = _build_augmented_sample_ids(split_sample_ids, augmentation_factor)

        split_data[split] = {
            "X": split_X,
            "y": split_y,
            "sample_ids": split_sample_ids,
            "rows": split_rows,
        }

    return split_data


__all__ = ["augment_data", "augment_train_split_only"]
