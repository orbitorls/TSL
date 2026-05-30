"""Training helpers for preprocessing manifest sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.features import BASIC_FEATURE_DIM, FEATURE_SCHEMA_VERSION
from src.core.normalizer import (
    PreprocessingManifest,
    save_preprocessing_manifest,
    sha256_file,
)


def build_preprocessing_manifest(
    *,
    mean: np.ndarray | list[float],
    std: np.ndarray | list[float],
    dataset: str,
    split_strategy: str,
    target_frames: int,
    seq_mode: bool,
    label_map_path: str | Path | None = None,
    label_metadata: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> PreprocessingManifest:
    """Build the Task 4 preprocessing manifest for canonical training outputs."""
    label_hash = sha256_file(label_map_path) if label_map_path is not None else None
    manifest = PreprocessingManifest(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_level="basic",
        feature_dim=BASIC_FEATURE_DIM,
        feature_order=FEATURE_SCHEMA_VERSION,
        normalization_type="zscore",
        mean=np.asarray(mean, dtype=np.float64).tolist(),
        std=np.asarray(std, dtype=np.float64).tolist(),
        target_frames=int(target_frames),
        seq_mode=bool(seq_mode),
        dataset=str(dataset),
        split_strategy=str(split_strategy),
        label_map_path=str(label_map_path) if label_map_path is not None else None,
        label_map_hash=label_hash,
        label_metadata=label_metadata or {},
        source_metadata=source_metadata or {},
    )
    manifest.validate()
    return manifest


def save_training_preprocessing_manifest(
    checkpoint_path: str | Path,
    *,
    mean: np.ndarray | list[float],
    std: np.ndarray | list[float],
    dataset: str,
    split_strategy: str,
    target_frames: int,
    seq_mode: bool,
    label_map_path: str | Path | None = None,
    label_metadata: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write preprocessing_manifest.json beside a training checkpoint."""
    manifest = build_preprocessing_manifest(
        mean=mean,
        std=std,
        dataset=dataset,
        split_strategy=split_strategy,
        target_frames=target_frames,
        seq_mode=seq_mode,
        label_map_path=label_map_path,
        label_metadata=label_metadata,
        source_metadata=source_metadata,
    )
    return save_preprocessing_manifest(checkpoint_path, manifest)
