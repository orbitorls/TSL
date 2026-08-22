"""Core normalization utilities for TSL-51."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.features import BASIC_FEATURE_DIM, FEATURE_SCHEMA_VERSION

PREPROCESSING_MANIFEST_FILENAME = "preprocessing_manifest.json"
SUPPORTED_NORMALIZATION_TYPE = "zscore"
SUPPORTED_FEATURE_LEVEL = "basic"
SUPPORTED_FEATURE_ORDER = FEATURE_SCHEMA_VERSION


@dataclass
class PreprocessingManifest:
    """Serializable preprocessing contract shared by training and inference."""

    normalization_type: str
    mean: list[float]
    std: list[float]
    target_frames: int
    seq_mode: bool
    dataset: str
    split_strategy: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_level: str = SUPPORTED_FEATURE_LEVEL
    feature_dim: int = BASIC_FEATURE_DIM
    feature_order: str = SUPPORTED_FEATURE_ORDER
    source_metadata: dict[str, Any] = field(default_factory=dict)
    label_map_path: str | None = None
    label_map_hash: str | None = None
    label_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, checkpoint_input_dim: int | None = None) -> None:
        """Validate this phase's supported preprocessing contract."""
        if self.feature_level == "enhanced":
            raise ValueError("feature_level='enhanced' is not supported for preprocessing manifests")
        if self.feature_level != SUPPORTED_FEATURE_LEVEL:
            raise ValueError(f"Unsupported feature_level={self.feature_level!r}; expected 'basic'")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "feature_schema_version mismatch: "
                f"expected {FEATURE_SCHEMA_VERSION!r}, got {self.feature_schema_version!r}"
            )
        if self.feature_dim != BASIC_FEATURE_DIM:
            raise ValueError(f"feature_dim mismatch: expected {BASIC_FEATURE_DIM}, got {self.feature_dim}")
        if self.feature_order != SUPPORTED_FEATURE_ORDER:
            raise ValueError(
                f"feature_order mismatch: expected {SUPPORTED_FEATURE_ORDER!r}, got {self.feature_order!r}"
            )
        if self.normalization_type != SUPPORTED_NORMALIZATION_TYPE:
            raise ValueError(
                "normalization_type mismatch: "
                f"expected {SUPPORTED_NORMALIZATION_TYPE!r}, got {self.normalization_type!r}"
            )
        if len(self.mean) != self.feature_dim:
            raise ValueError(f"mean length {len(self.mean)} does not match feature_dim {self.feature_dim}")
        if len(self.std) != self.feature_dim:
            raise ValueError(f"std length {len(self.std)} does not match feature_dim {self.feature_dim}")
        if checkpoint_input_dim is not None and int(checkpoint_input_dim) != self.feature_dim:
            raise ValueError(
                "checkpoint input_dim mismatch: "
                f"checkpoint has {int(checkpoint_input_dim)}, manifest feature_dim is {self.feature_dim}"
            )
        if int(self.target_frames) <= 0:
            raise ValueError("target_frames must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable manifest dictionary."""
        return {
            "feature_schema_version": self.feature_schema_version,
            "feature_level": self.feature_level,
            "feature_dim": int(self.feature_dim),
            "feature_order": self.feature_order,
            "normalization_type": self.normalization_type,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "target_frames": int(self.target_frames),
            "seq_mode": bool(self.seq_mode),
            "dataset": str(self.dataset),
            "split_strategy": str(self.split_strategy),
            "source_metadata": dict(self.source_metadata),
            "label_map_path": self.label_map_path,
            "label_map_hash": self.label_map_hash,
            "label_metadata": dict(self.label_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingManifest:
        """Create a manifest from a JSON-compatible dictionary."""
        return cls(
            feature_schema_version=str(data.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
            feature_level=str(data.get("feature_level", SUPPORTED_FEATURE_LEVEL)),
            feature_dim=int(data.get("feature_dim", BASIC_FEATURE_DIM)),
            feature_order=str(data.get("feature_order", SUPPORTED_FEATURE_ORDER)),
            normalization_type=str(data["normalization_type"]),
            mean=[float(v) for v in data["mean"]],
            std=[float(v) for v in data["std"]],
            target_frames=int(data["target_frames"]),
            seq_mode=bool(data["seq_mode"]),
            dataset=str(data.get("dataset", "unknown")),
            split_strategy=str(data.get("split_strategy", "unknown")),
            source_metadata=dict(data.get("source_metadata") or {}),
            label_map_path=data.get("label_map_path"),
            label_map_hash=data.get("label_map_hash"),
            label_metadata=dict(data.get("label_metadata") or {}),
        )


def manifest_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the sidecar preprocessing manifest path for a checkpoint."""
    return Path(checkpoint_path).with_name(PREPROCESSING_MANIFEST_FILENAME)


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 hash for label maps or other small metadata files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_preprocessing_manifest(
    checkpoint_path: str | Path,
    manifest: PreprocessingManifest,
    *,
    validate: bool = True,
) -> Path:
    """Write ``preprocessing_manifest.json`` beside a checkpoint."""
    if validate:
        manifest.validate()
    output_path = manifest_path_for_checkpoint(checkpoint_path)
    output_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_preprocessing_manifest(path: str | Path, *, validate: bool = True) -> PreprocessingManifest:
    """Load and validate a preprocessing manifest JSON file."""
    manifest = PreprocessingManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    if validate:
        manifest.validate()
    return manifest


def load_preprocessing_manifest_for_checkpoint(
    checkpoint_path: str | Path,
    checkpoint: dict[str, Any],
) -> PreprocessingManifest | None:
    """Load the sidecar manifest for a checkpoint when present."""
    manifest_path = manifest_path_for_checkpoint(checkpoint_path)
    if not manifest_path.exists():
        return None
    manifest = load_preprocessing_manifest(manifest_path)
    checkpoint_input_dim = checkpoint.get("input_dim")
    manifest.validate(checkpoint_input_dim=checkpoint_input_dim)
    return manifest


def resolve_checkpoint_preprocessing(
    checkpoint_path: str | Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Resolve preprocessing metadata, preferring the sidecar manifest if present."""
    manifest = load_preprocessing_manifest_for_checkpoint(checkpoint_path, checkpoint)
    if manifest is not None:
        return {
            "mean": np.asarray(manifest.mean, dtype=np.float32),
            "std": np.asarray(manifest.std, dtype=np.float32),
            "input_dim": int(manifest.feature_dim),
            "feature_level": manifest.feature_level,
            "seq_mode": bool(manifest.seq_mode),
            "target_frames": int(manifest.target_frames),
            "manifest": manifest,
        }

    mean = checkpoint.get("normalization_mean", checkpoint.get("mean"))
    std = checkpoint.get("normalization_std", checkpoint.get("std"))
    if mean is None or std is None:
        raise KeyError("Checkpoint missing normalization stats (mean/std)")
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    config = checkpoint.get("config", {})
    return {
        "mean": mean_arr,
        "std": std_arr,
        "input_dim": int(checkpoint.get("input_dim", int(mean_arr.shape[0]))),
        "feature_level": str(config.get("feature_level", "basic")),
        "seq_mode": bool(checkpoint.get("seq_mode", False)),
        "target_frames": int(checkpoint.get("target_frames", 30)),
        "manifest": None,
    }



class Normalizer:
    """Z-score normalizer with statistics tracking."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, features: np.ndarray) -> Normalizer:
        """Fit normalizer to training data."""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)  # Avoid div by zero
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted statistics."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        return (features - self.mean) / self.std  # type: ignore[no-any-return]

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        return normalized * self.std + self.mean  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for checkpoint."""
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Normalizer:
        """Deserialize from dict."""
        mean = np.array(data["mean"]) if data.get("mean") is not None else None
        std = np.array(data["std"]) if data.get("std") is not None else None
        return cls(mean=mean, std=std)
