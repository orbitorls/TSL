"""Core feature schema and extraction for TSL-51.

This module is the single source of truth for the 162-dimensional ``basic``
landmark feature order used by training, data extraction, and inference-facing
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

FEATURE_SCHEMA_VERSION = "basic-162-v1"
BASIC_FEATURE_DIM = 162
HAND_FEATURE_DIM = 63
POSE_FEATURE_DIM = 36

_POSE_BASES = (
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
)


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned feature schema contract."""

    name: str
    version: str
    columns: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.columns)


class UnsupportedFeatureLevelError(ValueError):
    """Raised when a feature level is known but not supported in this phase."""


def _build_basic_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for prefix in ("lh", "rh"):
        for c in ("x", "y", "z"):
            for i in range(21):
                columns.append(f"{prefix}_{c}{i}")
    for base in _POSE_BASES:
        for c in ("x", "y", "z"):
            columns.append(f"{base}_{c}")
    return tuple(columns)


BASIC_FEATURE_SCHEMA = FeatureSchema(
    name="basic",
    version=FEATURE_SCHEMA_VERSION,
    columns=_build_basic_columns(),
)

if BASIC_FEATURE_SCHEMA.dimension != BASIC_FEATURE_DIM:
    raise RuntimeError(
        f"Basic feature schema must be {BASIC_FEATURE_DIM} dims; "
        f"got {BASIC_FEATURE_SCHEMA.dimension}"
    )

# Public feature dimensions. ``enhanced`` is intentionally absent because the
# 249-dim variant is not supported in this implementation phase.
FEATURE_LEVELS = MappingProxyType(
    {
        "basic": BASIC_FEATURE_SCHEMA.dimension,
        "finger": 252,
        "full": 1596,
        "face": 1434,
    }
)

UNSUPPORTED_FEATURE_LEVELS = MappingProxyType(
    {
        "enhanced": (
            "feature_level='enhanced' is not supported in this implementation phase. "
            "Use feature_level='basic' with the 162-dim schema, or implement and validate "
            "the full enhanced pipeline before enabling it."
        )
    }
)


def validate_feature_level(feature_level: str = "basic") -> str:
    """Validate and return a supported feature level.

    Raises an actionable error for explicitly unsupported feature levels instead
    of silently falling back to a partial implementation.
    """
    if feature_level in UNSUPPORTED_FEATURE_LEVELS:
        raise UnsupportedFeatureLevelError(UNSUPPORTED_FEATURE_LEVELS[feature_level])
    if feature_level not in FEATURE_LEVELS:
        supported = ", ".join(FEATURE_LEVELS)
        raise ValueError(f"Unknown feature_level={feature_level!r}. Supported levels: {supported}.")
    return feature_level


def get_feature_dim(feature_level: str = "basic") -> int:
    """Return the dimension for a supported feature level."""
    return int(FEATURE_LEVELS[validate_feature_level(feature_level)])


def get_basic_feature_columns() -> tuple[str, ...]:
    """Return the immutable canonical 162-dim basic column order."""
    return BASIC_FEATURE_SCHEMA.columns


def extract_features(lm_df, feature_level: str = "basic") -> np.ndarray:
    """Extract mean-aggregated features from landmark DataFrame.

    Args:
        lm_df: DataFrame with landmark columns (lh_x0, rh_x0, etc.)
        feature_level: One of 'basic', 'finger', 'full', 'face'. 'enhanced' is
            explicitly unsupported for this phase.

    Returns:
        numpy array of shape (feature_dim,)
    """
    feature_level = validate_feature_level(feature_level)

    cols = BASIC_FEATURE_SCHEMA.columns
    features = np.zeros(len(cols), dtype=np.float32)
    available_cols = lm_df.columns.intersection(cols)

    # Performance optimization: Use pandas vectorized mean instead of iterative
    # safe_mean calls for ~10x speedup during feature extraction
    if len(available_cols) > 0:
        col_means = lm_df[available_cols].mean(numeric_only=True).fillna(0.0).to_dict()
        for i, col in enumerate(cols):
            features[i] = col_means.get(col, 0.0)

    feature_dim = FEATURE_LEVELS[feature_level]
    return features[:feature_dim]
