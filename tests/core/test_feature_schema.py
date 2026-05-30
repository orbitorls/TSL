"""Contract tests for the canonical 162-dim basic feature schema."""

import numpy as np
import pandas as pd
import pytest

from src.core.features import (
    BASIC_FEATURE_DIM,
    BASIC_FEATURE_SCHEMA,
    FEATURE_LEVELS,
    FEATURE_SCHEMA_VERSION,
    UnsupportedFeatureLevelError,
    extract_features,
)
from src.data.feature_extraction import (
    FEATURE_DIMS,
    FeatureExtractor,
    extract_features_from_landmark_df,
    extract_sequence_from_landmark_df,
)
from src.train.augment import augment_data


def test_basic_schema_is_versioned_and_ordered() -> None:
    assert FEATURE_SCHEMA_VERSION
    assert BASIC_FEATURE_SCHEMA.name == "basic"
    assert BASIC_FEATURE_SCHEMA.version == FEATURE_SCHEMA_VERSION
    assert BASIC_FEATURE_SCHEMA.dimension == BASIC_FEATURE_DIM == 162
    assert len(BASIC_FEATURE_SCHEMA.columns) == 162

    assert BASIC_FEATURE_SCHEMA.columns[0:21] == tuple(f"lh_x{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[21:42] == tuple(f"lh_y{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[42:63] == tuple(f"lh_z{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[63:84] == tuple(f"rh_x{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[84:105] == tuple(f"rh_y{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[105:126] == tuple(f"rh_z{i}" for i in range(21))
    assert BASIC_FEATURE_SCHEMA.columns[126:129] == (
        "l_shoulder_x",
        "l_shoulder_y",
        "l_shoulder_z",
    )
    assert BASIC_FEATURE_SCHEMA.columns[-3:] == (
        "mouth_left_x",
        "mouth_left_y",
        "mouth_left_z",
    )


def test_data_feature_dims_reuse_core_feature_levels() -> None:
    assert FEATURE_DIMS is FEATURE_LEVELS
    assert FEATURE_DIMS["basic"] == BASIC_FEATURE_SCHEMA.dimension


def test_basic_extractors_use_identical_schema_order() -> None:
    values = {col: [float(idx)] for idx, col in enumerate(BASIC_FEATURE_SCHEMA.columns)}
    df = pd.DataFrame(values)

    core_features = extract_features(df, "basic")
    data_features = extract_features_from_landmark_df(df, "basic")
    sequence = extract_sequence_from_landmark_df(df, "basic", target_frames=1)

    expected = np.arange(BASIC_FEATURE_SCHEMA.dimension, dtype=np.float32)
    np.testing.assert_array_equal(core_features, expected)
    np.testing.assert_array_equal(data_features, expected)
    np.testing.assert_array_equal(sequence[0], expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda df: extract_features(df, "enhanced"),
        lambda df: extract_features_from_landmark_df(df, "enhanced"),
        lambda df: extract_sequence_from_landmark_df(df, "enhanced", target_frames=1),
        lambda df: FeatureExtractor("enhanced"),
    ],
)
def test_enhanced_feature_level_is_explicitly_unsupported(call) -> None:
    with pytest.raises(UnsupportedFeatureLevelError, match="enhanced.*not supported.*basic"):
        call(pd.DataFrame())


def test_augmentation_rejects_non_basic_feature_vectors_before_flip() -> None:
    X = np.zeros((2, 249), dtype=np.float32)
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="basic.*162"):
        augment_data(X, y, ["a", "b"], augmentation_factor=1)


def test_augmentation_flips_basic_schema_hands_only(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "choice", lambda choices: "flip")
    X = np.arange(162, dtype=np.float32).reshape(1, 162)
    y = np.array([0])

    X_aug, y_aug = augment_data(X, y, ["a"], augmentation_factor=1)

    assert y_aug.tolist() == [0, 0]
    flipped = X_aug[1]
    np.testing.assert_array_equal(flipped[:63], X[0, 63:126])
    np.testing.assert_array_equal(flipped[63:126], X[0, :63])
    np.testing.assert_array_equal(flipped[126:], X[0, 126:])
