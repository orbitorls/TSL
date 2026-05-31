"""Tests for core features module."""

import numpy as np
import pandas as pd

from src.core import FEATURE_LEVELS, extract_features


def test_feature_levels():
    """Test FEATURE_LEVELS dictionary."""
    assert FEATURE_LEVELS['basic'] == 162
    assert FEATURE_LEVELS['enhanced'] == 249
    assert FEATURE_LEVELS['finger'] == 252
    assert FEATURE_LEVELS['full'] == 1596
    assert FEATURE_LEVELS['face'] == 1434


def test_extract_features_basic():
    """Test basic feature extraction."""
    # Create sample DataFrame with landmark columns
    data = {}
    for i in range(21):
        data[f'lh_x{i}'] = [0.5] * 10
        data[f'lh_y{i}'] = [0.5] * 10
        data[f'lh_z{i}'] = [0.0] * 10
        data[f'rh_x{i}'] = [0.5] * 10
        data[f'rh_y{i}'] = [0.5] * 10
        data[f'rh_z{i}'] = [0.0] * 10

    for base in ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
                'l_wrist', 'r_wrist', 'lbrow_outer', 'lbrow_inner',
                'rbrow_inner', 'rbrow_outer', 'mouth_right', 'mouth_left']:
        for c in ['x', 'y', 'z']:
            data[f'{base}_{c}'] = [0.5] * 10

    df = pd.DataFrame(data)
    features = extract_features(df, 'basic')

    assert features.shape == (162,)
    assert not np.isnan(features).any()


def test_extract_features_missing_columns():
    """Test with missing columns defaults to 0."""
    df = pd.DataFrame({'lh_x0': [0.5] * 10})
    features = extract_features(df, 'basic')

    assert features.shape == (162,)
    # First feature (lh_x0) should be 0.5 (present)
    assert features[0] == 0.5
    # Second feature (lh_y0) should be 0.0 (missing, defaults to 0)
    assert features[1] == 0.0
