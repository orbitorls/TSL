"""Tests for feature extraction output shapes."""

import unittest

import numpy as np
import pandas as pd

from src.data.feature_extraction import FEATURE_DIMS, extract_features_from_landmark_df


class FeatureExtractionTest(unittest.TestCase):
    def test_basic_shape(self):
        df = pd.DataFrame()
        feats = extract_features_from_landmark_df(df, feature_level='basic')
        self.assertEqual(feats.shape, (FEATURE_DIMS['basic'],))

    def test_finger_shape(self):
        df = pd.DataFrame()
        feats = extract_features_from_landmark_df(df, feature_level='finger')
        self.assertEqual(feats.shape, (FEATURE_DIMS['finger'],))

    def test_nonzero_basic(self):
        rows = 10
        data = {f'lh_x{i}': np.random.rand(rows) for i in range(21)}
        data.update({f'lh_y{i}': np.random.rand(rows) for i in range(21)})
        data.update({f'lh_z{i}': np.random.rand(rows) for i in range(21)})
        df = pd.DataFrame(data)
        feats = extract_features_from_landmark_df(df, feature_level='basic')
        self.assertEqual(feats.shape, (162,))
        self.assertFalse(np.all(feats == 0.0))


if __name__ == '__main__':
    unittest.main()
