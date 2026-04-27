import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FEATURE_DIMS, FeatureExtractor, extract_features_from_landmark_df
from src.data.loader import load_local_dataset
from src.train.models import MODEL_CLASSES


class SharedModulesTest(unittest.TestCase):
    def test_feature_dims_exported(self):
        self.assertEqual(FEATURE_DIMS['basic'], 162)
        self.assertEqual(FEATURE_DIMS['face'], 1434)

    def test_feature_extraction_on_empty_dataframe(self):
        df = pd.DataFrame()
        features = extract_features_from_landmark_df(df, feature_level='basic')
        self.assertEqual(features.shape, (FEATURE_DIMS['basic'],))
        self.assertTrue(np.all(features == 0.0))

    def test_feature_extractor_wrapper(self):
        df = pd.DataFrame()
        extractor = FeatureExtractor(feature_level='basic')
        features = extractor.extract_from_dataframe(df)
        self.assertEqual(features.shape, (FEATURE_DIMS['basic'],))

    def test_model_registry_contains_gru(self):
        self.assertIn('gru', MODEL_CLASSES)
        self.assertIn('mlp', MODEL_CLASSES)

    def test_load_local_dataset_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            load_local_dataset(Path('nonexistent_file.npz'), use_cache=False)


if __name__ == '__main__':
    unittest.main()
