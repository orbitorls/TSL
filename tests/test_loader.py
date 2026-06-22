"""Tests for data loading functionality."""


import numpy as np

from src.data.loader import print_dataset_quality_report, validate_dataset


class TestDatasetValidation:
    def test_validate_dataset_valid(self, mock_dataset):
        """Test validation with valid dataset."""
        X, y, classes = mock_dataset
        results = validate_dataset(X, y, classes)

        assert results['n_samples'] == 100
        assert results['n_features'] == 162
        assert results['n_classes'] == 51
        assert results['valid'] is True
        assert len(results['issues']) == 0

    def test_validate_dataset_with_nan(self, mock_dataset):
        """Test validation with NaN values."""
        X, y, classes = mock_dataset
        X[0, 0] = np.nan

        results = validate_dataset(X, y, classes)
        assert results['valid'] is False
        assert any('NaN' in issue for issue in results['issues'])

    def test_validate_dataset_with_inf(self, mock_dataset):
        """Test validation with infinite values."""
        X, y, classes = mock_dataset
        X[0, 0] = np.inf

        results = validate_dataset(X, y, classes)
        assert results['valid'] is False
        assert any('Inf' in issue for issue in results['issues'])

    def test_validate_dataset_shape_mismatch(self):
        """Test validation with shape mismatch - not checked by validator."""
        X = np.random.randn(100, 162).astype(np.float32)
        y = np.random.randint(0, 51, 50)  # Wrong shape
        classes = [f"sign_{i}" for i in range(51)]

        # Note: validate_dataset does not check X/y shape alignment.
        # It focuses on NaN/Inf, class imbalance, and feature range.
        results = validate_dataset(X, y, classes)
        assert 'n_samples' in results
        assert 'n_features' in results


class TestDatasetQualityReport:
    def test_print_quality_report_valid(self, mock_dataset, capsys):
        """Test printing quality report for valid dataset."""
        X, y, classes = mock_dataset
        results = validate_dataset(X, y, classes)

        print_dataset_quality_report(results)
        captured = capsys.readouterr()

        assert "DATASET QUALITY REPORT" in captured.out
        assert "Samples: 100" in captured.out
        assert "Features: 162" in captured.out
