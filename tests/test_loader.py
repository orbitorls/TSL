"""Tests for data loading functionality."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.data.loader import validate_dataset, print_dataset_quality_report


class TestDatasetValidation:
    def test_validate_dataset_valid(self, mock_dataset):
        """Test validation with valid dataset."""
        X, y, classes = mock_dataset
        results = validate_dataset(X, y, classes)
        
        assert results['n_samples'] == 100
        assert results['n_features'] == 162
        assert results['n_classes'] == 51
        assert results['has_nan'] == False
        assert results['has_inf'] == False
    
    def test_validate_dataset_with_nan(self, mock_dataset):
        """Test validation with NaN values."""
        X, y, classes = mock_dataset
        X[0, 0] = np.nan
        
        results = validate_dataset(X, y, classes)
        assert results['has_nan'] == True
    
    def test_validate_dataset_with_inf(self, mock_dataset):
        """Test validation with infinite values."""
        X, y, classes = mock_dataset
        X[0, 0] = np.inf
        
        results = validate_dataset(X, y, classes)
        assert results['has_inf'] == True
    
    def test_validate_dataset_shape_mismatch(self):
        """Test validation with shape mismatch."""
        X = np.random.randn(100, 162).astype(np.float32)
        y = np.random.randint(0, 51, 50)  # Wrong shape
        classes = [f"sign_{i}" for i in range(51)]
        
        results = validate_dataset(X, y, classes)
        assert results['valid'] == False
        assert 'shape_mismatch' in results['errors']


class TestDatasetQualityReport:
    def test_print_quality_report_valid(self, mock_dataset, capsys):
        """Test printing quality report for valid dataset."""
        X, y, classes = mock_dataset
        results = validate_dataset(X, y, classes)
        
        print_dataset_quality_report(results)
        captured = capsys.readouterr()
        
        assert "Dataset Quality Report" in captured.out
        assert "Samples: 100" in captured.out
        assert "Features: 162" in captured.out
