"""Tests for inference functionality."""

import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.train.models import GRUModel


class TestModelLoading:
    def test_load_model_structure(self, sample_batch, sample_classes):
        """Test model loading structure."""
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        model.eval()
        
        # Create mock checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'classes': sample_classes,
            'mean': np.zeros(162),
            'std': np.ones(162),
        }
        
        # Test loading
        loaded_model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        loaded_model.load_state_dict(checkpoint['state_dict'])
        loaded_model.eval()
        
        # Test forward pass
        x = torch.FloatTensor(sample_batch)
        with torch.no_grad():
            logits = loaded_model(x)
        assert logits.shape == (4, 51)
    
    def test_prediction_shapes(self, sample_features, sample_classes, device):
        """Test prediction output shapes."""
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        model = model.to(device)
        model.eval()
        
        # Normalize features
        mean = np.zeros(162)
        std = np.ones(162)
        x = (sample_features - mean) / std
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(x)
            probs = logits.softmax(dim=1)
            pred = probs.argmax(dim=1).item()
        
        assert 0 <= pred < 51
        assert probs.shape == (1, 51)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


class TestNormalization:
    def test_normalization_zero_mean_unit_std(self, sample_features):
        """Test normalization with zero mean and unit std."""
        mean = np.zeros(162)
        std = np.ones(162)
        normalized = (sample_features - mean) / std
        
        assert np.allclose(normalized, sample_features)
    
    def test_normalization_custom_mean_std(self, sample_features):
        """Test normalization with custom mean and std."""
        mean = np.ones(162)
        std = 2.0 * np.ones(162)
        normalized = (sample_features - mean) / std
        
        expected = (sample_features - 1.0) / 2.0
        assert np.allclose(normalized, expected)
