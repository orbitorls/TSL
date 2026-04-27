"""Shared pytest fixtures for TSL tests."""

import numpy as np
import torch
import pytest

from src.train.models import GRUModel, MLP


@pytest.fixture
def sample_features():
    """Sample 162-dim feature vector for testing."""
    return np.random.randn(162).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Sample batch of features (batch_size=4, 162 features)."""
    return np.random.randn(4, 162).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Sample labels for 51 classes (batch_size=4)."""
    return np.random.randint(0, 51, 4)


@pytest.fixture
def gru_model():
    """GRU model instance for testing."""
    model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    return model


@pytest.fixture
def mlp_model():
    """MLP model instance for testing."""
    model = MLP(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    return model


@pytest.fixture
def sample_classes():
    """Sample class names."""
    return [f"sign_{i}" for i in range(51)]


@pytest.fixture
def device():
    """Device for testing (CPU by default, CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing data loading."""
    X = np.random.randn(100, 162).astype(np.float32)
    y = np.random.randint(0, 51, 100)
    classes = [f"sign_{i}" for i in range(51)]
    return X, y, classes
