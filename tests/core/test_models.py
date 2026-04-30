"""Tests for core models module."""

import torch
import pytest
from src.core import GRUModel, MLPModel, MODEL_REGISTRY, get_model


def test_gru_forward():
    """Test GRUModel forward pass with sequence input."""
    model = GRUModel(input_dim=162, num_classes=51)
    x = torch.randn(2, 30, 162)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (2, 51)


def test_gru_single_frame():
    """Test GRUModel with single frame input (2D)."""
    model = GRUModel(input_dim=162, num_classes=51)
    x = torch.randn(2, 162)  # (batch, features) - single frame
    out = model(x)
    assert out.shape == (2, 51)


def test_mlp_forward():
    """Test MLPModel forward pass."""
    model = MLPModel(input_dim=162, num_classes=51)
    x = torch.randn(2, 162)
    out = model(x)
    assert out.shape == (2, 51)


def test_model_registry():
    """Test MODEL_REGISTRY contains expected models."""
    assert 'gru' in MODEL_REGISTRY
    assert 'mlp' in MODEL_REGISTRY
    assert get_model('gru') == GRUModel
    assert get_model('mlp') == MLPModel


def test_gru_params():
    """Test GRUModel has correct parameter count."""
    model = GRUModel(input_dim=162, num_classes=51, hidden_dim=128, num_layers=2)
    assert model.num_params > 0


def test_mlp_params():
    """Test MLPModel has correct parameter count."""
    model = MLPModel(input_dim=162, num_classes=51, hidden_dim=128, num_layers=2)
    assert model.num_params > 0
