"""Tests for training functionality."""

import numpy as np
import torch
import pytest
import torch.nn as nn

from src.train.models import GRUModel
from src.train.trainer import Trainer
from src.train.config import TrainingConfig


class TestTrainer:
    def test_trainer_initialization(self, mock_dataset, device):
        """Test trainer initialization."""
        X, y, classes = mock_dataset
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        config = TrainingConfig(epochs=2, batch_size=16)
        
        trainer = Trainer(model, X, y, classes, config, device=device)
        assert trainer.model is not None
        assert trainer.config.epochs == 2
    
    def test_trainer_forward_pass(self, mock_dataset, device):
        """Test trainer forward pass."""
        X, y, classes = mock_dataset
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        config = TrainingConfig(epochs=1, batch_size=16)
        
        trainer = Trainer(model, X, y, classes, config, device=device)
        
        # Test single forward pass
        batch_X = torch.FloatTensor(X[:16]).to(device)
        logits = trainer.model(batch_X)
        assert logits.shape == (16, 51)
    
    def test_trainer_backward_pass(self, mock_dataset, device):
        """Test trainer backward pass."""
        X, y, classes = mock_dataset
        model = GRUModel(input_dim=162, num_classes=51, hidden_dim=64, num_layers=2)
        config = TrainingConfig(epochs=1, batch_size=16)
        
        trainer = Trainer(model, X, y, classes, config, device=device)
        
        batch_X = torch.FloatTensor(X[:16]).to(device)
        batch_y = torch.LongTensor(y[:16]).to(device)
        logits = trainer.model(batch_X)
        loss = nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTrainingConfig:
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.model == "gru"
        assert config.hidden_dim == 256
        assert config.num_layers == 3
        assert config.learning_rate == 1e-3
    
    def test_preset_quick(self):
        """Test quick preset configuration."""
        from src.train.config import PresetConfig
        config = PresetConfig.quick()
        assert config.epochs == 5
        assert config.hidden_dim == 128
        assert config.n_folds == 2
    
    def test_preset_default(self):
        """Test default preset configuration."""
        from src.train.config import PresetConfig
        config = PresetConfig.default()
        assert config.epochs == 50
        assert config.hidden_dim == 256
        assert config.n_folds == 5
