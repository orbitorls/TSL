"""Core model definitions for TSL-51.

Single source of truth for model architectures.
"""

import torch
import torch.nn as nn
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseModel(Protocol):
    """Protocol for all TSL-51 models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def num_params(self) -> int:
        ...


class GRUModel(nn.Module):
    """Bidirectional GRU for temporal sequence modeling.

    Architecture:
    - Bidirectional GRU (3 layers, 256 hidden by default)
    - LayerNorm on output
    - Dropout for regularization
    - Linear classifier
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim) if single frame
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLPModel(nn.Module):
    """Multi-Layer Perceptron with LayerNorm and GELU activation.

    Architecture:
    - Linear input -> hidden
    - LayerNorm -> GELU -> Dropout (repeated for each layer)
    - Final Linear to num_classes
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


MODEL_REGISTRY = {
    'gru': GRUModel,
    'mlp': MLPModel,
    'mopgru': GRUModel,  # Alias for backward compatibility
}


def get_model(name: str):
    """Get model class by name."""
    return MODEL_REGISTRY.get(name, GRUModel)
