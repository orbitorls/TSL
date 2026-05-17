"""Core model definitions for TSL-51.

Single source of truth for model architectures.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class BaseModel(Protocol):
    """Protocol for all TSL-51 models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def num_params(self) -> int: ...


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
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
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
        return self.fc(out)  # type: ignore[no-any-return]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SmallGRUModel(nn.Module):
    """Compact GRU: ~600K params (vs 5.4M full)"""

    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)  # type: ignore[no-any-return]

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
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # type: ignore[no-any-return]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MOPGRU(nn.Module):
    """Modified GRU (MOPGRU) - alias for backward compatibility."""

    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)  # type: ignore[no-any-return]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridGRUTransformer(nn.Module):
    """Hybrid GRU + Transformer Encoder for complex sequences."""

    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        nhead: int = 8,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        proj_out = self.proj(gru_out)
        trans_out = self.transformer(proj_out)
        out = trans_out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)  # type: ignore[no-any-return]

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CTCModel(nn.Module):
    """CTC-based model for sentence-level recognition."""

    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.blank_idx = 0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, _lengths=None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.fc(out)
        return nn.functional.log_softmax(out, dim=-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Backward compatibility alias
MLP = MLPModel

MODEL_REGISTRY = {
    "gru": GRUModel,
    "gru_small": SmallGRUModel,
    "mlp": MLPModel,
    "mlpmodel": MLPModel,
    "mopgru": MOPGRU,
    "hybrid": HybridGRUTransformer,
    "ctc": CTCModel,
}


def get_model(name: str) -> type[nn.Module]:
    """Get model class by name."""
    return MODEL_REGISTRY.get(name, GRUModel)
