"""Core model definitions for TSL-51.

Single source of truth for model architectures.
Supports attention pooling for better temporal modeling.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseModel(Protocol):
    """Protocol for all TSL-51 models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def num_params(self) -> int:
        ...


class AttentionPooling(nn.Module):
    """Self-attention pooling for sequence aggregation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)  # (batch, seq_len, 1)
        pooled = (x * weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class GRUModel(nn.Module):
    """Bidirectional GRU with attention pooling.

    Architecture:
    - Bidirectional GRU (3 layers, 256 hidden by default)
    - Optional attention pooling over sequence
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
        use_attention_pooling: bool = True
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
        self.use_attention = use_attention_pooling
        if use_attention_pooling:
            self.attention = AttentionPooling(hidden_dim * 2)
        else:
            # Create a no-op attention for consistent attribute access
            self.attention = nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim) if single frame
        out, _ = self.gru(x)  # (batch, seq, hidden*2)

        if self.use_attention:
            out = self.attention(out)  # (batch, hidden*2)
        else:
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


class CNN1DModel(nn.Module):
    """1D-CNN Temporal Convolutional Network.

    Architecture:
    - Linear input projection
    - Stacked 1D convolutions with batch norm, ReLU, dropout
    - Adaptive average pooling over sequence
    - Linear classifier
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        kernel_size: int = 3
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        conv_layers = []
        current_dim = hidden_dim
        for i in range(num_layers):
            in_ch = current_dim
            out_ch = min(hidden_dim * (2 if i < num_layers - 1 else 1), 512)
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = out_ch
        self.conv_layers = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim) -> (batch, seq=1, input_dim)
        x = self.input_proj(x)  # (batch, seq, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq)
        x = self.conv_layers(x)  # (batch, hidden_dim, seq)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return self.fc(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(2.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalAttentionModel(nn.Module):
    """Transformer encoder with multi-head attention for temporal modeling.

    Architecture:
    - Linear input projection
    - Sinusoidal positional encoding
    - Stacked transformer encoder layers
    - Mean pooling over sequence
    - LayerNorm + Linear classifier
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        nhead: int = 4
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, seq, hidden_dim)
        x = x.mean(dim=1)  # (batch, hidden_dim)
        x = self.norm(x)
        return self.fc(self.dropout(x))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ResidualMLPBlock(nn.Module):
    """MLP block with residual (skip) connection using GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class ResidualMLPModel(nn.Module):
    """MLP with residual (skip) connections using Pre-Norm.

    Architecture:
    - Linear input projection
    - Stacked ResidualMLPBlocks
    - LayerNorm
    - Linear classifier
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch, seq, dim) -> (batch, dim)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.fc(self.dropout(x))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LightweightModel(nn.Module):
    """MobileNet-style lightweight model for edge deployment (< 100K params).

    Architecture:
    - Linear input projection 162 -> 64
    - Depthwise separable convolutions (DW + PW)
    - Global average pooling
    - LayerNorm + Linear classifier
    """
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        ch1 = base_channels      # 32
        ch2 = base_channels * 2  # 64
        ch3 = base_channels * 4  # 128

        # Input projection: 162 -> 64
        self.input_proj = nn.Linear(input_dim, ch1)  # 162 -> 32

        # Block 1: depthwise separable conv 32 -> 64
        self.conv1_dw = nn.Conv1d(ch1, ch1, 3, padding=1, groups=ch1)
        self.conv1_pw = nn.Conv1d(ch1, ch2, 1)
        self.bn1 = nn.BatchNorm1d(ch2)

        # Block 2: depthwise separable conv 64 -> 128
        self.conv2_dw = nn.Conv1d(ch2, ch2, 3, padding=1, groups=ch2)
        self.conv2_pw = nn.Conv1d(ch2, ch3, 1)
        self.bn2 = nn.BatchNorm1d(ch3)

        # Block 3: depthwise separable conv 128 -> 128
        self.conv3_dw = nn.Conv1d(ch3, ch3, 3, padding=1, groups=ch3)
        self.conv3_pw = nn.Conv1d(ch3, ch3, 1)
        self.bn3 = nn.BatchNorm1d(ch3)

        self.norm = nn.LayerNorm(ch3)
        self.fc = nn.Linear(ch3, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch, seq, dim) -> (batch, dim)
        # Project and reshape for Conv1d: (batch, 32) -> (batch, 32, 1)
        x = self.input_proj(x).unsqueeze(-1)
        # Block 1: 32 -> 64
        x = torch.relu(self.bn1(self.conv1_pw(self.conv1_dw(x))))
        # Block 2: 64 -> 128
        x = torch.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        # Block 3: 128 -> 128
        x = torch.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        # Global average pool: (batch, 128, 1) -> (batch, 128)
        x = x.mean(dim=-1)
        return self.fc(self.dropout(self.norm(x)))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


MODEL_REGISTRY = {
    'gru': GRUModel,
    'mlp': MLPModel,
    'mopgru': GRUModel,  # Alias for backward compatibility
    'cnn1d': CNN1DModel,
    'temporal_attention': TemporalAttentionModel,
    'resmlp': ResidualMLPModel,
    'lightweight': LightweightModel,
}


def get_model(name: str):
    """Get model class by name."""
    return MODEL_REGISTRY.get(name, GRUModel)
