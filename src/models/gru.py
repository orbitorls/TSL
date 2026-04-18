# src/models/gru.py - GRU Model Architecture
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """Bidirectional GRU for temporal sequence modeling.
    
    Architecture:
    - Bidirectional GRU (3 layers, 256 hidden by default)
    - LayerNorm on output
    - Dropout for regularization
    - Linear classifier
    
    Args:
        input_dim: Feature dimension (default 162 for MediaPipe landmarks)
        num_classes: Number of output classes
        hidden_dim: GRU hidden size (default 256)
        num_layers: Number of GRU layers (default 3)
        dropout: Dropout rate (default 0.3)
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, input_dim) if single frame
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)