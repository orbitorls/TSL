# src/models/mlp.py - MLP Model Architecture
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with LayerNorm and GELU activation.
    
    Architecture:
    - Linear input -> hidden
    - LayerNorm -> GELU -> Dropout (repeated for each layer)
    - Final Linear to num_classes
    
    Args:
        input_dim: Feature dimension (default 162 for MediaPipe landmarks)
        num_classes: Number of output classes
        hidden_dim: Hidden layer size (default 256)
        num_layers: Number of hidden layers (default 3)
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
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.net(x)