"""Shared model definitions for TSL training and inference.

Re-exports from src.core.models for backward compatibility.
All model implementations are maintained in src.core.models as the single source of truth.
"""

from src.core.models import (
    MLP,
    MODEL_REGISTRY,
    MOPGRU,
    CTCModel,
    GRUModel,
    HybridGRUTransformer,
    MLPModel,
    SmallGRUModel,
    get_model,
)

# Backward compatibility alias used by training scripts
MODEL_CLASSES = MODEL_REGISTRY

__all__ = [
    "GRUModel",
    "SmallGRUModel",
    "MLPModel",
    "MLP",
    "MOPGRU",
    "HybridGRUTransformer",
    "CTCModel",
    "MODEL_REGISTRY",
    "MODEL_CLASSES",
    "get_model",
]
