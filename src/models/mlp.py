"""Redirect to core.models for backward compatibility.

This module is deprecated. Use src.core instead.
"""

from src.core.models import MLPModel as MLP, MLPModel
from src.core.models import MODEL_REGISTRY

__all__ = ["MLP", "MLPModel", "MODEL_REGISTRY"]