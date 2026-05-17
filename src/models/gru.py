"""Redirect to core.models for backward compatibility.

This module is deprecated. Use src.core instead.
"""

from src.core.models import GRUModel, get_model

__all__ = ["GRUModel", "get_model"]
