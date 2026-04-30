"""Redirect to core.models for backward compatibility.

This module is deprecated. Use src.core instead.
"""

from src.core.models import GRUModel
from src.core.models import get_model

__all__ = ["GRUModel", "get_model"]