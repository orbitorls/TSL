"""TSL-51 Core Module

Single source of truth for features, models, and normalization.
"""

from src.core.features import FEATURE_LEVELS, extract_features
from src.core.models import (
    MODEL_REGISTRY,
    MOPGRU,
    CTCModel,
    GRUModel,
    HybridGRUTransformer,
    MLPModel,
    SmallGRUModel,
    get_model,
)


# Lazy import for Normalizer (to avoid heavy dependencies at import time)
def __getattr__(name):
    if name == "Normalizer":
        from src.core.normalizer import Normalizer

        return Normalizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FEATURE_LEVELS",
    "extract_features",
    "MODEL_REGISTRY",
    "get_model",
    "GRUModel",
    "SmallGRUModel",
    "MLPModel",
    "MOPGRU",
    "HybridGRUTransformer",
    "CTCModel",
    "Normalizer",
]
