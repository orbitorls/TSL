"""TSL-51 Core Module

Single source of truth for features, models, and normalization.
"""

# Import features first (no dependencies)
from src.core.features import FEATURE_LEVELS, extract_features

# Lazy imports for models and normalizer (to avoid circular dependencies)
def __getattr__(name):
    if name == "GRUModel":
        from src.core.models import GRUModel
        return GRUModel
    if name == "MLPModel":
        from src.core.models import MLPModel
        return MLPModel
    if name == "MODEL_REGISTRY":
        from src.core.models import MODEL_REGISTRY
        return MODEL_REGISTRY
    if name == "get_model":
        from src.core.models import get_model
        return get_model
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
    "MLPModel",
    "Normalizer",
]