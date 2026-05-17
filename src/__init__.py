# TSL (Thai Sign Language) - Core Package
#
# Organized structure:
# - src.core: Core models, features, and normalizer
# - src.models: Model architecture wrappers
# - src.data: Dataset loading and preprocessing
# - src.train: Training configuration and utilities
# - src.inference: Inference/Prediction
# - src.utils: Shared utility functions

__version__ = "0.1.0"

from src.core.models import (
    GRUModel,
    SmallGRUModel,
    MLPModel,
    MOPGRU,
    HybridGRUTransformer,
    CTCModel,
    MODEL_REGISTRY,
    get_model,
)
from src.core.features import FEATURE_LEVELS, extract_features

__all__ = [
    "__version__",
    "GRUModel",
    "SmallGRUModel",
    "MLPModel",
    "MOPGRU",
    "HybridGRUTransformer",
    "CTCModel",
    "MODEL_REGISTRY",
    "get_model",
    "FEATURE_LEVELS",
    "extract_features",
]
