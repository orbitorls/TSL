# TSL-51 Full Modernization Design

**Date:** 2026-04-30
**Status:** Approved
**Approach:** Layered Architecture with Moderate Modernization

## Overview

Refactor TSL-51 codebase to use layered architecture while maintaining backward compatibility with existing trained models.

## Goals

1. Single source of truth for features and models
2. Clean separation of concerns
3. Backward compatible with v1, v2 checkpoints
4. Testable structure
5. Extensible for future

## Architecture

```
src/
├── core/              # Shared, stable APIs
│   ├── __init__.py
│   ├── features.py    # Feature extraction (single source)
│   ├── models.py     # Model registry & definitions
│   └── normalizer.py # Normalization utilities
├── train/             # Training pipeline
│   ├── __init__.py
│   ├── cli.py        # CLI entry points
│   ├── trainer.py   # Trainer class
│   ├── config.py   # Configuration classes
│   └── evaluator.py # Metrics
├── inference/          # Inference
│   ├── __init__.py
│   └── predictor.py # TSLPredictor
└── utils/             # Utilities
    └── dataset_utils.py
```

## Key Components

### 1. Core Features

```python
# src/core/features.py
FEATURE_LEVELS = {
    'basic': 162,     # Hand + Pose
    'enhanced': 249,  # + geometric
    'finger': 258,    # + finger joints
    'full': 1596,    # + face
}

def extract_features(lm_df, feature_level='basic') -> np.ndarray
def extract_sequence(lm_df, feature_level, target_frames=30) -> np.ndarray
def normalize(features, mean, std) -> np.ndarray
def denormalize(features, mean, std) -> np.ndarray
```

### 2. Core Models

```python
# src/core/models.py
class BaseModel(Protocol):
    def forward(x: Tensor) -> Tensor
    @property
    def num_params(self) -> int

class GRUModel(nn.Module):
    """Bidirectional GRU - single implementation"""
    ...

class MLPModel(nn.Module):
    """MLP with LayerNorm - single implementation"""
    ...

MODEL_REGISTRY = {
    'gru': GRUModel,
    'mlp': MLPModel,
    'mopgru': GRUModel,  # Alias
    'hybrid': HybridGRUTransformer,
}
```

### 3. Trainer

```python
# src/train/trainer.py
class Trainer:
    def __init__(config: TrainingConfig)
    def train(train_data, val_data) -> TrainingResult
    def save_checkpoint(path)  # v3 format
    def load_checkpoint(path)  # v1-v3 support
```

### 4. Predictor

```python
# src/inference/predictor.py
class Predictor:
    def __init__(model_path, device=None)
    def predict(landmarks) -> (label, confidence)
    def predict_top_k(landmarks, k=5) -> list[(label, confidence)]
```

## Backward Compatibility

### Checkpoint Format

v3 format (current):
```python
{
    'model': 'gru',
    'state_dict': {...},
    'classes': [...],
    'normalization_mean': [...],
    'normalization_std': [...],
    'accuracy': 0.95,
    'config': {
        'model': 'gru',
        'hidden_dim': 256,
        'num_layers': 3,
        'feature_level': 'basic',
    }
}
```

v1/v2 legacy: Support via `load_checkpoint()` automatic detection

### Feature Extraction

- Support both old column format (`p_x0`) and new (`l_shoulder_x`)
- Auto-detect format from DataFrame columns

### Model Registry

```python
# Auto-discover from MODEL_REGISTRY
def get_model_class(name: str):
    return MODEL_REGISTRY.get(name, GRUModel)
```

## Testing Strategy

```
tests/
├── core/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_normalizer.py
├── train/
│   └── test_trainer.py
���── inference/
    └── test_predictor.py
```

## Migration Path

1. Create `src/core/` - new base layer
2. Update `src/train/` - use core
3. Update `src/inference/` - use core
4. Add compatibility shims for old imports
5. Add tests
6. Update CLI entry points

## Success Criteria

- [ ] All current training commands work
- [ ] All current inference commands work
- [ ] Backward compatible with existing .pt files
- [ ] Test coverage > 70%
- [ ] No duplicate code between modules

## Files to Create/Modify

### New Files
- `src/core/__init__.py`
- `src/core/features.py` (merged from preprocessing.py + feature_extraction.py)
- `src/core/models.py` (merged from models/*.py + train/models.py)
- `src/core/normalizer.py`
- `tests/`

### Modify
- `src/train/trainer.py` - use core
- `src/inference/predictor.py` - use core
- `pyproject.toml` - update entry points

### Deprecate (after migration)
- `src/data/preprocessing.py` - redirect to core
- `src/data/feature_extraction.py` - redirect to core
- `src/models/gru.py` - redirect to core
- `src/models/mlp.py` - redirect to core
- `src/train/models.py` - redirect to core