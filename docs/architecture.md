# Architecture Documentation

This document explains the technical architecture of the TSL-51 system.

## Overview

TSL-51 is a PyTorch-based system for Thai Sign Language recognition using MediaPipe landmarks. The system consists of three main components:

1. **Data Pipeline** - Feature extraction and dataset loading
2. **Training Pipeline** - Model training with K-Fold CV
3. **Inference Pipeline** - Real-time and batch prediction

## Feature Format

### 162-Dimensional Feature Vector

The system uses a 162-dimensional feature vector extracted from MediaPipe landmarks:

```
[0-62]    Left hand (21 points × 3 coordinates)
[63-125]  Right hand (21 points × 3 coordinates)
[126-161] Pose (12 points × 3 coordinates)
```

### Hand Landmarks (63 features each)

MediaPipe provides 21 landmarks for each hand:
- 0: Wrist
- 1-4: Thumb (tip, IP, MCP, CMC)
- 5-8: Index finger (tip, DIP, PIP, MCP)
- 9-12: Middle finger (tip, DIP, PIP, MCP)
- 13-16: Ring finger (tip, DIP, PIP, MCP)
- 17-20: Pinky finger (tip, DIP, PIP, MCP)

Each landmark has 3 coordinates (x, y, z).

### Pose Landmarks (36 features)

12 pose landmarks for upper body and face:
- Left/Right shoulder
- Left/Right elbow
- Left/Right wrist
- Left/Right brow (outer, inner)
- Mouth (left, right)

## Model Architectures

### GRU Model (Default)

```
Input (162) → GRU (bidirectional, 3 layers) → LayerNorm → Dropout → FC (51)
```

- Hidden dimension: 256 (configurable)
- Number of layers: 3 (configurable)
- Dropout: 0.3
- Output: 51 class logits

### MLP Model

```
Input (162) → [Linear → LayerNorm → GELU → Dropout] × N → FC (51)
```

- Hidden dimension: 256 (configurable)
- Number of layers: 3 (configurable)
- Dropout: 0.3
- Output: 51 class logits

### MOPGRU Model

Similar to GRU but with different pooling strategy.

### HybridGRUTransformer

```
Input (162) → GRU (bidirectional) → Projection → Transformer → FC (51)
```

- Hidden dimension: 256
- Transformer layers: 2
- Attention heads: 8
- Output: 51 class logits

### CTC Model

```
Input (162) → GRU (bidirectional) → FC (52) → LogSoftmax
```

- Includes blank token for sequence labeling
- Uses CTC loss

## Training Pipeline

### Data Flow

```
HuggingFace Dataset → Feature Extraction → Normalization → Augmentation → K-Fold Split → Training
```

### Per-Fold Normalization

To prevent data leakage, normalization parameters (mean, std) are computed separately for each fold:

```python
for fold in folds:
    train_idx, val_idx = split
    mean, std = compute_normalization(X[train_idx])
    X_train = (X[train_idx] - mean) / std
    X_val = (X[val_idx] - mean) / std
    train_model(X_train, y_train, X_val, y_val)
```

### Data Augmentation

Three augmentation strategies:
1. **Noise**: Add Gaussian noise (std=0.01)
2. **Scale**: Random scale (0.95-1.05)
3. **Flip**: Swap left/right hand landmarks

### Training Configuration

- Optimizer: AdamW
- Learning rate: 0.001 (configurable)
- Scheduler: OneCycleLR
- Loss: CrossEntropyLoss (with class weighting)
- Gradient clipping: Optional
- Mixed precision: AMP (if available)
- Early stopping: Patience=10

## Inference Pipeline

### Model Loading

```python
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['state_dict'])
mean = checkpoint['mean']
std = checkpoint['std']
classes = checkpoint['classes']
```

### Prediction Flow

```
Video → MediaPipe → 162 Features → Normalize → Model → Softmax → Argmax → Sign
```

### Normalization

Inference uses the same normalization parameters as training:

```python
x = (features - mean) / std
```

## Module Structure

### src/data/

- `loader.py` - Dataset loading functions
- `loader_expert.py` - Full expert dataset loader
- `feature_extraction.py` - Feature extraction utilities
- `extractor.py` - MediaPipe landmark extractor
- `preprocessing.py` - Data preprocessing

### src/train/

- `config.py` - Training configuration and presets
- `trainer.py` - Core training logic (Trainer class)
- `evaluator.py` - Evaluation metrics and reporting
- `models.py` - Model definitions
- `augment.py` - Data augmentation
- `visualize.py` - Training visualization
- `compat.py` - Compatibility shims (Windows UTF-8, AMP)
- `loop.py` - Training loop utilities
- `utils.py` - Training utilities

### src/inference/

- `runner.py` - General inference script
- `predict_video.py` - Video prediction
- `camera_translate.py` - Real-time camera translation
- `translate.py` - JSON translation

## Performance Considerations

### GPU Requirements

- Training requires CUDA GPU
- Minimum 4GB VRAM for default model
- 8GB+ VRAM for larger models

### Memory Optimization

- Use gradient accumulation for large batches
- Reduce batch size if OOM
- Use smaller model for faster training

### Inference Speed

- GRU: ~5ms per sample (CPU)
- MLP: ~2ms per sample (CPU)
- Hybrid: ~8ms per sample (CPU)

## Known Limitations

1. **Single Sign Recognition**: Current system recognizes isolated signs, not continuous sentences
2. **GPU Required**: Training script exits if CUDA unavailable
3. **Windows Encoding**: Thai characters require UTF-8 fix on Windows
4. **Dataset Size**: User sign dataset is small (547 samples)
