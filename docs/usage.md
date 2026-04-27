# Usage Guide

This guide provides detailed instructions for using the TSL-51 system.

## Training Models

### Basic Training

Train a model with default settings:

```bash
python train_tsl51_v3.py
```

This will:
- Load the TSL-51 user_sign dataset (547 samples)
- Train a GRU model with 5-fold cross-validation
- Save the best model to `models/tsl51_gru_YYYYMMDD_HHMMSS.pt`
- Generate results in `results/`

### Custom Training Parameters

```bash
python train_tsl51_v3.py \
    --dataset tsl51_expert_full \
    --model gru \
    --hidden 512 \
    --layers 4 \
    --epochs 100 \
    --batch 256 \
    --lr 0.001 \
    --augment 5 \
    --folds 5
```

### Using Presets

The `src/train/config.py` module provides preset configurations:

```python
from src.train.config import PresetConfig

# Quick testing (5 epochs, smaller model)
config = PresetConfig.quick()

# Default training
config = PresetConfig.default()

# Full CV with more epochs
config = PresetConfig.full_cv()

# Fast MLP training
config = PresetConfig.mlp_fast()

# Large dataset (45k+ samples)
config = PresetConfig.large_dataset()
```

## Working with Datasets

### Using HuggingFace Datasets

```python
from src.data.loader import load_tsl51_user_sign, load_tsl51_expert, load_tsl51_expert_full

# User sign dataset (547 samples)
X, y, classes = load_tsl51_user_sign()

# Expert dataset (1,155 original)
X, y, classes = load_tsl51_expert(include_augmented=False)

# Full expert dataset (~45k samples)
X, y, classes = load_tsl51_expert_full()
```

### Using Custom Datasets

#### CSV Format

Create a CSV file with:
- First column: label (sign name)
- Other columns: 162 feature values (landmarks)

```bash
python train_tsl51_v3.py --dataset local --data-path ./my_data.csv
```

#### NumPy Format

Save pre-processed data as `.npz`:

```python
import numpy as np
np.savez('my_data.npz', X=X, y=y, classes=classes)
```

```bash
python train_tsl51_v3.py --dataset local --data-path ./my_data.npz
```

## Inference

### Load Trained Model

```python
import torch
import numpy as np
from src.train.models import GRUModel

# Load checkpoint
checkpoint = torch.load('models/tsl51_gru_best.pt')
classes = checkpoint['classes']
mean = np.array(checkpoint['mean'])
std = np.array(checkpoint['std'])

# Load model
model = GRUModel(input_dim=162, num_classes=51)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

### Predict Single Sample

```python
def predict(landmarks, model, classes, mean, std):
    """Predict sign from 162-dim landmark vector."""
    x = np.array(landmarks, dtype=np.float32)
    x = (x - mean) / std
    x = torch.tensor(x).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1).item()
    
    return classes[pred], probs[0].numpy()

# Example
landmarks = np.random.randn(162)  # Your 162-dim feature vector
sign, probabilities = predict(landmarks, model, classes, mean, std)
print(f"Predicted: {sign}")
print(f"Probabilities: {probabilities}")
```

### Real-time Camera Translation

```bash
python camera_translate.py --model models/tsl51_gru_best.pt
```

### Video Prediction

```bash
python predict_video.py --input video.mp4 --model models/tsl51_gru_best.pt
```

### Batch Inference on JSON

```bash
python translate.py --model models/tsl51_gru_best.pt --input landmarks.json
```

## Hyperparameter Tuning

### Learning Rate

```bash
# Try different learning rates
python train_tsl51_v3.py --lr 0.0001  # Lower
python train_tsl51_v3.py --lr 0.01    # Higher
```

### Model Architecture

```bash
# Smaller model (faster training)
python train_tsl51_v3.py --hidden 128 --layers 2

# Larger model (better accuracy)
python train_tsl51_v3.py --hidden 512 --layers 4
```

### Data Augmentation

```bash
# No augmentation
python train_tsl51_v3.py --augment 0

# 2x augmentation
python train_tsl51_v3.py --augment 2

# 10x augmentation
python train_tsl51_v3.py --augment 10
```

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
python train_tsl51_v3.py --batch 32
```

### Slow Training

- Use smaller model: `--hidden 128 --layers 2`
- Reduce epochs: `--epochs 20`
- Use fewer folds: `--folds 3`

### Poor Accuracy

- Increase model size: `--hidden 512 --layers 4`
- Add augmentation: `--augment 5`
- Use expert dataset: `--dataset tsl51_expert_full`
- Increase epochs: `--epochs 100`

### CUDA Not Available

The training script requires CUDA GPU. If not available:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If you need CPU-only training, modify the script to remove the GPU check.
