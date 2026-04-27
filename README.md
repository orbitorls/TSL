# TSL-51: Thai Sign Language Recognition

A PyTorch-based system for recognizing Thai Sign Language (TSL) using MediaPipe landmarks. Supports training on isolated signs and real-time inference for translation.

## Features

- **Multiple Model Architectures**: GRU, MLP, MOPGRU, HybridGRUTransformer, CTC
- **GPU Training**: CUDA support with automatic mixed precision (AMP)
- **K-Fold Cross Validation**: Robust model evaluation with per-fold normalization
- **Data Augmentation**: Noise injection, scaling, and horizontal flipping
- **Real-time Translation**: Webcam-based sign language translation
- **Video Prediction**: Batch inference on video files
- **Multiple Datasets**: User sign, expert, combined, and full expert (~45k samples)

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/TSL.git
cd TSL
pip install -r requirements.txt
```

**Dependencies**: torch, numpy, pandas, matplotlib, scikit-learn, tqdm, huggingface_hub, opencv-python, mediapipe, pillow, textual

**GPU Requirement**: Training requires CUDA GPU. The script will exit if GPU is unavailable.

### Train a Model

```bash
# Default training (TSL-51 user_sign, 5-fold CV, GRU model)
python train_tsl51_v3.py

# Fast training with data augmentation
python train_tsl51_v3.py --dataset tsl51_user_sign --augment 5 --epochs 50 --batch 128

# Use full expert dataset (~45k samples)
python train_tsl51_v3.py --dataset tsl51_expert_full

# MLP model variant
python train_tsl51_v3.py --model mlp --hidden 256 --layers 3
```

### Inference

```bash
# Real-time camera translation
python camera_translate.py --model models/tsl51_gru_best.pt

# Predict from video file
python predict_video.py --input video.mp4 --model models/tsl51_gru_best.pt

# Inference on pre-extracted features
python inference.py --model models/tsl51_gru_best.pt --input data.npz
```

## Datasets

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| `tsl51_user_sign` | 547 | 51 | HuggingFace `Namonpas/thai-sign-language-tsl51` |
| `tsl51_expert` | 1,155 | 51 | Expert recordings (original) |
| `tsl51_expert_full` | ~45,000 | 51 | Expert with augmented data |
| `tsl51_combined` | 1,702 | 51 | user_sign + expert original |
| `local` | Custom | Custom | CSV or NumPy format |

**Feature Format**: 162-dimensional vector
- Left hand: 63 features (21 points × 3 coordinates)
- Right hand: 63 features (21 points × 3 coordinates)
- Pose: 36 features (12 points × 3 coordinates)

## Model Architectures

| Model | Params | Description | Best For |
|-------|--------|-------------|----------|
| GRU | ~500K | Bidirectional GRU with LayerNorm | General purpose |
| MLP | ~300K | Feedforward with LayerNorm + GELU | Fast inference |
| MOPGRU | ~500K | GRU variant with different pooling | Temporal patterns |
| Hybrid | ~800K | GRU + Transformer encoder | Complex sequences |
| CTC | ~500K | GRU with CTC loss | Sequence labeling |

## Directory Structure

```
TSL/
├── src/
│   ├── data/           # Data loading and preprocessing
│   │   ├── loader.py          # Dataset loading functions
│   │   ├── loader_expert.py   # Full expert dataset loader
│   │   ├── feature_extraction.py  # Feature extraction utilities
│   │   └── extractor.py       # MediaPipe landmark extractor
│   ├── train/          # Training modules
│   │   ├── config.py          # Training configuration
│   │   ├── trainer.py         # Core training logic
│   │   ├── evaluator.py       # Evaluation metrics
│   │   ├── models.py          # Model definitions
│   │   ├── augment.py         # Data augmentation
│   │   └── visualize.py       # Training visualization
│   └── inference/      # Inference modules
│       ├── runner.py          # General inference script
│       ├── predict_video.py   # Video prediction
│       ├── camera_translate.py # Real-time camera translation
│       └── translate.py       # JSON translation
├── train_tsl51_v3.py  # Main training script (CLI entry point)
├── tests/             # Unit and integration tests
├── models/            # Saved model checkpoints (.pt)
├── results/           # Training results (JSON, PNG, TXT)
└── .cache/tsl51/       # Cached dataset files
```

## Training Configuration

### Preset Configurations

```python
# Quick testing (5 epochs, smaller model)
python train_tsl51_v3.py --smoke

# Default (50 epochs, augmentation)
python train_tsl51_v3.py

# Full CV (100 epochs)
python train_tsl51_v3.py --folds 5 --epochs 100

# Large dataset (45k+ samples)
python train_tsl51_v3.py --dataset tsl51_expert_full --hidden 512 --layers 4
```

### Key Hyperparameters

- `--hidden`: Hidden dimension (default: 256)
- `--layers`: Number of layers (default: 3)
- `--dropout`: Dropout rate (default: 0.3)
- `--lr`: Learning rate (default: 0.001)
- `--batch`: Batch size (default: 64)
- `--augment`: Data augmentation factor (e.g., 2 = 2x samples)
- `--patience`: Early stopping patience (default: 10)

## Output Files

- `models/tsl51_gru_YYYYMMDD_HHMMSS.pt` - Saved model with timestamp
- `results/cv_YYYYMMDD_HHMMSS.json` - JSON results with metrics
- `results/results_YYYYMMDD_HHMMSS.png` - Visualization chart
- `results/report_YYYYMMDD_HHMMSS.txt` - Text report

## Model Usage (Inference)

```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('models/tsl51_gru_best.pt')
classes = checkpoint['classes']
mean = np.array(checkpoint['mean'])
std = np.array(checkpoint['std'])

# Load model architecture
from src.train.models import GRUModel
model = GRUModel(input_dim=162, num_classes=51)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Predict
landmarks = np.array(...)  # 162-dim feature vector
x = (landmarks - mean) / std
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    probs = model(x).softmax(dim=1)
    pred = probs.argmax(dim=1).item()
    sign = classes[pred]
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check src/

# Formatting
black src/
```

## Known Issues

- **Windows Fonts**: Matplotlib does not support emojis on Windows; plain text is used instead
- **GPU Required**: Training script exits if CUDA GPU is unavailable
- **Initial Download**: First dataset download from HuggingFace takes time (cached in `.cache/tsl51/`)

## License

[Add your license here]

## Citation

If you use this code, please cite:

```bibtex
@software{tsl51,
  title = {Thai Sign Language Recognition (TSL-51)},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/TSL}
}
```

## Acknowledgments

- Dataset: [Namonpas/thai-sign-language-tsl51](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51)
- MediaPipe: [Google MediaPipe](https://google.github.io/mediapipe/)
