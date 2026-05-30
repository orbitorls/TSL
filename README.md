# TSL-51: Thai Sign Language Recognition

A PyTorch-based system for recognizing Thai Sign Language (TSL) using MediaPipe landmarks. Designed for training on Google Colab with a clean, modular architecture.

## Features

- **Multiple Model Architectures**: GRU, MLP, MOPGRU, HybridGRUTransformer, CTC
- **Colab-Optimized**: Ready-to-use notebooks for cloud training
- **K-Fold Cross Validation**: Robust model evaluation with per-fold normalization
- **Data Augmentation**: Noise injection, scaling, and horizontal flipping
- **Video Inference**: Batch inference on video files
- **Multiple Datasets**: User sign, expert, combined, and full expert (~45k samples)

## Quick Start (Google Colab)

### 1. Clone Repository to Google Drive

```bash
# In Google Drive
git clone https://github.com/yourusername/TSL.git
cd TSL
```

### 2. Open Colab Notebooks

Navigate to the `colab/` directory and open the notebooks in order:

1. **01_setup.ipynb** - Environment setup and dependency installation
2. **02_train.ipynb** - Train your TSL-51 model
3. **03_evaluate.ipynb** - Evaluate model performance
4. **04_inference.ipynb** - Run inference on videos

### 3. Run Setup Notebook

Execute all cells in `01_setup.ipynb` to:
- Mount Google Drive
- Install dependencies (PyTorch, MediaPipe, etc.)
- Verify GPU availability
- Test module imports
- Create output directories

### 4. Train Model

Configure training parameters in `02_train.ipynb`:
```python
CONFIG = {
    'dataset': 'tsl51_user_sign',  # Options: tsl51_user_sign, tsl51_expert, tsl51_combined, tsl51_expert_full
    'model_type': 'gru',           # Options: gru, mlp
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50,
    'k_folds': 5,
    'augment_factor': 2
}
```

Run the notebook to train your model with K-Fold cross-validation. The best model will be saved to `models/`.

### 5. Evaluate and Inference

Use `03_evaluate.ipynb` to test your model on a held-out test set, and `04_inference.ipynb` to run inference on new videos.

## Datasets

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| `tsl51_user_sign` | 547 | 51 | HuggingFace `Namonpas/thai-sign-language-tsl51` |
| `tsl51_expert` | 1,155 | 51 | Expert recordings (original) |
| `tsl51_expert_full` | ~45,000 | 51 | Expert with augmented data |
| `tsl51_combined` | 1,702 | 51 | user_sign + expert original |
| `local` | Custom | Custom | CSV or NumPy format |

**Feature Format**: Configurable feature levels
- `basic` (162): Left hand 63 + Right hand 63 + Pose 36
- `finger` (252): Basic 162 + Finger joints 90
- `full` (1596): All landmarks including face mesh
- `face` (1434): Face mesh only

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
├── colab/                  # Google Colab notebooks
│   ├── 01_setup.ipynb     # Environment setup
│   ├── 02_train.ipynb     # Training pipeline
│   ├── 03_evaluate.ipynb  # Model evaluation
│   └── 04_inference.ipynb # Inference demo
├── src/                    # Core training modules
│   ├── core/              # Models & features
│   ├── data/              # Data loading
│   ├── train/             # Training logic
│   ├── inference/         # Inference modules
│   └── utils/             # Utilities
├── tsl_web/                # Flask real-time web translator
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── models/                 # Local model outputs (ignored except placeholders)
├── results/                # Local training results (ignored except placeholders)
├── artifacts/              # Local run artifacts and large outputs (ignored)
├── legacy/                 # Archived content
│   ├── root_scripts/      # Legacy root-level scripts
│   ├── papers/            # LaTeX papers
│   ├── tools/             # TUI app
│   ├── website/           # Next.js website
│   └── results/           # Old results
├── README.md
└── requirements.txt
```

Root-level Python files such as `predict_video.py`, `camera_translate.py`, and
`train_tsl51_v3.py` are compatibility shims. New implementation work should go
under `src/` or `tsl_web/`.

## Training Configuration

Training is configured directly in the Colab notebooks. See `02_train.ipynb` for the configuration dictionary:

```python
CONFIG = {
    'dataset': 'tsl51_user_sign',  # Options: tsl51_user_sign, tsl51_expert, tsl51_combined, tsl51_expert_full
    'model_type': 'gru',           # Options: gru, mlp
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50,
    'k_folds': 5,
    'augment_factor': 2,
    'early_stopping_patience': 10
}
```

## Output Files

- `models/tsl51_{model_type}_{timestamp}.pt` - Saved model checkpoint
- `results/cv_{timestamp}.json` - Cross-validation results with metrics
- `results/evaluation_{timestamp}.json` - Test set evaluation results
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/per_class_accuracy.png` - Per-class accuracy plot

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
from src.core.models import GRUModel
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
python -m pytest tests/
```

### Code Quality

```bash
# Linting
python -m ruff check src/ --no-fix

# Formatting
python -m black src/
```

## Known Issues

- **Initial Download**: First dataset download from HuggingFace takes time (cached in `.cache/tsl51/`)
- **Colab GPU**: Free Colab tier has limited GPU runtime; consider upgrading for longer training sessions

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
