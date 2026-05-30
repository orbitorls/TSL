# AGENTS.md - Thai Sign Language Recognition Project

## Project Overview

Thai Sign Language (TSL-51) recognition system using PyTorch. Designed for training on Google Colab with a clean, modular architecture. Supports training on isolated signs and inference for real-time translation.

## Architecture

### Colab-Focused Structure
The project has been restructured for optimal Google Colab workflow:

- **`colab/`** - Google Colab notebooks
  - `01_setup.ipynb` - Environment setup and dependency installation
  - `02_train.ipynb` - Training pipeline with K-Fold CV
  - `03_evaluate.ipynb` - Model evaluation and metrics
  - `04_inference.ipynb` - Video inference demo

- **`src/core/`** - Single source of truth for models and features
  - `models.py` - All model architectures (GRU, MLP, MOPGRU, HybridGRUTransformer, CTC)
  - `features.py` - Feature dimension constants and extraction
  - `normalizer.py` - Data normalization utilities

- **`src/data/`** - Data loading and processing
  - `loader.py` - Dataset loading functions (user_sign, expert, combined, full expert ~45k)
  - `loader_expert.py` - Full expert dataset loader
  - `feature_extraction.py` - Feature extraction utilities
  - `extractor.py` - MediaPipe landmark extractor

- **`src/train/`** - Training modules
  - `config.py` - Training configuration and presets
  - `trainer.py` - Core training logic (Trainer class)
  - `evaluator.py` - Evaluation metrics and reporting
  - `models.py` - Re-exports from src.core.models for backward compatibility
  - `augment.py` - Data augmentation
  - `visualize.py` - Training visualization

- **`src/inference/`** - Inference modules
  - `runner.py` - General inference script
  - `predict_video.py` - Video prediction
  - `camera_translate.py` - Real-time camera translation
  - `translate.py` - JSON translation

- **`src/utils/`** - Shared utility functions
  - `dataset_utils.py` - safe_mean, safe_std, validation helpers
  - `security.py` - File path validation for security

- **`tsl_web/`** - Flask real-time web translator
  - `app.py` - Web app, model loading, `/predict` inference endpoint
  - `static/` - Browser-side camera, hand detection, and UI code
  - `templates/` - HTML shell
  - `models/` - Small web task assets that must be packaged with the app

Root-level Python files are compatibility shims only. Put new implementation in
`src/` or `tsl_web/`.

### Legacy Archive

All legacy content has been moved to `legacy/`:

- **`legacy/root_scripts/`** - Legacy root-level Python scripts (moved during restructuring)
- **`legacy/papers/`** - LaTeX papers and academic writing
- **`legacy/tools/`** - TUI application and tools
- **`legacy/website/`** - Next.js website
- **`legacy/scripts_archive/`** - Historical scripts archive
- **`legacy/root_archive/`** - Additional archived scripts
- **`legacy/models/`** - Old model checkpoints
- **`legacy/results/`** - Old training results

## Key Files

### Colab Notebooks
- **`colab/01_setup.ipynb`** - Environment setup, dependency installation, Google Drive mounting
- **`colab/02_train.ipynb`** - Training pipeline with K-Fold CV and configurable parameters
- **`colab/03_evaluate.ipynb`** - Model evaluation, confusion matrix, per-class accuracy
- **`colab/04_inference.ipynb`** - Video inference with MediaPipe landmark extraction

### Core Modules
- **`src/data/loader.py`** - Modular data loading with validation and quality metrics
- **`src/train/config.py`** - Training configuration with presets (quick, default, full_cv, mlp_fast, large_dataset)
- **`src/train/trainer.py`** - Core training logic with gradient clipping, mixed precision, early stopping
- **`src/core/models.py`** - All model architectures (GRU, MLP, MOPGRU, HybridGRUTransformer, CTC)

### Dependencies

```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm huggingface_hub opencv-python mediapipe pillow
```

See also: `requirements.txt` for the canonical dependency list.

## Common Commands (Colab)

### Training

Training is done through Colab notebooks. Configure parameters in `02_train.ipynb`:

```python
CONFIG = {
    'dataset': 'tsl51_user_sign',
    'model_type': 'gru',
    'hidden_dim': 256,
    'num_layers': 3,
    'epochs': 50,
    'k_folds': 5
}
```

### Evaluation

Run `03_evaluate.ipynb` to:
- Load trained model
- Evaluate on test set
- Generate confusion matrix
- Calculate per-class accuracy

### Inference

Run `04_inference.ipynb` to:
- Upload video file
- Extract MediaPipe landmarks
- Run model inference
- Display prediction timeline

## Architecture Notes

### Dataset
- **Source**: HuggingFace `Namonpas/thai-sign-language-tsl51`
- **User Sign**: 547 videos, 51 classes (single signs)
- **Expert Original**: 1,155 videos, 51 classes
- **Expert Full**: ~45,000 samples including augmented data
- **Combined**: 1,702 samples (user_sign + expert original)
- **Sentence Data**: 252 videos, 76 unique sentences (3-6 signs per sentence)
- **Features**: 162-dimensional (63 left hand + 63 right hand + 36 pose landmarks)

### Dataset Options
- `tsl51_user_sign` - Default, 547 samples
- `tsl51_expert` - 1,155 original expert samples (or ~45k with --include-augmented)
- `tsl51_expert_full` - Full expert dataset with all augmented data (~45k samples)
- `tsl51_combined` - Combined user_sign + expert (1,702 samples)
- `local` - Custom local dataset

### Models
- **GRU**: Bidirectional, default 256 hidden, 3 layers
- **MLP**: Feedforward with LayerNorm and GELU/ReLU
- Input: 162 features → Output: 51 classes

### Training Pipeline
1. Load from HuggingFace or local cache (`.cache/tsl51/`)
2. Optional data augmentation (noise, scale, flip)
3. Train/test split (if specified)
4. K-Fold CV with per-fold normalization (no data leakage)
5. Class weighting for imbalance
6. Early stopping with patience
7. Saves model + visualization report

## Important Conventions

### Colab GPU
Colab notebooks automatically detect and use GPU when available. Free tier has limited GPU runtime; consider upgrading for longer training sessions.

### Output Structure
```
models/
  tsl51_{model_type}_YYYYMMDD_HHMMSS.pt    # Local saved model with timestamp
results/
  cv_YYYYMMDD_HHMMSS.json                  # JSON results
  evaluation_YYYYMMDD_HHMMSS.json          # Evaluation results
  confusion_matrix.png                      # Confusion matrix visualization
  per_class_accuracy.png                    # Per-class accuracy plot
artifacts/
  runs/                                    # Local run folders and large outputs
```

`models/`, `results/`, and `artifacts/` outputs are ignored by Git. Commit only
small placeholders or intentionally packaged runtime assets.

## Gotchas

1. **No emojis in matplotlib** - Windows fonts don't support emojis; use plain text only
2. **Cache first run** - Initial dataset download from HuggingFace takes time
3. **MediaPipe dependency** - Video prediction requires `mediapipe` and `opencv-python`
4. **Sentence vs Sign** - Project supports both isolated sign (current) and sentence-level (planned)
5. **Feature dimension consistency** - `src.core.features.FEATURE_LEVELS` and `src.data.feature_extraction.FEATURE_DIMS` are kept in sync as single source of truth

## Feature Format
Model expects 162 features in order:
1. Left hand: lh_x0-20, lh_y0-20, lh_z0-20 (63)
2. Right hand: rh_x0-20, rh_y0-20, rh_z0-20 (63)
3. Pose: shoulders, elbows, wrists, brows, mouth (36)
