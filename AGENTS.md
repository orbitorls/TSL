# AGENTS.md - Thai Sign Language Recognition Project

## Project Overview

Thai Sign Language (TSL-51) recognition system using PyTorch. Supports training on isolated signs and inference for real-time translation.

## Architecture

### Modular Structure
The project has been refactored with a clean modular architecture:

- **`src/data/`** - Data loading and processing
  - `loader.py` - Dataset loading functions (user_sign, expert, combined, full expert ~45k)
  - `loader_expert.py` - Full expert dataset loader
  - `feature_extraction.py` - Feature extraction utilities
  - `extractor.py` - MediaPipe landmark extractor

- **`src/train/`** - Training modules
  - `config.py` - Training configuration and presets
  - `trainer.py` - Core training logic (Trainer class)
  - `evaluator.py` - Evaluation metrics and reporting
  - `models.py` - Model definitions (GRU, MLP, MOPGRU, HybridGRUTransformer, CTC)
  - `augment.py` - Data augmentation
  - `visualize.py` - Training visualization

- **`src/inference/`** - Inference modules
  - `runner.py` - General inference script
  - `predict_video.py` - Video prediction
  - `camera_translate.py` - Real-time camera translation
  - `translate.py` - JSON translation

### Legacy Shims (Root Level)

The following root-level files are legacy shims for backward compatibility:

- `inference.py` → `src.inference.runner`
- `predict_video.py` → `src.inference.predict_video`
- `camera_translate.py` → `src.inference.camera_translate`
- `translate.py` → `src.inference.translate`
- `tsl_tasks_extractor.py` → `src.data.extractor`

## Key Files

### Training
- **`train_tsl51_v3.py`** - Main training script with K-Fold CV, data augmentation, GPU support (CLI entry point)
- **`src/data/loader.py`** - Modular data loading with validation and quality metrics
- **`src/train/config.py`** - Training configuration with presets (quick, default, full_cv, mlp_fast, large_dataset)
- **`src/train/trainer.py`** - Core training logic with gradient clipping, mixed precision, early stopping

### Inference & Prediction
- **`inference.py`** - Load trained models and run inference on pre-extracted features
- **`predict_video.py`** - Predict Thai words from video files using MediaPipe landmarks
- **`camera_translate.py`** - Real-time sign language translation from webcam
- **`translate.py`** - Single model translation from JSON landmark files

### Dependencies (Inferred)
```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm huggingface_hub opencv-python mediapipe pillow
```

See also: `requirements.txt` for the canonical dependency list.

## Common Commands

### Training
```bash
# Default training (5-fold CV, GRU model, user_sign dataset)
python train_tsl51_v3.py

# Training with full expert dataset (~45k samples)
python train_tsl51_v3.py --dataset tsl51_expert_full

# Fast training with augmentation
python train_tsl51_v3.py --layers 3 --epochs 50 --batch 128 --augment 5 --test-split 0.2

# MLP model variant
python train_tsl51_v3.py --model mlp --hidden 256 --layers 3

# Using preset configurations (via src/train/config.py)
# Quick: python train_tsl51_v3.py --smoke
# Default: python train_tsl51_v3.py
# Full CV: python train_tsl51_v3.py --folds 5 --epochs 100
```

### Inference
```bash
# Run inference with trained model
python inference.py --model models/tsl51_gru_best.pt --input data.npz

# Predict from video
python predict_video.py --input video.mp4 --model models/tsl51_gru_best.pt

# Real-time camera translation
python camera_translate.py --model models/tsl51_gru_best.pt
```

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

### GPU Requirement
The training script **requires CUDA GPU**. It exits immediately if GPU unavailable:
```python
if not torch.cuda.is_available():
    sys.exit(1)
```

### Windows Encoding
All scripts include Windows UTF-8 fix for Thai characters:
```python
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

### Output Structure
```
models/
  tsl51_gru_YYYYMMDD_HHMMSS.pt    # Saved model with timestamp
results/
  cv_YYYYMMDD_HHMMSS.json         # JSON results
  results_YYYYMMDD_HHMMSS.png     # Visualization chart
  report_YYYYMMDD_HHMMSS.txt      # Text report
.cache/tsl51/
  user_sign_data.npz              # Cached dataset
```

## Gotchas

1. **No requirements.txt** - Install dependencies manually (see Dependencies section above)
2. **No emojis in matplotlib** - Windows fonts don't support emojis; use plain text only
2. **Cache first run** - Initial dataset download from HuggingFace takes time
3. **MediaPipe dependency** - Video prediction requires `mediapipe` and `opencv-python`
4. **Sentence vs Sign** - Project supports both isolated sign (current) and sentence-level (planned)

## Feature Format
Model expects 162 features in order:
1. Left hand: lh_x0-20, lh_y0-20, lh_z0-20 (63)
2. Right hand: rh_x0-20, rh_y0-20, rh_z0-20 (63)
3. Pose: shoulders, elbows, wrists, brows, mouth (36)
