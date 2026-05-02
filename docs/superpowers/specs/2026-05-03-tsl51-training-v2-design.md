# TSL-51 Training System v2 Design

> **Goal:** Achieve >80% accuracy on 51-class Thai Sign Language recognition using ~50k+ samples with proper data pipeline and model training.

**Architecture:** Data-first incremental approach — load augmented data first, measure improvement, then optimize model if needed.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace datasets, MediaPipe, scikit-learn

---

## 1. Problem Analysis

### Current State
- **Model:** GRU (5.4M params) trained on 4,813 samples / 51 classes
- **Current accuracy:** ~4.2% (baseline random = 2%)
- **Issue:** Severe overfitting — too many parameters for too few samples

### Root Cause
| Factor | Current | Target |
|--------|---------|--------|
| Samples per class | ~94 | ~1,000+ |
| Total samples | 5k | 50k+ |
| Model params | 5.4M | 500K-2M |

---

## 2. Data Pipeline

### 2.1 Data Sources

```
HuggingFace Dataset: Namonpas/thai-sign-language-tsl51
├── user_sign/           (~5k samples, original)
├── expert_primary/      (~1k samples, original)
├── expert_scraped/      (~45k samples, PRE-AUGMENTED)
└── augmented/           (additional augmentation available)
```

### 2.2 Data Loading Strategy

**Phase 1: Load Augmented Data**
```python
# src/data/loader.py - New function
def load_tsl51_full(
    include_augmented: bool = True,
    max_samples: int = None,
    force_download: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full TSL-51 dataset including augmented samples.
    Expected: ~45k+ samples when fully loaded.
    """
```

**Phase 2: Optional User Data Collection**
- Use MediaPipe to collect additional videos
- Standardize to 162 features (hand + pose landmarks)
- Append to existing dataset

### 2.3 Train/Val/Test Split

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model learning |
| Val | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation (held-out) |

**Stratified split** to maintain class balance.

---

## 3. Model Architecture

### 3.1 Primary Model: GRU

```python
class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 128,      # Reduced from 256
        num_layers: int = 2,        # Reduced from 3
        dropout: float = 0.4,        # Increased for regularization
    ):
```

| Parameter | Current | New |
|-----------|---------|-----|
| hidden_dim | 256 | 128 |
| num_layers | 3 | 2 |
| dropout | 0.3 | 0.4 |
| Est. params | 5.4M | ~600K |

### 3.2 Baseline Model: MLP

```python
class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 162,
        num_classes: int = 51,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
```

**Rationale:** Simpler model for comparison; may generalize better with more data.

---

## 4. Training Configuration

### 4.1 TrainingConfig Presets

```python
@dataclass
class TrainingConfig:
    # Model
    model: str = "gru"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.4

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    patience: int = 20

    # Data
    dataset: str = "tsl51_full"
    test_split: float = 0.15
    val_split: float = 0.15

    # Augmentation (runtime, not pre-augmented)
    augmentation_factor: int = 0  # Don't double-augment
    noise_level: float = 0.005
    scale_range: tuple = (0.95, 1.05)

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True
```

### 4.2 Regularization

| Technique | Setting |
|-----------|---------|
| Dropout | 0.4 |
| Weight decay | 1e-4 |
| Early stopping | patience=20 |
| Gradient clipping | max_norm=1.0 |
| Label smoothing | 0.1 (optional) |

---

## 5. Data Flow

```
1. Load augmented data (~45k samples)
       ↓
2. Extract 162 features (hand + pose)
       ↓
3. Stratified train/val/test split (70/15/15)
       ↓
4. 5-Fold Cross-Validation
       ├── Fold 1: Train on 4 folds, validate on 1
       ├── Fold 2: Train on 4 folds, validate on 1
       ├── ...
       └── Fold 5: Train on 4 folds, validate on 1
       ↓
5. Select best fold by validation accuracy
       ↓
6. Train final model on all training data
       ↓
7. Evaluate on held-out test set
       ↓
8. Export best model (.pt with metadata)
```

---

## 6. Expected Results

| Metric | Current | Expected |
|--------|---------|----------|
| Accuracy | 4.2% | >80% |
| Precision | 0.2% | >75% |
| Recall | 4.2% | >75% |
| F1 Score | 0.3% | >75% |

### Success Criteria
- [ ] Test accuracy >80%
- [ ] Consistent across all 5 folds (std < 5%)
- [ ] No severe class imbalance issues

### Failure Conditions
- [ ] Test accuracy <50% after 2 weeks
- [ ] Severe overfitting (train acc 99%, val acc <30%)
- [ ] Data loading failures

---

## 7. Implementation Order

### Phase 1: Data Pipeline (Week 1)
1. Update `load_tsl51_full()` to load 45k augmented samples
2. Verify feature extraction for all samples
3. Add stratified split function
4. Run baseline train with current model → measure

### Phase 2: Model Optimization (Week 2)
1. If accuracy <60%: Reduce model size
2. If accuracy 60-80%: Tune hyperparameters
3. If accuracy >80%: Proceed to export

### Phase 3: Finalization
1. Train on full dataset
2. Export optimized model
3. Update inference pipeline

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `src/data/loader.py` | Add `load_tsl51_full()` |
| `src/core/models.py` | Add smaller model variants |
| `src/train/config.py` | Add new presets |
| `src/train/trainer.py` | Support val split |
| `train_tsl51_v2.py` | Update CLI args |
| `tests/test_loader.py` | Add data loading tests |

---

## 9. Testing Plan

### Unit Tests
- [ ] Data loading returns correct shape
- [ ] Feature extraction handles edge cases
- [ ] Stratified split maintains class balance

### Integration Tests
- [ ] Full pipeline runs end-to-end
- [ ] Model trains without errors
- [ ] Export produces valid .pt file

### Performance Tests
- [ ] Training completes within 24 hours (GPU)
- [ ] Inference <100ms per sample
