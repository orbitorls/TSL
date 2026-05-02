# TSL-51 Training System v2 Implementation Plan

**Goal:** Achieve >80% accuracy by loading 45k+ augmented samples and optimizing model/training config

**Architecture:** Data-first incremental approach - load augmented data first, measure improvement, then optimize model if needed.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace datasets, scikit-learn

---

## Task 1: Add load_tsl51_full() Function

**Files:**
- Modify: `src/data/loader.py`

- [ ] **Step 1: Add load_tsl51_full() function stub**

```python
def load_tsl51_full(
    include_augmented: bool = True,
    max_samples: Optional[int] = None,
    force_download: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load full TSL-51 dataset including pre-augmented expert data.
    
    Returns:
        Tuple of (X, y, classes) arrays
    """
```

- [ ] **Step 2: Implement function body**

```python
    cache_file = CACHE_DIR / "full_dataset.npz"
    
    if cache_file.exists() and not force_download:
        print(f"Loading from combined cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['classes']
    
    # Load from sources
    X_user, y_user, _ = load_tsl51_user_sign(force_download=force_download)
    X_expert, y_expert, classes = load_tsl51_expert(
        include_augmented=include_augmented,
        force_download=force_download
    )
    
    # Combine
    X_full = np.vstack([X_user, X_expert])
    y_full = np.concatenate([y_user, y_expert])
    
    # Cache
    np.savez(cache_file, X=X_full, y=y_full, classes=classes)
    
    return X_full, y_full, classes
```

- [ ] **Step 3: Run verification**

```bash
python -c "from src.data.loader import load_tsl51_full; print('OK')"
```

Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/data/loader.py
git commit -m "feat(data): add load_tsl51_full for 45k+ samples"
```

---

## Task 2: Add Stratified Split Function

**Files:**
- Modify: `src/data/loader.py`

- [ ] **Step 1: Add stratified_split()**

```python
def stratified_split(
    X: np.ndarray, y: np.ndarray, classes: np.ndarray,
    val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
):
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

- [ ] **Step 2: Commit**

```bash
git add src/data/loader.py
git commit -m "feat(data): add stratified_split for train/val/test"
```

---

## Task 3: Add Small GRU Model

**Files:**
- Modify: `src/core/models.py`

- [ ] **Step 1: Add SmallGRUModel class**

```python
class SmallGRUModel(nn.Module):
    """Compact GRU: ~600K params (vs 5.4M full)"""
    def __init__(self, input_dim=162, num_classes=51, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.norm(out[:, -1, :])
        return self.fc(self.dropout(out))
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 2: Register model**

```python
MODEL_REGISTRY = {
    'gru': GRUModel,
    'gru_small': SmallGRUModel,
    'mlp': MLPModel,
}
```

- [ ] **Step 3: Commit**

```bash
git add src/core/models.py
git commit -m "feat(model): add SmallGRUModel (~600K params)"
```

---

## Task 4: Update train_tsl51_v3.py CLI

**Files:**
- Modify: `train_tsl51_v3.py`

- [ ] **Step 1: Add tsl51_full to dataset choices**

```python
parser.add_argument("--dataset", ..., 
    choices=[..., "tsl51_full", ...])
```

- [ ] **Step 2: Add load case**

```python
elif args.dataset == "tsl51_full":
    from src.data.loader import load_tsl51_full
    X, y, classes = load_tsl51_full(...)
```

- [ ] **Step 3: Commit**

```bash
git add train_tsl51_v3.py
git commit -m "cli: add tsl51_full dataset"
```

---

## Task 5: Run Baseline Training

- [ ] **Step 1: Test data loading**

```bash
python -c "
from src.data.loader import load_tsl51_full, validate_dataset
X, y, classes = load_tsl51_full(include_augmented=True, force_download=True)
print(f'Samples: {len(X)}, Classes: {len(classes)}')
"
```

- [ ] **Step 2: Run training**

```bash
python train_tsl51_v3.py \
    --dataset tsl51_full \
    --model gru_small \
    --hidden 128 --layers 2 --dropout 0.4 \
    --epochs 100 --batch 128 --patience 20 --folds 5
```

- [ ] **Step 3: Record results**

```
CV Accuracy: XX%
F1 Score: XX%
```

---

## Verification Checklist

- [ ] `load_tsl51_full()` loads ~45k samples
- [ ] `SmallGRUModel` has ~600K parameters
- [ ] Training completes without errors
- [ ] **CV Accuracy >60%** (target: >80%)

---

## Next Steps (if accuracy <80%)

1. **60-80%:** Tune hyperparameters
2. **50-60%:** Try larger model or more augmentation  
3. **<50%:** Debug data pipeline