# TSL-51 Training System v2 + Website Integration Design

**Date:** 2026-05-02  
**Status:** Approved  
**Version:** 1.0

## 1. Overview

ปรับปรุงระบบ training ทั้งหมดแบบ modular plugin-based architecture และเชื่อมต่อกับ website เพื่อ model management และ results visualization

## 2. Goals

- Production-ready training system
- Maximum GPU performance (AMP, optimized dataloader)
- Multi-experiment tracking support (W&B, MLflow, TensorBoard)
- Website integration for model management & visualization

## 3. Architecture

### 3.1 Training System (`src/train_v2/`)

```
src/train_v2/
├── __init__.py
├── engine/
│   ├── __init__.py
│   ├── trainer.py       # Core trainer with callback system
│   ├── callbacks.py     # Callback implementations
│   └── state.py         # Training state management
├── config/
│   ├── __init__.py
│   ├── schema.py        # Pydantic config validation
│   └── presets/         # YAML presets
├── models/
│   ├── __init__.py
│   └── registry.py      # Model registry
├── tracking/
│   ├── __init__.py
│   ├── base.py          # Abstract tracker interface
│   ├── wandb_tracker.py
│   ├── mlflow_tracker.py
│   └── tensorboard_tracker.py
├── augmentation/
│   ├── __init__.py
│   ├── spatial.py       # Scale, flip, noise
│   └── temporal.py      # Time-based augmentations
└── utils/
    ├── __init__.py
    └── logging.py
```

### 3.2 Website Integration (`tsl-website/`)

```
tsl-website/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── inference/route.ts      # POST inference
│   │   │   ├── models/route.ts         # CRUD models
│   │   │   ├── experiments/route.ts   # List experiments
│   │   │   └── results/route.ts        # Fetch metrics
│   │   ├── dashboard/
│   │   │   ├── page.tsx               # Metrics dashboard
│   │   │   └── experiments/[id]/page.tsx
│   │   └── results/
│   │       └── [modelId]/page.tsx     # Confusion matrix, etc.
│   └── components/
│       ├── ModelUploader.tsx
│       ├── MetricsDashboard.tsx
│       ├── ConfusionMatrix.tsx
│       └── ExperimentHistory.tsx
└── lib/
    └── model-registry.ts  # Model management client
```

## 4. Core Components

### 4.1 Callback System

```python
class Callback(ABC):
    def on_train_start(self, state: TrainerState): pass
    def on_epoch_start(self, state: TrainerState, epoch: int): pass
    def on_batch_end(self, state: TrainerState, batch: int, metrics: dict): pass
    def on_epoch_end(self, state: TrainerState, epoch: int, metrics: dict): pass
    def on_train_end(self, state: TrainerState): pass

# Built-in callbacks
- EarlyStoppingCallback(patience=10, metric="val_acc")
- ModelCheckpointCallback(dir="checkpoints/", metric="val_acc")
- LRSchedulerCallback(scheduler)
- MetricsLoggerCallback()
- WandbCallback(project, entity)
- MLflowCallback(experiment_name)
- TensorBoardCallback(log_dir)
```

### 4.2 Tracker Interface

```python
class BaseTracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int): pass

    @abstractmethod
    def log_params(self, params: dict): pass

    @abstractmethod
    def finish(self): pass
```

### 4.3 Config Schema

```python
@dataclass
class TrainingConfig:
    model: str = "gru"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10

    # Performance
    use_amp: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2

    # Tracking
    trackers: List[str] = field(default_factory=lambda: ["tensorboard"])
    project_name: str = "tsl51"

    # Augmentation
    augmentation_factor: int = 0
    noise_level: float = 0.01
    scale_range: tuple = (0.95, 1.05)
    use_mixup: bool = False

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True
```

## 5. Implementation Phases

### Phase 1: Training System Core
- [ ] Core trainer with callback system
- [ ] Config validation (Pydantic)
- [ ] Model registry
- [ ] Basic callbacks (EarlyStopping, Checkpoint)

### Phase 2: Tracking Integration
- [ ] Base tracker interface
- [ ] TensorBoard tracker
- [ ] WandB tracker
- [ ] MLflow tracker

### Phase 3: Advanced Augmentation
- [ ] Spatial augmentations
- [ ] Mixup/Cutmix
- [ ] Temporal augmentations

### Phase 4: Website API
- [ ] Model upload/management API
- [ ] Experiment listing API
- [ ] Results/metrics API
- [ ] Inference endpoint

### Phase 5: Website Components
- [ ] ModelUploader component
- [ ] MetricsDashboard
- [ ] ConfusionMatrix visualization
- [ ] ExperimentHistory

## 6. Backward Compatibility

- เปลี่ยนชื่อ `src/train/` → `src/train_v1/`
- `src/train/` เป็น symlink หรือ redirect ไปที่ `train_v2/`
- CLI commands เหมือนเดิม (`tsl-train`, `tsl-inference`)
- Existing models & configs ยังใช้งานได้

## 7. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/inference` | Real-time inference |
| GET | `/api/models` | List models |
| POST | `/api/models` | Upload model |
| GET | `/api/models/[id]` | Get model details |
| GET | `/api/experiments` | List experiments |
| GET | `/api/results/[modelId]` | Get metrics & confusion matrix |

## 8. Success Criteria

- Training speed improved 20%+ with AMP & optimized dataloader
- All 3 tracking tools work interchangeably
- Website can upload, view, and compare model results
- System is testable and maintainable