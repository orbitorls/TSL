"""Training configuration for TSL-51 models.

This module provides configuration classes for training TSL-51 models,
making it easier to manage hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Training configuration parameters.

    Attributes:
        model: Model architecture ('mlp', 'gru', 'mopgru', 'hybrid', 'ctc')
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs per fold
        patience: Early stopping patience
        seed: Random seed
        device: Device to use ('cuda' or 'cpu')
        use_amp: Use automatic mixed precision
        gradient_clip_value: Gradient clipping value (None to disable)
        use_gradient_accumulation: Use gradient accumulation
        accumulation_steps: Number of steps to accumulate gradients
    """
    model: str = "gru"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    patience: int = 10
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    gradient_clip_value: Optional[float] = 1.0  # Enable gradient clipping by default
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 1

    # Data augmentation
    augmentation_factor: int = 0
    noise_level: float = 0.01
    scale_range: Optional[tuple] = None  # None means use default (0.95, 1.05)

    # Enhanced training options
    label_smoothing: float = 0.1  # New: Label smoothing factor
    use_cosine_annealing: bool = True  # New: Use cosine annealing instead of OneCycleLR
    mixup_alpha: float = 0.0  # New: Mixup augmentation (0 = disabled)
    use_attention_pooling: bool = True  # New: Use attention pooling in GRU

    # Dataset
    dataset: str = "tsl51_user_sign"
    max_samples: Optional[int] = None
    test_split: float = 0.0
    force_download: bool = False

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True

    # Feature extraction
    feature_level: str = "basic"
    target_frames: int = 30

    # LR Scheduling
    lr_scheduler: str = "one_cycle"
    warmup_epochs: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 3

    # SWA (Stochastic Weight Averaging)
    use_swa: bool = False
    swa_start_epoch: int = 20
    swa_lr: float = 0.0001

    # Augmentation method
    augment_method: str = "basic"
    cutout_ratio: float = 0.1

    # Analytics
    save_confusion_matrix: bool = False
    smooth_curves: bool = False

    # Checkpointing
    save_best_f1: bool = False


@dataclass
class PresetConfig:
    """Predefined training configurations for common use cases."""

    @staticmethod
    def quick() -> TrainingConfig:
        """Quick training for testing (5 epochs, smaller model)."""
        return TrainingConfig(
            model="gru",
            hidden_dim=128,
            num_layers=2,
            epochs=5,
            batch_size=64,
            n_folds=2,
            patience=3,
            augmentation_factor=0,
        )

    @staticmethod
    def accuracy_focus() -> TrainingConfig:
        """Accuracy-focused configuration for best performance."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            dropout=0.25,
            learning_rate=5e-4,
            batch_size=64,
            epochs=100,
            n_folds=5,
            patience=15,
            augmentation_factor=3,
            # Enhanced training
            label_smoothing=0.1,
            use_cosine_annealing=True,
            mixup_alpha=0.2,
            use_attention_pooling=True,
            gradient_clip_value=1.0,
            # Feature level
            feature_level="dynamics",
        )

    @staticmethod
    def default() -> TrainingConfig:
        """Default training configuration."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            epochs=50,
            batch_size=128,
            n_folds=5,
            patience=10,
            augmentation_factor=5,
        )
    
    @staticmethod
    def full_cv() -> TrainingConfig:
        """Full K-Fold CV with more epochs."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            epochs=100,
            batch_size=128,
            n_folds=5,
            patience=15,
            augmentation_factor=10,
        )
    
    @staticmethod
    def mlp_fast() -> TrainingConfig:
        """Fast MLP training."""
        return TrainingConfig(
            model="mlp",
            hidden_dim=256,
            num_layers=3,
            epochs=50,
            batch_size=128,
            n_folds=5,
            patience=10,
            augmentation_factor=5,
        )
    
    @staticmethod
    def large_dataset() -> TrainingConfig:
        """Configuration for large datasets (45k+ samples)."""
        return TrainingConfig(
            model="gru",
            hidden_dim=512,
            num_layers=4,
            epochs=100,
            batch_size=256,
            n_folds=5,
            patience=20,
            augmentation_factor=0,  # No augmentation needed for large dataset
            use_gradient_accumulation=True,
            accumulation_steps=2,
        )


def get_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from argparse arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        TrainingConfig instance
    """
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return TrainingConfig(
        model=args.model,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=device,
        use_amp=HAS_AMP if 'HAS_AMP' in globals() else True,
        dataset=args.dataset,
        max_samples=args.samples,
        test_split=args.test_split,
        force_download=args.force_download,
        n_folds=args.folds,
        feature_level=args.feature_level,
        target_frames=args.target_frames,
        augmentation_factor=args.augment,
        # New LR scheduling parameters
        lr_scheduler=getattr(args, 'lr_scheduler', 'one_cycle'),
        warmup_epochs=getattr(args, 'warmup_epochs', 5),
        plateau_factor=getattr(args, 'plateau_factor', 0.5),
        plateau_patience=getattr(args, 'plateau_patience', 3),
        # SWA parameters
        use_swa=getattr(args, 'use_swa', False),
        swa_start_epoch=getattr(args, 'swa_start_epoch', 20),
        swa_lr=getattr(args, 'swa_lr', 0.0001),
        # Augmentation method
        augment_method=getattr(args, 'augment_method', 'basic'),
        cutout_ratio=getattr(args, 'cutout_ratio', 0.1),
        # Analytics
        save_confusion_matrix=getattr(args, 'save_confusion_matrix', False),
        smooth_curves=getattr(args, 'smooth_curves', False),
        save_best_f1=getattr(args, 'save_best_f1', False),
    )


# Check for AMP availability (will be set when imported)
HAS_AMP = True  # Placeholder, will be checked at runtime
