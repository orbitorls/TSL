"""Training configuration for TSL-51 models.

This module provides configuration classes for training TSL-51 models,
making it easier to manage hyperparameters and settings.
"""

from dataclasses import dataclass

from src.train.compat import HAS_AMP


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
    gradient_clip_value: float | None = None
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 1

    # Data augmentation
    augmentation_factor: int = 0
    noise_level: float = 0.01
    scale_range: tuple = (0.95, 1.05)

    # Dataset
    dataset: str = "tsl51_user_sign"
    max_samples: int | None = None
    test_split: float = 0.0
    force_download: bool = False

    # Cross-validation
    n_folds: int = 5
    stratified: bool = True

    # Feature extraction
    feature_level: str = "basic"
    seq_mode: bool = True
    target_frames: int = 30


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
    def default() -> TrainingConfig:
        """Default training configuration with seq_mode and augmentation."""
        return TrainingConfig(
            model="gru",
            hidden_dim=256,
            num_layers=3,
            epochs=50,
            batch_size=128,
            n_folds=5,
            patience=10,
            feature_level="basic",
            seq_mode=True,
            target_frames=30,
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
        use_amp=HAS_AMP,
        dataset=args.dataset,
        max_samples=args.samples,
        test_split=args.test_split,
        force_download=args.force_download,
        n_folds=args.folds,
        feature_level=args.feature_level,
        target_frames=args.target_frames,
        augmentation_factor=args.augment,
    )
