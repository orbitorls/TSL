"""Core training logic for TSL-51 models.

This module contains the training loop and related functions extracted from
train_tsl51_v3.py for better modularity.
"""

import logging
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from src.core.models import MODEL_REGISTRY as MODEL_CLASSES
from src.train.evaluator import compute_metrics

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for TSL-51 models.

    This class handles the training loop, validation, and model checkpointing.
    """

    def __init__(
        self,
        config_or_model: TrainingConfig | nn.Module,
        X_train: Any | None = None,
        y_train: Any | None = None,
        classes: Any | None = None,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ):
        """Initialize trainer.

        Args:
            config_or_model: Either a ``TrainingConfig`` (new API) or a pre-built model (legacy API).
            X_train: Optional training features (legacy API).
            y_train: Optional training labels (legacy API).
            classes: Optional class labels (legacy API, currently unused by trainer internals).
            config: Legacy API config object when ``config_or_model`` is a model.
            device: Override device ('cuda' or 'cpu').
        """
        if isinstance(config_or_model, TrainingConfig):
            self.config = config_or_model
            if X_train is not None and not isinstance(X_train, str):
                raise TypeError(
                    "Legacy positional training data is only accepted when passing a model as first arg"
                )
            resolved_device = device or (X_train if isinstance(X_train, str) else self.config.device)
            self.model: Any = None
            self.train_data = None
            self.train_labels = None
            self.classes = None
        else:
            self.model = config_or_model
            if config is None:
                raise TypeError("Trainer legacy API requires config as the 5th positional argument")

            self.config = config
            self.train_data = np.asarray(X_train) if X_train is not None else None
            self.train_labels = np.asarray(y_train) if y_train is not None else None
            self.classes = classes
            resolved_device = device or self.config.device

        if isinstance(resolved_device, torch.device):
            self.device = resolved_device
        else:
            self.device = torch.device(resolved_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.optimizer: Any = None
        self.scheduler: Any = None
        self.scaler: Any = None
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.best_state: dict[str, Any] | None = None

    @staticmethod
    def _resolve_accumulation_steps(config: TrainingConfig) -> int:
        if not config.use_gradient_accumulation:
            return 1
        return max(1, int(config.accumulation_steps or 1))

    @staticmethod
    def _resolve_num_workers(device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        workers = (os.cpu_count() or 1) // 2
        return max(1, min(4, workers))

    @staticmethod
    def _to_ndarray(array_like: Any, dtype) -> np.ndarray:
        if isinstance(array_like, np.ndarray):
            return array_like.astype(dtype, copy=False)
        return np.asarray(array_like, dtype=dtype)

    def _configure_scheduler(self, train_loader: DataLoader, accumulation_steps: int) -> None:
        if self.optimizer is None:
            return

        steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
        )

    def _ensure_optimizer(self) -> None:
        if self.optimizer is not None or self.model is None:
            return

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

        self.scaler = None
        if self.config.use_amp and self.device.type == "cuda":
            try:
                from torch.amp.grad_scaler import GradScaler

                self.scaler = GradScaler()
            except ImportError:
                try:
                    from torch.cuda.amp import GradScaler  # type: ignore[assignment]

                    self.scaler = GradScaler()
                except ImportError:
                    self.scaler = None
                    logger.warning("AMP not available, training without mixed precision")

    def setup_model(self, input_dim: int, num_classes: int) -> None:
        """Setup model, optimizer, scheduler, and scaler."""
        # Get model class.
        model_class = MODEL_CLASSES.get(self.config.model, MODEL_CLASSES["gru"])

        # Create and move model to selected device.
        self.model = model_class(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        # Setup optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

        # Setup gradient scaler for mixed precision.
        self.scaler = None
        if self.config.use_amp and self.device.type == "cuda":
            try:
                from torch.amp.grad_scaler import GradScaler

                self.scaler = GradScaler()
            except ImportError:
                try:
                    from torch.cuda.amp import GradScaler  # type: ignore[assignment]

                    self.scaler = GradScaler()
                except ImportError:
                    self.scaler = None
                    logger.warning("AMP not available, training without mixed precision")

    def _build_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        num_workers = self._resolve_num_workers(self.device)
        common_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": num_workers,
            "pin_memory": num_workers > 0,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **common_kwargs,
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader, criterion, accumulation_steps: int = 1):
        """Train for one epoch.

        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialised before training")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        accumulated_steps = 0

        is_amp_enabled = self.config.use_amp and self.scaler is not None and self.device.type == "cuda"

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device, non_blocking=self.device.type == "cuda")
            y_batch = y_batch.to(self.device, non_blocking=self.device.type == "cuda")
            if accumulated_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            accumulated_steps += 1

            should_step = ((batch_idx + 1) % accumulation_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            )

            if is_amp_enabled:
                with torch.autocast("cuda"):
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch) / accumulation_steps
                self.scaler.scale(loss).backward()

                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.config.gradient_clip_value is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    accumulated_steps = 0
            else:
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch) / accumulation_steps
                loss.backward()

                if should_step:
                    if self.config.gradient_clip_value is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_value
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    accumulated_steps = 0

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total if total else 0.0

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        """Validate the model.

        Returns:
            Tuple of (loss, accuracy, predictions, labels, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model must be initialised before validation")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device, non_blocking=self.device.type == "cuda")
                y_batch = y_batch.to(self.device, non_blocking=self.device.type == "cuda")

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = probs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

                all_preds.extend(predicted.detach().cpu().numpy())
                all_labels.extend(y_batch.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        if len(val_loader) == 0:
            return 0.0, 0.0, np.array([]), np.array([]), np.array([])

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total if total else 0.0

        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def train(self, X_train, y_train, X_val, y_val, classes, fold_idx: int = 0) -> dict[str, Any]:
        """Train the model on given train/val split.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            classes: Class names
            fold_idx: Current fold index

        Returns:
            Dictionary containing training results
        """
        X_train_arr = self._to_ndarray(X_train, np.float32)
        y_train_arr = self._to_ndarray(y_train, np.int64).astype(np.int64)
        X_val_arr = self._to_ndarray(X_val, np.float32)
        y_val_arr = self._to_ndarray(y_val, np.int64).astype(np.int64)

        if len(X_train_arr) == 0 or len(y_train_arr) == 0:
            raise ValueError("Training dataset is empty")
        if len(X_val_arr) == 0 or len(y_val_arr) == 0:
            raise ValueError("Validation dataset is empty")
        if len(X_train_arr) != len(y_train_arr):
            raise ValueError("X_train and y_train length mismatch")
        if len(X_val_arr) != len(y_val_arr):
            raise ValueError("X_val and y_val length mismatch")

        train_loader, val_loader = self._build_dataloaders(X_train_arr, y_train_arr, X_val_arr, y_val_arr)

        input_dim = int(X_train_arr.shape[-1])
        num_classes = int(len(classes)) if classes is not None else int(y_train_arr.max()) + 1
        if self.model is None:
            self.setup_model(input_dim=input_dim, num_classes=num_classes)
        else:
            self.model = self.model.to(self.device)
        if self.model is None:
            raise RuntimeError("Failed to initialize model")

        # Setup training controls for the active model.
        self._ensure_optimizer()
        if self.optimizer is None:
            raise RuntimeError("Failed to initialize optimizer")

        accumulation_steps = self._resolve_accumulation_steps(self.config)
        self._configure_scheduler(train_loader, accumulation_steps)

        # Setup weighted loss for class imbalance.
        # Use explicit class indices so missing classes are handled explicitly.
        present_classes = np.asarray(sorted({int(v) for v in np.unique(y_train_arr)}), dtype=np.int64)
        # Guard against labels outside configured class range.
        present_classes = present_classes[(present_classes >= 0) & (present_classes < num_classes)]
        if present_classes.size == 0:
            raise ValueError("No valid class labels found in training targets")

        present_weights = compute_class_weight(
            "balanced",
            classes=present_classes,
            y=y_train_arr,
        ).astype(np.float32)
        class_weights = np.ones(num_classes, dtype=np.float32)
        class_weights[present_classes] = present_weights
        class_weights_tensor = torch.from_numpy(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        self.best_state = None
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, accumulation_steps)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Fold {fold_idx} Epoch {epoch + 1}/{self.config.epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Early stopping and best-model tracking.
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_acc
                self.patience_counter = 0
                self.best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "seq_mode": bool(getattr(self.config, "seq_mode", False)),
                    "target_frames": int(getattr(self.config, "target_frames", 30)),
                    "model": str(self.config.model),
                    "num_classes": num_classes,
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model_state_dict"])

        # Final evaluation via evaluator
        val_loss, val_acc, val_preds, val_labels, val_probs = self.validate(val_loader, criterion)
        metrics = compute_metrics(val_labels, val_preds, classes, y_probs=val_probs)
        self.best_val_f1 = metrics["f1_score"]

        results = {
            "fold": fold_idx,
            "history": history,
            "best_val_acc": self.best_val_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_score": metrics["f1_score"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_top3_acc": metrics["top3_accuracy"],
            "val_top5_acc": metrics["top5_accuracy"],
            "per_class_metrics": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "most_confused": metrics["most_confused"],
            "model_state": self.best_state,
        }

        return results
