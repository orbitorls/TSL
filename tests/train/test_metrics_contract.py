from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.features import BASIC_FEATURE_DIM, FEATURE_SCHEMA_VERSION
from src.train.config import TrainingConfig
from src.train.trainer import Trainer

CLASSES = np.array(["hello", "thanks", "water"])
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES.tolist())}


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_idx, label in enumerate(CLASSES.tolist()):
        for family_idx in range(4):
            video_id = f"vid_{class_idx}_{family_idx}"
            rows.append(
                {
                    "sample_id": f"sample-{video_id}",
                    "video_id": video_id,
                    "landmark_path": f"landmarks/{video_id}.csv",
                    "sign_clean": label,
                    "is_augmented": False,
                }
            )
    return rows


def _dataset(rows: list[dict[str, object]]):
    sample_ids = [str(row["sample_id"]) for row in rows]
    X = np.stack(
        [np.full((BASIC_FEATURE_DIM,), idx + 1, dtype=np.float32) for idx, _row in enumerate(rows)]
    )
    y = np.array([LABEL_TO_INDEX[str(row["sign_clean"])] for row in rows], dtype=np.int64)
    return SimpleNamespace(X=X, y=y, classes=CLASSES, rows=rows, sample_ids=sample_ids)


def _base_pipeline_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="unit-dataset",
        output_dir=tmp_path,
        model="mlp",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=4,
        epochs=1,
        patience=1,
        seed=7,
        device="cpu",
        use_amp=False,
        gradient_clip_value=None,
        use_gradient_accumulation=False,
        accumulation_steps=1,
        augmentation_factor=0,
        noise_level=0.01,
        scale_range=(0.95, 1.05),
        feature_level="basic",
        seq_mode=False,
        target_frames=30,
        val_size=0.25,
        test_size=0.25,
        real_world_mode=True,
        split_strategy="video_family_holdout",
    )


def test_trainer_prefers_macro_f1_over_accuracy_for_best_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.train.trainer as trainer_module

    config = TrainingConfig(
        model="mlp",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=2,
        epochs=2,
        patience=2,
        device="cpu",
        use_amp=False,
        seq_mode=False,
    )
    trainer = Trainer(config)

    X_train = np.zeros((4, BASIC_FEATURE_DIM), dtype=np.float32)
    y_train = np.array([0, 1, 2, 0], dtype=np.int64)
    X_val = np.zeros((3, BASIC_FEATURE_DIM), dtype=np.float32)
    y_val = np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(
        Trainer,
        "_build_dataloaders",
        lambda _self, _X_train, _y_train, _X_val, _y_val: ([object()], [object()]),
    )
    monkeypatch.setattr(Trainer, "train_epoch", lambda _self, _loader, _criterion, _accumulation_steps=1: (0.1, 50.0))

    validation_returns = iter(
        [
            (0.5, 92.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
            (0.4, 75.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
            (0.4, 75.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
        ]
    )
    monkeypatch.setattr(Trainer, "validate", lambda _self, _loader, _criterion: next(validation_returns))

    def fake_setup_model(self, input_dim: int, num_classes: int) -> None:
        self.model = torch.nn.Linear(input_dim, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scaler = None

    monkeypatch.setattr(Trainer, "setup_model", fake_setup_model)

    metrics_by_call = iter(
        [
            {
                "accuracy": 92.0,
                "precision": 91.0,
                "recall": 90.0,
                "f1_score": 89.0,
                "top3_accuracy": 100.0,
                "top5_accuracy": 100.0,
                "per_class": {},
                "confusion_matrix": np.eye(3, dtype=int),
                "most_confused": [],
                "macro": {"f1": 55.0},
            },
            {
                "accuracy": 75.0,
                "precision": 74.0,
                "recall": 73.0,
                "f1_score": 72.0,
                "top3_accuracy": 100.0,
                "top5_accuracy": 100.0,
                "per_class": {},
                "confusion_matrix": np.eye(3, dtype=int),
                "most_confused": [],
                "macro": {"f1": 70.0},
            },
            {
                "accuracy": 75.0,
                "precision": 74.0,
                "recall": 73.0,
                "f1_score": 72.0,
                "top3_accuracy": 100.0,
                "top5_accuracy": 100.0,
                "per_class": {},
                "confusion_matrix": np.eye(3, dtype=int),
                "most_confused": [],
                "macro": {"f1": 70.0},
            },
        ]
    )
    monkeypatch.setattr(trainer_module, "compute_metrics", lambda *_args, **_kwargs: next(metrics_by_call))

    result = trainer.train(X_train, y_train, X_val, y_val, CLASSES, fold_idx=0)

    assert trainer.best_state is not None
    assert trainer.best_state["val_acc"] == 75.0
    assert trainer.best_state["val_macro_f1"] == 70.0
    assert result["primary_metric_name"] == "macro_f1"
    assert result["primary_metric"] == 70.0
    assert result["best_val_acc"] == 75.0


def test_trainer_best_state_snapshots_best_epoch_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.train.trainer as trainer_module

    config = TrainingConfig(
        model="mlp",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        batch_size=2,
        epochs=2,
        patience=2,
        device="cpu",
        use_amp=False,
        seq_mode=False,
    )
    trainer = Trainer(config)

    X_train = np.zeros((4, BASIC_FEATURE_DIM), dtype=np.float32)
    y_train = np.array([0, 1, 2, 0], dtype=np.int64)
    X_val = np.zeros((3, BASIC_FEATURE_DIM), dtype=np.float32)
    y_val = np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(Trainer, "_build_dataloaders", lambda _self, *_args: ([object()], [object()]))

    epoch_states: list[dict[str, torch.Tensor]] = []

    def fake_train_epoch(self, _loader, _criterion, _accumulation_steps=1):
        with torch.no_grad():
            self.model.weight.fill_(float(len(epoch_states) + 1))
            self.model.bias.fill_(float(len(epoch_states) + 1))
        epoch_states.append(copy.deepcopy(self.model.state_dict()))
        return 0.1, 50.0

    monkeypatch.setattr(Trainer, "train_epoch", fake_train_epoch)

    validation_returns = iter(
        [
            (0.5, 92.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
            (0.4, 75.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
            (0.4, 75.0, np.array([0, 1, 2]), np.array([0, 1, 2]), np.eye(3)),
        ]
    )
    monkeypatch.setattr(Trainer, "validate", lambda _self, _loader, _criterion: next(validation_returns))

    def fake_setup_model(self, input_dim: int, num_classes: int) -> None:
        self.model = torch.nn.Linear(input_dim, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scaler = None

    monkeypatch.setattr(Trainer, "setup_model", fake_setup_model)

    metrics_by_call = iter(
        [
            {"accuracy": 92.0, "precision": 91.0, "recall": 90.0, "f1_score": 89.0, "top3_accuracy": 100.0, "top5_accuracy": 100.0, "per_class": {}, "confusion_matrix": np.eye(3, dtype=int), "most_confused": [], "macro": {"f1": 70.0}},
            {"accuracy": 75.0, "precision": 74.0, "recall": 73.0, "f1_score": 72.0, "top3_accuracy": 100.0, "top5_accuracy": 100.0, "per_class": {}, "confusion_matrix": np.eye(3, dtype=int), "most_confused": [], "macro": {"f1": 60.0}},
            {"accuracy": 92.0, "precision": 91.0, "recall": 90.0, "f1_score": 89.0, "top3_accuracy": 100.0, "top5_accuracy": 100.0, "per_class": {}, "confusion_matrix": np.eye(3, dtype=int), "most_confused": [], "macro": {"f1": 70.0}},
        ]
    )
    monkeypatch.setattr(trainer_module, "compute_metrics", lambda *_args, **_kwargs: next(metrics_by_call))

    result = trainer.train(X_train, y_train, X_val, y_val, CLASSES, fold_idx=0)

    best_weight = result["model_state"]["model_state_dict"]["weight"]
    final_epoch_weight = epoch_states[-1]["weight"]
    best_epoch_weight = epoch_states[0]["weight"]

    assert torch.equal(best_weight, best_epoch_weight)
    assert not torch.equal(best_weight, final_epoch_weight)


@pytest.mark.parametrize(
    ("missing_key", "expected_match"),
    [
        ("val_acc", "val_acc"),
        ("val_f1_score", "val_f1_score"),
        ("val_precision", "val_precision"),
        ("val_recall", "val_recall"),
        ("val_top3_acc", "val_top3_acc"),
        ("val_top5_acc", "val_top5_acc"),
        ("per_class_metrics", "per_class_metrics"),
        ("confusion_matrix", "confusion_matrix"),
    ],
)
def test_pipeline_rejects_incomplete_secondary_metrics_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
    expected_match: str,
) -> None:
    from src.train import pipeline

    rows = _rows()
    dataset = _dataset(rows)

    def fake_train_split_only(X, y, sample_ids, manifest, _classes, **_kwargs):
        train_ids = [row["sample_id"] for row in manifest["splits"]["train"]]
        val_ids = [row["sample_id"] for row in manifest["splits"]["val"]]
        test_ids = [row["sample_id"] for row in manifest["splits"]["test"]]
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

        def select(ids):
            indices = [index_by_id[str(sample_id)] for sample_id in ids]
            return X[indices], y[indices], [str(sample_id) for sample_id in ids]

        train_X, train_y, train_sample_ids = select(train_ids)
        val_X, val_y, val_sample_ids = select(val_ids)
        test_X, test_y, test_sample_ids = select(test_ids)
        return {
            "train": {"X": train_X, "y": train_y, "sample_ids": train_sample_ids, "rows": manifest["splits"]["train"]},
            "val": {"X": val_X, "y": val_y, "sample_ids": val_sample_ids, "rows": manifest["splits"]["val"]},
            "test": {"X": test_X, "y": test_y, "sample_ids": test_sample_ids, "rows": manifest["splits"]["test"]},
        }

    class IncompleteMetricsTrainer:
        def __init__(self, config):
            self.config = config
            self.model = torch.nn.Linear(BASIC_FEATURE_DIM, len(CLASSES))

        def train(self, _X_train, _y_train, _X_val, _y_val, _classes, fold_idx=0):
            result = {
                "fold": fold_idx,
                "val_loss": 0.25,
                "val_acc": 82.0,
                "val_f1_score": 79.0,
                "val_macro_f1": 77.5,
                "primary_metric_name": "macro_f1",
                "primary_metric": 77.5,
                "val_precision": 80.0,
                "val_recall": 78.0,
                "val_top3_acc": 95.0,
                "val_top5_acc": 100.0,
                "per_class_metrics": {"hello": {"accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0}},
                "confusion_matrix": np.eye(len(CLASSES), dtype=int),
                "most_confused": [],
                "model_state": {"model_state_dict": self.model.state_dict(), "epoch": 0},
            }
            result.pop(missing_key)
            return result

    monkeypatch.setattr(pipeline, "augment_train_split_only", fake_train_split_only)
    monkeypatch.setattr(pipeline, "Trainer", IncompleteMetricsTrainer)

    with pytest.raises(pipeline.PipelineConfigError, match=expected_match):
        pipeline.run_training_pipeline(_base_pipeline_config(tmp_path), dataset=dataset)


def test_pipeline_rejects_missing_macro_f1_before_serialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.train import pipeline

    rows = _rows()
    dataset = _dataset(rows)

    def fake_train_split_only(X, y, sample_ids, manifest, _classes, **_kwargs):
        train_ids = [row["sample_id"] for row in manifest["splits"]["train"]]
        val_ids = [row["sample_id"] for row in manifest["splits"]["val"]]
        test_ids = [row["sample_id"] for row in manifest["splits"]["test"]]
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

        def select(ids):
            indices = [index_by_id[str(sample_id)] for sample_id in ids]
            return X[indices], y[indices], [str(sample_id) for sample_id in ids]

        train_X, train_y, train_sample_ids = select(train_ids)
        val_X, val_y, val_sample_ids = select(val_ids)
        test_X, test_y, test_sample_ids = select(test_ids)
        return {
            "train": {"X": train_X, "y": train_y, "sample_ids": train_sample_ids, "rows": manifest["splits"]["train"]},
            "val": {"X": val_X, "y": val_y, "sample_ids": val_sample_ids, "rows": manifest["splits"]["val"]},
            "test": {"X": test_X, "y": test_y, "sample_ids": test_sample_ids, "rows": manifest["splits"]["test"]},
        }

    class MissingMacroTrainer:
        def __init__(self, config):
            self.config = config
            self.model = torch.nn.Linear(BASIC_FEATURE_DIM, len(CLASSES))

        def train(self, _X_train, _y_train, _X_val, _y_val, _classes, fold_idx=0):
            return {
                "fold": fold_idx,
                "val_acc": 84.0,
                "val_f1_score": 80.0,
                "primary_metric_name": "macro_f1",
                "model_state": {"model_state_dict": self.model.state_dict(), "epoch": 0},
            }

    monkeypatch.setattr(pipeline, "augment_train_split_only", fake_train_split_only)
    monkeypatch.setattr(pipeline, "Trainer", MissingMacroTrainer)

    with pytest.raises(pipeline.PipelineConfigError, match="macro_f1"):
        pipeline.run_training_pipeline(_base_pipeline_config(tmp_path), dataset=dataset)


def test_pipeline_serializes_grouped_split_aware_metrics_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.train import pipeline

    rows = _rows()
    dataset = _dataset(rows)

    def fake_train_split_only(X, y, sample_ids, manifest, _classes, **_kwargs):
        train_ids = [row["sample_id"] for row in manifest["splits"]["train"]]
        val_ids = [row["sample_id"] for row in manifest["splits"]["val"]]
        test_ids = [row["sample_id"] for row in manifest["splits"]["test"]]
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

        def select(ids):
            indices = [index_by_id[str(sample_id)] for sample_id in ids]
            return X[indices], y[indices], [str(sample_id) for sample_id in ids]

        train_X, train_y, train_sample_ids = select(train_ids)
        val_X, val_y, val_sample_ids = select(val_ids)
        test_X, test_y, test_sample_ids = select(test_ids)
        return {
            "train": {"X": train_X, "y": train_y, "sample_ids": train_sample_ids, "rows": manifest["splits"]["train"]},
            "val": {"X": val_X, "y": val_y, "sample_ids": val_sample_ids, "rows": manifest["splits"]["val"]},
            "test": {"X": test_X, "y": test_y, "sample_ids": test_sample_ids, "rows": manifest["splits"]["test"]},
        }

    class ContractTrainer:
        def __init__(self, config):
            self.config = config
            self.model = torch.nn.Linear(BASIC_FEATURE_DIM, len(CLASSES))

        def train(self, _X_train, _y_train, _X_val, _y_val, _classes, fold_idx=0):
            return {
                "fold": fold_idx,
                "val_loss": 0.25,
                "val_acc": 82.0,
                "val_f1_score": 79.0,
                "val_macro_f1": 77.5,
                "primary_metric_name": "macro_f1",
                "primary_metric": 77.5,
                "val_precision": 80.0,
                "val_recall": 78.0,
                "val_top3_acc": 95.0,
                "val_top5_acc": 100.0,
                "per_class_metrics": {
                    "hello": {"accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0}
                },
                "confusion_matrix": np.eye(len(CLASSES), dtype=int),
                "most_confused": [],
                "model_state": {"model_state_dict": self.model.state_dict(), "epoch": 0},
            }

    monkeypatch.setattr(pipeline, "augment_train_split_only", fake_train_split_only)
    monkeypatch.setattr(pipeline, "Trainer", ContractTrainer)

    result = pipeline.run_training_pipeline(_base_pipeline_config(tmp_path), dataset=dataset)

    metrics_path = Path(result["artifacts"]["metrics"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert payload["primary_metric_name"] == "macro_f1"
    assert payload["primary_metric"] == 77.5
    assert payload["dataset"] == "unit-dataset"
    assert payload["seed"] == 7
    assert payload["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert payload["checkpoint_path"] == result["artifacts"]["checkpoint"]
    assert payload["preprocessing_manifest_path"] == result["artifacts"]["preprocessing_manifest"]
    assert payload["split_metadata"]["requested_strategy"] == "video_family_holdout"
    assert payload["split_metadata"]["persisted_strategy"] == "video_family_holdout"
    assert payload["split_metadata"]["group_key"] == "video_family_id"
    assert payload["split_metadata"]["sample_counts"]["train"] > 0
    assert payload["metrics"]["macro_f1"] == 77.5
    assert payload["metrics"]["weighted_f1"] == 79.0
    assert payload["metrics"]["accuracy"] == 82.0
    assert payload["metrics"]["top3_accuracy"] == 95.0
    assert payload["metrics"]["top5_accuracy"] == 100.0
    assert payload["metrics"]["per_class"] == {
        "hello": {"accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0}
    }
    assert payload["metrics"]["confusion_matrix"]["data"] == np.eye(len(CLASSES), dtype=int).tolist()
    assert payload["metrics"]["confusion_matrix"]["path"] is None
    assert payload["artifacts"]["checkpoint"] == result["artifacts"]["checkpoint"]
    assert payload["artifacts"]["split_manifest"] == result["artifacts"]["split_manifest"]
