from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.core.features import BASIC_FEATURE_DIM

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


def test_pipeline_writes_canonical_training_artifacts(tmp_path: Path, monkeypatch) -> None:
    from src.train import pipeline

    rows = _rows()
    dataset = _dataset(rows)
    calls: dict[str, Any] = {}

    def fake_train_split_only(X, y, sample_ids, manifest, classes, **kwargs):
        calls["augment_kwargs"] = kwargs
        train_ids = [row["sample_id"] for row in manifest["splits"]["train"]]
        val_ids = [row["sample_id"] for row in manifest["splits"]["val"]]
        test_ids = [row["sample_id"] for row in manifest["splits"]["test"]]
        index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

        def select(ids):
            indices = [index_by_id[str(sample_id)] for sample_id in ids]
            return X[indices], y[indices], [str(sample_id) for sample_id in ids]

        train_X, train_y, train_sample_ids = select(train_ids)
        if kwargs["augmentation_factor"] > 0:
            train_X = np.concatenate([train_X, train_X + 0.5], axis=0)
            train_y = np.concatenate([train_y, train_y], axis=0)
            train_sample_ids = train_sample_ids + [f"{sample_id}__aug_1" for sample_id in train_sample_ids]
        val_X, val_y, val_sample_ids = select(val_ids)
        test_X, test_y, test_sample_ids = select(test_ids)
        return {
            "train": {"X": train_X, "y": train_y, "sample_ids": train_sample_ids, "rows": manifest["splits"]["train"]},
            "val": {"X": val_X, "y": val_y, "sample_ids": val_sample_ids, "rows": manifest["splits"]["val"]},
            "test": {"X": test_X, "y": test_y, "sample_ids": test_sample_ids, "rows": manifest["splits"]["test"]},
        }

    class FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.model = torch.nn.Linear(BASIC_FEATURE_DIM, len(CLASSES))

        def train(self, X_train, y_train, X_val, y_val, classes, fold_idx=0):
            calls["train_shape"] = X_train.shape
            calls["val_shape"] = X_val.shape
            calls["train_mean_after_norm"] = float(np.mean(X_train))
            return {
                "fold": fold_idx,
                "val_acc": 75.0,
                "val_f1_score": 70.0,
                "val_macro_f1": 66.5,
                    "val_precision": 75.0,
                    "val_recall": 75.0,
                    "val_top3_acc": 100.0,
                    "val_top5_acc": 100.0,
                    "per_class_metrics": {},
                    "confusion_matrix": [],
                "primary_metric_name": "macro_f1",
                "primary_metric": 66.5,
                "model_state": {"model_state_dict": self.model.state_dict(), "epoch": 0},
            }

    monkeypatch.setattr(pipeline, "augment_train_split_only", fake_train_split_only)
    monkeypatch.setattr(pipeline, "Trainer", FakeTrainer)

    config = SimpleNamespace(
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
        augmentation_factor=1,
        noise_level=0.01,
        scale_range=(0.95, 1.05),
        feature_level="basic",
        seq_mode=False,
        target_frames=30,
        val_size=0.25,
        test_size=0.25,
    )

    result = pipeline.run_training_pipeline(config, dataset=dataset)

    assert result["primary_metric_name"] == "macro_f1"
    assert result["primary_metric"] == 66.5
    assert calls["augment_kwargs"]["augmentation_factor"] == 1
    assert calls["train_shape"][0] > len(result["split_manifest"]["splits"]["train"])
    assert abs(calls["train_mean_after_norm"]) < 1e-5

    artifact_paths = result["artifacts"]
    checkpoint_path = Path(artifact_paths["checkpoint"])
    metrics_path = Path(artifact_paths["metrics"])
    split_manifest_path = Path(artifact_paths["split_manifest"])
    preprocessing_manifest_path = Path(artifact_paths["preprocessing_manifest"])
    label_map_path = Path(artifact_paths["label_map"])

    for path in (checkpoint_path, metrics_path, split_manifest_path, preprocessing_manifest_path, label_map_path):
        assert path.exists(), path

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    assert label_map == {"hello": 0, "thanks": 1, "water": 2}

    preprocessing = json.loads(preprocessing_manifest_path.read_text(encoding="utf-8"))
    assert preprocessing["label_map_path"] == str(label_map_path)
    assert preprocessing["label_map_hash"]
    assert preprocessing["label_metadata"]["classes"] == CLASSES.tolist()
    assert preprocessing["split_strategy"] == "video_family_holdout"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["primary_metric_name"] == "macro_f1"
    assert metrics["primary_metric"] == 66.5
    assert metrics["artifacts"]["label_map"] == str(label_map_path)

    saved_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["split_strategy"] == "video_family_holdout"
    assert saved_manifest["group_key"] == "video_family_id"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["classes"] == CLASSES.tolist()
    assert checkpoint["label_to_idx"] == label_map
    assert checkpoint["primary_metric_name"] == "macro_f1"
    assert checkpoint["input_dim"] == BASIC_FEATURE_DIM


def test_pipeline_requires_grouped_split_for_real_world_mode(tmp_path: Path, monkeypatch) -> None:
    from src.train import pipeline

    rows = _rows()
    dataset = _dataset(rows)

    class ShouldNotTrain:
        def __init__(self, _config):
            raise AssertionError("pipeline should reject random split before training starts")

    monkeypatch.setattr(pipeline, "Trainer", ShouldNotTrain)

    config = SimpleNamespace(
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
        feature_level="basic",
        seq_mode=False,
        target_frames=30,
        val_size=0.25,
        test_size=0.25,
        real_world_mode=True,
        split_strategy="random",
    )

    with pytest.raises(pipeline.PipelineConfigError, match="real-world mode.*grouped|video-family"):
        pipeline.run_training_pipeline(config, dataset=dataset)


def test_train_cli_normal_path_does_not_import_legacy_training_script() -> None:
    tree = ast.parse(Path("src/cli/train.py").read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert "legacy.root_scripts.train_tsl51_v3" not in imported_modules
