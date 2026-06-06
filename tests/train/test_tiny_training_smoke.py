from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.core.features import BASIC_FEATURE_DIM
from src.train.pipeline import run_training_pipeline

CLASSES = np.array(["hello", "thanks", "water", "yes", "no", "sorry"])
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


def _dataset(rows: list[dict[str, object]]) -> SimpleNamespace:
    sample_ids = [str(row["sample_id"]) for row in rows]
    features = np.stack(
        [
            np.full((BASIC_FEATURE_DIM,), idx + 1, dtype=np.float32)
            + np.linspace(0.0, 0.01, BASIC_FEATURE_DIM, dtype=np.float32)
            for idx, _row in enumerate(rows)
        ]
    )
    labels = np.array([LABEL_TO_INDEX[str(row["sign_clean"])] for row in rows], dtype=np.int64)
    return SimpleNamespace(X=features, y=labels, classes=CLASSES, rows=rows, sample_ids=sample_ids)


def test_tiny_training_pipeline_writes_deterministic_artifacts(tmp_path: Path) -> None:
    dataset = _dataset(_rows())
    output_dir = tmp_path / "tiny-training-run"
    config = SimpleNamespace(
        dataset="unit-dataset",
        output_dir=output_dir,
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
        noise_level=0.0,
        scale_range=(1.0, 1.0),
        feature_level="basic",
        seq_mode=False,
        target_frames=30,
        val_size=0.25,
        test_size=0.25,
        split_strategy="video_family_holdout",
        real_world_mode=True,
        timestamp="tiny-smoke",
    )

    result = run_training_pipeline(config, dataset=dataset)

    assert result["primary_metric_name"] == "macro_f1"
    assert result["requested_split_strategy"] == "video_family_holdout"

    artifact_paths = {name: Path(raw_path) for name, raw_path in result["artifacts"].items()}
    expected_names = {
        "checkpoint",
        "metrics",
        "split_manifest",
        "preprocessing_manifest",
        "label_map",
    }
    assert set(artifact_paths) == expected_names

    for path in artifact_paths.values():
        assert path.exists(), path
        assert output_dir in (path, *path.parents)

    checkpoint_path = artifact_paths["checkpoint"]
    assert checkpoint_path.name == "tsl51_mlp_tiny-smoke.pt"

    metrics = json.loads(artifact_paths["metrics"].read_text(encoding="utf-8"))
    split_manifest = json.loads(artifact_paths["split_manifest"].read_text(encoding="utf-8"))
    label_map = json.loads(artifact_paths["label_map"].read_text(encoding="utf-8"))
    preprocessing = json.loads(artifact_paths["preprocessing_manifest"].read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["primary_metric_name"] == "macro_f1"
    assert metrics["artifacts"]["checkpoint"] == str(checkpoint_path)
    assert metrics["artifacts"]["split_manifest"] == str(artifact_paths["split_manifest"])

    assert split_manifest["split_strategy"] == "video_family_holdout"
    assert split_manifest["group_key"] == "video_family_id"
    assert split_manifest["sample_counts"]["train"] > 0
    assert split_manifest["sample_counts"]["val"] > 0
    assert split_manifest["sample_counts"]["test"] > 0
    assert all(not row["augmented_row"] for row in split_manifest["splits"]["val"])
    assert all(not row["augmented_row"] for row in split_manifest["splits"]["test"])

    assert label_map == {label: idx for idx, label in enumerate(CLASSES.tolist())}

    assert preprocessing["feature_level"] == "basic"
    assert preprocessing["feature_schema_version"] == "basic-162-v1"
    assert preprocessing["split_strategy"] == "video_family_holdout"
    assert preprocessing["label_map_path"] == str(artifact_paths["label_map"])
    assert preprocessing["source_metadata"]["split_manifest_path"] == str(artifact_paths["split_manifest"])

    assert checkpoint["classes"] == CLASSES.tolist()
    assert checkpoint["label_to_idx"] == label_map
    assert checkpoint["primary_metric_name"] == "macro_f1"
    assert checkpoint["input_dim"] == BASIC_FEATURE_DIM
    assert len(checkpoint["mean"]) == BASIC_FEATURE_DIM
    assert len(checkpoint["std"]) == BASIC_FEATURE_DIM
