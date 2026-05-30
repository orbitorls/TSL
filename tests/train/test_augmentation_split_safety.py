"""Tests for train-only augmentation after grouped splitting."""

import numpy as np
import pytest

from src.train.augment import augment_data, augment_train_split_only
from src.train.splits import build_grouped_split_manifest


CLASSES = np.array(["hello", "thanks", "water"])
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES.tolist())}


def _rows():
    rows = []
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


def _dataset_from_rows(rows):
    sample_ids = []
    X = []
    y = []
    for idx, row in enumerate(rows):
        sample_ids.append(row["sample_id"])
        X.append(np.full((162,), idx + 1, dtype=np.float32))
        y.append(LABEL_TO_INDEX[row["sign_clean"]])
    return np.stack(X), np.array(y, dtype=np.int64), sample_ids


def _expected_split_arrays(manifest, split, all_sample_ids, X, y):
    index_by_id = {sample_id: idx for idx, sample_id in enumerate(all_sample_ids)}
    split_rows = manifest["splits"][split]
    indices = [index_by_id[row["sample_id"]] for row in split_rows]
    return (
        [row["sample_id"] for row in split_rows],
        X[indices],
        y[indices],
    )


def test_augmentation_applies_only_to_train_split(monkeypatch):
    rows = _rows()
    X, y, sample_ids = _dataset_from_rows(rows)
    manifest = build_grouped_split_manifest(
        rows,
        dataset_name="unit",
        seed=7,
        val_size=0.25,
        test_size=0.25,
    )

    monkeypatch.setattr(np.random, "choice", lambda _choices: "scale")
    monkeypatch.setattr(np.random, "uniform", lambda _low, _high: 2.0)

    split_data = augment_train_split_only(
        X,
        y,
        sample_ids,
        manifest,
        CLASSES,
        augmentation_factor=1,
    )

    train_ids, train_X, train_y = _expected_split_arrays(manifest, "train", sample_ids, X, y)
    val_ids, val_X, val_y = _expected_split_arrays(manifest, "val", sample_ids, X, y)
    test_ids, test_X, test_y = _expected_split_arrays(manifest, "test", sample_ids, X, y)

    assert split_data["train"]["sample_ids"][: len(train_ids)] == train_ids
    assert split_data["train"]["X"].shape[0] == len(train_ids) * 2
    assert split_data["train"]["y"].shape[0] == len(train_y) * 2
    assert split_data["train"]["sample_ids"][len(train_ids) :] == [
        f"{sample_id}__aug_1" for sample_id in train_ids
    ]

    assert split_data["val"]["sample_ids"] == val_ids
    assert np.array_equal(split_data["val"]["X"], val_X)
    assert np.array_equal(split_data["val"]["y"], val_y)

    assert split_data["test"]["sample_ids"] == test_ids
    assert np.array_equal(split_data["test"]["X"], test_X)
    assert np.array_equal(split_data["test"]["y"], test_y)


def test_augmentation_preserves_original_only_validation_and_test_ids(monkeypatch):
    rows = _rows()
    X, y, sample_ids = _dataset_from_rows(rows)
    manifest = build_grouped_split_manifest(
        rows,
        dataset_name="unit",
        seed=11,
        val_size=0.25,
        test_size=0.25,
    )

    monkeypatch.setattr(np.random, "choice", lambda _choices: "noise")
    monkeypatch.setattr(np.random, "normal", lambda _mean, _std, shape: np.ones(shape, dtype=np.float32))

    split_data = augment_train_split_only(
        X,
        y,
        sample_ids,
        manifest,
        CLASSES,
        augmentation_factor=2,
    )

    original_holdout_ids = {
        split: [row["sample_id"] for row in manifest["splits"][split]]
        for split in ("val", "test")
    }

    for split in ("val", "test"):
        assert split_data[split]["sample_ids"] == original_holdout_ids[split]
        assert all("__aug_" not in sample_id for sample_id in split_data[split]["sample_ids"])
        assert split_data[split]["X"].shape[0] == manifest["sample_counts"][split]
        assert split_data[split]["y"].shape[0] == manifest["sample_counts"][split]


def test_flip_rejects_non_basic_schema_before_augmentation_runs(monkeypatch):
    X = np.zeros((3, 200), dtype=np.float32)
    y = np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(np.random, "choice", lambda _choices: "flip")

    with pytest.raises(ValueError, match="basic 162-dim"):
        augment_data(X, y, CLASSES, augmentation_factor=1)
