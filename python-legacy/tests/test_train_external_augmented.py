from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train_external_augmented.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_external_augmented_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cache(labels: list[str], y: list[int], *, splits: list[str] | None = None) -> dict[str, np.ndarray]:
    data = {
        "X": np.zeros((len(y), 60, 162), dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "class_names": np.asarray(labels, dtype=object),
    }
    if splits is not None:
        data["split"] = np.asarray(splits, dtype=object)
    return data


def test_align_external_labels_remaps_to_base_order() -> None:
    module = load_module()
    external = cache(["b", "a"], [0, 1, 0])

    X, y = module.align_external_labels(external, ["a", "b", "c"])

    assert X.shape == (3, 60, 162)
    assert y.tolist() == [1, 0, 1]


def test_align_external_labels_rejects_unknown_label() -> None:
    module = load_module()
    external = cache(["a", "missing"], [0, 1])

    with pytest.raises(ValueError, match="outside base classes"):
        module.align_external_labels(external, ["a", "b"])


def test_split_external_cache_uses_train_and_val_only() -> None:
    module = load_module()
    external = cache(["a", "b"], [0, 1, 0], splits=["train", "val", "train"])

    split = module.split_external_cache(external, ["a", "b"])

    assert split["train"][0].shape[0] == 2
    assert split["train"][1].tolist() == [0, 0]
    assert split["val"][0].shape[0] == 1
    assert split["val"][1].tolist() == [1]


def test_split_external_cache_rejects_external_test() -> None:
    module = load_module()
    external = cache(["a"], [0], splits=["external_test"])

    with pytest.raises(ValueError, match="external_test"):
        module.split_external_cache(external, ["a"])


def test_split_external_cache_requires_fallback_when_metadata_missing() -> None:
    module = load_module()
    external = cache(["a"], [0])

    with pytest.raises(ValueError, match="missing split metadata"):
        module.split_external_cache(external, ["a"])

    split = module.split_external_cache(external, ["a"], fallback_split="train")
    assert split["train"][1].tolist() == [0]


def test_collect_external_splits_merges_multiple_caches() -> None:
    module = load_module()
    first = cache(["a", "b"], [0], splits=["train"])
    second = cache(["b", "a"], [0, 1], splits=["val", "train"])

    split = module.collect_external_splits([first, second], ["a", "b"])

    assert split["train"][1].tolist() == [0, 0]
    assert split["val"][1].tolist() == [1]


def test_require_external_val_rejects_empty_validation_split() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="external val"):
        module.validate_external_validation_requirement(True, 0)

    module.validate_external_validation_requirement(True, 1)
    module.validate_external_validation_requirement(False, 0)


def test_validate_tsl51_classes_requires_51_labels() -> None:
    module = load_module()

    module.validate_tsl51_classes([f"label_{idx}" for idx in range(51)])
    with pytest.raises(ValueError, match="51 classes"):
        module.validate_tsl51_classes(["a", "b"])


def test_validate_training_args_reuse_requires_base_artifacts() -> None:
    module = load_module()

    module.validate_tsl51_training_args("refit", None)
    with pytest.raises(ValueError, match="requires --base-artifact-dir"):
        module.validate_tsl51_training_args("reuse", None)
