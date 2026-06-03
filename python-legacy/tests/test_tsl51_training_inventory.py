from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA = REPO_ROOT / "data" / "tsl51_raw_local" / "metadata" / "combined_metadata.csv"
ARTIFACT_LABELS = REPO_ROOT / "artifacts" / "tsl51" / "tsl51_labels.json"
EXPECTED_MISSING_FROM_CURRENT_ARTIFACT = {
    "เกิด_var_2",
    "ตลาด_var_4",
    "ขอบคุณ_เปิดมือสองข้าง",
    "ด้วยกัน_var_1",
}


def load_metadata_sign_ids(path: Path = METADATA) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (row.get("sign_id") or "").strip()
            for row in rows
            if (row.get("sign_id") or "").strip() and (row.get("sign_id") or "").strip() != "null_act"
        }


def load_artifact_labels(path: Path = ARTIFACT_LABELS) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {str(raw[str(i)]) for i in range(len(raw))}
    return {str(value) for value in raw}


def test_tsl51_metadata_has_51_non_null_classes() -> None:
    assert len(load_metadata_sign_ids()) == 51


def test_current_artifact_is_missing_expected_metadata_classes() -> None:
    metadata_labels = load_metadata_sign_ids()
    artifact_labels = load_artifact_labels()

    missing = metadata_labels - artifact_labels

    assert len(artifact_labels) == 47
    assert missing == EXPECTED_MISSING_FROM_CURRENT_ARTIFACT




def load_train_local_all(module_name: str = "train_local_all_for_test"):
    import importlib.util
    import sys

    module_path = REPO_ROOT / "python-legacy" / "scripts" / "train_local_all.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_balanced_tsl51_sample_limit_keeps_all_classes() -> None:
    train_local_all = load_train_local_all("train_local_all_balanced_test")

    rows = []
    for label in ("a", "b", "c"):
        for idx in range(3):
            rows.append({"sign_id": label, "landmark_path": f"{label}_{idx}.csv"})

    limited = train_local_all._limit_tsl51_rows_balanced(rows, max_samples=6)

    assert {row["sign_id"] for row in limited} == {"a", "b", "c"}
    assert len(limited) == 6



def test_balanced_tsl51_sample_limit_prefers_user_sign_landmarks() -> None:
    train_local_all = load_train_local_all("train_local_all_priority_test")

    rows = [
        {"sign_id": "a", "landmark_path": "landmarks/expert_scraped/a.csv"},
        {"sign_id": "a", "landmark_path": "landmarks/expert_primary_01/a.csv"},
        {"sign_id": "a", "landmark_path": "landmarks/user_sign/a.csv"},
        {"sign_id": "b", "landmark_path": "landmarks/expert_primary_01/b.csv"},
        {"sign_id": "b", "landmark_path": "landmarks/user_sign/b.csv"},
    ]

    limited = train_local_all._limit_tsl51_rows_balanced(rows, max_samples=2)

    assert [row["landmark_path"] for row in limited] == [
        "landmarks/user_sign/a.csv",
        "landmarks/user_sign/b.csv",
    ]


def test_full_tsl51_class_mode_rejects_missing_classes() -> None:
    train_local_all = load_train_local_all("train_local_all_full_test")

    with pytest.raises(RuntimeError, match="missing TSL51 classes"):
        train_local_all._validate_full_tsl51_classes(
            expected_classes={"a", "b", "c"},
            available_classes={"a", "b"},
        )
