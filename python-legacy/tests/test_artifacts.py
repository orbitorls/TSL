from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.artifacts import ensure_required_files, load_labels


def _workspace_tmp(name: str) -> Path:
    path = Path("test_tmp_manual") / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_ensure_required_files_reports_all_missing_paths() -> None:
    tmp_path = _workspace_tmp("missing_paths")
    existing = tmp_path / "labels.json"
    existing.write_text("{}", encoding="utf-8")
    missing_model = tmp_path / "model.keras"
    missing_scaler = tmp_path / "scaler.pkl"

    with pytest.raises(FileNotFoundError) as exc:
        ensure_required_files(
            {
                "model": missing_model,
                "labels": existing,
                "scaler": missing_scaler,
            },
            hint="train first",
        )

    message = str(exc.value)
    assert "model: " in message
    assert "scaler: " in message
    assert str(missing_model) in message
    assert str(missing_scaler) in message
    assert "train first" in message


def test_load_labels_accepts_list_and_dict() -> None:
    tmp_path = _workspace_tmp("labels_valid")
    list_path = tmp_path / "labels_list.json"
    list_path.write_text(json.dumps(["ก", "ข"], ensure_ascii=False), encoding="utf-8")
    assert load_labels(list_path) == {"0": "ก", "1": "ข"}

    dict_path = tmp_path / "labels_dict.json"
    dict_path.write_text(
        json.dumps({"0": "ก", "1": "ข"}, ensure_ascii=False), encoding="utf-8"
    )
    assert load_labels(dict_path) == {"0": "ก", "1": "ข"}


def test_load_labels_rejects_invalid_shape() -> None:
    tmp_path = _workspace_tmp("labels_invalid")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"zero": "ก"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="numeric string keys"):
        load_labels(labels_path)
