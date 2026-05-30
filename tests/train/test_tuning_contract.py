from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from src.core.features import FEATURE_SCHEMA_VERSION
from src.train.config import (
    TASK9_BASELINE_MACRO_F1,
    TRUSTED_TUNING_MIN_DELTA,
    TuningCandidate,
    build_trusted_tuning_report,
    build_trusted_tuning_run_configs,
    select_trusted_tuning_result,
)


def _mapping(value: object) -> Mapping[str, Any]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, Any], value)


def _candidate_runs(report: Mapping[str, object]) -> list[Mapping[str, Any]]:
    runs = report["candidate_runs"]
    assert isinstance(runs, list)
    return [cast(Mapping[str, Any], item) for item in runs]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _baseline_metrics(tmp_path: Path) -> dict[str, object]:
    split_manifest_path = _write_json(
        tmp_path / "split_manifest.json",
        {
            "dataset_name": "tsl51_user_sign",
            "seed": 42,
            "split_strategy": "video_family_holdout",
            "group_key": "video_family_id",
            "sample_counts": {"train": 383, "val": 82, "test": 82},
            "group_counts": {"train": 383, "val": 82, "test": 82},
            "splits": {"train": [], "val": [], "test": []},
        },
    )
    preprocessing_manifest_path = _write_json(
        tmp_path / "preprocessing_manifest.json",
        {
            "dataset": "tsl51_user_sign",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_level": "basic",
            "target_frames": 30,
            "seq_mode": True,
        },
    )
    return {
        "dataset": "tsl51_user_sign",
        "model": "gru",
        "seed": 42,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_level": "basic",
        "primary_metric_name": "macro_f1",
        "primary_metric": TASK9_BASELINE_MACRO_F1,
        "metrics": {"macro_f1": TASK9_BASELINE_MACRO_F1},
        "split_manifest_path": str(split_manifest_path),
        "preprocessing_manifest_path": str(preprocessing_manifest_path),
    }


def _candidate(run_config: dict[str, object]) -> TuningCandidate:
    candidate = run_config["candidate"]
    assert isinstance(candidate, TuningCandidate)
    return candidate


def _result(run_config: dict[str, object], macro_f1: float) -> dict[str, object]:
    return {
        "candidate_name": _candidate(run_config).name,
        "primary_metric_name": "macro_f1",
        "primary_metric": macro_f1,
        "metrics": {
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1 - 1.0,
            "accuracy": macro_f1 + 5.0,
            "precision": macro_f1 - 2.0,
            "recall": macro_f1 - 3.0,
            "top3_accuracy": min(100.0, macro_f1 + 20.0),
            "top5_accuracy": min(100.0, macro_f1 + 25.0),
        },
        "artifacts": {
            "checkpoint": f"artifacts/runs/task10/{_candidate(run_config).name}.pt",
            "metrics": f"artifacts/runs/task10/{_candidate(run_config).name}/metrics.json",
            "split_manifest": run_config["split_manifest_path"],
            "preprocessing_manifest": run_config["preprocessing_manifest_path"],
            "label_map": f"artifacts/runs/task10/{_candidate(run_config).name}/label_map.json",
        },
        "split_manifest_hash": run_config["split_manifest_hash"],
        "feature_schema_version": run_config["feature_schema_version"],
        "feature_level": run_config["feature_level"],
        "preprocessing_schema": run_config["preprocessing_schema"],
    }


def test_controlled_tuning_candidates_are_fixed_manual_and_share_trusted_metadata(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)

    run_configs = build_trusted_tuning_run_configs(baseline)

    assert [_candidate(item).name for item in run_configs] == [
        "mlp_basic_baseline",
        "gru_sequence",
        "gru_regularized",
    ]
    assert [_candidate(item).config.model for item in run_configs] == ["mlp", "gru", "gru"]
    assert [_candidate(item).complexity_rank for item in run_configs] == [0, 1, 2]
    assert all(item["primary_metric_name"] == "macro_f1" for item in run_configs)
    assert all(item["split_strategy"] == "video_family_holdout" for item in run_configs)
    assert all(item["feature_schema_version"] == FEATURE_SCHEMA_VERSION for item in run_configs)
    assert all(item["feature_level"] == "basic" for item in run_configs)

    split_hashes = {item["split_manifest_hash"] for item in run_configs}
    preprocessing_schemas = {item["preprocessing_schema"] for item in run_configs}
    assert len(split_hashes) == 1
    assert len(preprocessing_schemas) == 1
    assert next(iter(split_hashes))
    assert next(iter(preprocessing_schemas)) == "basic-162-v1|basic|target_frames=30|seq_mode=True"


def test_complex_candidate_must_clear_two_point_macro_f1_gate(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)
    run_configs = build_trusted_tuning_run_configs(baseline)
    just_below_gate = TASK9_BASELINE_MACRO_F1 + TRUSTED_TUNING_MIN_DELTA - 0.01
    results = [
        _result(run_configs[0], TASK9_BASELINE_MACRO_F1 - 1.0),
        _result(run_configs[1], just_below_gate),
        _result(run_configs[2], just_below_gate),
    ]

    selection = select_trusted_tuning_result(results, baseline)

    assert selection.selected_name == "task9_baseline"
    assert selection.accepted_tuned_candidate is False
    assert selection.selected_macro_f1 == pytest.approx(TASK9_BASELINE_MACRO_F1)
    assert "did not clear" in selection.rationale


def test_candidate_at_two_point_macro_f1_gate_is_selected(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)
    run_configs = build_trusted_tuning_run_configs(baseline)
    threshold = TASK9_BASELINE_MACRO_F1 + TRUSTED_TUNING_MIN_DELTA
    results = [
        _result(run_configs[0], TASK9_BASELINE_MACRO_F1 + 0.5),
        _result(run_configs[1], threshold),
        _result(run_configs[2], threshold - 0.1),
    ]

    selection = select_trusted_tuning_result(results, baseline)

    assert selection.selected_name == "gru_sequence"
    assert selection.accepted_tuned_candidate is True
    assert selection.selected_macro_f1 == pytest.approx(threshold)
    assert ">= 2.00" in selection.rationale


def test_candidate_results_must_use_same_split_and_preprocessing_contract(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)
    run_configs = build_trusted_tuning_run_configs(baseline)
    results = [
        _result(run_configs[0], TASK9_BASELINE_MACRO_F1 + 3.0),
        _result(run_configs[1], TASK9_BASELINE_MACRO_F1 + 4.0),
        _result(run_configs[2], TASK9_BASELINE_MACRO_F1 + 5.0),
    ]
    results[2]["split_manifest_hash"] = "different-hash"

    with pytest.raises(ValueError, match="same split manifest hash"):
        select_trusted_tuning_result(results, baseline)

    results[2]["split_manifest_hash"] = results[0]["split_manifest_hash"]
    results[1]["preprocessing_schema"] = "basic-162-v1|basic|target_frames=15|seq_mode=True"

    with pytest.raises(ValueError, match="same preprocessing schema"):
        select_trusted_tuning_result(results, baseline)


def test_trusted_tuning_report_records_each_candidate_run_detail(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)
    run_configs = build_trusted_tuning_run_configs(baseline)
    results = [
        _result(run_configs[0], TASK9_BASELINE_MACRO_F1 + 0.5),
        _result(run_configs[1], TASK9_BASELINE_MACRO_F1 + 1.0),
        _result(run_configs[2], TASK9_BASELINE_MACRO_F1 + 1.5),
    ]

    report = build_trusted_tuning_report(run_configs, results, baseline)

    assert report["primary_metric_name"] == "macro_f1"
    baseline_payload = _mapping(report["baseline"])
    selection_payload = _mapping(report["selection"])
    candidate_runs = _candidate_runs(report)

    assert baseline_payload["name"] == "task9_baseline"
    assert selection_payload["selected_name"] == "task9_baseline"
    assert selection_payload["accepted_tuned_candidate"] is False
    assert "did not clear" in str(selection_payload["rationale"])
    assert [entry["candidate_name"] for entry in candidate_runs] == [
        "mlp_basic_baseline",
        "gru_sequence",
        "gru_regularized",
    ]

    split_hashes = {entry["split_manifest_hash"] for entry in candidate_runs}
    preprocessing_schemas = {entry["preprocessing_schema"] for entry in candidate_runs}
    feature_schemas = {entry["feature_schema_version"] for entry in candidate_runs}
    assert len(split_hashes) == 1
    assert preprocessing_schemas == {"basic-162-v1|basic|target_frames=30|seq_mode=True"}
    assert feature_schemas == {FEATURE_SCHEMA_VERSION}

    for entry in candidate_runs:
        command = str(entry["command"])
        config_payload = _mapping(entry["config"])
        grouped_macro_f1 = float(entry["grouped_macro_f1"])
        secondary_metrics = _mapping(entry["secondary_metrics"])
        artifacts = _mapping(entry["artifacts"])

        assert command.startswith("python -m src.cli.train ")
        assert "--split-strategy video_family_grouped" in command
        assert "--primary-metric macro_f1" in command
        assert "--gradient-clip-value" not in command
        assert config_payload["model"] in {"mlp", "gru"}
        assert entry["seed"] == 42
        assert entry["primary_metric_name"] == "macro_f1"
        assert isinstance(entry["grouped_macro_f1"], float)
        assert entry["feature_level"] == "basic"
        assert secondary_metrics == {
            "weighted_f1": pytest.approx(grouped_macro_f1 - 1.0),
            "accuracy": pytest.approx(grouped_macro_f1 + 5.0),
            "precision": pytest.approx(grouped_macro_f1 - 2.0),
            "recall": pytest.approx(grouped_macro_f1 - 3.0),
            "top3_accuracy": pytest.approx(min(100.0, grouped_macro_f1 + 20.0)),
            "top5_accuracy": pytest.approx(min(100.0, grouped_macro_f1 + 25.0)),
        }
        assert set(artifacts) == {
            "checkpoint",
            "metrics",
            "split_manifest",
            "preprocessing_manifest",
            "label_map",
        }
        assert all(artifacts.values())

    regularized = next(
        entry for entry in candidate_runs if entry["candidate_name"] == "gru_regularized"
    )
    regularized_config = _mapping(regularized["config"])
    assert regularized_config["gradient_clip_value"] == 1.0


def test_trusted_tuning_report_selects_candidate_only_after_two_point_gate(tmp_path: Path) -> None:
    baseline = _baseline_metrics(tmp_path)
    run_configs = build_trusted_tuning_run_configs(baseline)
    threshold = TASK9_BASELINE_MACRO_F1 + TRUSTED_TUNING_MIN_DELTA
    results = [
        _result(run_configs[0], TASK9_BASELINE_MACRO_F1 + 0.5),
        _result(run_configs[1], threshold),
        _result(run_configs[2], threshold - 0.1),
    ]

    report = build_trusted_tuning_report(run_configs, results, baseline)

    selection_payload = _mapping(report["selection"])

    assert selection_payload["selected_name"] == "gru_sequence"
    assert selection_payload["accepted_tuned_candidate"] is True
    assert selection_payload["selected_macro_f1"] == pytest.approx(threshold)
    assert selection_payload["required_macro_f1"] == pytest.approx(threshold)
    assert ">= 2.00" in str(selection_payload["rationale"])


def test_tuning_contract_does_not_introduce_hpo_dependency_or_state() -> None:
    source = Path("src/train/config.py").read_text(encoding="utf-8").lower()
    forbidden_tokens = ("optuna", "mlflow", "ray.tune", "hyperopt", "trial.suggest", "study_name")
    assert all(token not in source for token in forbidden_tokens)

    tree = ast.parse(Path("src/train/config.py").read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported_roots.isdisjoint({"optuna", "mlflow", "ray", "hyperopt"})
