from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_registry_modules():
    package = types.ModuleType("tsl_translate")
    package.__path__ = [str(SRC / "tsl_translate")]
    sys.modules["tsl_translate"] = package

    tracks_spec = importlib.util.spec_from_file_location(
        "tsl_translate.tracks",
        SRC / "tsl_translate" / "tracks.py",
    )
    assert tracks_spec is not None and tracks_spec.loader is not None
    tracks_module = importlib.util.module_from_spec(tracks_spec)
    sys.modules["tsl_translate.tracks"] = tracks_module
    tracks_spec.loader.exec_module(tracks_module)

    registry_spec = importlib.util.spec_from_file_location(
        "tsl_translate.registry",
        SRC / "tsl_translate" / "registry.py",
    )
    assert registry_spec is not None and registry_spec.loader is not None
    registry_module = importlib.util.module_from_spec(registry_spec)
    sys.modules["tsl_translate.registry"] = registry_module
    registry_spec.loader.exec_module(registry_module)
    return registry_module.ModelRegistry, tracks_module.TRACKS


ModelRegistry, TRACKS = load_registry_modules()


def write_artifact_stub(path: Path, label_count: int) -> None:
    path.mkdir(parents=True)
    (path / "tsl51_model.keras").write_bytes(b"stub")
    (path / "tsl51_scaler.pkl").write_bytes(b"stub")
    labels = {str(idx): f"label_{idx}" for idx in range(label_count)}
    (path / "tsl51_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False),
        encoding="utf-8",
    )


def write_manifest(path: Path, **values: object) -> None:
    payload = {"track": "tsl51_word_signs", "num_classes": 51}
    payload.update(values)
    (path / "tsl51_model_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_tsl51_registry_prefers_candidate_with_external_holdout_evidence(tmp_path: Path) -> None:
    baseline = tmp_path / ".tools" / "train_runs_baseline" / "artifacts" / "tsl51"
    external = tmp_path / ".tools" / "train_runs_external" / "artifacts" / "tsl51"
    write_artifact_stub(baseline, 51)
    write_artifact_stub(external, 51)
    write_manifest(baseline, test_accuracy=0.92, external_augmented=False)
    write_manifest(external, test_accuracy=0.91, external_augmented=True, external_val_samples=8)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [Path(candidate.name) for candidate in candidates] == [
        Path(".tools") / "train_runs_external" / "artifacts" / "tsl51",
        Path(".tools") / "train_runs_baseline" / "artifacts" / "tsl51",
    ]


def test_tsl51_registry_discovers_and_prefers_reviewed_external_experiment(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / ".tools" / "tsl51_experiments" / "full51_v2" / "artifacts" / "tsl51"
    external = (
        tmp_path
        / ".tools"
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51"
    )
    write_artifact_stub(baseline, 51)
    write_artifact_stub(external, 51)
    write_manifest(baseline, test_accuracy=0.9245283007621765)
    write_manifest(
        external,
        external_augmented=True,
        external_val_samples=0,
        test_accuracy=0.9746835231781006,
    )

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [Path(candidate.name) for candidate in candidates] == [
        Path(".tools")
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51",
        Path(".tools") / "tsl51_experiments" / "full51_v2" / "artifacts" / "tsl51",
    ]


def test_tsl51_registry_keeps_canonical_artifact_candidate_name_platform_agnostic(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "artifacts" / "tsl51"
    write_artifact_stub(valid, 51)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [Path(candidate.name) for candidate in candidates] == [Path("artifacts") / "tsl51"]
