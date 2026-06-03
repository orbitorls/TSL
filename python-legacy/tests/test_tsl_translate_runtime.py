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


def write_eval_summary(path: Path, artifact_dir: Path, samples_file: str, **values: object) -> None:
    path.mkdir(parents=True)
    payload = {
        "samples_file": samples_file,
        "artifact_dir": str(artifact_dir),
        "total_samples": 5,
        "top1_accuracy": 1.0,
        "top3_accuracy": 1.0,
    }
    payload.update(values)
    (path / "summary.json").write_text(
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



def test_tsl51_registry_prefers_reviewed_external_evidence_before_mtime(
    tmp_path: Path,
) -> None:
    proven = (
        tmp_path
        / ".tools"
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51"
    )
    newer = (
        tmp_path
        / ".tools"
        / "tsl51_experiments"
        / "full51_v4_prelabel_weighted"
        / "artifacts"
        / "tsl51"
    )
    write_artifact_stub(proven, 51)
    write_artifact_stub(newer, 51)
    write_manifest(proven, external_augmented=True, external_val_samples=0, test_accuracy=0.9746835231781006)
    write_manifest(newer, external_augmented=True, external_val_samples=0, test_accuracy=0.9746835231781006)
    write_eval_summary(
        tmp_path
        / "reports"
        / "current_reviewed_external_eval"
        / "full51_v3_external_weighted_recheck",
        Path(".tools")
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51",
        str(Path("work") / "reviewed_external_assets" / "external_test_samples.csv"),
        total_samples=5,
        top1_accuracy=1.0,
        top3_accuracy=1.0,
    )

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [Path(candidate.name) for candidate in candidates] == [
        Path(".tools")
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51",
        Path(".tools")
        / "tsl51_experiments"
        / "full51_v4_prelabel_weighted"
        / "artifacts"
        / "tsl51",
    ]


def test_tsl51_registry_prefers_webcam_finetuned_model_after_finetune(
    tmp_path: Path,
) -> None:
    """After webcam fine-tune, v4_webcam_seed (external_val_samples>0) must
    automatically outrank v3 even if v3 has higher internal accuracy.
    This guards against regressions that would cause the runtime to fall back
    to an unvalidated studio model after the user records their webcam data."""
    v3 = (
        tmp_path
        / ".tools"
        / "tsl51_experiments"
        / "full51_v3_external_weighted"
        / "artifacts"
        / "tsl51"
    )
    v4_webcam = (
        tmp_path
        / ".tools"
        / "tsl51_experiments"
        / "full51_v4_webcam_seed"
        / "artifacts"
        / "tsl51"
    )
    write_artifact_stub(v3, 51)
    write_artifact_stub(v4_webcam, 51)
    # v3: higher internal accuracy but no real webcam validation
    write_manifest(v3, external_augmented=True, external_val_samples=0, test_accuracy=0.9747)
    # v4_webcam: lower internal accuracy but validated on real webcam holdout
    write_manifest(v4_webcam, external_augmented=True, external_val_samples=12, test_accuracy=0.8500)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [Path(candidate.name) for candidate in candidates] == [
        Path(".tools") / "tsl51_experiments" / "full51_v4_webcam_seed" / "artifacts" / "tsl51",
        Path(".tools") / "tsl51_experiments" / "full51_v3_external_weighted" / "artifacts" / "tsl51",
    ], (
        "full51_v4_webcam_seed must rank first because it has external_val_samples>0 "
        "(validated on real webcam holdout), even though v3 has higher internal accuracy. "
        "If this fails, the runtime will silently fall back to a studio-domain model "
        "after the user records webcam data."
    )
