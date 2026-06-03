from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tsl_translate.tracks import TrackSpec


@dataclass
class ArtifactCandidate:
    name: str
    model: Path
    labels: Path
    scaler: Path
    manifest: Path | None


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _discover_dirs(self, track: TrackSpec) -> list[Path]:
        dirs: list[Path] = []
        tools = self.root / ".tools"
        if tools.exists():
            for run_dir in sorted(tools.glob("train_runs_*"), reverse=True):
                for p in run_dir.glob("**/artifacts/*"):
                    if p.is_dir() and p.name == track.key:
                        dirs.append(p)
            if track.key == "tsl51":
                experiments = tools / "tsl51_experiments"
                for p in sorted(experiments.glob("full51_*/artifacts/tsl51"), reverse=True):
                    if p.is_dir():
                        dirs.append(p)
        dirs.append(self.root / "artifacts" / track.key)
        unique: list[Path] = []
        seen: set[str] = set()
        for d in dirs:
            k = str(d.resolve()) if d.exists() else str(d)
            if k not in seen:
                unique.append(d)
                seen.add(k)
        return unique

    def discover(self, track: TrackSpec) -> list[ArtifactCandidate]:
        out: list[ArtifactCandidate] = []
        for d in self._discover_dirs(track):
            model_keras = d / track.default_model
            model_tflite = model_keras.with_suffix(".tflite")
            labels = d / track.default_labels
            scaler = d / track.default_scaler
            manifest = d / track.manifest
            model = model_keras if model_keras.exists() else model_tflite
            if model.exists() and labels.exists() and scaler.exists():
                out.append(
                    ArtifactCandidate(
                        name=str(d.relative_to(self.root)) if d.exists() else str(d),
                        model=model,
                        labels=labels,
                        scaler=scaler,
                        manifest=manifest if manifest.exists() else None,
                    )
                )
        return sorted(out, key=lambda candidate: _candidate_rank(candidate, track), reverse=True)


def discover_candidates(registry: ModelRegistry, track: TrackSpec) -> list[ArtifactCandidate]:
    return registry.discover(track)


def _manifest_json(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _candidate_rank(
    candidate: ArtifactCandidate,
    track: TrackSpec,
) -> tuple[int, int, float, tuple[int, float, float, int], float]:
    manifest = _manifest_json(candidate.manifest) if candidate.manifest else {}
    external_validated = 0
    external_augmented = 0
    if track.key == "tsl51" and manifest.get("external_augmented") is True:
        external_augmented = 1
        try:
            external_val_samples = int(manifest.get("external_val_samples") or 0)
        except (TypeError, ValueError):
            external_val_samples = 0
        external_validated = 1 if external_val_samples > 0 else 0
    try:
        accuracy = float(manifest.get("test_accuracy") or 0.0)
    except (TypeError, ValueError):
        accuracy = 0.0
    reviewed_external = _reviewed_external_rank(candidate, track)
    try:
        modified = candidate.model.stat().st_mtime
    except OSError:
        modified = 0.0
    return external_validated, external_augmented, accuracy, reviewed_external, modified


def _reviewed_external_rank(candidate: ArtifactCandidate, track: TrackSpec) -> tuple[int, float, float, int]:
    if track.key != "tsl51":
        return 0, 0.0, 0.0, 0
    root = _candidate_root(candidate)
    if root is None:
        return 0, 0.0, 0.0, 0
    reports = root / "reports"
    if not reports.exists():
        return 0, 0.0, 0.0, 0

    candidate_name = _normalised_path(candidate.name)
    best = (0, 0.0, 0.0, 0)
    for summary_path in reports.glob("**/summary.json"):
        summary = _manifest_json(summary_path)
        artifact_dir = _normalised_path(str(summary.get("artifact_dir") or ""))
        samples_file = _normalised_path(str(summary.get("samples_file") or ""))
        report_dir = _normalised_path(str(summary_path.parent.relative_to(root)))
        if artifact_dir != candidate_name:
            continue
        if "reviewed_external" not in samples_file and "reviewed_external" not in report_dir:
            continue
        try:
            top1 = float(summary.get("top1_accuracy") or 0.0)
            top3 = float(summary.get("top3_accuracy") or 0.0)
            total_samples = int(summary.get("total_samples") or 0)
        except (TypeError, ValueError):
            continue
        best = max(best, (1, top1, top3, total_samples))
    return best


def _candidate_root(candidate: ArtifactCandidate) -> Path | None:
    name_parts = Path(candidate.name).parts
    if not name_parts:
        return None
    artifact_dir = candidate.model.parent
    parent_index = len(name_parts) - 1
    try:
        return artifact_dir.parents[parent_index]
    except IndexError:
        return None


def _normalised_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()
