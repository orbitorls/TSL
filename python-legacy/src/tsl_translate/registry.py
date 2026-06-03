from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tsl_translate.tracks import TrackSpec

logger = logging.getLogger(__name__)


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
            experiments = tools / "tsl51_experiments"
            if experiments.exists():
                run_dirs = sorted(
                    (p for p in experiments.glob("*") if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in run_dirs:
                    p = run_dir / "artifacts" / track.key
                    if p.is_dir():
                        dirs.append(p)
            for run_dir in sorted(tools.glob("train_runs_*"), reverse=True):
                for p in run_dir.glob("**/artifacts/*"):
                    if p.is_dir() and p.name == track.key:
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
                if not _labels_match_expected_count(labels, track.expected_num_classes):
                    # Warn but still include — a class-count mismatch means the
                    # artifact was trained with --tsl51-class-mode=observed and
                    # is missing some classes.  It can still be used for inference
                    # on the classes it does have.  The correct fix is to retrain
                    # with --tsl51-class-mode=full so all 51 classes are present.
                    try:
                        raw = json.loads(labels.read_text(encoding="utf-8"))
                        actual = len(raw) if isinstance(raw, (list, dict)) else "?"
                    except Exception:
                        actual = "?"
                    logger.warning(
                        "Artifact %s has %s labels but track '%s' expects %s. "
                        "Retrain with --tsl51-class-mode=full to fix. "
                        "Including this artifact with reduced class coverage.",
                        str(d),
                        actual,
                        track.key,
                        track.expected_num_classes,
                    )
                if not _manifest_is_eligible(manifest, track):
                    continue
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


def _labels_match_expected_count(labels_path: Path, expected_count: int | None) -> bool:
    if expected_count is None:
        return True
    try:
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(raw, list):
        return len(raw) == expected_count
    if isinstance(raw, dict):
        return len(raw) == expected_count
    return False


def _manifest_json(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _manifest_is_eligible(manifest_path: Path, track: TrackSpec) -> bool:
    manifest = _manifest_json(manifest_path)
    if track.key == "tsl51" and manifest.get("external_augmented") is True:
        return int(manifest.get("external_val_samples") or 0) > 0
    return True


def _candidate_rank(candidate: ArtifactCandidate, track: TrackSpec) -> tuple[int, int, float, float]:
    manifest = _manifest_json(candidate.manifest) if candidate.manifest else {}
    external_validated = 0
    clean_rank = 1
    if track.key == "tsl51" and manifest.get("external_augmented") is True:
        try:
            external_val_samples = int(manifest.get("external_val_samples") or 0)
        except (TypeError, ValueError):
            external_val_samples = 0
        external_validated = 1 if external_val_samples > 0 else 0
        clean_rank = 0
    try:
        accuracy = float(manifest.get("test_accuracy") or 0.0)
    except (TypeError, ValueError):
        accuracy = 0.0
    try:
        modified = candidate.model.stat().st_mtime
    except OSError:
        modified = 0.0
    return external_validated, clean_rank, accuracy, modified
