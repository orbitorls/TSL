from __future__ import annotations

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
        return out


def discover_candidates(registry: ModelRegistry, track: TrackSpec) -> list[ArtifactCandidate]:
    return registry.discover(track)
