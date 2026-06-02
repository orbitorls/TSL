"""Helpers for loading and validating training artifacts used by webcam demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def ensure_required_files(paths: Mapping[str, str | Path], hint: str = "") -> None:
    """Raise a readable error listing every missing required artifact."""
    missing = [(name, Path(path)) for name, path in paths.items() if not Path(path).exists()]
    if not missing:
        return

    lines = ["Required artifact file(s) not found:"]
    lines.extend(f"  - {name}: {path}" for name, path in missing)
    if hint:
        lines.append(hint)
    raise FileNotFoundError("\n".join(lines))


def load_labels(labels_path: str | Path) -> dict[str, str]:
    """Load labels as a ``{\"0\": label}`` mapping, accepting list or dict JSON."""
    path = Path(labels_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        labels = {str(i): str(value) for i, value in enumerate(raw)}
    elif isinstance(raw, dict):
        labels = {str(key): str(value) for key, value in raw.items()}
    else:
        raise ValueError(f"{path} must contain a list or object of labels")

    expected = {str(i) for i in range(len(labels))}
    actual = set(labels)
    if actual != expected:
        raise ValueError(
            f"{path} must use contiguous numeric string keys 0..{len(labels) - 1}"
        )
    return labels
