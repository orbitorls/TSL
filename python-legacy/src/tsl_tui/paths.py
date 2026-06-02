"""Repo paths, dataset discovery (train_paths), and artifact status."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LEGACY_ROOT.parent
CONFIG_PATH = LEGACY_ROOT / "config.local.json"
CONFIG_EXAMPLE_PATH = LEGACY_ROOT / "config.local.json.example"
TRAIN_SCRIPT = LEGACY_ROOT / "scripts" / "train_local_all.py"
INGEST_SCRIPT = LEGACY_ROOT / "scripts" / "ingest_local.py"

if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from src.train_paths import (  # noqa: E402
    discover_fs_zip,
    resolve_fs_dataset_root,
    resolve_tsl51_metadata_dir,
)


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        release = platform.uname().release.lower()
    except Exception:
        return False
    return "microsoft" in release or "wsl" in release


def is_windows_native() -> bool:
    return sys.platform == "win32" and not is_wsl()


def _expand_user(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def load_local_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    import json

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def default_fs_dataset_root() -> Path:
    cfg = load_local_config()
    requested = Path(
        cfg.get("fs_dataset_root", REPO_ROOT / "data" / "fingerspelling")
    )
    return resolve_fs_dataset_root(REPO_ROOT, _expand_user(requested))


def default_tsl51_metadata_dir() -> Path:
    cfg = load_local_config()
    requested = Path(
        cfg.get("tsl51_metadata_dir", REPO_ROOT / "data" / "tsl51" / "metadata")
    )
    return resolve_tsl51_metadata_dir(REPO_ROOT, _expand_user(requested))


def default_fs_zip() -> Path | None:
    cfg = load_local_config()
    explicit = cfg.get("fs_zip")
    exp_path = _expand_user(explicit) if explicit else None
    return discover_fs_zip(REPO_ROOT, exp_path)


def default_artifact_dir() -> Path:
    cfg = load_local_config()
    raw = cfg.get("artifact_dir", REPO_ROOT / "artifacts")
    return _expand_user(raw)


def artifact_track_ready(artifact_dir: Path, track: str) -> bool:
    sub = artifact_dir / track
    if not sub.is_dir():
        return False
    if track == "fingerspelling":
        return (sub / "model.keras").is_file() or (sub / "labels.json").is_file()
    return (sub / "tsl51_model.keras").is_file() or (sub / "tsl51_labels.json").is_file()


@dataclass(frozen=True)
class EnvironmentStatus:
    repo_root: Path
    legacy_root: Path
    platform_label: str
    windows_native: bool
    wsl: bool
    config_exists: bool
    fs_dataset: Path
    fs_dataset_ok: bool
    fs_zip: Path | None
    fs_zip_ok: bool
    tsl51_metadata: Path
    tsl51_metadata_ok: bool
    artifact_dir: Path
    fs_artifacts_ok: bool
    tsl51_artifacts_ok: bool
    wsl_hint: str

    def lines(self) -> list[str]:
        def mark(ok: bool) -> str:
            return "[green]OK[/]" if ok else "[yellow]missing[/]"

        rows = [
            f"Repo root:     {self.repo_root}",
            f"Legacy root:   {self.legacy_root}",
            f"Platform:      {self.platform_label}",
            f"Config:        {mark(self.config_exists)} {CONFIG_PATH.name}",
            f"FS dataset:    {mark(self.fs_dataset_ok)} {self.fs_dataset}",
            f"FS zip:        {mark(self.fs_zip_ok)} {self.fs_zip or '(none)'}",
            f"TSL-51 meta:   {mark(self.tsl51_metadata_ok)} {self.tsl51_metadata}",
            f"Artifacts:     {self.artifact_dir}",
            f"  fingerspelling {mark(self.fs_artifacts_ok)}",
            f"  tsl51          {mark(self.tsl51_artifacts_ok)}",
        ]
        if self.windows_native:
            rows.append("")
            rows.append("[red]Windows native:[/] TensorFlow GPU training is not supported here.")
            rows.append(self.wsl_hint)
        return rows


def gather_status() -> EnvironmentStatus:
    fs = default_fs_dataset_root()
    tsl51 = default_tsl51_metadata_dir()
    z = default_fs_zip()
    art = default_artifact_dir()
    plat = platform.system()
    if is_wsl():
        plat = f"WSL ({plat})"
    elif is_windows_native():
        plat = "Windows (native)"

    wsl_repo = str(REPO_ROOT).replace("\\", "/")
    if len(wsl_repo) >= 2 and wsl_repo[1] == ":":
        drive = wsl_repo[0].lower()
        wsl_repo = f"/mnt/{drive}{wsl_repo[2:]}"

    wsl_hint = (
        f"Use WSL:  wsl bash -lc 'cd {wsl_repo}/python-legacy && "
        f"source ~/venvs/tsl/bin/activate && python -m tsl_tui'"
    )

    return EnvironmentStatus(
        repo_root=REPO_ROOT,
        legacy_root=LEGACY_ROOT,
        platform_label=plat,
        windows_native=is_windows_native(),
        wsl=is_wsl(),
        config_exists=CONFIG_PATH.is_file(),
        fs_dataset=fs,
        fs_dataset_ok=fs.is_dir(),
        fs_zip=z,
        fs_zip_ok=z is not None and z.is_file(),
        tsl51_metadata=tsl51,
        tsl51_metadata_ok=tsl51.is_dir(),
        artifact_dir=art,
        fs_artifacts_ok=artifact_track_ready(art, "fingerspelling"),
        tsl51_artifacts_ok=artifact_track_ready(art, "tsl51"),
        wsl_hint=wsl_hint,
    )


def config_to_argv(extra: list[str] | None = None) -> list[str]:
    """Map config.local.json keys to train_local_all.py CLI flags."""
    argv: list[str] = []
    for key, value in load_local_config().items():
        if key.startswith("_") or value is None:
            continue
        argv.append(f"--{key.replace('_', '-')}")
        argv.append(str(value))
    if extra:
        argv.extend(extra)
    return argv
