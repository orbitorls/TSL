"""Dataset path discovery for local training (shared by train_local_all.py)."""

from __future__ import annotations

from pathlib import Path

FS_DATASET_DIR_ALIASES = (
    "fingerspelling",
    "one_stage_tfs_ready",
    "one_stage_tfs_local",
    "one_stage_tfs_local_v2",
    "one_stage_tfs",
)
FS_ZIP_ALIASES = ("one_stage_tfs_fresh.zip", "one_stage_tfs_local.zip")
TSL51_METADATA_DIR_ALIASES = (
    ("tsl51", "metadata"),
    ("tsl51_raw_local", "metadata"),
    ("tsl51_raw", "metadata"),
)


def resolve_fs_dataset_root(repo_root: Path, requested: Path) -> Path:
    if requested.is_dir():
        return requested
    for name in FS_DATASET_DIR_ALIASES:
        candidate = repo_root / "data" / name
        if candidate.is_dir():
            print(f"[fs] --fs-dataset-root not found ({requested}); using {candidate}")
            return candidate
    return requested


def resolve_tsl51_metadata_dir(repo_root: Path, requested: Path) -> Path:
    if requested.is_dir():
        return requested
    for parts in TSL51_METADATA_DIR_ALIASES:
        candidate = repo_root / "data" / Path(*parts)
        if candidate.is_dir():
            print(
                f"[tsl51] --tsl51-metadata-dir not found ({requested}); using {candidate}"
            )
            return candidate
    return requested


def discover_fs_zip(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    for name in FS_ZIP_ALIASES:
        candidate = repo_root / "data" / name
        if candidate.is_file():
            print(f"[fs] using dataset zip: {candidate}")
            return candidate
    return explicit
