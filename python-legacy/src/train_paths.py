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
FS_DYNAMIC_CLIP_DIR_ALIASES = (
    "fingerspelling_dynamic",
    "fingerspelling_clips",
    "fingerspelling_dynamic_local",
    "fingerspelling_dynamic_clips",
)
FS_DYNAMIC_MANIFEST_ALIASES = (
    "fingerspelling_dynamic_manifest.csv",
    "manifest.csv",
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


def resolve_fsd_dataset_root(repo_root: Path, requested: Path) -> Path:
    if requested.is_dir():
        return requested
    for name in FS_DYNAMIC_CLIP_DIR_ALIASES:
        candidate = repo_root / "data" / name
        if candidate.is_dir():
            print(
                f"[fsd] --fs-dynamic-dataset-root not found ({requested}); using {candidate}"
            )
            return candidate
    return requested


def discover_fsd_manifest(repo_root: Path, dataset_root: Path) -> Path:
    """Find the first existing manifest CSV inside the dataset dir or its direct subfolders.

    Returns the matched path (which may or may not exist on disk — callers MUST
    ``is_file()``-check before reading).  When the dataset directory itself does
    not exist, the function still returns a candidate path so the caller can
    raise a clear ``FileNotFoundError``.
    """
    del repo_root  # signature kept for parity with discover_fs_zip
    for name in FS_DYNAMIC_MANIFEST_ALIASES:
        candidate = dataset_root / name
        if candidate.is_file():
            print(f"[fsd] using manifest: {candidate}")
            return candidate
    if dataset_root.is_dir():
        for sub in sorted(dataset_root.iterdir()):
            if not sub.is_dir():
                continue
            for name in FS_DYNAMIC_MANIFEST_ALIASES:
                candidate = sub / name
                if candidate.is_file():
                    print(f"[fsd] using manifest: {candidate}")
                    return candidate
    return dataset_root / FS_DYNAMIC_MANIFEST_ALIASES[0]


def discover_fs_zip(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    for name in FS_ZIP_ALIASES:
        candidate = repo_root / "data" / name
        if candidate.is_file():
            print(f"[fs] using dataset zip: {candidate}")
            return candidate
    return explicit
