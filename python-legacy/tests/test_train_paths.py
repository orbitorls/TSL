"""Dataset path discovery for train_local_all."""

from __future__ import annotations

from pathlib import Path

from src.train_paths import discover_fs_zip, resolve_fs_dataset_root, resolve_tsl51_metadata_dir


def test_resolve_fs_dataset_root_falls_back_to_ready(tmp_path: Path) -> None:
    data = tmp_path / "data"
    ready = data / "one_stage_tfs_ready"
    ready.mkdir(parents=True)
    requested = data / "fingerspelling"
    assert resolve_fs_dataset_root(tmp_path, requested) == ready


def test_discover_fs_zip_from_data_dir(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    zip_path = data / "one_stage_tfs_fresh.zip"
    zip_path.write_bytes(b"PK\x05\x06")
    assert discover_fs_zip(tmp_path, None) == zip_path


def test_resolve_tsl51_metadata_dir_alias(tmp_path: Path) -> None:
    data = tmp_path / "data"
    local_meta = data / "tsl51_raw_local" / "metadata"
    local_meta.mkdir(parents=True)
    requested = data / "tsl51" / "metadata"
    assert resolve_tsl51_metadata_dir(tmp_path, requested) == local_meta
