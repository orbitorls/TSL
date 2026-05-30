"""Grouped split manifest utilities for TSL isolated-sign training.

The split unit is a video family, not an individual row. This prevents the
same real-world recording, or its synthetic variants, from appearing across
train/validation/test boundaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
import random
from typing import Any, cast

AUGMENTATION_TOKENS = (
    "blur",
    "brightness",
    "comb",
    "contrast",
    "noise",
    "rotation",
    "scale",
    "original",
)

SPLIT_NAMES = ("train", "val", "test")


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read a value from a dict-like or pandas-record-like row."""
    if isinstance(row, Mapping):
        return row.get(key, default)
    get = getattr(row, "get", None)
    if callable(get):
        return get(key, default)
    return getattr(row, key, default)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _clean_string(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _basename_without_suffix(path_or_id: str) -> str:
    name = Path(path_or_id.replace("\\", "/")).name
    return Path(name).stem if Path(name).suffix else name


def strip_known_augmentation_suffix(value: str) -> str:
    """Strip only known synthetic augmentation tokens from a basename/video id.

    Plain user recording IDs such as ``น้ำ_var_1_10`` are intentionally left
    intact because ``_var_`` identifies a real user-sign variant, not a
    synthetic augmentation.
    """
    base = _basename_without_suffix(value)
    for token in AUGMENTATION_TOKENS:
        marker = f"_{token}"
        idx = base.find(marker)
        if idx > 0:
            return base[:idx]
    return base


def derive_video_family_id(row: Any) -> str:
    """Derive the primary split group key for a metadata row.

    ``video_id`` is authoritative when present. If it is absent, the basename
    of ``landmark_path`` is used. Known synthetic augmentation suffixes are
    stripped from either source.
    """
    source = _clean_string(_row_get(row, "video_id"))
    if not source:
        source = _clean_string(_row_get(row, "landmark_path"))
    if not source:
        source = _clean_string(_row_get(row, "sample_id"))
    if not source:
        raise ValueError("Cannot derive video_family_id without video_id, landmark_path, or sample_id")
    family_id = strip_known_augmentation_suffix(source)
    if not family_id:
        raise ValueError("Derived empty video_family_id")
    return family_id


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _filename_has_augmentation_token(row: Any) -> bool:
    candidates = [_clean_string(_row_get(row, "landmark_path")), _clean_string(_row_get(row, "video_id"))]
    for candidate in candidates:
        if not candidate:
            continue
        base = _basename_without_suffix(candidate)
        for token in AUGMENTATION_TOKENS:
            if f"_{token}" in base:
                return True
    return False


def is_augmented_row(row: Any) -> bool:
    """Return whether a row is synthetic augmentation.

    Authoritative order: an explicit ``is_augmented`` flag wins when present;
    otherwise known filename/video-id augmentation tokens are used. ``_var_``
    is deliberately not an augmentation token.
    """
    explicit = _row_get(row, "is_augmented")
    if not _is_missing(explicit):
        return _coerce_bool(explicit)
    return _filename_has_augmentation_token(row)


def _normalise_row(row: Any, index: int) -> dict[str, Any]:
    family_id = derive_video_family_id(row)
    sample_id = _clean_string(_row_get(row, "sample_id")) or _clean_string(_row_get(row, "video_id"))
    if not sample_id:
        sample_id = _clean_string(_row_get(row, "landmark_path")) or f"row-{index}"
    label = _clean_string(_row_get(row, "sign_clean")) or _clean_string(_row_get(row, "label"))
    if not label:
        raise ValueError(f"Row {index} is missing sign_clean/label metadata")

    return {
        "sample_id": sample_id,
        "video_id": _clean_string(_row_get(row, "video_id")),
        "video_family_id": family_id,
        "label": label,
        "landmark_path": _clean_string(_row_get(row, "landmark_path")),
        "augmented_row": is_augmented_row(row),
    }


def _normalise_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    normalised = [_normalise_row(row, idx) for idx, row in enumerate(rows)]
    if not normalised:
        raise ValueError("Cannot build split manifest from zero rows")
    return normalised


def _stable_group_order(groups: Mapping[str, list[dict[str, Any]]], seed: int) -> list[str]:
    rng = random.Random(seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    return group_ids


def _target_count(total_groups: int, fraction: float) -> int:
    if fraction <= 0 or total_groups <= 0:
        return 0
    count = round(total_groups * fraction)
    return max(1, min(total_groups, count))


def _assign_holdout_groups(
    groups: Mapping[str, list[dict[str, Any]]],
    forced_train: set[str],
    seed: int,
    val_size: float,
    test_size: float,
) -> dict[str, str]:
    available = [group_id for group_id in _stable_group_order(groups, seed) if group_id not in forced_train]
    val_target = _target_count(len(available), val_size)
    remaining_after_val = max(0, len(available) - val_target)
    test_target = _target_count(remaining_after_val, test_size / max(1e-12, 1 - val_size))

    assignments = {group_id: "train" for group_id in groups}
    for group_id in available[:val_target]:
        assignments[group_id] = "val"
    for group_id in available[val_target : val_target + test_target]:
        assignments[group_id] = "test"
    return assignments


def _class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def _build_summary(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {split: _class_counts(list(rows)) for split, rows in splits.items()}


def build_grouped_split_manifest(
    rows: Iterable[Any],
    *,
    dataset_name: str,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, Any]:
    """Build a deterministic train/val/test manifest grouped by video family.

    Families containing augmented rows are assigned wholly to train. This keeps
    synthetic variants train-only without leaking their original family into
    validation or test.
    """
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be in [0, 1)")
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be in [0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    normalised = _normalise_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalised:
        groups[row["video_family_id"]].append(row)

    forced_train = {
        family_id
        for family_id, group_rows in groups.items()
        if any(row["augmented_row"] for row in group_rows)
    }
    assignments = _assign_holdout_groups(groups, forced_train, seed, val_size, test_size)

    splits = {split: [] for split in SPLIT_NAMES}
    for family_id in sorted(groups):
        split = assignments[family_id]
        splits[split].extend(sorted(groups[family_id], key=lambda row: row["sample_id"]))

    for split in SPLIT_NAMES:
        splits[split] = sorted(splits[split], key=lambda row: (row["video_family_id"], row["sample_id"]))

    manifest = {
        "dataset_name": dataset_name,
        "seed": seed,
        "split_strategy": "video_family_holdout",
        "group_key": "video_family_id",
        "total_samples": len(normalised),
        "total_groups": len(groups),
        "sample_counts": {split: len(splits[split]) for split in SPLIT_NAMES},
        "group_counts": {
            split: len({row["video_family_id"] for row in splits[split]}) for split in SPLIT_NAMES
        },
        "class_counts": _build_summary(splits),
        "splits": splits,
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject split manifests that leak families or hold out augmented rows."""
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Manifest missing splits mapping")

    family_to_split: dict[str, str] = {}
    sample_ids: set[str] = set()
    for split in SPLIT_NAMES:
        rows = cast(Iterable[Any], splits.get(split, []))
        for row in rows:
            family_id = _clean_string(_row_get(row, "video_family_id"))
            if not family_id:
                raise ValueError(f"Missing video_family_id in {split} split")
            if split in {"val", "test"} and _coerce_bool(_row_get(row, "augmented_row", False)):
                raise ValueError(f"{split} split contains augmented row {row!r}")
            previous = family_to_split.get(family_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"video_family_id leakage: {family_id!r} appears in {previous} and {split}"
                )
            family_to_split[family_id] = split

            sample_id = _clean_string(_row_get(row, "sample_id"))
            if sample_id:
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
                sample_ids.add(sample_id)


def get_manifest_split_rows(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Return validated manifest rows for a single split.

    This accessor keeps downstream pipeline code anchored to the canonical
    manifest contract instead of reaching into arbitrary split structures.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")

    validate_split_manifest(manifest)
    splits = cast(Mapping[str, Iterable[Any]], manifest["splits"])
    rows = []
    for row in splits.get(split, []):
        if not isinstance(row, Mapping):
            raise ValueError(f"Manifest row for split {split!r} must be mapping-like")
        rows.append(dict(row))
    return rows


__all__ = [
    "AUGMENTATION_TOKENS",
    "build_grouped_split_manifest",
    "derive_video_family_id",
    "get_manifest_split_rows",
    "is_augmented_row",
    "strip_known_augmentation_suffix",
    "validate_split_manifest",
]

