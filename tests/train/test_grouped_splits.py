"""Tests for grouped train/val/test split manifests."""

import pytest

from src.train.splits import (
    build_grouped_split_manifest,
    derive_video_family_id,
    is_augmented_row,
    validate_split_manifest,
)


def _rows():
    rows = []
    classes = ["hello", "thanks", "water"]
    for class_idx, label in enumerate(classes):
        for family_idx in range(4):
            video_id = f"vid_{class_idx}_{family_idx}"
            rows.append(
                {
                    "sample_id": f"sample-{video_id}",
                    "video_id": video_id,
                    "landmark_path": f"landmarks/{video_id}.csv",
                    "sign_clean": label,
                    "is_augmented": False,
                }
            )
    rows.extend(
        [
            {
                "sample_id": "aug-from-flag",
                "video_id": "vid_0_0_noise_k1",
                "landmark_path": "landmarks/vid_0_0_noise_k1.csv",
                "sign_clean": "hello",
                "is_augmented": True,
            },
            {
                "sample_id": "aug-from-path",
                "landmark_path": "landmarks/vid_1_0_brightness_1.2x.csv",
                "sign_clean": "thanks",
            },
        ]
    )
    return rows


def _families(manifest, split):
    return {row["video_family_id"] for row in manifest["splits"][split]}


def test_video_family_prefers_video_id_and_strips_known_augmentation_suffixes():
    assert derive_video_family_id(
        {
            "video_id": "vid_0193_noise_k1",
            "landmark_path": "landmarks/ignored.csv",
            "sign_clean": "ignored",
        }
    ) == "vid_0193"
    assert derive_video_family_id(
        {"landmark_path": "landmarks/vid_0193_brightness_1.2x.csv", "sign_clean": "ignored"}
    ) == "vid_0193"
    assert derive_video_family_id(
        {"landmark_path": "landmarks/น้ำ_var_1_10.csv", "sign_clean": "น้ำ"}
    ) == "น้ำ_var_1_10"


def test_augmented_detection_prefers_explicit_flag_and_uses_conservative_filename_fallback():
    assert is_augmented_row({"is_augmented": True, "landmark_path": "landmarks/plain.csv"}) is True
    assert is_augmented_row({"is_augmented": False, "landmark_path": "landmarks/vid_1_noise_k1.csv"}) is False
    assert is_augmented_row({"landmark_path": "landmarks/vid_1_rotation_10.csv"}) is True
    assert is_augmented_row({"landmark_path": "landmarks/น้ำ_var_1_10.csv"}) is False


def test_grouped_split_manifest_is_deterministic_and_group_disjoint():
    first = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=7, val_size=0.25, test_size=0.25
    )
    second = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=7, val_size=0.25, test_size=0.25
    )

    assert first == second
    assert first["dataset_name"] == "unit"
    assert first["seed"] == 7
    assert first["split_strategy"] == "video_family_holdout"
    assert first["group_key"] == "video_family_id"

    train_families = _families(first, "train")
    val_families = _families(first, "val")
    test_families = _families(first, "test")
    assert train_families.isdisjoint(val_families)
    assert train_families.isdisjoint(test_families)
    assert val_families.isdisjoint(test_families)

    validate_split_manifest(first)


def test_validation_and_test_splits_exclude_augmented_rows():
    manifest = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=3, val_size=0.25, test_size=0.25
    )

    assert any(row["augmented_row"] for row in manifest["splits"]["train"])
    assert all(not row["augmented_row"] for row in manifest["splits"]["val"])
    assert all(not row["augmented_row"] for row in manifest["splits"]["test"])
    validate_split_manifest(manifest)


def test_validate_split_manifest_rejects_augmented_validation_or_test_rows():
    manifest = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=3, val_size=0.25, test_size=0.25
    )
    manifest["splits"]["val"].append(
        {
            "sample_id": "bad-aug",
            "video_id": "vid_bad_noise_k1",
            "video_family_id": "vid_bad",
            "label": "hello",
            "landmark_path": "landmarks/vid_bad_noise_k1.csv",
            "augmented_row": True,
        }
    )

    with pytest.raises(ValueError, match="augmented"):
        validate_split_manifest(manifest)


def test_validate_split_manifest_rejects_cross_split_family_leakage():
    manifest = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=3, val_size=0.25, test_size=0.25
    )
    leaked = dict(manifest["splits"]["train"][0])
    leaked["sample_id"] = "leaked-copy"
    leaked["augmented_row"] = False
    manifest["splits"]["test"].append(leaked)

    with pytest.raises(ValueError, match="leakage"):
        validate_split_manifest(manifest)


def test_manifest_summary_fields_include_sample_group_and_class_counts():
    manifest = build_grouped_split_manifest(
        _rows(), dataset_name="unit", seed=11, val_size=0.25, test_size=0.25
    )

    assert manifest["sample_counts"] == {
        split: len(rows) for split, rows in manifest["splits"].items()
    }
    assert manifest["group_counts"] == {
        split: len({row["video_family_id"] for row in rows})
        for split, rows in manifest["splits"].items()
    }
    assert set(manifest["class_counts"]) == {"train", "val", "test"}
    assert sum(manifest["class_counts"]["train"].values()) == manifest["sample_counts"]["train"]
    assert manifest["total_samples"] == len(_rows())
