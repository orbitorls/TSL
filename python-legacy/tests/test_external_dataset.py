from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_dataset import (  # noqa: E402
    ManifestError,
    build_fingerspelling_cache,
    build_tsl51_cache,
    load_manifest,
)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "video_id",
        "path",
        "track",
        "label",
        "start_s",
        "end_s",
        "source_url",
        "split",
        "license_note",
        "quality_status",
        "source_type",
        "rights_status",
        "rights_evidence",
        "consent_id",
        "signer_id",
        "session_id",
        "camera_id",
        "camera_angle",
        "take_id",
        "segment_id",
        "reviewer_id",
        "reviewed_at",
        "review_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "video_id": "clip-a",
        "path": "videos/clip-a.mp4",
        "track": "fingerspelling",
        "label": "KO_KAI",
        "start_s": 1.0,
        "end_s": 2.0,
        "source_url": "https://example.test/clip-a",
        "split": "train",
        "license_note": "local research only",
        "quality_status": "reviewed",
        "source_type": "own_recording",
        "rights_status": "own_internal_consent",
        "rights_evidence": "consent:demo",
        "consent_id": "consent-demo",
        "signer_id": "signer-a",
        "session_id": "session-a",
        "camera_id": "cam-a",
        "camera_angle": "front",
        "take_id": "take-a",
        "segment_id": "seg-a",
        "reviewer_id": "reviewer-a",
        "reviewed_at": "2026-06-02",
        "review_notes": "ok",
    }
    row.update(overrides)
    return row


def test_manifest_rejects_missing_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(label="")])

    with pytest.raises(ManifestError, match="label"):
        load_manifest(manifest, known_labels={"KO_KAI"})


def test_manifest_rejects_invalid_time_range(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(start_s=4.0, end_s=3.0)])

    with pytest.raises(ManifestError, match="end_s"):
        load_manifest(manifest, known_labels={"KO_KAI"})


def test_manifest_rejects_unknown_class(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(label="UNKNOWN")])

    with pytest.raises(ManifestError, match="unknown label"):
        load_manifest(manifest, known_labels={"KO_KAI"})


def test_manifest_rejects_video_id_split_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(split="train"),
            base_row(split="external_test", label="BOR_BAI_MAI"),
        ],
    )

    with pytest.raises(ManifestError, match="multiple splits"):
        load_manifest(manifest, known_labels={"KO_KAI", "BOR_BAI_MAI"})


def test_manifest_allowed_tracks_filters_mixed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(track="fingerspelling", label="KO_KAI"),
            base_row(
                video_id="clip-b",
                track="tsl51",
                label="สวัสดี_อายุเท่ากันหรือน้อยกว่า",
            ),
        ],
    )

    rows = load_manifest(
        manifest,
        known_labels={"KO_KAI", "สวัสดี_อายุเท่ากันหรือน้อยกว่า"},
        allowed_tracks={"tsl51"},
    )

    assert len(rows) == 1
    assert rows[0].track == "tsl51"




def test_strict_manifest_rejects_unreviewed_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(quality_status="prelabel")])

    with pytest.raises(ManifestError, match="quality_status must be reviewed"):
        load_manifest(manifest, known_labels={"KO_KAI"}, strict_training=True)


def test_strict_manifest_rejects_unknown_rights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(rights_status="unknown")])

    with pytest.raises(ManifestError, match="rights_status"):
        load_manifest(manifest, known_labels={"KO_KAI"}, strict_training=True)


def test_strict_manifest_rejects_youtube_without_rights_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(
                source_type="youtube_permission",
                rights_status="permission_granted_internal",
                rights_evidence="",
            )
        ],
    )

    with pytest.raises(ManifestError, match="rights_evidence"):
        load_manifest(manifest, known_labels={"KO_KAI"}, strict_training=True)


def test_strict_manifest_rejects_session_split_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(video_id="clip-a", split="train", session_id="same-session"),
            base_row(video_id="clip-b", split="val", session_id="same-session"),
        ],
    )

    with pytest.raises(ManifestError, match="session_id"):
        load_manifest(manifest, known_labels={"KO_KAI"}, strict_training=True)


def test_strict_manifest_rejects_conflicting_segment_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(label="KO_KAI"),
            base_row(label="BOR_BAI_MAI"),
        ],
    )

    with pytest.raises(ManifestError, match="conflicting labels"):
        load_manifest(
            manifest,
            known_labels={"KO_KAI", "BOR_BAI_MAI"},
            strict_training=True,
        )


def test_strict_manifest_allows_non_overlapping_segments_with_different_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            base_row(label="KO_KAI", start_s=1.0, end_s=2.0),
            base_row(label="BOR_BAI_MAI", start_s=2.1, end_s=3.0),
        ],
    )

    rows = load_manifest(
        manifest,
        known_labels={"KO_KAI", "BOR_BAI_MAI"},
        strict_training=True,
    )

    assert [row.label for row in rows] == ["KO_KAI", "BOR_BAI_MAI"]


def test_strict_manifest_keeps_extended_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row(source_type="consented_recording", segment_id="seg-001")])

    rows = load_manifest(manifest, known_labels={"KO_KAI"}, strict_training=True)

    assert rows[0].source_type == "consented_recording"
    assert rows[0].rights_status == "own_internal_consent"
    assert rows[0].segment_id == "seg-001"
    labels = ["KO_KAI", "BOR_BAI_MAI"]
    fs_features = [
        ("KO_KAI", np.ones(63, dtype=np.float32)),
        ("BOR_BAI_MAI", np.zeros(63, dtype=np.float32)),
    ]
    fs_cache = build_fingerspelling_cache(fs_features, labels)

    assert fs_cache["X"].shape == (2, 63)
    assert fs_cache["y"].tolist() == [0, 1]
    assert fs_cache["class_names"].tolist() == labels

    tsl_features = [
        ("KO_KAI", np.ones((60, 162), dtype=np.float32)),
        ("BOR_BAI_MAI", np.zeros((60, 162), dtype=np.float32)),
    ]
    tsl_cache = build_tsl51_cache(tsl_features, labels)

    assert tsl_cache["X"].shape == (2, 60, 162)
    assert tsl_cache["y"].tolist() == [0, 1]
    assert tsl_cache["class_names"].tolist() == labels
