"""External video dataset contracts for fingerspelling and TSL-51.

The functions in this module intentionally avoid importing MediaPipe,
TensorFlow, or OpenCV so manifest validation and cache-shape tests stay fast.
Heavy extraction lives in repository scripts and calls these helpers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FINGERSPELLING_FEATURE_DIM = 63
TSL51_FEATURE_DIM = 162
TSL51_SEQ_LEN = 60

MANIFEST_FIELDS = (
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
)
OPTIONAL_MANIFEST_FIELDS = (
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
)
TRACKS = frozenset({"fingerspelling", "tsl51"})
SPLITS = frozenset({"train", "val", "external_test"})
QUALITY_STATUSES = frozenset({"raw", "prelabel", "reviewed", "rejected"})
SOURCE_TYPES = frozenset(
    {
        "hf_tsl51",
        "own_recording",
        "consented_recording",
        "youtube_permission",
        "youtube_cc",
        "other_public",
    }
)
RIGHTS_STATUSES = frozenset(
    {
        "cc_by_nc_sa_4_0",
        "own_internal_consent",
        "permission_granted_internal",
        "clear_cc_internal",
        "rejected",
        "unknown",
    }
)


class ManifestError(ValueError):
    """Raised when an external dataset manifest is unsafe to use."""


@dataclass(frozen=True)
class ManifestRow:
    video_id: str
    path: Path
    track: str
    label: str
    start_s: float
    end_s: float
    source_url: str
    split: str
    license_note: str
    quality_status: str
    source_type: str = ""
    rights_status: str = ""
    rights_evidence: str = ""
    consent_id: str = ""
    signer_id: str = ""
    session_id: str = ""
    camera_id: str = ""
    camera_angle: str = ""
    take_id: str = ""
    segment_id: str = ""
    reviewer_id: str = ""
    reviewed_at: str = ""
    review_notes: str = ""


def _require_text(row: dict[str, str], field: str, line_no: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ManifestError(f"line {line_no}: missing {field}")
    return value


def _parse_float(row: dict[str, str], field: str, line_no: int) -> float:
    value = _require_text(row, field, line_no)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ManifestError(f"line {line_no}: {field} must be numeric") from exc
    if parsed < 0:
        raise ManifestError(f"line {line_no}: {field} must be >= 0")
    return parsed


def _segments_overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> bool:
    return first_start < second_end and second_start < first_end


def load_label_set(path: str | Path) -> set[str]:
    """Load labels from a list/dict JSON artifact."""
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {str(value) for value in raw}
    if isinstance(raw, dict):
        return {str(value) for value in raw.values()}
    raise ManifestError(f"{path} must contain a label list or dict")


def load_manifest(
    path: str | Path,
    known_labels: set[str],
    *,
    allowed_tracks: set[str] | None = None,
    require_reviewed: bool = False,
    strict_training: bool = False,
) -> list[ManifestRow]:
    """Load and validate an external video manifest.

    A video id is intentionally restricted to one split to prevent clip leakage
    across train/validation/external-test partitions.
    """
    manifest_path = Path(path)
    tracks = allowed_tracks or set(TRACKS)
    rows: list[ManifestRow] = []
    video_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    signer_train_splits: dict[str, set[str]] = {}
    segment_labels: dict[tuple[str, str], list[tuple[float, float, str, int]]] = {}

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ManifestError(f"manifest missing field(s): {', '.join(missing_fields)}")

        for idx, raw in enumerate(reader, start=2):
            video_id = _require_text(raw, "video_id", idx)
            video_path = Path(_require_text(raw, "path", idx))
            track = _require_text(raw, "track", idx)
            label = _require_text(raw, "label", idx)
            start_s = _parse_float(raw, "start_s", idx)
            end_s = _parse_float(raw, "end_s", idx)
            split = _require_text(raw, "split", idx)
            quality_status = _require_text(raw, "quality_status", idx)
            source_type = (raw.get("source_type") or "").strip()
            rights_status = (raw.get("rights_status") or "").strip()
            rights_evidence = (raw.get("rights_evidence") or "").strip()
            consent_id = (raw.get("consent_id") or "").strip()
            signer_id = (raw.get("signer_id") or "").strip()
            session_id = (raw.get("session_id") or "").strip()
            camera_id = (raw.get("camera_id") or "").strip()
            camera_angle = (raw.get("camera_angle") or "").strip()
            take_id = (raw.get("take_id") or "").strip()
            segment_id = (raw.get("segment_id") or "").strip()
            reviewer_id = (raw.get("reviewer_id") or "").strip()
            reviewed_at = (raw.get("reviewed_at") or "").strip()
            review_notes = (raw.get("review_notes") or "").strip()

            if track not in TRACKS:
                raise ManifestError(f"line {idx}: unsupported track {track!r}")
            if track not in tracks:
                continue
            if split not in SPLITS:
                raise ManifestError(f"line {idx}: unsupported split {split!r}")
            if quality_status not in QUALITY_STATUSES:
                raise ManifestError(
                    f"line {idx}: unsupported quality_status {quality_status!r}"
                )
            if require_reviewed and quality_status != "reviewed":
                raise ManifestError(f"line {idx}: quality_status must be reviewed")
            if strict_training:
                if quality_status != "reviewed":
                    raise ManifestError(f"line {idx}: quality_status must be reviewed")
                if not source_type:
                    raise ManifestError(f"line {idx}: missing source_type")
                if source_type not in SOURCE_TYPES:
                    raise ManifestError(f"line {idx}: unsupported source_type {source_type!r}")
                if not rights_status:
                    raise ManifestError(f"line {idx}: missing rights_status")
                if rights_status not in RIGHTS_STATUSES:
                    raise ManifestError(f"line {idx}: unsupported rights_status {rights_status!r}")
                if rights_status in {"unknown", "rejected"}:
                    raise ManifestError(f"line {idx}: rights_status must be approved for training")
                if source_type in {"youtube_permission", "youtube_cc"} and not rights_evidence:
                    raise ManifestError(f"line {idx}: rights_evidence is required for YouTube rows")
            if label not in known_labels:
                raise ManifestError(f"line {idx}: unknown label {label!r}")
            if end_s <= start_s:
                raise ManifestError(f"line {idx}: end_s must be greater than start_s")

            if strict_training:
                segment_key = (video_id, video_path.as_posix().lower())
                for existing_start, existing_end, existing_label, existing_line in segment_labels.get(segment_key, []):
                    if existing_label != label and _segments_overlap(
                        existing_start,
                        existing_end,
                        start_s,
                        end_s,
                    ):
                        raise ManifestError(
                            f"line {idx}: conflicting labels for overlapping segment "
                            f"{video_id!r} ({existing_line}: {existing_label!r}, {idx}: {label!r})"
                        )
                segment_labels.setdefault(segment_key, []).append((start_s, end_s, label, idx))

            existing_split = video_splits.get(video_id)
            if existing_split is not None and existing_split != split:
                raise ManifestError(
                    f"video_id {video_id!r} appears in multiple splits: "
                    f"{existing_split!r}, {split!r}"
                )
            video_splits[video_id] = split
            if strict_training and session_id:
                existing_session_split = session_splits.get(session_id)
                if existing_session_split is not None and existing_session_split != split:
                    raise ManifestError(
                        f"session_id {session_id!r} appears in multiple splits: "
                        f"{existing_session_split!r}, {split!r}"
                    )
                session_splits[session_id] = split
            if strict_training and signer_id:
                splits = signer_train_splits.setdefault(signer_id, set())
                splits.add(split)
                if "external_test" in splits and ("train" in splits or "val" in splits):
                    raise ManifestError(
                        f"signer_id {signer_id!r} appears in external_test and training/validation splits"
                    )

            rows.append(
                ManifestRow(
                    video_id=video_id,
                    path=video_path,
                    track=track,
                    label=label,
                    start_s=start_s,
                    end_s=end_s,
                    source_url=(raw.get("source_url") or "").strip(),
                    split=split,
                    license_note=(raw.get("license_note") or "").strip(),
                    quality_status=quality_status,
                    source_type=source_type,
                    rights_status=rights_status,
                    rights_evidence=rights_evidence,
                    consent_id=consent_id,
                    signer_id=signer_id,
                    session_id=session_id,
                    camera_id=camera_id,
                    camera_angle=camera_angle,
                    take_id=take_id,
                    segment_id=segment_id,
                    reviewer_id=reviewer_id,
                    reviewed_at=reviewed_at,
                    review_notes=review_notes,
                )
            )

    return rows


def _label_indices(labels: list[str]) -> dict[str, int]:
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    return {label: idx for idx, label in enumerate(labels)}


def build_fingerspelling_cache(
    samples: Iterable[tuple[str, np.ndarray]], labels: list[str]
) -> dict[str, np.ndarray]:
    """Build an NPZ-ready fingerspelling cache from labeled 63-D frames."""
    label_to_idx = _label_indices(labels)
    X: list[np.ndarray] = []
    y: list[int] = []
    for label, feature in samples:
        arr = np.asarray(feature, dtype=np.float32)
        if arr.shape != (FINGERSPELLING_FEATURE_DIM,):
            raise ValueError(
                f"fingerspelling feature must be ({FINGERSPELLING_FEATURE_DIM},), got {arr.shape}"
            )
        X.append(arr)
        y.append(label_to_idx[label])
    return {
        "track": np.asarray("fingerspelling", dtype=object),
        "feature_dim": np.asarray(FINGERSPELLING_FEATURE_DIM, dtype=np.int32),
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "class_names": np.asarray(labels, dtype=object),
    }


def build_tsl51_cache(
    samples: Iterable[tuple[str, np.ndarray]], labels: list[str]
) -> dict[str, np.ndarray]:
    """Build an NPZ-ready TSL-51 cache from labeled 60x162 sequences."""
    label_to_idx = _label_indices(labels)
    X: list[np.ndarray] = []
    y: list[int] = []
    for label, sequence in samples:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape != (TSL51_SEQ_LEN, TSL51_FEATURE_DIM):
            raise ValueError(
                f"tsl51 sequence must be ({TSL51_SEQ_LEN}, {TSL51_FEATURE_DIM}), got {arr.shape}"
            )
        X.append(arr)
        y.append(label_to_idx[label])
    return {
        "track": np.asarray("tsl51", dtype=object),
        "feature_dim": np.asarray(TSL51_FEATURE_DIM, dtype=np.int32),
        "seq_len": np.asarray(TSL51_SEQ_LEN, dtype=np.int32),
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "class_names": np.asarray(labels, dtype=object),
    }
