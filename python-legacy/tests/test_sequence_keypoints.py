"""
Unit tests for src.sequence_keypoints — the shared contract for TSL-51 word-level
holistic keypoint extraction and sequence buffering.

These tests pin down the public contract of:
  - SequenceBuffer
  - extract_holistic_frame

so that training (notebook) and inference (webcam_word_demo.py) cannot silently
drift the 162-D feature representation or padding convention.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make the project root importable regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_keypoints import (  # noqa: E402  (import after sys.path tweak)
    CSV_FACE_LANDMARKS,
    CSV_POSE_LANDMARKS,
    FACE_INDICES,
    FEATURE_DIM,
    LEFT_SHOULDER_IDX,
    NUM_HAND_LANDMARKS,
    POSE_INDICES,
    RIGHT_SHOULDER_IDX,
    SEQ_LEN_DEFAULT,
    SequenceBuffer,
    csv_to_sequence,
    extract_holistic_frame,
    pad_truncate_sequence,
    read_landmark_csv,
    tsl51_csv_column_names,
)


# ── Stub builders ─────────────────────────────────────────────────────────────
# MediaPipe Holistic ``results`` is duck-typed for unit tests:
#   results.pose_landmarks         : NormalizedLandmarkList | None
#   results.face_landmarks         : NormalizedLandmarkList | None
#   results.left_hand_landmarks    : NormalizedLandmarkList | None
#   results.right_hand_landmarks   : NormalizedLandmarkList | None


def make_landmark_list(coords_by_index: dict[int, tuple[float, float, float]]) -> SimpleNamespace:
    """Build a fake landmark list addressable by MediaPipe index."""
    max_idx = max(coords_by_index.keys()) if coords_by_index else -1
    landmarks = []
    for i in range(max_idx + 1):
        x, y, z = coords_by_index.get(i, (0.0, 0.0, 0.0))
        landmarks.append(SimpleNamespace(x=float(x), y=float(y), z=float(z)))
    return SimpleNamespace(landmark=landmarks)


def make_hand(coords_21x3: np.ndarray) -> SimpleNamespace:
    """Wrap a (21, 3) array as a fake hand NormalizedLandmarkList."""
    assert coords_21x3.shape == (NUM_HAND_LANDMARKS, 3)
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            for c in coords_21x3
        ]
    )


def make_holistic_results(
    *,
    pose: SimpleNamespace | None = None,
    face: SimpleNamespace | None = None,
    left_hand: SimpleNamespace | None = None,
    right_hand: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pose_landmarks=pose,
        face_landmarks=face,
        left_hand_landmarks=left_hand,
        right_hand_landmarks=right_hand,
    )


# ── SequenceBuffer ────────────────────────────────────────────────────────────

def test_sequence_buffer_shape_and_leading_zero_pad() -> None:
    """get_padded() is (seq_len, 162) with leading zeros when not full."""
    seq_len = 5
    buf = SequenceBuffer(seq_len)
    frame = np.arange(FEATURE_DIM, dtype=np.float32)

    buf.push(frame)
    out = buf.get_padded()

    assert out.shape == (seq_len, FEATURE_DIM)
    assert out.dtype == np.float32
    assert np.all(out[: seq_len - 1] == 0.0)
    np.testing.assert_array_equal(out[-1], frame)


def test_sequence_buffer_is_full_after_seq_len_pushes() -> None:
    """is_full() becomes True only after seq_len frames are pushed."""
    seq_len = 4
    buf = SequenceBuffer(seq_len)
    frame = np.ones(FEATURE_DIM, dtype=np.float32)

    assert len(buf) == 0
    for i in range(seq_len - 1):
        buf.push(frame * i)
        assert len(buf) == i + 1
        assert not buf.is_full()

    buf.push(frame * (seq_len - 1))
    assert len(buf) == seq_len
    assert buf.is_full()


def test_sequence_buffer_reset_clears() -> None:
    """reset() empties the buffer and is_full() returns False again."""
    buf = SequenceBuffer(3)
    frame = np.ones(FEATURE_DIM, dtype=np.float32)

    for _ in range(3):
        buf.push(frame)
    assert buf.is_full()

    buf.reset()
    assert not buf.is_full()
    assert np.all(buf.get_padded() == 0.0)


def test_sequence_buffer_end_to_end_three_frames() -> None:
    """Three pushes leave zeros in the first (seq_len - 3) rows."""
    seq_len = SEQ_LEN_DEFAULT
    buf = SequenceBuffer(seq_len)

    frames = [
        np.full(FEATURE_DIM, 1.0, dtype=np.float32),
        np.full(FEATURE_DIM, 2.0, dtype=np.float32),
        np.full(FEATURE_DIM, 3.0, dtype=np.float32),
    ]
    for f in frames:
        buf.push(f)

    out = buf.get_padded()
    assert out.shape == (seq_len, FEATURE_DIM)
    assert np.all(out[: seq_len - 3] == 0.0)
    np.testing.assert_array_equal(out[-3], frames[0])
    np.testing.assert_array_equal(out[-2], frames[1])
    np.testing.assert_array_equal(out[-1], frames[2])


# ── extract_holistic_frame ────────────────────────────────────────────────────

def test_extract_returns_none_when_both_hands_missing() -> None:
    """No hands at all → None (caller should skip this frame)."""
    results = make_holistic_results(
        pose=make_landmark_list({11: (0.3, 0.4, 0.0), 12: (0.7, 0.4, 0.0)}),
        left_hand=None,
        right_hand=None,
    )
    assert extract_holistic_frame(results) is None


def test_extract_returns_162d_float32_with_one_hand(rng: np.random.Generator) -> None:
    """At least one visible hand → (162,) float32 vector."""
    coords = rng.uniform(0.0, 1.0, size=(NUM_HAND_LANDMARKS, 3)).astype(np.float32)
    results = make_holistic_results(
        pose=make_landmark_list({11: (0.3, 0.4, 0.0), 12: (0.7, 0.4, 0.0)}),
        left_hand=make_hand(coords),
        right_hand=None,
    )

    out = extract_holistic_frame(results)
    assert out is not None
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.float32


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_shoulder_midpoint_subtraction() -> None:
    """
    Shoulders at (0.3, 0.4, 0) and (0.7, 0.4, 0) → anchor (0.5, 0.4, 0).
    A landmark placed exactly at the anchor must become ~0 after subtraction.
    """
    left_shoulder = (0.3, 0.4, 0.0)
    right_shoulder = (0.7, 0.4, 0.0)
    anchor = (0.5, 0.4, 0.0)

    pose = make_landmark_list(
        {
            LEFT_SHOULDER_IDX: left_shoulder,
            RIGHT_SHOULDER_IDX: right_shoulder,
            13: anchor,  # left elbow at shoulder midpoint
        }
    )
    hand = make_hand(np.full((NUM_HAND_LANDMARKS, 3), anchor, dtype=np.float32))

    results = make_holistic_results(pose=pose, left_hand=hand, right_hand=None)
    out = extract_holistic_frame(results)
    assert out is not None

    # Pose block: first two landmarks are shoulders; third is elbow at anchor.
    left_shoulder_out = out[0:3]
    right_shoulder_out = out[3:6]
    elbow_out = out[6:9]

    np.testing.assert_allclose(left_shoulder_out, [-0.2, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(right_shoulder_out, [0.2, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(elbow_out, np.zeros(3), atol=1e-5)

    # Left-hand wrist (first hand landmark) also sits at anchor → ~0.
    left_hand_start = 18 + 18  # pose + face
    np.testing.assert_allclose(out[left_hand_start : left_hand_start + 3], np.zeros(3), atol=1e-5)


# ── TSL-51 CSV contract ───────────────────────────────────────────────────────

def _make_tsl51_csv_rows(num_frames: int = 2) -> str:
    """Minimal TSL-51 CSV text with shoulders at known positions."""
    cols = ["frame", "t_ms"] + tsl51_csv_column_names()
    lines = [",".join(cols)]
    for frame_idx in range(num_frames):
        values = [str(frame_idx), str(frame_idx * 33)]
        for prefix in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"):
            if prefix == "l_shoulder":
                values.extend(["0.3", "0.4", "0.0"])
            elif prefix == "r_shoulder":
                values.extend(["0.7", "0.4", "0.0"])
            else:
                values.extend(["0.5", "0.4", "0.0"])
        for _ in range(6):
            values.extend(["0.5", "0.4", "0.0"])
        for _ in range(NUM_HAND_LANDMARKS * 2):
            values.extend(["0.5", "0.4", "0.0"])
        lines.append(",".join(values))
    return "\n".join(lines)


def test_tsl51_csv_column_names_count() -> None:
    assert len(tsl51_csv_column_names()) == FEATURE_DIM


def test_read_landmark_csv_shoulder_anchor_and_shape() -> None:
    csv_text = _make_tsl51_csv_rows(num_frames=2)
    arr = read_landmark_csv(csv_text)
    assert arr.shape == (2, FEATURE_DIM)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr[0, 0:3], [-0.2, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(arr[0, 3:6], [0.2, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(arr[0, 6:9], np.zeros(3), atol=1e-5)


def test_csv_to_sequence_leading_pad() -> None:
    seq = csv_to_sequence(_make_tsl51_csv_rows(num_frames=2), seq_len=4)
    assert seq.shape == (4, FEATURE_DIM)
    assert np.all(seq[:2] == 0.0)
    assert not np.all(seq[2:] == 0.0)


# ── Order-consistency: CSV path == live-inference path ────────────────────────

def _make_distinct_coords(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return (n, 3) array where every x value is unique, so order swaps are detectable."""
    coords = rng.uniform(0.1, 0.9, size=(n, 3)).astype(np.float32)
    # Assign distinct x values 0.01, 0.02, ... to guarantee detectability
    for i in range(n):
        coords[i, 0] = (i + 1) * 0.01
    return coords


def test_extract_holistic_frame_matches_read_landmark_csv(rng: np.random.Generator) -> None:
    """
    The 162-D vector from ``extract_holistic_frame`` (live inference) must be
    identical index-for-index to the vector produced by ``read_landmark_csv``
    (training CSV path) for the same landmark values.

    This test catches any drift in hand-landmark ordering between the two code
    paths (e.g. block vs interleaved) — the core bug that caused TSL-51 0% accuracy.
    Each hand landmark is given a unique x value so that even a single index swap
    produces a measurable difference.
    """
    # Build per-landmark coords with distinct x values so ordering swaps are detectable.
    # lh: x = 0.01..0.21;  rh: x = 0.51..0.71  (non-overlapping, non-zero)
    lh_coords = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    rh_coords = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
    for i in range(NUM_HAND_LANDMARKS):
        lh_coords[i] = [(i + 1) * 0.01, (i + 1) * 0.005, 0.0]
        rh_coords[i] = [0.5 + (i + 1) * 0.01, (i + 1) * 0.003, 0.0]

    # Fixed pose landmarks: shoulders at known positions, other pose landmarks off-origin.
    # POSE_INDICES = [11, 12, 13, 14, 15, 16]; set each to a unique coordinate.
    pose_by_idx: dict[int, tuple[float, float, float]] = {}
    for slot, mp_idx in enumerate(POSE_INDICES):
        pose_by_idx[mp_idx] = (0.3 if mp_idx == LEFT_SHOULDER_IDX else
                               0.7 if mp_idx == RIGHT_SHOULDER_IDX else 0.5 + slot * 0.02,
                               0.4, 0.0)
    pose_lml = make_landmark_list(pose_by_idx)

    # Face landmarks: FACE_INDICES = [105, 70, 300, 334, 61, 291]; use distinct values.
    face_by_idx: dict[int, tuple[float, float, float]] = {}
    for slot, mp_idx in enumerate(FACE_INDICES):
        face_by_idx[mp_idx] = (0.2 + slot * 0.03, 0.6 + slot * 0.01, 0.0)
    face_lml = make_landmark_list(face_by_idx)

    # ── Path 1: live inference via extract_holistic_frame ─────────────────────
    results = make_holistic_results(
        pose=pose_lml,
        face=face_lml,
        left_hand=make_hand(lh_coords),
        right_hand=make_hand(rh_coords),
    )
    live_feat = extract_holistic_frame(results)
    assert live_feat is not None, "extract_holistic_frame returned None unexpectedly"

    # ── Path 2: training CSV path via read_landmark_csv ───────────────────────
    # Build one CSV row using the same landmark values.
    col_names = tsl51_csv_column_names()
    pose_prefix_to_mp_idx = dict(zip(CSV_POSE_LANDMARKS, POSE_INDICES))
    face_prefix_to_mp_idx = dict(zip(CSV_FACE_LANDMARKS, FACE_INDICES))

    row: dict[str, str] = {}
    for prefix in CSV_POSE_LANDMARKS:
        c = pose_by_idx[pose_prefix_to_mp_idx[prefix]]
        row[f"{prefix}_x"] = str(c[0])
        row[f"{prefix}_y"] = str(c[1])
        row[f"{prefix}_z"] = str(c[2])
    for prefix in CSV_FACE_LANDMARKS:
        c = face_by_idx[face_prefix_to_mp_idx[prefix]]
        row[f"{prefix}_x"] = str(c[0])
        row[f"{prefix}_y"] = str(c[1])
        row[f"{prefix}_z"] = str(c[2])
    for i in range(NUM_HAND_LANDMARKS):
        row[f"lh_x{i}"] = str(lh_coords[i, 0])
        row[f"lh_y{i}"] = str(lh_coords[i, 1])
        row[f"lh_z{i}"] = str(lh_coords[i, 2])
        row[f"rh_x{i}"] = str(rh_coords[i, 0])
        row[f"rh_y{i}"] = str(rh_coords[i, 1])
        row[f"rh_z{i}"] = str(rh_coords[i, 2])

    header = "frame,t_ms," + ",".join(col_names)
    data_row = "0,0," + ",".join(row[c] for c in col_names)
    csv_text = header + "\n" + data_row
    csv_feat = read_landmark_csv(csv_text)
    assert csv_feat.shape == (1, FEATURE_DIM)

    np.testing.assert_allclose(
        live_feat,
        csv_feat[0],
        atol=1e-4,
        err_msg=(
            "extract_holistic_frame and read_landmark_csv produced different 162-D vectors "
            "for the same landmarks. This indicates a hand-ordering mismatch (e.g. block vs "
            "interleaved) between the training CSV path and the live inference path."
        ),
    )
