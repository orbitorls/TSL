"""
Regression tests for the canonical clip-to-sequence pipeline.

These tests verify that:
1. ``resample_frames`` is the single, consistent function used across
   training (``csv_to_sequence``) and evaluation pipelines.
2. The behaviour of ``resample_frames`` is correct for all three cases:
   T == seq_len, T < seq_len (leading pad), T > seq_len (uniform resample).
3. The "head truncation" bug is absent — a 120-frame array resampled to 60
   must NOT equal ``arr[:60]`` (which would mean only the clip start is seen).

Background
----------
The train/serve skew was caused by ``pad_truncate_sequence`` doing
``features[:seq_len]`` (head truncation) while both offline callers
(``evaluate_tsl51_video.py`` and ``build_external_dataset.py``) did
``np.linspace`` uniform resampling.  This produced 0% top-1 accuracy
under the uniform eval strategy because the model was trained on leading
frames (often hand-absent setup frames) and evaluated on the full clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_keypoints import (  # noqa: E402
    FEATURE_DIM,
    SEQ_LEN_DEFAULT,
    csv_to_sequence,
    pad_truncate_sequence,
    read_landmark_csv,
    resample_frames,
    tsl51_csv_column_names,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_csv_rows(num_frames: int) -> str:
    """Build a minimal TSL-51 CSV with ``num_frames`` rows of distinctive values.

    Each frame row has all 162 feature values set to ``float(frame_index) + 1``
    so that head-truncation versus uniform-resampling produce different results
    when num_frames > seq_len.
    """
    col_names = tsl51_csv_column_names()
    header = "frame,t_ms," + ",".join(col_names)
    lines = [header]
    for i in range(num_frames):
        val = float(i + 1)
        row = f"{i},{i * 33}," + ",".join(str(val) for _ in col_names)
        lines.append(row)
    return "\n".join(lines)


def _make_feature_array(num_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (num_frames, FEATURE_DIM) float32 array with distinct values per frame."""
    arr = rng.uniform(0.01, 1.0, size=(num_frames, FEATURE_DIM)).astype(np.float32)
    # Make each row distinct so we can detect which frames were selected
    for i in range(num_frames):
        arr[i, 0] = float(i + 1)
    return arr


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── resample_frames: basic shape and dtype ────────────────────────────────────

def test_resample_frames_exact_length_is_identity(rng: np.random.Generator) -> None:
    """When T == seq_len, resample_frames returns the same values (no reorder)."""
    arr = _make_feature_array(SEQ_LEN_DEFAULT, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT)
    assert out.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr)


def test_resample_frames_short_zero_pads(rng: np.random.Generator) -> None:
    """When T < seq_len, output is leading-zero-padded with frames at the end."""
    arr = _make_feature_array(30, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT)
    assert out.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    assert out.dtype == np.float32
    # First 30 rows must be all zeros (leading pad)
    assert np.all(out[:30] == 0.0), "Expected leading zero-padding for short clips"
    # Last 30 rows must match the input frames
    np.testing.assert_array_equal(out[30:], arr)


def test_resample_frames_long_is_uniform_not_head(rng: np.random.Generator) -> None:
    """When T > seq_len, uniform resampling must NOT equal head-truncation.

    This is the core regression: the old ``features[:seq_len]`` bug would
    cause training to see only the first 60 frames (often setup / hand-absent),
    while evaluation saw the full clip uniformly sampled.
    """
    arr = _make_feature_array(120, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT)
    assert out.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    assert out.dtype == np.float32

    # Must NOT be the head-truncated result
    head = arr[:SEQ_LEN_DEFAULT]
    assert not np.array_equal(out, head), (
        "resample_frames returned arr[:seq_len] (head truncation) — "
        "the train/serve skew bug is still present"
    )

    # Must include the final frame (proves uniform coverage reaches clip end)
    final_frame = arr[-1]
    assert np.array_equal(out[-1], final_frame), (
        "resample_frames must include the last frame when T > seq_len "
        "(uniform resampling property)"
    )

    # Verify against the canonical linspace formula
    expected_idx = np.linspace(0, 119, SEQ_LEN_DEFAULT).round().astype(int)
    np.testing.assert_array_equal(out, arr[expected_idx])


def test_resample_frames_empty_returns_zeros() -> None:
    """T == 0 must return an all-zeros array without error."""
    arr = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    out = resample_frames(arr, SEQ_LEN_DEFAULT)
    assert out.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    assert np.all(out == 0.0)


def test_resample_frames_rejects_wrong_shape() -> None:
    """Bad feature dimension must raise ValueError, not silently produce garbage."""
    bad = np.zeros((10, FEATURE_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match=str(FEATURE_DIM)):
        resample_frames(bad, SEQ_LEN_DEFAULT)


# ── csv_to_sequence and pad_truncate_sequence delegate to resample_frames ─────

def test_csv_to_sequence_long_matches_resample_frames() -> None:
    """csv_to_sequence on N > seq_len rows must equal resample_frames(read_landmark_csv(...)).

    This test confirms that the training data loader (csv_to_sequence) and the
    canonical resampler share the same logic — i.e. the pipeline is unified.
    """
    n_frames = 90  # > SEQ_LEN_DEFAULT (60)
    csv_text = _make_csv_rows(n_frames)

    # Path 1: csv_to_sequence (the training loader path)
    seq_via_csv = csv_to_sequence(csv_text, SEQ_LEN_DEFAULT)

    # Path 2: read_landmark_csv → resample_frames (explicit canonical path)
    raw = read_landmark_csv(csv_text)
    assert raw.shape == (n_frames, FEATURE_DIM), f"Expected ({n_frames}, {FEATURE_DIM}), got {raw.shape}"
    seq_via_resample = resample_frames(raw, SEQ_LEN_DEFAULT)

    assert seq_via_csv.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    np.testing.assert_array_equal(
        seq_via_csv,
        seq_via_resample,
        err_msg=(
            "csv_to_sequence and resample_frames(read_landmark_csv(...)) produced "
            "different results — the training loader is not using the canonical function."
        ),
    )


def test_pad_truncate_sequence_delegates_to_resample_frames(rng: np.random.Generator) -> None:
    """pad_truncate_sequence must produce the same output as resample_frames.

    The wrapper exists for backward compatibility; it must not silently diverge.
    """
    arr = _make_feature_array(120, rng)
    assert np.array_equal(
        pad_truncate_sequence(arr, SEQ_LEN_DEFAULT),
        resample_frames(arr, SEQ_LEN_DEFAULT),
    )

    # Also verify the short-clip branch
    short_arr = _make_feature_array(20, rng)
    assert np.array_equal(
        pad_truncate_sequence(short_arr, SEQ_LEN_DEFAULT),
        resample_frames(short_arr, SEQ_LEN_DEFAULT),
    )


# ── resample_frames: stretch=True (live-serve path) ───────────────────────────

def test_resample_frames_stretch_short_has_no_zero_frames(rng: np.random.Generator) -> None:
    """stretch=True on T<seq_len must produce all-real frames (no leading zeros)."""
    arr = _make_feature_array(20, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT, stretch=True)
    assert out.shape == (SEQ_LEN_DEFAULT, FEATURE_DIM)
    assert out.dtype == np.float32
    # Every frame must be non-zero (real data, not padding)
    all_zero_rows = np.all(out == 0.0, axis=1)
    assert not np.any(all_zero_rows), (
        "stretch=True must not produce zero-padded frames; "
        f"found {all_zero_rows.sum()} all-zero rows"
    )


def test_resample_frames_stretch_covers_full_span(rng: np.random.Generator) -> None:
    """stretch=True must include both first and last frames (uniform coverage)."""
    arr = _make_feature_array(20, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT, stretch=True)
    np.testing.assert_array_equal(out[0], arr[0], err_msg="First frame must be arr[0]")
    np.testing.assert_array_equal(out[-1], arr[-1], err_msg="Last frame must be arr[-1]")


def test_resample_frames_stretch_false_still_zero_pads(rng: np.random.Generator) -> None:
    """Default stretch=False must still produce leading-zero-padding for T<seq_len."""
    arr = _make_feature_array(30, rng)
    out = resample_frames(arr, SEQ_LEN_DEFAULT, stretch=False)
    assert np.all(out[:30] == 0.0), "stretch=False must leading-zero-pad short clips"
    np.testing.assert_array_equal(out[30:], arr)


def test_resample_frames_stretch_long_same_as_no_stretch(rng: np.random.Generator) -> None:
    """stretch=True and stretch=False must agree when T >= seq_len (uniform downsample)."""
    arr = _make_feature_array(120, rng)
    assert np.array_equal(
        resample_frames(arr, SEQ_LEN_DEFAULT, stretch=True),
        resample_frames(arr, SEQ_LEN_DEFAULT, stretch=False),
    )
