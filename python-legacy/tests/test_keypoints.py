"""
Unit tests for src.keypoints — the single source of truth for hand keypoint
extraction and normalization that BOTH training (notebook) and inference
(webcam_demo.py) rely on.

These tests pin down the public contract of:
  - normalize_landmarks
  - extract_hand_landmarks
  - extract_and_normalize

so that any future refactor (e.g. mirror augmentation helpers, rotation
augmentation) cannot silently drift the feature representation away from
what the trained model expects.
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

from src.keypoints import (  # noqa: E402  (import after sys.path tweak)
    FEATURE_SIZE,
    NUM_LANDMARKS,
    extract_and_normalize,
    extract_hand_landmarks,
    normalize_landmarks,
)


# ── Stub builders ─────────────────────────────────────────────────────────────
# MediaPipe's `results` object is a protobuf-ish structure. For unit testing
# we only need the duck-typed shape:
#   results.multi_hand_landmarks : list[HandLandmarks] | None
#   results.multi_handedness     : list[Handedness] | None
# where HandLandmarks.landmark is a list of objects with .x .y .z attributes,
# and Handedness.classification[0].score is a float.


def make_hand(coords_21x3: np.ndarray) -> SimpleNamespace:
    """Wrap a (21, 3) array of (x, y, z) coords as a fake HandLandmarks object."""
    assert coords_21x3.shape == (NUM_LANDMARKS, 3)
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                  for c in coords_21x3]
    )


def make_handedness(score: float) -> SimpleNamespace:
    """Fake a MediaPipe handedness entry with a single classification score."""
    return SimpleNamespace(classification=[SimpleNamespace(score=float(score))])


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG so every test sees the same 'random' hand."""
    return np.random.default_rng(42)


@pytest.fixture
def valid_landmarks_flat(rng: np.random.Generator) -> np.ndarray:
    """
    A plausible 63-d landmark vector: 21 random points in [0, 1]^3, with
    landmark 9 nudged so it is guaranteed not to coincide with the wrist
    (otherwise hand_span would be zero).
    """
    coords = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    # Force a non-trivial offset between wrist (0) and mid-finger MCP (9).
    coords[9] = coords[0] + np.array([0.3, 0.2, 0.1], dtype=np.float32)
    return coords.flatten()


# ── normalize_landmarks ───────────────────────────────────────────────────────

def test_normalize_returns_correct_shape_and_dtype(valid_landmarks_flat: np.ndarray) -> None:
    """Output of normalize_landmarks is always (63,) float32 for valid input."""
    out = normalize_landmarks(valid_landmarks_flat)
    assert out is not None
    assert out.shape == (FEATURE_SIZE,)
    assert out.dtype == np.float32


def test_normalize_places_wrist_at_origin(valid_landmarks_flat: np.ndarray) -> None:
    """Landmark 0 (wrist) must be at (0, 0, 0) after normalization."""
    out = normalize_landmarks(valid_landmarks_flat)
    assert out is not None
    wrist_xyz = out[0:3]
    np.testing.assert_allclose(wrist_xyz, np.zeros(3), atol=1e-5)


def test_normalize_scales_hand_span_to_unit(valid_landmarks_flat: np.ndarray) -> None:
    """
    Distance from wrist (landmark 0) to mid-finger MCP (landmark 9) must be
    exactly 1.0 after normalization, because the function divides by that
    distance to make the representation scale-invariant.
    """
    out = normalize_landmarks(valid_landmarks_flat)
    assert out is not None
    landmark_9_xyz = out[9 * 3:9 * 3 + 3]
    span = float(np.linalg.norm(landmark_9_xyz))
    assert span == pytest.approx(1.0, abs=1e-5)


def test_normalize_returns_none_for_degenerate_input() -> None:
    """
    When all 21 landmarks share the same coordinates (collapsed hand), the
    span from wrist to landmark 9 is exactly 0 and normalization is
    mathematically undefined. The function must signal this with None so the
    caller drops the sample instead of feeding a NaN/zero vector to the model.

    We use 0.5 (not 0.0) so the test does not accidentally start at the
    origin — that would still produce span=0 but for the wrong reason.
    """
    collapsed = np.ones(FEATURE_SIZE, dtype=np.float32) * 0.5
    assert normalize_landmarks(collapsed) is None


# ── extract_hand_landmarks ────────────────────────────────────────────────────

def test_extract_returns_none_when_no_hand_detected() -> None:
    """If MediaPipe found no hand, multi_hand_landmarks is None → return None."""
    class _StubResults:
        multi_hand_landmarks = None
        multi_handedness = None

    assert extract_hand_landmarks(_StubResults()) is None


def test_extract_returns_none_for_empty_multi_hand_landmarks() -> None:
    """Empty list (no hands) should also yield None, not an empty array."""
    results = SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[])
    assert extract_hand_landmarks(results) is None


def test_extract_picks_highest_confidence_hand(rng: np.random.Generator) -> None:
    """
    When two hands are detected, extract_hand_landmarks must select the one
    with the higher handedness classification score, so the downstream
    classifier only ever sees one hand per frame.
    """
    coords_lo = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    coords_hi = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)

    results = SimpleNamespace(
        multi_hand_landmarks=[make_hand(coords_lo), make_hand(coords_hi)],
        multi_handedness=[make_handedness(0.4), make_handedness(0.9)],
    )

    out = extract_hand_landmarks(results)
    assert out is not None
    assert out.shape == (FEATURE_SIZE,)
    np.testing.assert_allclose(out, coords_hi.flatten(), atol=1e-6)


# ── extract_and_normalize (end-to-end) ────────────────────────────────────────

def test_extract_and_normalize_end_to_end(rng: np.random.Generator) -> None:
    """
    Smoke test for the full pipeline: given a valid stub with one detected
    hand, extract_and_normalize should yield a (63,) vector that satisfies
    the same wrist@origin and unit-span invariants as normalize_landmarks.
    """
    coords = rng.uniform(0.0, 1.0, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    coords[9] = coords[0] + np.array([0.25, 0.15, 0.05], dtype=np.float32)

    results = SimpleNamespace(
        multi_hand_landmarks=[make_hand(coords)],
        multi_handedness=[make_handedness(0.99)],
    )

    out = extract_and_normalize(results)
    assert out is not None
    assert out.shape == (FEATURE_SIZE,)
    assert out.dtype == np.float32

    np.testing.assert_allclose(out[0:3], np.zeros(3), atol=1e-5)
    span = float(np.linalg.norm(out[9 * 3:9 * 3 + 3]))
    assert span == pytest.approx(1.0, abs=1e-5)


def test_extract_and_normalize_returns_none_when_no_hand() -> None:
    """No hand → no features. extract_and_normalize short-circuits to None."""
    results = SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[])
    assert extract_and_normalize(results) is None
