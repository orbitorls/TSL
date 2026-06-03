from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tsl_translate.session import PredictService  # noqa: E402
from tsl_translate.registry import ModelRegistry  # noqa: E402
from tsl_translate.tracks import TRACKS  # noqa: E402
from webcam_word_demo import TSL51_TRACK, choose_display_label  # noqa: E402
from webcam_word_demo import SIGN_END_FRAMES_DEFAULT, MIN_SIGN_FRAMES_DEFAULT  # noqa: E402
from sequence_keypoints import resample_frames, FEATURE_DIM  # noqa: E402


def test_tsl51_track_defaults_match_runtime_contract() -> None:
    track = TRACKS["tsl51"]

    assert TSL51_TRACK is track
    assert track.expected_feature_dim == 162
    assert track.expected_seq_len == 60
    assert track.expected_num_classes == 51
    assert track.default_model == "tsl51_model.keras"
    assert track.default_scaler == "tsl51_scaler.pkl"
    assert track.default_labels == "tsl51_labels.json"


def write_artifact_stub(path: Path, label_count: int) -> None:
    path.mkdir(parents=True)
    (path / "tsl51_model.keras").write_bytes(b"stub")
    (path / "tsl51_scaler.pkl").write_bytes(b"stub")
    labels = {str(idx): f"label_{idx}" for idx in range(label_count)}
    (path / "tsl51_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False),
        encoding="utf-8",
    )


def write_manifest(path: Path, **values: object) -> None:
    payload = {"track": "tsl51_word_signs", "num_classes": 51}
    payload.update(values)
    (path / "tsl51_model_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_tsl51_registry_prefers_candidate_with_external_holdout_evidence(tmp_path: Path) -> None:
    baseline = tmp_path / ".tools" / "train_runs_baseline" / "artifacts" / "tsl51"
    external = tmp_path / ".tools" / "train_runs_external" / "artifacts" / "tsl51"
    write_artifact_stub(baseline, 51)
    write_artifact_stub(external, 51)
    write_manifest(baseline, test_accuracy=0.92, external_augmented=False)
    write_manifest(external, test_accuracy=0.91, external_augmented=True, external_val_samples=8)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [candidate.name for candidate in candidates] == [
        str(Path(".tools") / "train_runs_external" / "artifacts" / "tsl51"),
        str(Path(".tools") / "train_runs_baseline" / "artifacts" / "tsl51"),
    ]


def test_tsl51_registry_keeps_artifacts_with_expected_class_count(tmp_path: Path) -> None:
    valid = tmp_path / "artifacts" / "tsl51"
    write_artifact_stub(valid, 51)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [candidate.name for candidate in candidates] == ["artifacts\\tsl51"]


def test_predict_service_set_alpha_keeps_smoother_when_unchanged() -> None:
    service = PredictService(TRACKS["tsl51"], alpha=0.4)
    first_smoothed = service.smoother.update(np.array([0.2, 0.8], dtype=np.float32))
    smoother = service.smoother

    service.set_alpha(0.4)
    second_smoothed = service.smoother.update(np.array([0.8, 0.2], dtype=np.float32))

    assert service.smoother is smoother
    assert np.allclose(first_smoothed, np.array([0.2, 0.8], dtype=np.float32))
    assert np.allclose(second_smoothed, np.array([0.44, 0.56], dtype=np.float32))


def test_predict_service_set_alpha_replaces_smoother_when_changed() -> None:
    service = PredictService(TRACKS["tsl51"], alpha=0.4)
    service.smoother.update(np.array([0.2, 0.8], dtype=np.float32))
    smoother = service.smoother

    service.set_alpha(0.8)
    smoothed = service.smoother.update(np.array([0.8, 0.2], dtype=np.float32))

    assert service.smoother is not smoother
    assert service.smoother.alpha == 0.8
    assert np.allclose(smoothed, np.array([0.8, 0.2], dtype=np.float32))


def test_choose_display_label_returns_unknown_below_threshold() -> None:
    label, overlay, committed = choose_display_label(
        {"0": "สวัสดี"},
        pred_idx=0,
        confidence=0.69,
        threshold=0.70,
    )

    assert label == "?"
    assert "Unknown" in overlay
    assert "รอท่าชัดเจน" in overlay
    assert committed is False


def test_choose_display_label_returns_thai_label_at_threshold() -> None:
    label, overlay, committed = choose_display_label(
        {"0": "สวัสดี"},
        pred_idx=0,
        confidence=0.70,
        threshold=0.70,
    )

    assert label == "สวัสดี"
    assert "สวัสดี" in overlay
    assert committed is True


# ---------------------------------------------------------------------------
# Sign-boundary detection tests
# ---------------------------------------------------------------------------

def test_sign_boundary_resample_uses_canonical_function() -> None:
    """resample_frames on N > 60 frames must cover the full segment.

    Specifically the last output frame must equal sign_frames[-1], confirming
    uniform resampling (not head-truncation) is used.
    """
    rng = np.random.default_rng(42)
    n_frames = 90
    sign_frames = [rng.random(FEATURE_DIM).astype(np.float32) for _ in range(n_frames)]

    result = resample_frames(np.asarray(sign_frames, dtype=np.float32), 60)

    assert result.shape == (60, FEATURE_DIM), (
        f"Expected shape (60, {FEATURE_DIM}), got {result.shape}"
    )
    # The last resampled index must map to sign_frames[-1] (index n_frames-1).
    # np.linspace(0, n_frames-1, 60).round() ends at n_frames-1.
    assert np.allclose(result[-1], sign_frames[-1]), (
        "Last resampled frame does not equal sign_frames[-1]; "
        "uniform resampling must cover the full temporal span."
    )


def test_sign_boundary_resample_short_sign_zero_pads() -> None:
    """resample_frames on 20 frames (< 60) must produce (60, 162) with leading zeros.

    The 20 actual frames must sit at the end of the output, matching
    SequenceBuffer.get_padded() behaviour used during live inference.
    """
    rng = np.random.default_rng(7)
    n_frames = 20
    sign_frames = [rng.random(FEATURE_DIM).astype(np.float32) for _ in range(n_frames)]
    arr = np.asarray(sign_frames, dtype=np.float32)

    result = resample_frames(arr, 60)

    assert result.shape == (60, FEATURE_DIM), (
        f"Expected shape (60, {FEATURE_DIM}), got {result.shape}"
    )
    # Leading (60 - 20) = 40 rows must be zero
    assert np.all(result[:40] == 0.0), "Leading zero-pad rows must be all zeros."
    # Trailing 20 rows must match the original frames
    assert np.allclose(result[40:], arr), (
        "Trailing frames do not match the original sign_frames."
    )


def test_sign_boundary_constants_are_sane() -> None:
    """SIGN_END_FRAMES_DEFAULT and MIN_SIGN_FRAMES_DEFAULT must be in sane ranges."""
    assert 5 <= SIGN_END_FRAMES_DEFAULT <= 20, (
        f"SIGN_END_FRAMES_DEFAULT={SIGN_END_FRAMES_DEFAULT} is outside [5, 20]."
    )
    assert 10 <= MIN_SIGN_FRAMES_DEFAULT <= 30, (
        f"MIN_SIGN_FRAMES_DEFAULT={MIN_SIGN_FRAMES_DEFAULT} is outside [10, 30]."
    )
