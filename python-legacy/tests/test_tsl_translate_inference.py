from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tsl_translate.inference as inference_module  # noqa: E402
from sequence_keypoints import FEATURE_DIM  # noqa: E402
from tsl_translate.inference import InferenceSettings, process_rgb_frame  # noqa: E402
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService  # noqa: E402
from tsl_translate.tracks import TRACKS  # noqa: E402


class IdentityScaler:
    def transform(self, data: np.ndarray) -> np.ndarray:
        return data


class FixedPredictor:
    def predict(self, _data: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.9], dtype=np.float32)


class AmbiguousPredictor:
    def predict(self, _data: np.ndarray) -> np.ndarray:
        return np.array([0.45, 0.55], dtype=np.float32)


class ModeratePredictor:
    """High enough to pass threshold but below early-commit bar (threshold + 0.1)."""

    def predict(self, _data: np.ndarray) -> np.ndarray:
        return np.array([0.75, 0.24], dtype=np.float32)


class DummyHolistic:
    def process(self, _rgb: np.ndarray):
        return object()


class DummyRuntime(MediaPipeRuntime):
    def __init__(self) -> None:
        self.holistic = DummyHolistic()
        self.hands = None


def make_loaded_model(*, predictor: object | None = None) -> LoadedModel:
    return LoadedModel(
        predictor=predictor or FixedPredictor(),
        labels={"0": "label_0", "1": "label_1"},
        scaler=IdentityScaler(),
        backend="test",
        model_path=Path("model.keras"),
        labels_path=Path("labels.json"),
        scaler_path=Path("scaler.pkl"),
        load_time_ms=0.0,
    )


def _patch_motion(monkeypatch, *, motion: float = 0.02) -> None:
    monkeypatch.setattr(
        inference_module,
        "extract_holistic_frame",
        lambda _results: np.ones(FEATURE_DIM, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_module,
        "_extract_hand_coords",
        lambda _results: np.ones(126, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_module,
        "_mean_hand_displacement",
        lambda _prev, _curr: motion,
    )


def test_tsl51_fast_sign_preview_gate_uses_half_min_frames(monkeypatch) -> None:
    """Fast mode (commit_on_preview=True): preview after 2 buffering frames, commit on 2nd preview."""
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model(predictor=ModeratePredictor())
    settings = InferenceSettings(
        threshold=0.7,
        alpha=0.4,
        top_k=3,
        motion_min=0.008,
        min_sign_frames=3,
        sign_end_frames=5,
        commit_on_preview=True,
    )
    _patch_motion(monkeypatch)
    monotonic_values = iter([0.05 * i for i in range(1, 10)])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(monotonic_values))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for expected in range(1, 3):
        result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        assert result.status == "buffering"
        assert result.buffering == f"{expected}/3"
        assert result.committed_label is None

    preview_result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert preview_result.status == "previewing"
    assert preview_result.label == "label_0"
    assert preview_result.committed_label is None

    commit_result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert commit_result.status == "predicted"
    assert commit_result.committed_label == "label_0"


def test_tsl51_accuracy_mode_does_not_commit_on_preview(monkeypatch) -> None:
    """Accuracy mode: preview mid-sign but commit only at sign-end."""
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model()
    settings = InferenceSettings(
        threshold=0.7,
        alpha=0.4,
        top_k=3,
        motion_min=0.008,
        min_sign_frames=3,
        sign_end_frames=2,
        commit_on_preview=False,
    )
    _patch_motion(monkeypatch)
    times = iter([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(times))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(2):
        assert process_rgb_frame(track, loaded, service, runtime, rgb, settings).status == "buffering"

    preview = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert preview.status == "previewing"
    assert preview.committed_label is None

    still_preview = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert still_preview.status == "previewing"
    assert still_preview.committed_label is None

    # low motion begins
    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda _p, _c: 0.0)
    low1 = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert low1.committed_label is None

    committed = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert committed.status == "predicted"
    assert committed.committed_label == "label_1"


def test_tsl51_sign_end_uses_seq_buf_when_motion_frames_short(monkeypatch) -> None:
    """Commit at sign-end using seq_buf when motion-gated sign_frames are too few."""
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model()
    settings = InferenceSettings(
        threshold=0.7,
        min_sign_frames=6,
        sign_end_frames=2,
        commit_on_preview=False,
        prefer_seq_buf_on_commit=True,
    )
    _patch_motion(monkeypatch, motion=0.02)
    times = iter([0.05 * i for i in range(1, 30)])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(times))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    # Build seq_buf with enough hand frames while motion gate only gets a few sign_frames
    for _ in range(8):
        process_rgb_frame(track, loaded, service, runtime, rgb, settings)

    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda _p, _c: 0.0)
    low1 = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert low1.committed_label is None

    committed = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert committed.committed_label == "label_1"


def test_tsl51_margin_gate_rejects_ambiguous_prediction(monkeypatch) -> None:
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model(predictor=AmbiguousPredictor())
    settings = InferenceSettings(
        threshold=0.5,
        min_confidence_margin=0.12,
        min_sign_frames=2,
        sign_end_frames=2,
        commit_on_preview=False,
    )
    _patch_motion(monkeypatch)
    times = iter([0.1, 0.2, 0.3, 0.4, 0.5])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(times))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    preview = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert preview.status == "low_confidence"
    assert preview.committed_label is None


def test_tsl51_smoother_resets_on_new_sign(monkeypatch) -> None:
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model()
    settings = InferenceSettings(
        threshold=0.7,
        min_sign_frames=2,
        sign_end_frames=2,
        commit_on_preview=False,
    )
    _patch_motion(monkeypatch)
    times = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(times))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert service.smoother._smoothed is not None

    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda _p, _c: 0.0)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)

    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda _p, _c: 0.02)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert service.sign_frames
    assert service.smoother._smoothed is None


def test_tsl51_with_min_sign_frames_one_shows_signing_not_predicted(monkeypatch) -> None:
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model()
    settings = InferenceSettings(threshold=0.7, alpha=0.4, top_k=3, motion_min=0.008)
    settings.min_sign_frames = 1
    settings.sign_end_frames = 8

    monkeypatch.setattr(
        inference_module,
        "extract_holistic_frame",
        lambda _results: np.ones(FEATURE_DIM, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_module,
        "_extract_hand_coords",
        lambda _results: np.ones(126, dtype=np.float32),
    )

    result = process_rgb_frame(
        track,
        loaded,
        service,
        runtime,
        np.zeros((8, 8, 3), dtype=np.uint8),
        settings,
    )

    assert result.status == "signing"
    assert result.buffering is None
    assert result.committed_label is None


def test_tsl51_preview_then_commit_after_stable_predictions(monkeypatch) -> None:
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model(predictor=ModeratePredictor())
    settings = InferenceSettings(
        threshold=0.7,
        alpha=0.4,
        top_k=3,
        motion_min=0.008,
        min_sign_frames=8,
        sign_end_frames=8,
        commit_on_preview=True,
    )
    _patch_motion(monkeypatch)
    monotonic_values = iter([0.00, 0.05, 0.10, 0.15, 0.20, 0.33, 0.52, 0.68])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(monotonic_values))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    preview_result = None
    for _ in range(5):
        preview_result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)

    assert preview_result is not None
    assert preview_result.status == "previewing"
    assert preview_result.label == "label_0"
    assert preview_result.committed_label is None

    interim = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert interim.status == "predicted"
    assert interim.label == "label_0"
    assert interim.committed_label == "label_0"

    commit_result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert commit_result.status == "signing"
    assert commit_result.committed_label is None


def test_tsl51_previews_within_three_frames_when_min_sign_frames_is_three(monkeypatch) -> None:
    track = TRACKS["tsl51"]
    service = PredictService(track, alpha=0.4)
    runtime = DummyRuntime()
    loaded = make_loaded_model()
    settings = InferenceSettings(
        threshold=0.7,
        alpha=0.4,
        top_k=3,
        motion_min=0.008,
        min_sign_frames=3,
        sign_end_frames=8,
        commit_on_preview=False,
    )
    _patch_motion(monkeypatch)
    monotonic_values = iter([0.05, 0.10, 0.15])
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(monotonic_values))

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(2):
        result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        assert result.status == "buffering"

    preview_result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert preview_result.status == "previewing"
    assert preview_result.label == "label_1"
    assert preview_result.committed_label is None


class DummyHands:
    def process(self, _rgb: np.ndarray):
        return object()


class FsDummyRuntime(DummyRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.hands = DummyHands()


class FsHighMarginPredictor:
    def predict(self, _data: np.ndarray) -> np.ndarray:
        return np.array([0.88, 0.08], dtype=np.float32)


class FsLowMarginPredictor:
    def predict(self, _data: np.ndarray) -> np.ndarray:
        return np.array([0.52, 0.48], dtype=np.float32)


def make_fs_loaded_model(*, predictor: object) -> LoadedModel:
    return LoadedModel(
        predictor=predictor,
        labels={"0": "KO_KAI", "1": "BOR_BAI_MAI"},
        scaler=IdentityScaler(),
        backend="test",
        model_path=Path("model.keras"),
        labels_path=Path("labels.json"),
        scaler_path=Path("scaler.pkl"),
        load_time_ms=0.0,
    )


def _patch_fs_hand(monkeypatch) -> None:
    monkeypatch.setattr(
        inference_module,
        "extract_and_normalize",
        lambda _results: np.ones(63, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_module,
        "landmarks_from_hands",
        lambda _results: {"left_hand": None, "right_hand": None},
    )
    monkeypatch.setattr(
        inference_module,
        "hands_detected_from_hands",
        lambda _results: {"left": True, "right": False},
    )


def test_fingerspelling_rejects_low_margin(monkeypatch) -> None:
    track = TRACKS["fingerspelling"]
    service = PredictService(track, alpha=1.0)
    runtime = FsDummyRuntime()
    loaded = make_fs_loaded_model(predictor=FsLowMarginPredictor())
    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        min_confidence_margin=0.10,
        prediction_stable_frames=2,
    )
    _patch_fs_hand(monkeypatch)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert result.status == "low_confidence"
    assert result.committed_label is None
    assert "Unknown" in result.label


def test_fingerspelling_stable_frames_commit(monkeypatch) -> None:
    track = TRACKS["fingerspelling"]
    service = PredictService(track, alpha=1.0)
    runtime = FsDummyRuntime()
    loaded = make_fs_loaded_model(predictor=FsHighMarginPredictor())
    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        min_confidence_margin=0.10,
        prediction_stable_frames=2,
    )
    _patch_fs_hand(monkeypatch)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    first = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert first.status == "previewing"
    assert first.label == "KO_KAI"
    assert first.committed_label is None

    second = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert second.status == "predicted"
    assert second.committed_label == "KO_KAI"


def test_fingerspelling_no_hand_resets_candidate(monkeypatch) -> None:
    track = TRACKS["fingerspelling"]
    service = PredictService(track, alpha=1.0)
    runtime = FsDummyRuntime()
    loaded = make_fs_loaded_model(predictor=FsHighMarginPredictor())
    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        min_confidence_margin=0.10,
        prediction_stable_frames=2,
    )
    _patch_fs_hand(monkeypatch)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    monkeypatch.setattr(inference_module, "extract_and_normalize", lambda _results: None)
    process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    monkeypatch.setattr(
        inference_module,
        "extract_and_normalize",
        lambda _results: np.ones(63, dtype=np.float32),
    )
    again = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert again.status == "previewing"
    assert again.committed_label is None
