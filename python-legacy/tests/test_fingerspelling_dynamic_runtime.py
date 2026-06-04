from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tsl_translate.inference as inference_module  # noqa: E402
from tsl_translate.inference import InferenceSettings, process_rgb_frame  # noqa: E402
from tsl_translate.session import (  # noqa: E402
    LoadedModel,
    MediaPipeRuntime,
    PredictService,
)
from tsl_translate.tracks import TRACKS  # noqa: E402


class IdentityScaler:
    def transform(self, data: np.ndarray) -> np.ndarray:
        return data


class RecordingPredictor:
    """Predictor that records the shape of its last input and returns a fixed vector."""

    def __init__(self, probs: np.ndarray) -> None:
        self.probs = probs.astype(np.float32, copy=False)
        self.shapes: list[tuple[int, ...]] = []
        self.call_count = 0

    def predict(self, data: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.shapes.append(tuple(int(v) for v in data.shape))
        return self.probs


class DummyHands:
    def process(self, _rgb: np.ndarray):
        return object()


class DummyRuntime(MediaPipeRuntime):
    def __init__(self) -> None:
        self.hands = DummyHands()
        self.holistic = None


def _make_runtime() -> DummyRuntime:
    return DummyRuntime()


def _make_loaded(predictor: object) -> LoadedModel:
    return LoadedModel(
        predictor=predictor,
        labels={"0": "KO_KAI", "1": "BOR_BAI_MAI"},
        scaler=IdentityScaler(),
        backend="test",
        model_path=Path("fs_dynamic_model.keras"),
        labels_path=Path("fs_dynamic_labels.json"),
        scaler_path=Path("fs_dynamic_scaler.pkl"),
        load_time_ms=0.0,
    )


def _patch_dynamic_hand(
    monkeypatch,
    *,
    hand_feat: np.ndarray | None = None,
    hand_coords: np.ndarray | None = None,
    motion: float = 0.02,
) -> None:
    """Patch the runtime helpers used by the dynamic branch of process_rgb_frame."""
    if hand_feat is None:
        hand_feat = np.ones(63, dtype=np.float32)
    if hand_coords is None:
        hand_coords = np.ones(126, dtype=np.float32)
    monkeypatch.setattr(
        inference_module,
        "extract_and_normalize",
        lambda _results: hand_feat,
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
    monkeypatch.setattr(
        inference_module,
        "_extract_hand_coords",
        lambda _results: hand_coords,
    )
    monkeypatch.setattr(
        inference_module,
        "_mean_hand_displacement",
        lambda _prev, _curr: motion,
    )


def test_fingerspelling_dynamic_predictor_shape() -> None:
    track = TRACKS["fingerspelling_dynamic"]
    service = PredictService(track, alpha=1.0)

    assert service.seq_buf.seq_len == 30
    assert service.seq_buf.feature_dim == 63

    for _ in range(30):
        service.seq_buf.push(np.random.RandomState(0).randn(63).astype(np.float32))
    assert service.seq_buf.is_full()

    predictor = RecordingPredictor(np.array([0.9, 0.1], dtype=np.float32))
    loaded = _make_loaded(predictor)
    runtime = _make_runtime()

    seq = service.seq_buf.get_padded()
    seq_scaled = loaded.scaler.transform(seq).reshape(1, 30, 63)
    probs = predictor.predict(seq_scaled)

    assert predictor.shapes == [(1, 30, 63)]
    assert probs.shape == (2,)


def test_fingerspelling_dynamic_buffers_until_full(monkeypatch) -> None:
    track = TRACKS["fingerspelling_dynamic"]
    service = PredictService(track, alpha=1.0)
    runtime = _make_runtime()

    predictor = RecordingPredictor(np.array([0.9, 0.1], dtype=np.float32))
    loaded = _make_loaded(predictor)

    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        motion_min=0.008,
        min_sign_frames=12,
        sign_end_frames=2,
        commit_on_preview=True,
        prediction_stable_frames=1,
        min_confidence_margin=0.10,
    )
    _patch_dynamic_hand(monkeypatch)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    for i in range(1, 30):
        result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        assert predictor.call_count == 0, (
            f"predictor must not be called before the buffer is full (frame {i})"
        )
        assert result.committed_label is None

    final = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert predictor.call_count == 1
    assert predictor.shapes == [(1, 30, 63)]
    assert final.status == "predicted"
    assert final.committed_label is not None


def test_fingerspelling_dynamic_low_margin_rejected(monkeypatch) -> None:
    track = TRACKS["fingerspelling_dynamic"]
    service = PredictService(track, alpha=1.0)
    runtime = _make_runtime()

    predictor = RecordingPredictor(np.array([0.51, 0.49], dtype=np.float32))
    loaded = _make_loaded(predictor)

    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        motion_min=0.008,
        min_sign_frames=12,
        sign_end_frames=2,
        commit_on_preview=True,
        prediction_stable_frames=1,
        min_confidence_margin=0.10,
    )
    _patch_dynamic_hand(monkeypatch)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    result = None
    for _ in range(30):
        result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)

    assert predictor.call_count >= 1
    assert result is not None
    assert result.status == "low_confidence"
    assert result.committed_label is None
    assert "Unknown" in result.label


def test_fingerspelling_dynamic_sign_end_commit(monkeypatch) -> None:
    track = TRACKS["fingerspelling_dynamic"]
    service = PredictService(track, alpha=1.0)
    runtime = _make_runtime()

    predictor = RecordingPredictor(np.array([0.9, 0.1], dtype=np.float32))
    loaded = _make_loaded(predictor)

    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        motion_min=0.008,
        min_sign_frames=12,
        sign_end_frames=3,
        commit_on_preview=False,
        prediction_stable_frames=1,
        min_confidence_margin=0.10,
    )
    _patch_dynamic_hand(monkeypatch, motion=0.02)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    high_motion_results: list = []
    for _ in range(30):
        high_motion_results.append(
            process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        )
    assert predictor.call_count == 1
    assert all(r.committed_label is None for r in high_motion_results)

    monkeypatch.setattr(
        inference_module, "_mean_hand_displacement", lambda _p, _c: 0.0
    )
    low_motion_results: list = []
    for _ in range(3):
        low_motion_results.append(
            process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        )

    assert predictor.call_count == 2
    commit_frame = low_motion_results[-1]
    assert commit_frame.status == "predicted"
    assert commit_frame.committed_label is not None


def test_fingerspelling_dynamic_no_hand_resets(monkeypatch) -> None:
    track = TRACKS["fingerspelling_dynamic"]
    service = PredictService(track, alpha=1.0)
    runtime = _make_runtime()

    predictor = RecordingPredictor(np.array([0.9, 0.1], dtype=np.float32))
    loaded = _make_loaded(predictor)

    settings = InferenceSettings(
        threshold=0.7,
        alpha=1.0,
        motion_min=0.008,
        min_sign_frames=12,
        sign_end_frames=2,
        commit_on_preview=True,
        prediction_stable_frames=1,
        min_confidence_margin=0.10,
    )
    _patch_dynamic_hand(monkeypatch, motion=0.02)

    counter = {"i": 0}

    def advancing_monotonic() -> float:
        i = counter["i"]
        counter["i"] += 1
        if i < 30:
            t = float(i) * 0.01
        elif i < 33:
            t = 0.30 + float(i - 30) * 0.6
        else:
            t = 0.30 + 3 * 0.6 + float(i - 33) * 0.01
        return t

    monkeypatch.setattr(inference_module.time, "monotonic", advancing_monotonic)

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(30):
        process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert predictor.call_count == 1

    monkeypatch.setattr(
        inference_module, "extract_and_normalize", lambda _results: None
    )
    for _ in range(3):
        process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert predictor.call_count == 1, (
        "no-hand frames must not trigger extra predictor calls once the sign "
        "has been committed and state is cleared"
    )

    monkeypatch.setattr(
        inference_module,
        "extract_and_normalize",
        lambda _results: np.ones(63, dtype=np.float32),
    )
    for _ in range(10):
        process_rgb_frame(track, loaded, service, runtime, rgb, settings)
    assert predictor.call_count == 1
