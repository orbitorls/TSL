from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tsl_translate.inference as inference_module
from tsl_translate.inference import InferenceSettings, process_rgb_frame
from tsl_translate.session import LoadedModel, PredictService
from tsl_translate.tracks import TRACKS


class DummyScaler:
    def transform(self, values):
        return values


class DummyPredictor:
    def predict(self, values):
        return np.array([[0.75, 0.24]], dtype=np.float32)


class DummyHolistic:
    def process(self, rgb):
        return object()


class DummyRuntime:
    def __init__(self) -> None:
        self.holistic = DummyHolistic()


def test_tsl51_fast_mode_commits_on_second_preview(monkeypatch) -> None:
    """Fast mode (commit_on_preview=True): frame 3 previews, frame 4 commits."""
    frame = np.ones(162, dtype=np.float32)
    motion_times = iter([0.05 * i for i in range(1, 20)])

    monkeypatch.setattr(inference_module, "extract_holistic_frame", lambda results: frame)
    monkeypatch.setattr(inference_module, "_extract_hand_coords", lambda results: np.ones(126, dtype=np.float32))
    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda prev, curr: 0.02)
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(motion_times))

    loaded = LoadedModel(
        predictor=DummyPredictor(),
        labels={"0": "ทดสอบ", "1": "อื่น"},
        scaler=DummyScaler(),
        backend="keras",
        model_path=Path("model.keras"),
        labels_path=Path("labels.json"),
        scaler_path=Path("scaler.pkl"),
        load_time_ms=0.0,
    )
    service = PredictService(TRACKS["tsl51"], alpha=0.4)
    runtime = DummyRuntime()
    settings = InferenceSettings(
        threshold=0.7,
        alpha=0.4,
        top_k=3,
        motion_min=0.008,
        sign_end_frames=5,
        min_sign_frames=3,
        commit_on_preview=True,
    )

    for expected in range(1, 3):
        result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, np.zeros((4, 4, 3), dtype=np.uint8), settings)
        assert result.status == "buffering"
        assert result.buffering == f"{expected}/3"
        assert result.committed_label is None

    result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, np.zeros((4, 4, 3), dtype=np.uint8), settings)
    assert result.status == "previewing"
    assert result.label == "ทดสอบ"
    assert result.committed_label is None

    result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, np.zeros((4, 4, 3), dtype=np.uint8), settings)
    assert result.status == "predicted"
    assert result.buffering is None
    assert result.committed_label == "ทดสอบ"

    # Moderate confidence: pred index 0 -> "ทดสอบ" at 0.75

    result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, np.zeros((4, 4, 3), dtype=np.uint8), settings)
    assert result.status == "signing"
    assert result.committed_label is None


def test_tsl51_accuracy_mode_commits_at_sign_end(monkeypatch) -> None:
    """Accuracy defaults: preview mid-sign, commit only after low-motion sign-end."""
    frame = np.ones(162, dtype=np.float32)
    motion_times = iter([0.05 * i for i in range(1, 30)])

    monkeypatch.setattr(inference_module, "extract_holistic_frame", lambda results: frame)
    monkeypatch.setattr(inference_module, "_extract_hand_coords", lambda results: np.ones(126, dtype=np.float32))
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(motion_times))

    motion = {"value": 0.02}

    def _motion(_prev, _curr):
        return motion["value"]

    monkeypatch.setattr(inference_module, "_mean_hand_displacement", _motion)

    loaded = LoadedModel(
        predictor=DummyPredictor(),
        labels={"0": "ทดสอบ", "1": "อื่น"},
        scaler=DummyScaler(),
        backend="keras",
        model_path=Path("model.keras"),
        labels_path=Path("labels.json"),
        scaler_path=Path("scaler.pkl"),
        load_time_ms=0.0,
    )
    service = PredictService(TRACKS["tsl51"], alpha=0.4)
    runtime = DummyRuntime()
    settings = InferenceSettings(
        threshold=0.7,
        min_sign_frames=10,
        sign_end_frames=3,
        commit_on_preview=False,
    )

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    for expected in range(1, 6):
        result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
        assert result.status == "buffering"
        assert result.committed_label is None

    for _ in range(5):
        result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
        assert result.committed_label is None

    result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
    assert result.status == "previewing"
    assert result.committed_label is None

    motion["value"] = 0.0
    for _ in range(2):
        result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
        assert result.committed_label is None

    result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
    assert result.status == "predicted"
    assert result.committed_label == "ทดสอบ"


def test_early_commit_eligible_threshold_and_margin() -> None:
    settings = InferenceSettings(
        threshold=0.62,
        min_confidence_margin=0.10,
        commit_on_preview=True,
    )
    assert inference_module._early_commit_eligible(
        settings, allow_commit=True, conf=0.96, margin=0.94
    )
    assert not inference_module._early_commit_eligible(
        settings, allow_commit=True, conf=0.68, margin=0.38
    )


def test_tsl51_balanced_early_commit_clears_sign_frames(monkeypatch) -> None:
    """High-confidence preview commits once and clears sign_frames (skips sign-end wait)."""
    frame = np.ones(162, dtype=np.float32)
    motion_times = iter([0.05 * i for i in range(1, 30)])
    predict_calls = {"n": 0}

    monkeypatch.setattr(inference_module, "extract_holistic_frame", lambda results: frame)
    monkeypatch.setattr(inference_module, "_extract_hand_coords", lambda results: np.ones(126, dtype=np.float32))
    monkeypatch.setattr(inference_module, "_mean_hand_displacement", lambda prev, curr: 0.02)
    monkeypatch.setattr(inference_module.time, "monotonic", lambda: next(motion_times))

    class RampingPredictor:
        def predict(self, values):
            predict_calls["n"] += 1
            if predict_calls["n"] < 6:
                return np.array([[0.68, 0.30]], dtype=np.float32)
            return np.array([[0.96, 0.02]], dtype=np.float32)

    loaded = LoadedModel(
        predictor=RampingPredictor(),
        labels={"0": "ทดสอบ", "1": "อื่น"},
        scaler=DummyScaler(),
        backend="keras",
        model_path=Path("model.keras"),
        labels_path=Path("labels.json"),
        scaler_path=Path("scaler.pkl"),
        load_time_ms=0.0,
    )
    service = PredictService(TRACKS["tsl51"], alpha=0.4)
    runtime = DummyRuntime()
    settings = InferenceSettings(
        threshold=0.75,
        min_sign_frames=4,
        sign_end_frames=3,
        min_confidence_margin=0.10,
        commit_on_preview=True,
    )
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)

    high_commit = None
    for _ in range(15):
        result = process_rgb_frame(TRACKS["tsl51"], loaded, service, runtime, rgb, settings)
        if result.committed_label and result.confidence >= 0.9:
            high_commit = result
            break

    assert high_commit is not None
    assert high_commit.committed_label == "ทดสอบ"
    assert len(service.sign_frames) == 0
