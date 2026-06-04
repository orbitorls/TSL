#!/usr/bin/env python3
"""Simulate per-sign commit latency for TSL-51 inference presets (unit-test harness)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY = REPO_ROOT / "python-legacy"
SRC = LEGACY / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tsl_translate.inference as inference_module  # noqa: E402
from tsl_translate.inference import InferenceSettings, process_rgb_frame  # noqa: E402
from tsl_translate.session import LoadedModel, PredictService  # noqa: E402
from tsl_translate.tracks import TRACKS  # noqa: E402

PRESETS: dict[str, InferenceSettings] = {
    "accurate": InferenceSettings(
        threshold=0.65,
        min_sign_frames=6,
        sign_end_frames=5,
        min_confidence_margin=0.12,
        commit_on_preview=False,
        transcript_stable_frames=2,
        transcript_debounce_s=0.4,
    ),
    "balanced": InferenceSettings(
        threshold=0.62,
        min_sign_frames=4,
        sign_end_frames=3,
        min_confidence_margin=0.10,
        commit_on_preview=True,
        transcript_stable_frames=1,
        transcript_debounce_s=0.2,
    ),
    "fast": InferenceSettings(
        threshold=0.55,
        min_sign_frames=3,
        sign_end_frames=5,
        min_confidence_margin=0.08,
        commit_on_preview=True,
        transcript_stable_frames=1,
        transcript_debounce_s=0.2,
    ),
}


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


@dataclass
class PresetLatency:
    preset: str
    frames_to_commit: int
    frames_sign_end_path: int | None


def _simulate_commit(settings: InferenceSettings) -> PresetLatency:
    frame = np.ones(162, dtype=np.float32)
    motion_times = iter(0.05 * i for i in range(1, 80))
    motion = {"value": 0.02}

    inference_module.extract_holistic_frame = lambda results: frame  # type: ignore[method-assign]
    inference_module._extract_hand_coords = lambda results: np.ones(126, dtype=np.float32)  # type: ignore[method-assign]
    inference_module._mean_hand_displacement = lambda prev, curr: motion["value"]  # type: ignore[method-assign]
    inference_module.time.monotonic = lambda: next(motion_times)  # type: ignore[method-assign]

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
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    track = TRACKS["tsl51"]

    frames_to_commit = 0
    sign_end_frame: int | None = None
    for frame_idx in range(1, 60):
        result = process_rgb_frame(track, loaded, service, runtime, rgb, settings)
        if result.committed_label:
            frames_to_commit = frame_idx
            break
        if motion["value"] == 0.0 and sign_end_frame is None and frame_idx > settings.min_sign_frames:
            sign_end_frame = frame_idx
        if frame_idx == settings.min_sign_frames + 2:
            motion["value"] = 0.0

    return PresetLatency(
        preset="",
        frames_to_commit=frames_to_commit,
        frames_sign_end_path=sign_end_frame,
    )


def main() -> int:
    out: dict[str, object] = {"presets": {}, "notes": []}
    baseline: int | None = None
    for name, settings in PRESETS.items():
        row = _simulate_commit(settings)
        row.preset = name
        out["presets"][name] = asdict(row)
        if name == "accurate":
            baseline = row.frames_to_commit
        if baseline and row.frames_to_commit:
            speedup = (baseline - row.frames_to_commit) / baseline if baseline else 0.0
            out["presets"][name]["frames_saved_vs_accurate"] = baseline - row.frames_to_commit
            out["presets"][name]["fraction_faster_vs_accurate"] = round(speedup, 3)

    out["notes"].append(
        "frames_to_commit counts motion+stillness frames until committed_label; "
        "lower is faster. Run evaluate_webcam_holdout.py for Top-1 accuracy gate."
    )
    report_path = REPO_ROOT / "reports" / "tsl51_preset_latency_sim.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
