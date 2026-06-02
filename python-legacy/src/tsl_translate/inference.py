from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from keypoints import extract_and_normalize
from sequence_keypoints import extract_holistic_frame
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService
from tsl_translate.tracks import TrackSpec
from webcam_runtime import EMABuffer, format_topk


@dataclass
class InferenceSettings:
    threshold: float = 0.7
    alpha: float = 0.4
    top_k: int = 3
    motion_min: float = 0.008


@dataclass
class TopKEntry:
    label: str
    p: float


@dataclass
class FrameResult:
    label: str
    confidence: float
    topk: list[TopKEntry] = field(default_factory=list)
    topk_text: str = ""
    status: str = "ready"
    buffering: str | None = None
    fps: float = 0.0


def _extract_hand_coords(results) -> np.ndarray | None:
    """Return fixed-size (126,) array of left+right hand xyz; zero-fill missing hands.

    Using a fixed size ensures ``prev.shape == curr.shape`` even when the number
    of detected hands changes between frames (1↔2), which would otherwise cause
    ``_mean_hand_displacement`` to return 0.0 and block every prediction.
    """
    left = getattr(results, "left_hand_landmarks", None)
    right = getattr(results, "right_hand_landmarks", None)
    if left is None and right is None:
        return None
    out = np.zeros(126, dtype=np.float32)
    for slot, hand in enumerate((left, right)):
        if hand is not None:
            base = slot * 63
            for j, lm in enumerate(hand.landmark):
                out[base + j * 3] = lm.x
                out[base + j * 3 + 1] = lm.y
                out[base + j * 3 + 2] = lm.z
    return out


def _mean_hand_displacement(prev: np.ndarray | None, curr: np.ndarray | None) -> float:
    if prev is None or curr is None or prev.shape != curr.shape:
        return 0.0
    n_pts = prev.shape[0] // 3
    p = prev.reshape(n_pts, 3)
    c = curr.reshape(n_pts, 3)
    return float(np.mean(np.linalg.norm(c - p, axis=1)))


def _topk_entries(smoothed: np.ndarray, labels: dict[str, str], top_k: int) -> tuple[str, list[TopKEntry]]:
    text, pairs = format_topk(smoothed, labels, top_k)
    entries = [TopKEntry(label=label, p=prob) for label, prob in pairs]
    return text, entries


def process_rgb_frame(
    track: TrackSpec,
    loaded: LoadedModel,
    service: PredictService,
    mp_runtime: MediaPipeRuntime,
    rgb: np.ndarray,
    settings: InferenceSettings,
) -> FrameResult:
    service.set_alpha(settings.alpha)
    t0 = time.perf_counter()

    overlay = ""
    topk_text = ""
    topk: list[TopKEntry] = []
    conf = 0.0
    status = "ready"
    buffering: str | None = None

    if track.key == "fingerspelling":
        results = mp_runtime.hands.process(rgb)
        feat = extract_and_normalize(results)
        if feat is None:
            service.smoother.reset()
            overlay = "ไม่พบมือ"
            status = "no_hand"
        else:
            feat_scaled = loaded.scaler.transform(feat.reshape(1, -1))
            probs = loaded.predictor.predict(feat_scaled)
            smoothed = service.smoother.update(probs)
            pred_idx = int(np.argmax(smoothed))
            conf = float(smoothed[pred_idx])
            pred_label = loaded.labels.get(str(pred_idx), "?")
            overlay = pred_label if conf >= settings.threshold else "?"
            topk_text, topk = _topk_entries(smoothed, loaded.labels, settings.top_k)
    else:
        results = mp_runtime.holistic.process(rgb)
        feat = extract_holistic_frame(results)
        now = time.monotonic()
        if feat is None:
            if now - service.last_hand_time > 0.5:
                service.seq_buf.reset()
                service.smoother.reset()
                service.prev_hand_coords = None
            overlay = "ไม่พบมือ"
            status = "no_hand"
        else:
            service.last_hand_time = now
            service.seq_buf.push(feat)
            hand_coords = _extract_hand_coords(results)
            motion = _mean_hand_displacement(service.prev_hand_coords, hand_coords)
            service.prev_hand_coords = hand_coords

            if (
                service.seq_buf.is_full()
                and motion >= settings.motion_min
                and (now - service.last_prediction_time) >= 0.35
            ):
                service.last_prediction_time = now
                seq = service.seq_buf.get_padded()
                seq_scaled = loaded.scaler.transform(seq).reshape(1, seq.shape[0], seq.shape[1])
                probs = loaded.predictor.predict(seq_scaled)
                smoothed = service.smoother.update(probs)
                pred_idx = int(np.argmax(smoothed))
                conf = float(smoothed[pred_idx])
                pred_label = loaded.labels.get(str(pred_idx), "?")
                overlay = pred_label if conf >= settings.threshold else "?"
                topk_text, topk = _topk_entries(smoothed, loaded.labels, settings.top_k)
            else:
                n = len(service.seq_buf)
                total = service.seq_buf.seq_len
                overlay = f"กำลังเก็บเฟรม {n}/{total}"
                status = "buffering"
                buffering = f"{n}/{total}"

    fps = 1.0 / max(1e-6, (time.perf_counter() - t0))
    return FrameResult(
        label=overlay,
        confidence=conf,
        topk=topk,
        topk_text=topk_text,
        status=status,
        buffering=buffering,
        fps=fps,
    )
