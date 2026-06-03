from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from keypoints import extract_and_normalize
from sequence_keypoints import extract_holistic_frame, resample_frames
from tsl_translate.session import LoadedModel, MediaPipeRuntime, PredictService
from tsl_translate.tracks import TrackSpec
from webcam_runtime import EMABuffer, format_topk


@dataclass
class InferenceSettings:
    threshold: float = 0.7
    alpha: float = 0.4
    top_k: int = 3
    motion_min: float = 0.008
    sign_end_frames: int = 8   # consecutive low-motion frames → sign ended
    min_sign_frames: int = 15  # minimum sign frames for a valid prediction
    live_predict_cooldown_s: float = 0.35
    preview_interval_s: float = 0.12
    min_confidence_margin: float = 0.12  # top1 - top2 minimum to accept prediction
    commit_on_preview: bool = False  # accuracy mode: commit at sign-end only
    stretch_live: bool = False  # up-sample short sign_frames to seq_len (live path)
    segment_mode: str = "motion"  # "motion" = sign_frames boundary; "rolling" = full seq_buf
    prefer_seq_buf_on_commit: bool = True  # sign-end uses seq_buf (closer to offline eval)


@dataclass
class TopKEntry:
    label: str
    p: float


@dataclass
class FrameResult:
    label: str
    confidence: float
    committed_label: str | None = None
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


def _preview_start_frames(settings: InferenceSettings) -> int:
    return max(2, int(np.ceil(settings.min_sign_frames / 2.0)))


def _confidence_margin(smoothed: np.ndarray, pred_idx: int) -> float:
    if smoothed.size < 2:
        return float(smoothed[pred_idx]) if smoothed.size else 0.0
    top2 = np.partition(smoothed, -2)[-2:]
    return float(np.max(top2) - np.min(top2))


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
    committed_label: str | None = None
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
            if conf >= settings.threshold:
                overlay = pred_label
                status = "predicted"
            else:
                overlay = "Unknown / รอท่าชัดเจน"
                status = "low_confidence"
            topk_text, topk = _topk_entries(smoothed, loaded.labels, settings.top_k)
    else:
        results = mp_runtime.holistic.process(rgb)
        feat = extract_holistic_frame(results)
        now = time.monotonic()
        seq_len = track.expected_seq_len or service.seq_buf.seq_len

        def _commit_frames() -> list[np.ndarray]:
            """Frames used for final sign-end / hands-lost prediction."""
            buf = service.seq_buf.as_list()
            if settings.prefer_seq_buf_on_commit and len(buf) >= settings.min_sign_frames:
                return buf
            if settings.segment_mode == "rolling" and service.seq_buf.is_full():
                return buf
            return service.sign_frames

        def _predict_on_frames(
            frames: list[np.ndarray],
            *,
            allow_commit: bool,
            stretch: bool = False,
            relax_margin: bool = False,
        ) -> None:
            nonlocal overlay, conf, committed_label, status, topk_text, topk, buffering
            seg = resample_frames(np.asarray(frames, dtype=np.float32), seq_len, stretch=stretch)
            seq_scaled = loaded.scaler.transform(seg).reshape(1, seq_len, seg.shape[1])
            probs = np.asarray(loaded.predictor.predict(seq_scaled), dtype=np.float32).reshape(-1)
            smoothed = service.smoother.update(probs)
            pred_idx = int(np.argmax(smoothed))
            conf = float(smoothed[pred_idx])
            margin = _confidence_margin(smoothed, pred_idx)
            pred_label = loaded.labels.get(str(pred_idx), "?")
            buffering = None
            confident = (
                conf >= settings.threshold
                and (relax_margin or margin >= settings.min_confidence_margin)
                and pred_label not in ("", "?")
            )
            if confident:
                overlay = pred_label
                if service.candidate_label == pred_label:
                    service.candidate_count += 1
                else:
                    service.candidate_label = pred_label
                    service.candidate_count = 1
                can_preview_commit = (
                    allow_commit
                    and settings.commit_on_preview
                    and not service.committed_this_sign
                    and service.candidate_count >= 2
                )
                can_sign_end_commit = (
                    not allow_commit
                    and not service.committed_this_sign
                )
                if can_preview_commit or can_sign_end_commit:
                    committed_label = pred_label
                    status = "predicted"
                    service.committed_this_sign = True
                    service.last_prediction_time = now
                elif allow_commit:
                    status = "previewing"
                else:
                    status = "predicted"
            else:
                overlay = "Unknown / รอท่าชัดเจน"
                status = "low_confidence"
                service.candidate_label = None
                service.candidate_count = 0
            topk_text, topk = _topk_entries(smoothed, loaded.labels, settings.top_k)

        if feat is None:
            # No hands — predict on accumulated sign frames before resetting
            if (
                (
                    len(service.sign_frames) >= settings.min_sign_frames
                    or (
                        settings.prefer_seq_buf_on_commit
                        and len(service.seq_buf.as_list()) >= settings.min_sign_frames
                    )
                )
                and not service.committed_this_sign
            ):
                _predict_on_frames(
                    _commit_frames(),
                    allow_commit=False,
                    stretch=settings.stretch_live,
                    relax_margin=True,
                )
                if conf >= settings.threshold and overlay not in ("", "?", "Unknown / รอท่าชัดเจน"):
                    committed_label = overlay
                    status = "predicted"
                    service.committed_this_sign = True
                    service.last_prediction_time = now
            if now - service.last_hand_time > 0.5:
                service.reset()
            else:
                service.sign_frames.clear()
                service.low_motion_count = 0
                service.candidate_label = None
                service.candidate_count = 0
            overlay = overlay or "ไม่พบมือ"
            status = status if status != "ready" else "no_hand"
        else:
            service.last_hand_time = now
            service.seq_buf.push(feat)
            hand_coords = _extract_hand_coords(results)
            motion = _mean_hand_displacement(service.prev_hand_coords, hand_coords)
            if service.prev_hand_coords is None and hand_coords is not None:
                motion = settings.motion_min
            service.prev_hand_coords = hand_coords

            if motion >= settings.motion_min:
                if not service.sign_frames:
                    service.committed_this_sign = False  # new sign starting
                    service.candidate_label = None
                    service.candidate_count = 0
                    service.smoother.reset()
                service.sign_frames.append(feat)
                service.low_motion_count = 0
                n_sign = len(service.sign_frames)
                preview_gate = _preview_start_frames(settings)
                if settings.min_sign_frames <= 1 and n_sign == 1:
                    status = "signing"
                    overlay = f"กำลังเซ็น {n_sign} เฟรม"
                    buffering = None
                elif n_sign <= preview_gate:
                    status = "buffering"
                    overlay = f"กำลังเก็บเฟรม {n_sign}/{settings.min_sign_frames}"
                    buffering = f"{n_sign}/{settings.min_sign_frames}"
                else:
                    status = "signing"
                    overlay = f"กำลังเซ็น {n_sign} เฟรม"
                    buffering = None
                    if (
                        (
                            n_sign >= settings.min_sign_frames
                            or now - service.last_preview_time >= settings.preview_interval_s
                        )
                        and not service.committed_this_sign
                    ):
                        service.last_preview_time = now
                        frames = (
                            service.seq_buf.as_list()
                            if settings.segment_mode == "rolling" and service.seq_buf.is_full()
                            else service.sign_frames
                        )
                        _predict_on_frames(frames, allow_commit=True, stretch=settings.stretch_live)
            else:
                service.low_motion_count += 1
                n = len(service.seq_buf)
                total = service.seq_buf.seq_len
                status = "buffering" if not service.seq_buf.is_full() else "waiting_motion"
                overlay = f"กำลังเก็บเฟรม {n}/{total}"
                buffering = f"{n}/{total}"

            # Sign ended: enough low-motion frames AND enough sign content
            buf_frames = len(service.seq_buf.as_list())
            has_enough = len(service.sign_frames) >= settings.min_sign_frames or (
                settings.prefer_seq_buf_on_commit and buf_frames >= settings.min_sign_frames
            )
            sign_ended = (
                service.low_motion_count >= settings.sign_end_frames
                and has_enough
                and not service.committed_this_sign
            )
            if sign_ended:
                frames = _commit_frames()
                _predict_on_frames(
                    frames,
                    allow_commit=False,
                    stretch=settings.stretch_live,
                    relax_margin=True,
                )
                if conf >= settings.threshold and overlay not in ("", "?", "Unknown / รอท่าชัดเจน"):
                    committed_label = overlay
                    status = "predicted"
                    service.committed_this_sign = True
                    service.last_prediction_time = now
                service.sign_frames.clear()
                service.candidate_label = None
                service.candidate_count = 0

    fps = 1.0 / max(1e-6, (time.perf_counter() - t0))
    return FrameResult(
        label=overlay,
        confidence=conf,
        committed_label=committed_label,
        topk=topk,
        topk_text=topk_text,
        status=status,
        buffering=buffering,
        fps=fps,
    )
