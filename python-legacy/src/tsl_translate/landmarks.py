"""Serialize MediaPipe landmarks for web overlay (normalized image coords, unmirrored)."""

from __future__ import annotations

from typing import Any


def _point(lm: Any, *, include_visibility: bool = False) -> list[float]:
    out = [round(float(lm.x), 5), round(float(lm.y), 5), round(float(lm.z), 5)]
    if include_visibility:
        vis = float(getattr(lm, "visibility", 1.0) or 1.0)
        out.append(round(vis, 5))
    return out


def _landmark_list(lm_list: Any | None, *, include_visibility: bool = False) -> list[list[float]] | None:
    if lm_list is None:
        return None
    return [_point(lm, include_visibility=include_visibility) for lm in lm_list.landmark]


def hands_detected_from_holistic(results: Any) -> dict[str, bool]:
    return {
        "left": getattr(results, "left_hand_landmarks", None) is not None,
        "right": getattr(results, "right_hand_landmarks", None) is not None,
    }


def landmarks_from_holistic(results: Any) -> dict[str, list[list[float]] | None]:
    return {
        "pose": _landmark_list(getattr(results, "pose_landmarks", None), include_visibility=True),
        "face": _landmark_list(getattr(results, "face_landmarks", None)),
        "left_hand": _landmark_list(getattr(results, "left_hand_landmarks", None)),
        "right_hand": _landmark_list(getattr(results, "right_hand_landmarks", None)),
    }


def hands_detected_from_hands(results: Any) -> dict[str, bool]:
    detected = {"left": False, "right": False}
    if not getattr(results, "multi_hand_landmarks", None):
        return detected
    handedness = getattr(results, "multi_handedness", None) or []
    for i, _hand in enumerate(results.multi_hand_landmarks):
        label = "Right"
        if i < len(handedness):
            label = handedness[i].classification[0].label
        if label == "Left":
            detected["left"] = True
        else:
            detected["right"] = True
    return detected


def landmarks_from_hands(results: Any) -> dict[str, list[list[float]] | None]:
    left: list[list[float]] | None = None
    right: list[list[float]] | None = None
    if getattr(results, "multi_hand_landmarks", None):
        handedness = getattr(results, "multi_handedness", None) or []
        for i, hand in enumerate(results.multi_hand_landmarks):
            label = "Right"
            if i < len(handedness):
                label = handedness[i].classification[0].label
            pts = [_point(lm) for lm in hand.landmark]
            if label == "Left":
                left = pts
            else:
                right = pts
    return {"pose": None, "face": None, "left_hand": left, "right_hand": right}
