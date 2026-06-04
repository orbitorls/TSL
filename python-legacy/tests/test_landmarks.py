from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsl_translate.landmarks import (  # noqa: E402
    hands_detected_from_hands,
    hands_detected_from_holistic,
    landmarks_from_hands,
    landmarks_from_holistic,
)


def _lm(x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _list(*points: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(landmark=points)


def test_landmarks_from_holistic_empty() -> None:
    results = SimpleNamespace(
        pose_landmarks=None,
        face_landmarks=None,
        left_hand_landmarks=None,
        right_hand_landmarks=None,
    )
    out = landmarks_from_holistic(results)
    assert out["pose"] is None
    assert out["left_hand"] is None
    assert hands_detected_from_holistic(results) == {"left": False, "right": False}


def test_landmarks_from_holistic_hands() -> None:
    results = SimpleNamespace(
        pose_landmarks=_list(_lm(0.5, 0.5, visibility=0.9)),
        face_landmarks=_list(_lm(0.4, 0.4)),
        left_hand_landmarks=_list(_lm(0.2, 0.3)),
        right_hand_landmarks=_list(_lm(0.8, 0.3)),
    )
    out = landmarks_from_holistic(results)
    assert out["pose"] is not None and len(out["pose"][0]) == 4
    assert out["face"] is not None and len(out["face"][0]) == 3
    assert out["left_hand"] is not None
    assert out["right_hand"] is not None
    assert hands_detected_from_holistic(results) == {"left": True, "right": True}


def test_landmarks_from_hands_multi() -> None:
    left_hand = _list(*(_lm(0.1 * i, 0.2) for i in range(21)))
    right_hand = _list(*(_lm(0.5 + 0.01 * i, 0.2) for i in range(21)))
    results = SimpleNamespace(
        multi_hand_landmarks=[left_hand, right_hand],
        multi_handedness=[
            SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.9)]),
            SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.8)]),
        ],
    )
    out = landmarks_from_hands(results)
    assert len(out["left_hand"] or []) == 21
    assert len(out["right_hand"] or []) == 21
    assert hands_detected_from_hands(results) == {"left": True, "right": True}


def test_frame_to_message_includes_landmarks() -> None:
    from translate_api.ws import inference as ws_module
    from tsl_translate.inference import FrameResult, TopKEntry

    result = FrameResult(
        label="test",
        confidence=0.9,
        landmarks={"pose": None, "face": None, "left_hand": [[0.1, 0.2, 0.0]], "right_hand": None},
        hands_detected={"left": True, "right": False},
    )
    msg = ws_module._frame_to_message(result, "", [0.9])
    assert msg["hands_detected"] == {"left": True, "right": False}
    assert msg["landmarks"]["left_hand"] == [[0.1, 0.2, 0.0]]
