"""
src/keypoints.py — single source of truth for hand keypoint extraction and normalization.

Imported by BOTH the Colab training notebook and webcam_demo.py so that training
and inference always use exactly the same feature representation.

Feature vector: 63 floats = 21 landmarks × (x, y, z), wrist-relative and scale-normalized.
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_SIZE = 63          # 21 landmarks × 3 coords
NUM_LANDMARKS = 21
WRIST_IDX = 0              # MediaPipe Hands: landmark 0 is the wrist


# ── Core functions ─────────────────────────────────────────────────────────────

def extract_hand_landmarks(results) -> np.ndarray | None:
    """
    Pull raw (x, y, z) coordinates from a MediaPipe Hands result object.

    Parameters
    ----------
    results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        The return value of mp.solutions.hands.Hands.process(image).

    Returns
    -------
    np.ndarray, shape (63,), dtype float32
        Flattened [x0,y0,z0, x1,y1,z1, …, x20,y20,z20] for the
        *highest-confidence* hand detected.
    None
        If results.multi_hand_landmarks is None or empty.
    """
    if not results.multi_hand_landmarks:
        return None

    # Pick the detection with the highest handedness score when multiple hands
    # are visible, so we always classify exactly one hand.
    if results.multi_handedness and len(results.multi_handedness) > 1:
        scores = [h.classification[0].score for h in results.multi_handedness]
        best_idx = int(np.argmax(scores))
    else:
        best_idx = 0

    hand_landmarks = results.multi_hand_landmarks[best_idx]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(arr: np.ndarray) -> np.ndarray | None:
    """
    Make landmarks wrist-relative and scale-normalized.

    1. Subtract the wrist (landmark 0) so the hand is always centred at origin.
    2. Divide by the Euclidean distance between the wrist (0) and the base of
       the middle finger (landmark 9) — a stable proxy for hand size.

    Parameters
    ----------
    arr : np.ndarray, shape (63,)
        Raw flattened landmark coordinates as returned by extract_hand_landmarks.

    Returns
    -------
    np.ndarray, shape (63,), dtype float32
        Normalised landmarks.
    None
        If the hand span is essentially zero (degenerate / junk detection).
        Callers must skip/drop this sample — never feed a zero-vector to the model.
    """
    arr = arr.reshape(NUM_LANDMARKS, 3)

    # Step 1 – wrist-relative
    wrist = arr[WRIST_IDX].copy()
    arr = arr - wrist

    # Step 2 – scale: distance from wrist to mid-finger MCP (landmark 9)
    hand_span = np.linalg.norm(arr[9])
    if hand_span < 1e-6:
        return None   # degenerate detection — caller must skip, not classify
    arr = arr / hand_span

    return arr.flatten().astype(np.float32)


def extract_and_normalize(results) -> np.ndarray | None:
    """
    Convenience wrapper: extract then normalize in one call.

    Returns None if no hand was detected (caller should skip this sample).
    """
    raw = extract_hand_landmarks(results)
    if raw is None:
        return None
    return normalize_landmarks(raw)
