"""
src/sequence_keypoints.py — single source of truth for TSL-51 holistic keypoint extraction
and sequence buffering.

Imported by BOTH the Colab training notebook (train_tsl51_word_signs.ipynb) and
webcam_word_demo.py so that training and inference always use exactly the same
162-D per-frame feature representation and sequence padding convention.

Feature vector: 162 floats per frame, canonical layout:
  - 6 pose landmarks × (x, y, z) = 18   (MediaPipe pose indices 11–16)
  - 6 face landmarks × (x, y, z) = 18   (MediaPipe face indices 105, 70, 300, 334, 61, 291)
  - 21 × (left-hand lm_i, right-hand lm_i) × (x, y, z) = 126   ← INTERLEAVED per landmark
    dims 36–38 = lh0, dims 39–41 = rh0, dims 42–44 = lh1, dims 45–47 = rh1, …

The interleaved hand ordering matches the Hugging Face TSL-51 landmark CSV column order
produced by ``tsl51_csv_column_names()`` and used during training.  Both ``extract_holistic_frame``
(live inference) and ``read_landmark_csv`` (training) MUST produce this exact layout so that
the StandardScaler and model weights apply correctly.

Anchor: subtract the midpoint of left/right shoulders (pose indices 11 and 12)
from ALL coordinates so the representation is translation-invariant.
"""

import numpy as np
from collections import deque

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_DIM = 162
SEQ_LEN_DEFAULT = 60

NUM_POSE_LANDMARKS = 6
NUM_FACE_LANDMARKS = 6
NUM_HAND_LANDMARKS = 21

# MediaPipe Holistic landmark indices (TSL-51 dataset convention)
POSE_INDICES = [11, 12, 13, 14, 15, 16]          # shoulders, elbows, wrists
FACE_INDICES = [105, 70, 300, 334, 61, 291]      # brows and mouth corners
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# TSL-51 landmark CSV layout (pose → face → left hand → right hand).
# Column names match Namonpas/thai-sign-language-tsl51 landmark CSVs on Hugging Face.
CSV_POSE_LANDMARKS = (
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
)
CSV_FACE_LANDMARKS = (
    "lbrow_outer",
    "lbrow_inner",
    "rbrow_inner",
    "rbrow_outer",
    "mouth_right",
    "mouth_left",
)
META_COLUMNS = frozenset({"frame", "t_ms", "Frame", "timestamp"})


def tsl51_csv_column_names() -> list[str]:
    """Return the 162 xyz column names in canonical interleaved order.

    Hand columns are interleaved per landmark:
    lh_x0, lh_y0, lh_z0, rh_x0, rh_y0, rh_z0, lh_x1, …, rh_x20, rh_y20, rh_z20.

    This matches both the Hugging Face TSL-51 landmark CSV layout (training) and the
    output of ``extract_holistic_frame`` (inference).
    """
    cols: list[str] = []
    for prefix in CSV_POSE_LANDMARKS + CSV_FACE_LANDMARKS:
        cols.extend((f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"))
    for i in range(NUM_HAND_LANDMARKS):
        for hand in ("lh", "rh"):
            cols.extend((f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"))
    if len(cols) != FEATURE_DIM:
        raise RuntimeError(f"Expected {FEATURE_DIM} CSV columns, got {len(cols)}")
    return cols


# ── Helpers ───────────────────────────────────────────────────────────────────

def _landmark_xyz(landmark_list, index: int) -> np.ndarray:
    """Return (x, y, z) for one landmark, or zeros if the list/index is missing."""
    if landmark_list is None:
        return np.zeros(3, dtype=np.float32)
    landmarks = landmark_list.landmark
    if index >= len(landmarks):
        return np.zeros(3, dtype=np.float32)
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _extract_pose_block(pose_landmarks) -> np.ndarray:
    """Six pose landmarks flattened to 18 floats."""
    coords = [_landmark_xyz(pose_landmarks, idx) for idx in POSE_INDICES]
    return np.concatenate(coords).astype(np.float32)


def _extract_face_block(face_landmarks) -> np.ndarray:
    """Six face landmarks flattened to 18 floats."""
    coords = [_landmark_xyz(face_landmarks, idx) for idx in FACE_INDICES]
    return np.concatenate(coords).astype(np.float32)


def _extract_hand_block(hand_landmarks) -> np.ndarray:
    """Twenty-one hand landmarks flattened to 63 floats, or zeros when missing."""
    if hand_landmarks is None:
        return np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def _shoulder_anchor(pose_landmarks) -> np.ndarray:
    """Midpoint of left (11) and right (12) shoulders; origin when pose is absent."""
    if pose_landmarks is None:
        return np.zeros(3, dtype=np.float32)
    left = _landmark_xyz(pose_landmarks, LEFT_SHOULDER_IDX)
    right = _landmark_xyz(pose_landmarks, RIGHT_SHOULDER_IDX)
    return ((left + right) / 2.0).astype(np.float32)


# ── Core functions ─────────────────────────────────────────────────────────────

def extract_holistic_frame(results) -> np.ndarray | None:
    """
    Build a 162-D float32 feature vector from a MediaPipe Holistic process() result.

    Parameters
    ----------
    results
        Return value of ``mp.solutions.holistic.Holistic.process(image)``.
        Duck-typed attributes used: ``pose_landmarks``, ``face_landmarks``,
        ``left_hand_landmarks``, ``right_hand_landmarks``.

    Returns
    -------
    np.ndarray, shape (162,), dtype float32
        Shoulder-anchored holistic landmarks in canonical interleaved order matching
        ``tsl51_csv_column_names()``:
          dims  0–17: 6 pose landmarks (x,y,z each)
          dims 18–35: 6 face landmarks (x,y,z each)
          dims 36–161: 21 × (lh_i x,y,z, rh_i x,y,z) — interleaved per landmark
    None
        When *both* ``left_hand_landmarks`` and ``right_hand_landmarks`` are
        missing.  If only one hand is missing, that hand's slots are zero-filled.
    """
    left = getattr(results, "left_hand_landmarks", None)
    right = getattr(results, "right_hand_landmarks", None)
    if left is None and right is None:
        return None

    pose = getattr(results, "pose_landmarks", None)
    face = getattr(results, "face_landmarks", None)

    # Interleave left and right hand per landmark: lh0, rh0, lh1, rh1, …, lh20, rh20
    # This matches tsl51_csv_column_names() and the training feature contract.
    lh = _extract_hand_block(left).reshape(NUM_HAND_LANDMARKS, 3)
    rh = _extract_hand_block(right).reshape(NUM_HAND_LANDMARKS, 3)
    hands = np.empty((NUM_HAND_LANDMARKS, 2, 3), dtype=np.float32)
    hands[:, 0, :] = lh
    hands[:, 1, :] = rh

    parts = [
        _extract_pose_block(pose),
        _extract_face_block(face),
        hands.reshape(NUM_HAND_LANDMARKS * 2 * 3),
    ]
    vec = np.concatenate(parts).astype(np.float32)

    anchor = _shoulder_anchor(pose)
    vec = (vec.reshape(-1, 3) - anchor).reshape(FEATURE_DIM)
    return vec.astype(np.float32)


class SequenceBuffer:
    """
    Fixed-length ring buffer for per-frame holistic features.

    ``get_padded()`` zero-pads *leading* frames when the buffer is not yet full,
    keeping the most recent frames at the end — matching TSL-51 training padding.
    """

    def __init__(self, seq_len: int, feature_dim: int = FEATURE_DIM) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self._frames: deque[np.ndarray] = deque(maxlen=seq_len)

    def __len__(self) -> int:
        """Return the number of frames currently buffered."""
        return len(self._frames)

    def push(self, frame: np.ndarray) -> None:
        """Append one ``(feature_dim,)`` frame; drop the oldest when full."""
        if frame.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected frame shape ({self.feature_dim},), got {frame.shape}"
            )
        self._frames.append(frame.astype(np.float32, copy=False))

    def get_padded(self) -> np.ndarray:
        """Return ``(seq_len, feature_dim)`` with leading zero-padding if not full."""
        out = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        n = len(self._frames)
        if n > 0:
            out[self.seq_len - n :] = np.stack(self._frames, axis=0)
        return out

    def is_full(self) -> bool:
        """True once ``seq_len`` frames have been pushed."""
        return len(self._frames) >= self.seq_len

    def as_list(self) -> list[np.ndarray]:
        """Return buffered frames oldest→newest."""
        return list(self._frames)

    def reset(self) -> None:
        """Clear all buffered frames."""
        self._frames.clear()


# ── TSL-51 CSV loading (training notebook) ────────────────────────────────────

def _parse_csv_float(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0.0
    return float(text)


def _anchor_feature_frames(flat: np.ndarray) -> np.ndarray:
    """Shoulder-anchor each row of shape ``(T, FEATURE_DIM)``."""
    out = np.empty_like(flat, dtype=np.float32)
    for t in range(flat.shape[0]):
        frame = flat[t].reshape(-1, 3)
        anchor = (frame[0] + frame[1]) / 2.0
        out[t] = (frame - anchor).reshape(FEATURE_DIM)
    return out


def read_landmark_csv(source) -> np.ndarray:
    """
    Parse a TSL-51 landmark CSV → ``(num_frames, FEATURE_DIM)`` float32.

    Selects CSV columns in canonical interleaved order (``tsl51_csv_column_names()``)
    and applies shoulder anchoring — producing the same 162-D layout as
    ``extract_holistic_frame``.  This is the layout that training uses; both
    functions MUST agree for the StandardScaler and model to work correctly.
    """
    import csv
    import io
    from pathlib import Path

    col_names = tsl51_csv_column_names()
    close_after = False

    if isinstance(source, bytes):
        handle = io.StringIO(source.decode("utf-8"))
    elif isinstance(source, (str, Path)):
        path = Path(source)
        text = str(source)
        if path.exists():
            handle = open(path, newline="", encoding="utf-8")
            close_after = True
        elif "\n" in text or text.lstrip().startswith("frame,"):
            handle = io.StringIO(text)
        else:
            raise FileNotFoundError(path)
    else:
        handle = io.StringIO(str(source))

    rows: list[list[float]] = []
    try:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        missing = [c for c in col_names if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing expected columns (first 5): {missing[:5]}"
            )
        for row in reader:
            rows.append([_parse_csv_float(row.get(c)) for c in col_names])
    finally:
        if close_after:
            handle.close()

    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    return _anchor_feature_frames(arr)


def resample_frames(
    features: np.ndarray,
    seq_len: int = SEQ_LEN_DEFAULT,
    *,
    stretch: bool = False,
) -> np.ndarray:
    """Canonical function: resample ``(T, FEATURE_DIM)`` → ``(seq_len, FEATURE_DIM)``.

    Resampling strategy
    -------------------
    * **T >= seq_len**: uniform resampling — ``seq_len`` evenly spaced indices
      picked by ``np.linspace(0, T-1, seq_len).round().astype(int)``.
    * **T < seq_len, stretch=False** (default): leading zero-pad so actual
      frames sit at the *end* of the window (matches ``SequenceBuffer.get_padded``
      and the training-data pipeline for offline clips).
    * **T < seq_len, stretch=True**: uniform *up*-sampling via the same
      ``np.linspace`` formula — repeats frames to fill ``seq_len``.  All frames
      are real; no zero prefix.  Use this in the live-serve path so that short
      signs produced by a ~15fps webcam receive the same temporal-coverage
      treatment as full-length training clips.
    * **T == 0**: returns an all-zeros array of shape ``(seq_len, FEATURE_DIM)``.

    Parameters
    ----------
    features:
        Float32 array of shape ``(T, FEATURE_DIM)``.
    seq_len:
        Desired output length (default ``SEQ_LEN_DEFAULT = 60``).
    stretch:
        When ``True`` and ``T < seq_len``, uniformly upsample (repeat frames)
        instead of leading-zero-padding.  Default ``False`` preserves the
        offline-pipeline behaviour.

    Returns
    -------
    np.ndarray, shape ``(seq_len, FEATURE_DIM)``, dtype float32.
    """
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected (T, {FEATURE_DIM}), got {features.shape}"
        )
    t = features.shape[0]
    if t == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    if t >= seq_len:
        idx = np.linspace(0, t - 1, seq_len).round().astype(int)
        return features[idx].astype(np.float32, copy=False)
    if stretch:
        idx = np.linspace(0, t - 1, seq_len).round().astype(int)
        return features[idx].astype(np.float32, copy=False)
    pad = np.zeros((seq_len - t, FEATURE_DIM), dtype=np.float32)
    return np.concatenate([pad, features], axis=0).astype(np.float32)


def pad_truncate_sequence(
    features: np.ndarray, seq_len: int = SEQ_LEN_DEFAULT
) -> np.ndarray:
    """Pad/truncate ``(T, FEATURE_DIM)`` → ``(seq_len, FEATURE_DIM)`` (leading pad).

    .. deprecated::
        This function now delegates to :func:`resample_frames`, which uses
        **uniform resampling** (not head-truncation) when ``T >= seq_len``.
        Prefer calling :func:`resample_frames` directly.  This wrapper is kept
        for backward compatibility with external callers (Rust golden parity
        tests, notebook inline copies, etc.) that only exercise the ``T <
        seq_len`` padding branch, where behaviour is unchanged.
    """
    return resample_frames(features, seq_len)


def csv_to_sequence(source, seq_len: int = SEQ_LEN_DEFAULT) -> np.ndarray:
    """Read one landmark CSV → ``(seq_len, FEATURE_DIM)`` for model input."""
    return resample_frames(read_landmark_csv(source), seq_len)
