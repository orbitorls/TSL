from __future__ import annotations

from dataclasses import dataclass

from keypoints import FEATURE_SIZE
from sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT


@dataclass(frozen=True)
class TrackSpec:
    key: str
    title: str
    runtime_track: str
    expected_feature_dim: int
    expected_seq_len: int | None
    default_model: str
    default_labels: str
    default_scaler: str
    manifest: str


TRACKS: dict[str, TrackSpec] = {
    "fingerspelling": TrackSpec(
        key="fingerspelling",
        title="Fingerspelling",
        runtime_track="thai_fingerspelling",
        expected_feature_dim=FEATURE_SIZE,
        expected_seq_len=None,
        default_model="model.keras",
        default_labels="labels.json",
        default_scaler="scaler.pkl",
        manifest="model_manifest.json",
    ),
    "tsl51": TrackSpec(
        key="tsl51",
        title="TSL-51",
        runtime_track="tsl51_word_signs",
        expected_feature_dim=FEATURE_DIM,
        expected_seq_len=SEQ_LEN_DEFAULT,
        default_model="tsl51_model.keras",
        default_labels="tsl51_labels.json",
        default_scaler="tsl51_scaler.pkl",
        manifest="tsl51_model_manifest.json",
    ),
}
