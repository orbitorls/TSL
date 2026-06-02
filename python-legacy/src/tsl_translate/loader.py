from __future__ import annotations

import time
from pathlib import Path

from tsl_translate.session import LoadedModel
from tsl_translate.tracks import TrackSpec
from webcam_runtime import load_demo_artifacts


def load_artifacts(
    track: TrackSpec,
    model_path: Path,
    labels_path: Path,
    scaler_path: Path,
    manifest_path: Path | None,
) -> LoadedModel:
    t0 = time.perf_counter()
    use_tflite = model_path.suffix.lower() == ".tflite"
    predictor, labels, scaler = load_demo_artifacts(
        str(model_path),
        str(labels_path),
        str(scaler_path),
        use_tflite=use_tflite,
        expected_feature_dim=track.expected_feature_dim,
        expected_sequence_len=track.expected_seq_len,
        track=track.runtime_track,
        manifest_path=manifest_path,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    return LoadedModel(
        predictor=predictor,
        labels=labels,
        scaler=scaler,
        backend="tflite" if use_tflite else "keras",
        model_path=model_path,
        labels_path=labels_path,
        scaler_path=scaler_path,
        load_time_ms=dt,
    )


def save_upload_bytes(
    upload_root: Path,
    track: TrackSpec,
    model_bytes: bytes,
    model_name: str,
    labels_bytes: bytes,
    labels_name: str,
    scaler_bytes: bytes,
    scaler_name: str,
    manifest_bytes: bytes | None = None,
    manifest_name: str | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    session_dir = upload_root / f"{track.key}_{int(time.time())}"
    session_dir.mkdir(parents=True, exist_ok=True)
    model_path = session_dir / model_name
    labels_path = session_dir / labels_name
    scaler_path = session_dir / scaler_name
    model_path.write_bytes(model_bytes)
    labels_path.write_bytes(labels_bytes)
    scaler_path.write_bytes(scaler_bytes)
    manifest_path: Path | None = None
    if manifest_bytes is not None and manifest_name:
        manifest_path = session_dir / manifest_name
        manifest_path.write_bytes(manifest_bytes)
    return model_path, labels_path, scaler_path, manifest_path
