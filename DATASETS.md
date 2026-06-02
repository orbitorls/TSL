# Dataset and Notebook Map

This project intentionally keeps two separate recognition tracks. Do not mix
artifacts between them.

**Rust (primary):** dataset ingest and training are moving into `crates/tsl-ingest`
and `crates/tsl-ml`. **Legacy Python:** notebooks, scripts, and webcam demos live
under `python-legacy/` until feature parity is complete.

## Rust commands (workspace)

From the repository root (requires [Rust](https://rustup.rs/) stable):

```bash
cargo test --workspace          # keypoint/sequence golden tests in tsl-core
cargo build -p tsl-cli --release
cargo run -p tsl-cli --bin tsl-train -- --help
```

Training and ingest CLIs are still stubs in places; see **Interim Rust state** in
`README.md`.

## 1. Thai Fingerspelling

- Dataset: One-Stage-TFS Thai One-Stage Fingerspelling Dataset
- **Legacy** notebook: `python-legacy/notebooks/train_all_datasets.ipynb` (`RUN_FINGERSPELLING=True`)
- **Legacy** live demo: `python-legacy/webcam_demo.py`
- Rust feature contract: `crates/tsl-core` (`keypoints` module, 63-D)
- Input features (Python): 63 MediaPipe Hands values from `python-legacy/src/keypoints.py`
- Output artifacts (canonical dir `artifacts/fingerspelling/`): `model.keras`, `model.tflite`,
  `labels.json`, `scaler.pkl`, `model_manifest.json`

Use this track when the user holds a single static Thai consonant hand shape.

### Local ingest (legacy Python)

```bash
cd python-legacy
python -m pip install -r requirements.txt
python scripts/ingest_local.py fingerspelling --zip /path/to/One-Stage-TFS.zip --output ../data/fingerspelling
```

## 2. TSL-51 Word Signs

- Dataset: `Namonpas/thai-sign-language-tsl51`
- **Legacy** notebook: `python-legacy/notebooks/train_all_datasets.ipynb` (`RUN_TSL51=True`)
- **Legacy** live demo: `python-legacy/webcam_word_demo.py`
- Rust feature contract: `crates/tsl-core` (`sequence` module, 162×60)
- Input features (Python): 60 frames × 162 MediaPipe Holistic values from
  `python-legacy/src/sequence_keypoints.py`
- Output artifacts (canonical dir `artifacts/tsl51/`): `tsl51_model.keras`, `tsl51_model.tflite`,
  `tsl51_labels.json`, `tsl51_scaler.pkl`, `tsl51_model_manifest.json`

Use this track for full word gestures with motion. The dataset license is
CC BY-NC-SA 4.0, so trained models from this track are non-commercial only.

### Local ingest (legacy Python)

```bash
cd python-legacy
python scripts/ingest_local.py tsl51 --output ../data/tsl51
```

Canonical paths (also used by `train_local_all.py` defaults and `config.local.json.example`):

| Purpose | Path |
|---------|------|
| Fingerspelling images | `data/fingerspelling/` |
| TSL-51 metadata CSVs | `data/tsl51/metadata/` |
| External fingerspelling clips | `data/external_fingerspelling/` |
| External TSL-51 clips | `data/external_tsl51/` |
| Fingerspelling artifacts | `artifacts/fingerspelling/` |
| TSL-51 artifacts | `artifacts/tsl51/` |
| WSL scratch (NPZ caches) | `~/tsl_training/` (`work-root`) |

Root `tests/` holds **golden JSON only** for Rust parity (`tests/golden/`). Python pytest lives under `python-legacy/tests/`.

## External Clip Pipeline

External video clips are a separate reviewed-data track for domain-shift testing
and fine-tuning. Keep the manifest schema identical for both tracks:

```csv
video_id,path,track,label,start_s,end_s,source_url,split,license_note,quality_status
```

Only `quality_status=reviewed` rows should enter training. Use `split` values
`train`, `val`, and `external_test`; the manifest parser rejects any `video_id`
that appears in more than one split.

Feature export:

```bash
python scripts/build_external_dataset.py --manifest reports/external_benchmark/external_manifest.csv --track fingerspelling --labels artifacts/fingerspelling/labels.json --out work/external_fingerspelling.npz --require-reviewed
python scripts/build_external_dataset.py --manifest reports/external_benchmark/external_manifest.csv --track tsl51 --labels artifacts/tsl51/tsl51_labels.json --out work/external_tsl51.npz --require-reviewed --min-detected-frames 12
```

Benchmark and report:

```bash
python scripts/evaluate_fingerspelling_video.py --video path/to/video.mp4 --artifact-dir artifacts/fingerspelling --samples path/to/samples.csv --out-dir reports/external_benchmark/fingerspelling_eval --top-k 3
python scripts/evaluate_tsl51_video.py --samples path/to/samples.csv --artifact-dir artifacts/tsl51 --out-dir reports/external_benchmark/tsl51_eval --strategy uniform --target-fps 30
python scripts/external_benchmark_report.py --fingerspelling reports/external_benchmark/fingerspelling_eval/predictions.csv --tsl51 reports/external_benchmark/tsl51_eval/predictions.csv --out-dir reports/external_benchmark
```

Augmented training starts from the original cache plus reviewed external cache:

```bash
python scripts/train_external_augmented.py --track tsl51 --base-cache work/tsl51_features.npz --external-cache work/external_tsl51.npz --out-dir work/tsl51_external_augmented --base-artifact-dir artifacts/tsl51
```

## Generated Files

The following files are regenerated and should not be committed:

- Python/test caches: `__pycache__/`, `.pytest_cache/`, `.pytest_tmp/`
- Feature caches: `keypoints_cache.npz`, `tsl51_features.npz`
- Model binaries and scalers: `*.keras`, `*.tflite`, `*.pkl`
- Evaluation plots: `training_curves.png`, `confusion_matrix.png`,
  `tsl51_training_curves.png`, `tsl51_confusion_matrix.png`
