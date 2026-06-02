# Repository Guidelines

## High-Signal Map

- Rust workspace root is canonical (`Cargo.toml`, `crates/*`).
- Workspace members:
  - `tsl-core`: 63-D / 162x60 feature contracts, scaler/labels logic, golden parity tests
  - `tsl-ingest`: One-Stage-TFS + TSL-51 metadata ingest helpers
  - `tsl-ml`: Candle training config/stub (`train_placeholder`)
  - `tsl-infer`: artifact loading, scaler/predictor wiring, EMA helpers
  - `tsl-vision`: landmark backend abstraction (stub + ONNX skeleton)
  - `tsl-cli`: runnable bins (`tsl-demo-fs`, `tsl-demo-51`, `tsl-train`, `tsl-ingest`)
- Legacy Python remains the full training/runtime path under `python-legacy/`.

## Canonical Commands (Repo Root)

- Preflight:
  - `python scripts/export_golden.py`
  - `cd python-legacy && python -m pytest -q`
- Smoke:
  - `cargo run -p tsl-cli --bin tsl-demo-fs -- --dry-run`
  - `cargo run -p tsl-cli --bin tsl-demo-51 -- --dry-run`
  - `cargo run -p tsl-cli --bin tsl-train -- --help`
- Full:
  - Preflight, then `cargo test --workspace`, then `cargo build --release -p tsl-cli`

CI (`.github/workflows/rust.yml`) runs `export_golden.py`, `cargo test --workspace`, and `cargo build --release -p tsl-cli` on `windows-latest`.

## Toolchain and Platform Reality

- `rust-toolchain.toml` sets `channel = "stable"` (not GNU-specific).
- Local GNU bootstrap is script-driven (`scripts/cargo_test_windows.ps1`), including MinGW setup and `rustup override set stable-x86_64-pc-windows-gnu`.
- One-shot local verification on Windows: `scripts/verify_all.ps1`.
- For GPU training, prefer WSL paths (`scripts/train_local_wsl.sh` / `.ps1`) over native Windows TensorFlow.

## Data and Artifacts Contracts

- Canonical dataset/artifact paths:
  - `data/fingerspelling/`
  - `data/tsl51/metadata/`
  - `artifacts/fingerspelling/`
  - `artifacts/tsl51/`
- `tests/golden/` is JSON fixture data for Rust parity only (not pytest tests).
- Golden fixture writer is `scripts/export_golden.py`; `python-legacy/scripts/export_golden.py` is just a redirect wrapper.

## Non-Obvious Gotchas

- Do not mix fingerspelling and TSL-51 artifacts (models/scalers/labels are track-specific).
- `tsl-train` is currently a Candle CPU/stub-oriented path; full training remains `python-legacy/scripts/train_local_all.py`.
- `python-legacy/src/train_paths.py` intentionally supports legacy alias folders (`one_stage_tfs_ready`, `tsl51_raw_local`, etc.).
- `tsl-vision` ONNX files live under `artifacts/vision/`, separate from model artifacts in `artifacts/fingerspelling` and `artifacts/tsl51`.

## Editing Conventions

- Prefer executable truth over prose: if README text conflicts with scripts/config/CI, follow scripts/config/CI.
- Keep changes focused; do not commit generated/cache/binary artifacts (`artifacts/`, `*.npz`, `*.keras`, `*.tflite`, `*.pkl`, pytest caches).
- Rust: `cargo fmt --all` before handoff; use `anyhow` for app-level errors and `thiserror` for typed library errors.
