# Convert TFLite / Keras models to ONNX for Rust inference

Rust demos (`tsl-demo-fs`, `tsl-demo-51`) load `*.onnx` via `tsl-infer` when built with `--features onnx`.

## One-off (Python, recommended)

```bash
pip install tf2onnx onnx tensorflow

# From repo root, after training artifacts exist:
python -m tf2onnx.convert --tflite model.tflite --output model.onnx
python -m tf2onnx.convert --tflite tsl51_model.tflite --output tsl51_model.onnx
```

For Keras:

```bash
python -m tf2onnx.convert --keras model.keras --output model.onnx
```

## Scaler migration

```bash
python scripts/migrate_scaler.py scaler.pkl -o scaler.json
python scripts/migrate_scaler.py tsl51_scaler.pkl -o tsl51_scaler.json
```

Place `*.onnx`, `labels.json`, and `scaler.json` in the repo root or `artifacts/` (gitignored).

## Build Rust with ONNX

```bash
cargo build --release -p tsl-infer --features onnx
cargo build --release -p tsl-cli --features onnx
```

If ONNX files are missing, demos fall back to a softmax stub and print warnings.
