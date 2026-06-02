# Golden tests only

This directory holds **JSON golden vectors** for Rust parity tests (`golden/`).

Python unit tests live under `python-legacy/tests/`. Regenerate goldens from the repo root:

```bash
python scripts/export_golden.py
cargo test --workspace
```
