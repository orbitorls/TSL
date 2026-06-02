#!/usr/bin/env bash
# Train both TSL tracks inside WSL2 with GPU (canonical paths under /mnt/d/TSL).
set -euo pipefail

REPO="${TSL_REPO_ROOT:-/mnt/d/TSL}"
LEGACY="${REPO}/python-legacy"
VENV="${TSL_VENV:-$HOME/venvs/tsl}"

if [[ ! -d "$LEGACY" ]]; then
  echo "[error] python-legacy not found at $LEGACY" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "$LEGACY"

echo "[train] preflight"
python scripts/train_local_all.py --preflight

echo "[train] full run (both tracks; auto-detects data/* aliases)"
python scripts/train_local_all.py \
  --tracks both \
  --work-root "${TSL_WORK_ROOT:-$HOME/tsl_training}" \
  --artifact-dir "${REPO}/artifacts"

echo "[train] done — artifacts under ${REPO}/artifacts/{fingerspelling,tsl51}"
echo "[hint] ONNX for Rust demos: see scripts/convert_tflite_to_onnx.md"
