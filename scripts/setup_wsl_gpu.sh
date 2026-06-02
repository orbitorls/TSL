#!/bin/bash
# setup_wsl_gpu.sh — One-shot setup for GPU training in WSL2 Ubuntu
# Run from Ubuntu terminal after first login:
#   bash /mnt/d/TSL/scripts/setup_wsl_gpu.sh
#
# Prerequisites: RTX 4060, Windows driver >= 450.80, WSL2 Ubuntu 22.04

set -e
echo "=== TSL-51 GPU Training Setup ==="
echo ""

# 1. System update
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git curl

# 2. Create Python venv
cd /mnt/d/TSL
python3.11 -m venv ~/venvs/tsl
source ~/venvs/tsl/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install TF with CUDA + other deps
# TF 2.16 on Linux supports CUDA 12.x natively
pip install tensorflow[and-cuda]==2.16.2
pip install mediapipe==0.10.14 opencv-python-headless scikit-learn joblib \
    huggingface_hub datasets pandas tqdm numpy"<2.0"

# 5. Verify GPU
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TF {tf.__version__} — GPUs: {gpus}')
if not gpus:
    print('WARNING: No GPU detected. Check nvidia-smi in WSL.')
else:
    print('GPU ready!')
"

echo ""
echo "=== Setup complete! Run training with ==="
echo "source ~/venvs/tsl/bin/activate"
echo "cd /mnt/d/TSL"
echo "python python-legacy/scripts/train_local_all.py \\"
echo "  --tracks tsl51 \\"
echo "  --tsl51-epochs 40 \\"
echo "  --tsl51-batch-size 32 \\"
echo "  --artifact-dir /mnt/d/TSL/artifacts/tsl51-fullset \\"
echo "  --work-root ~/tsl_training \\"
echo "  --tsl51-download-workers 16 \\"
echo "  --skip-tflite"
