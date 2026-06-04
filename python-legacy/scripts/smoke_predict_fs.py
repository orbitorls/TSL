"""Smoke test: load retrain artifacts and predict on a few test images.

Run from python-legacy dir:
  .venv/Scripts/python.exe scripts/smoke_predict_fs.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.keypoints import extract_and_normalize

ART = ROOT.parent / "artifacts" / "fingerspelling"
TEST_ROOT = ROOT.parent / "data" / "one_stage_tfs_ready" / "One-Stage-TFS Thai One-Stage Fingerspelling Dataset" / "Test set"

print(f"[load] {ART}")
model = tf.keras.models.load_model(ART / "model.keras")
scaler = joblib.load(ART / "scaler.pkl")
labels = {int(k): v for k, v in (eval(open(ART / 'labels.json').read())).items()}
print(f"[load] {len(labels)} classes")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4)

correct = 0
total = 0
per_class = {}
classes = sorted([p.name for p in TEST_ROOT.iterdir() if p.is_dir()])
print(f"[test] {len(classes)} classes under {TEST_ROOT.name}")

def _find_images(d: Path) -> list[Path]:
    out = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if out:
        return out
    out = []
    for sub in d.iterdir():
        if sub.is_dir():
            out.extend(_find_images(sub))
    return out

for cls in classes:
    cls_dir = TEST_ROOT / cls
    images = _find_images(cls_dir)
    if not images:
        continue
    n_correct = 0
    n_total = 0
    sample_preds = []
    for img_path in images[:30]:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        feat = extract_and_normalize(res)
        if feat is None:
            continue
        x = scaler.transform(feat.reshape(1, -1).astype(np.float32))
        pred = int(np.argmax(model.predict(x, verbose=0), axis=1)[0])
        pred_label = labels[pred]
        n_total += 1
        if pred_label == cls:
            n_correct += 1
        elif len(sample_preds) < 3:
            sample_preds.append((img_path.name, pred_label))
    per_class[cls] = (n_correct, n_total)
    correct += n_correct
    total += n_total
    if sample_preds:
        print(f"  {cls:15s} {n_correct:3d}/{n_total:3d}  mispreds: {sample_preds}")
    else:
        print(f"  {cls:15s} {n_correct:3d}/{n_total:3d}")

print(f"\n[totals] {correct}/{total} = {correct/total:.4f}" if total else "no samples")
hands.close()
