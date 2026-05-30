"""
TSL-51 Thai Sign Language Training Script
===========================================

A comprehensive training script for Thai Sign Language recognition using 
the TSL-51 dataset from HuggingFace.

FEATURES:
- GPU training with CUDA and Mixed Precision (AMP)
- K-Fold Cross Validation for robust evaluation
- Multiple model architectures (MLP, GRU)
- Class weighting for imbalanced data
- Per-fold normalization (no data leakage)
- Early stopping with patience
- Comprehensive results visualization
- Model auto-save functionality
- Support for multiple datasets

SUPPORTED DATASETS:
==================
1. tsl51_user_sign   - TSL-51 user_sign videos (547 samples, 51 classes)
2. tsl51_expert       - TSL-51 expert videos (1155 original, or ~45k with --include-augmented)
3. tsl51_combined     - Combined user_sign + expert (1702 samples)
4. local              - Custom local dataset (CSV/NumPy format)

DATASET FORMAT FOR LOCAL:
=========================
CSV file format:
- First column: label (sign name)
- Other columns: feature values (hand landmarks + pose)

Or use pre-processed NumPy (.npz):
- X: feature array (n_samples, n_features)
- y: label array (n_samples,)
- classes: class names array

USAGE:
======
# Default training (TSL-51 user_sign)
python train_tsl51_v3.py

# Use specific dataset
python train_tsl51_v3.py --dataset tsl51_user_sign
python train_tsl51_v3.py --dataset local --data-path ./my_data.npz

# MLP model with custom parameters
python train_tsl51_v3.py --model mlp --hidden 128 --layers 2

# Custom training configuration
python train_tsl51_v3.py --folds 3 --epochs 50 --batch 32 --lr 0.0005

# Data augmentation (2x samples, 3x samples, etc.)
python train_tsl51_v3.py --augment 2    # 547 → ~1094 samples
python train_tsl51_v3.py --augment 3    # 547 → ~1641 samples
python train_tsl51_v3.py --augment 5    # 547 → ~2735 samples

COMMAND-LINE ARGUMENTS:
=======================
--dataset         Dataset: 'tsl51_user_sign', 'tsl51_expert', 'local' (default: tsl51_user_sign)
--data-path       Path to local dataset file (required for 'local' dataset)
--folds           Number of CV folds (default: 5)
--model           Model type: 'mlp' or 'gru' (default: gru)
--hidden          Hidden dimension size (default: 256)
--layers          Number of layers (default: 3)
--dropout         Dropout rate (default: 0.3)
--epochs          Maximum epochs per fold (default: 30)
--batch           Batch size (default: 64)
--lr              Learning rate (default: 0.001)
--patience        Early stopping patience (default: 10)
--seed            Random seed (default: 42)
--no-cache         Force re-download and re-process data
--force-download   Force re-download even if cache exists
--augment          Data augmentation factor (e.g., 2 = 2x samples)

OUTPUT FILES:
=============
- results/*.json    - Full training results in JSON format
- results/*.png      - Visualization chart with metrics
- models/*.pt        - Saved PyTorch model weights

MODEL USAGE (Inference):
=========================
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Load model
checkpoint = torch.load('models/tsl51_gru_best.pt')
classes = checkpoint['classes']
mean = np.array(checkpoint['mean'])
std = np.array(checkpoint['std'])

# Load model architecture
model = GRUModel(input_dim=162, num_classes=51)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Inference function
def predict(landmarks):
    # landmarks: array of shape (162,) with hand + pose coordinates
    x = np.array(landmarks, dtype=np.float32)
    x = (x - mean) / std
    x = torch.tensor(x).unsqueeze(0)
    
    with torch.no_grad():
        probs = model(x).softmax(dim=1)
        pred = probs.argmax(dim=1).item()
    
    return classes[pred], probs[0].numpy()
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np

# Platform-specific compatibility
from src.train.compat import (
    setup_windows_encoding,
    setup_mkl_threads,
    HAS_AMP, GradScaler, autocast,
    HAS_TQDM, tqdm,
    HAS_MATPLOTLIB, plt,
)

setup_windows_encoding()
setup_mkl_threads()

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from src.utils.dataset_utils import safe_mean

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.train.evaluator import compute_metrics, aggregate_fold_results
from src.train.visualize import _pct, _fold_metric, _result_metric

PROJECT_DIR = Path(__file__).resolve().parent

# Module logger
logger = logging.getLogger(__name__)

# Check for CUDA and set device with CPU fallback
if not torch.cuda.is_available():
    print("="*70)
    print("WARNING: CUDA GPU not available. Falling back to CPU.")
    print("Training will proceed on CPU; automatic mixed precision (AMP) will be disabled.")
    print("="*70)
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# IMPORT SHARED COMPONENTS FROM SRC MODULES
# ============================================================================
from src.train.augment import augment_data
from src.core.models import GRUModel, MLP, MOPGRU, HybridGRUTransformer, CTCModel, MODEL_REGISTRY as MODEL_CLASSES
from src.data.feature_extraction import FeatureExtractor, sample_frames_uniform, pad_or_truncate


def estimate_params(model_type, hidden_dim, num_layers, input_dim, num_classes):
    """Estimate number of parameters in the model."""
    if model_type in ['gru', 'mopgru']:
        # GRU: bidirectional -> 2x hidden
        # GRU params: 3 * hidden_dim^2 * num_layers + 3 * hidden_dim * input_dim * 2
        return hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2
    elif model_type == 'hybrid':
        # GRU + Transformer
        gru_params = hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2
        # Transformer: nhead * hidden_dim^2 * 2 + hidden_dim * 4
        trans_params = 8 * hidden_dim * hidden_dim * 2 + hidden_dim * 4
        return gru_params + trans_params
    elif model_type == 'ctc':
        # CTC with extra class
        return hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2 + hidden_dim * num_classes
    else:
        # MLP: hidden^2 * layers + hidden * input + hidden * classes
        return hidden_dim * hidden_dim * (num_layers - 1) + hidden_dim * input_dim + hidden_dim * num_classes


def load_local_dataset(data_path, use_cache=True):
    """Load local dataset from CSV or NumPy file."""
    import pandas as pd
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        return None, None, None
    
    cache_file = CACHE_DIR / f"local_{data_path.stem}.npz"
    
    # Check if already cached
    if cache_file.exists() and use_cache:
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['classes']
    
    print(f"Loading local dataset: {data_path}")
    
    if data_path.suffix == '.npz':
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        classes = data['classes'] if 'classes' in data else np.array(sorted(set(y)))
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        # First column is label, rest are features
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values.astype(np.float32)
        classes = np.array(sorted(set(y)))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[c] for c in y])
    else:
        print("ERROR: Unsupported file format. Use .csv or .npz")
        return None, None, None
    
    # Handle string classes - but only if y is also string
    # If y is already int64, no conversion needed even if classes is string
    if classes.dtype.kind in ['U', 'S', 'O'] and y.dtype.kind in ['U', 'S', 'O']:
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[c] for c in y])
        classes = np.array(sorted(classes))
    
    # Replace NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Cache
    np.savez(cache_file, X=X, y=y, classes=classes)
    print(f"Loaded {len(X)} samples, {len(classes)} classes")
    
    return X, y, classes


def load_tsl51_user_sign(max_samples=None, force_download=False):
    """Load TSL-51 from user_sign_metadata with optional sampling."""
    cache_file = CACHE_DIR / "user_sign_data.npz"
    sample_cache = CACHE_DIR / f"user_sign_{max_samples}.npz" if max_samples else None
    
    # If requesting specific samples, use sample-specific cache
    if sample_cache and sample_cache.exists() and not force_download:
        print(f"Loading from cache: {sample_cache}")
        data = np.load(sample_cache, allow_pickle=True)
        return data['X'], data['y'], data['classes']
    
    # Load full dataset if not cached
    if cache_file.exists() and not force_download:
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        X_full, y_full, classes = data['X'], data['y'], data['classes']
    else:
        print("Downloading TSL-51 user_sign data...")
        
        from huggingface_hub import hf_hub_download
        import pandas as pd
        
        # Download user_sign_metadata
        meta_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename='metadata/user_sign_metadata.csv',
            repo_type='dataset'
        )
        metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
        
        # Count unique signs robustly: use len(set(...)) for compatibility
        # across pandas Series, numpy arrays, and any object with a values-like interface.
        try:
            raw = metadata['sign_clean']
            if hasattr(raw, 'values'):
                vals = np.asarray(raw.values)
            else:
                vals = np.asarray(raw)
            unique_signs = int(len(set(vals.tolist())))
        except Exception:
            try:
                unique_signs = int(len(set(np.asarray(metadata['sign_clean']).tolist())))
            except Exception:
                unique_signs = 0
        print(f"Metadata: {len(metadata)} videos, {unique_signs} signs")
        
        X_list, y_list = [], []
        
        iterator = metadata.iterrows()
        if HAS_TQDM and callable(tqdm):
            iterator = tqdm(iterator, total=len(metadata), desc="Processing")
        for idx, row in iterator:
            try:
                lm_path = row['landmark_path']
                sign = row['sign_clean']
                # Ensure we pass strings into hf_hub_download
                try:
                    lm_path = str(lm_path)
                except Exception:
                    lm_path = ''
                try:
                    sign = str(sign)
                except Exception:
                    sign = ''
                
                if pd.isna(sign) or pd.isna(lm_path):
                    continue
                
                # Download landmark file
                file_path = hf_hub_download(
                    repo_id='Namonpas/thai-sign-language-tsl51',
                    filename=str(lm_path),
                    repo_type='dataset'
                )
                
                lm_df = pd.read_csv(file_path)
                
                # Extract features (average across frames) using safe_mean
                features = []
                # Left hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'lh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(safe_mean(lm_df[col]))
                        else:
                            features.append(0.0)

                # Right hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'rh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(safe_mean(lm_df[col]))
                        else:
                            features.append(0.0)

                # Pose landmarks
                pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                            'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                            'mouth_right', 'mouth_left']
                for base in pose_cols:
                    for c in ['x', 'y', 'z']:
                        col = f'{base}_{c}'
                        if col in lm_df.columns:
                            features.append(safe_mean(lm_df[col]))
                        else:
                            features.append(0.0)
                
                X_list.append(np.nan_to_num(np.array(features, dtype=np.float32)))
                y_list.append(sign)
                
            except Exception as e:
                # Log and continue processing other entries
                logger.exception("Failed processing metadata row %s: %s", idx, e)
                continue
        
        if not X_list:
            print("ERROR: No samples extracted!")
            return None, None, None
        
        X_full = np.array(X_list)
        classes = np.array(sorted(set(y_list)))
        label_map = {c: i for i, c in enumerate(classes)}
        y_full = np.array([label_map[c] for c in y_list], dtype=np.int64)
        
        # Save full cache
        np.savez(cache_file, X=X_full, y=y_full, classes=classes)
        print(f"Cached full: {len(X_full)} samples, {len(classes)} classes")
    
    # Sample if needed (stratified to maintain class balance)
    if max_samples and len(X_full) > max_samples:
        print(f"Sampling {max_samples} samples (stratified)...")
        
        from sklearn.model_selection import train_test_split
        
        # Stratified sampling
        try:
            _, X_sampled, _, y_sampled = train_test_split(
                X_full, y_full,
                test_size=max_samples,
                stratify=y_full,
                random_state=42
            )
        except ValueError:
            # If stratification fails (small classes), use random sampling
            indices = np.random.choice(len(X_full), max_samples, replace=False)
            X_sampled, y_sampled = X_full[indices], y_full[indices]
        
        X, y = X_sampled, y_sampled
        
        # Cache the sampled version
        if sample_cache:
            np.savez(sample_cache, X=X, y=y, classes=classes)
            print(f"Cached sample: {sample_cache}")
    else:
        X, y = X_full, y_full
    
    print(f"Using {len(X)} samples, {len(classes)} classes")
    return X, y, classes


def load_tsl51_expert(include_augmented=False, max_samples=None, force_download=False):
    """Load TSL-51 from expert data (extracted from zip files).
    
    Args:
        include_augmented: If True, include pre-augmented data (45k+ samples)
                           If False, use only original data (~1155 samples)
        max_samples: Optional limit on number of samples
        force_download: Force re-download even if cached
    """
    import zipfile
    
    cache_suffix = "aug" if include_augmented else "orig"
    cache_file = CACHE_DIR / f"expert_{cache_suffix}_data.npz"
    sample_cache = CACHE_DIR / f"expert_{cache_suffix}_{max_samples}.npz" if max_samples else None
    
    # Check sample-specific cache first
    if sample_cache and sample_cache.exists() and not force_download:
        print(f"Loading from cache: {sample_cache}")
        data = np.load(sample_cache, allow_pickle=True)
        return data['X'], data['y'], data['classes']
    
    # Check full cache
    if cache_file.exists() and not force_download:
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        X_full, y_full, classes = data['X'], data['y'], data['classes']
    else:
        print("Loading TSL-51 expert data from zip files...")
        
        from huggingface_hub import hf_hub_download
        import pandas as pd
        
        # Download metadata
        meta_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename='metadata/expert_metadata.csv',
            repo_type='dataset'
        )
        metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
        
        # Filter by augmentation
        if not include_augmented:
            metadata = metadata[~metadata['is_augmented']]
        
        try:
            # np.asarray works for both pandas Series and numpy arrays
            vals = np.asarray(metadata['sign_clean'])
            unique_signs = int(len(set(vals.tolist())))
        except Exception:
            unique_signs = 0
        print(f"Metadata: {len(metadata)} videos, {unique_signs} signs")
        
        # Download and process from zip files
        zip_files = [
            ('landmarks/expert_scraped.zip', 'expert_scraped'),
            ('landmarks/expert_primary_02.zip', 'expert_primary'),
        ]
        
        X_list, y_list = [], []
        processed = set()
        
        for zip_filename, source_name in zip_files:
            print(f"Processing {source_name}...")
            
            try:
                zip_path = hf_hub_download(
                    repo_id='Namonpas/thai-sign-language-tsl51',
                    filename=zip_filename,
                    repo_type='dataset'
                )
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for idx, row in metadata.iterrows():
                        if len(X_list) >= (max_samples or float('inf')):
                            break
                        
                        video_id = row['video_id']
                        if video_id in processed:
                            continue
                        
                        sign = row.get('sign_clean')
                        lm_path_val = row.get('landmark_path')
                        # Guard against None and NaN without combining NDFrame with Python 'or'
                        _sign_bad = (sign is None) or (hasattr(sign, '__float__') and pd.isna(sign))
                        _lm_bad = (lm_path_val is None) or (hasattr(lm_path_val, '__float__') and pd.isna(lm_path_val))
                        if _sign_bad or _lm_bad:
                            continue
                        
                        # Get filename from landmark_path
                        lm_path = row.get('landmark_path', '')
                        try:
                            lm_path = str(lm_path)
                        except Exception:
                            lm_path = ''
                        lm_filename = lm_path.split('/')[-1] if '/' in lm_path else lm_path
                        zip_entry = f"landmarks/{lm_filename}"
                        
                        try:
                            with z.open(zip_entry) as f:
                                lm_df = pd.read_csv(f)
                            
                            # Extract 162 features
                            features = []
                            
                            # Left hand (21 * 3 = 63)
                            for i in range(21):
                                for c in ['x', 'y', 'z']:
                                    col = f'lh_{c}{i}'
                                    if col in lm_df.columns:
                                        features.append(float(safe_mean(lm_df[col])))
                                    else:
                                        features.append(0.0)
                            
                            # Right hand (21 * 3 = 63)
                            for i in range(21):
                                for c in ['x', 'y', 'z']:
                                    col = f'rh_{c}{i}'
                                    if col in lm_df.columns:
                                        features.append(float(safe_mean(lm_df[col])))
                                    else:
                                        features.append(0.0)
                            
                            # Pose (12 * 3 = 36)
                            pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                                        'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                                        'mouth_right', 'mouth_left']
                            for col_base in pose_cols:
                                for c in ['x', 'y', 'z']:
                                    col = f'{col_base}_{c}'
                                    if col in lm_df.columns:
                                        features.append(float(safe_mean(lm_df[col])))
                                    else:
                                        features.append(0.0)
                            
                            if len(features) == 162:
                                X_list.append(np.array(features, dtype=np.float32))
                                y_list.append(str(sign))
                                processed.add(video_id)
                                
                        except Exception as e:
                            # Log and continue; avoid silent swallowing
                            logger.debug("Skipping zip entry %s due to processing error: %s", zip_entry, e, exc_info=True)
                            continue
                
                print(f"  Extracted so far: {len(X_list)}")
                
            except Exception as e:
                print(f"  Error processing {source_name}: {e}")
                continue
        
        if not X_list:
            print("ERROR: No samples extracted!")
            return None, None, None
        
        X = np.array(X_list, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        classes = np.array(sorted(set(y_list)))
        label_map = {c: i for i, c in enumerate(classes)}
        y = np.array([label_map[c] for c in y_list], dtype=np.int64)
        
        # Save cache
        np.savez(cache_file, X=X, y=y, classes=classes)
        print(f"Saved to cache: {cache_file}")
    
    # Load from cache and apply sample limit
    data = np.load(cache_file, allow_pickle=True)
    X_full, y_full, classes = data['X'], data['y'], data['classes']
    
    if max_samples and len(X_full) > max_samples:
        indices = np.random.choice(len(X_full), max_samples, replace=False)
        X = X_full[indices]
        y = y_full[indices]
    else:
        X = X_full
        y = y_full
    
    return X, y, classes


def load_tsl51_combined(max_samples=None, force_download=False):
    """Load TSL-51 combining user_sign + expert data.
    
    Returns:
        Combined dataset from both sources (1,700+ samples)
    """
    print("Loading combined dataset (user_sign + expert)...")
    
    # Load both datasets
    X_user, y_user, classes_user = load_tsl51_user_sign(
        max_samples=None,
        force_download=force_download
    )
    X_expert, y_expert, classes_expert = load_tsl51_expert(
        include_augmented=False,
        max_samples=None,
        force_download=force_download
    )
    
    if X_user is None or X_expert is None:
        return None, None, None
    
    # Verify same classes
    if not np.array_equal(classes_user, classes_expert):
        print("WARNING: Class mismatch between datasets!")
        return None, None, None
    
    # Combine
    X_combined = np.vstack([X_user, X_expert])
    y_combined = np.concatenate([y_user, y_expert])
    classes = classes_user
    
    print(f"Combined: {len(X_combined)} samples ({len(X_user)} user + {len(X_expert)} expert)")
    print(f"Classes: {len(classes)}")
    
    # Sample if needed
    if max_samples and len(X_combined) > max_samples:
        from sklearn.model_selection import train_test_split
        _, X_sampled, _, y_sampled = train_test_split(
            X_combined, y_combined,
            test_size=max_samples,
            stratify=y_combined,
            random_state=42
        )
        X_combined, y_combined = X_sampled, y_sampled
        print(f"Sampled to {len(X_combined)} samples")
    
    return X_combined, y_combined, classes


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        if use_amp:
            # torch.amp.autocast_mode.autocast requires device_type as first positional arg (PyTorch >= 2.0).
            # torch.cuda.amp.autocast is deprecated but accepts the same signature.
            # Both default to CUDA when device_type='cuda' is passed.
            cm: Any = autocast('cuda')  # type: ignore[call-overload]

            with cm:
                out = model(X)
                loss = criterion(out, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback if scaler unavailable
                loss.backward()
                optimizer.step()
        else:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            softmax_out = torch.softmax(out, dim=1)
            correct += (softmax_out.argmax(1) == y).sum().item()
            total += y.size(0)
            preds.extend(softmax_out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
            probs.extend(softmax_out.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, targets, probs


def save_visualizations(results, fold_results, output_dir, args):
    """Save comprehensive training visualizations."""
    if not HAS_MATPLOTLIB:
        return
    if plt is None:
        return

    output_dir = Path(output_dir)
    # Metrics are normalized on-the-fly via _fold_metric / _result_metric helpers

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Title with gradient effect
    fig.suptitle('TSL-51 Thai Sign Language Training Report', 
                 fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)
    
    # Calculate dynamic y-axis limits
    all_accs = [_fold_metric(r, 'val_acc') for r in fold_results]
    all_metrics = [
        _result_metric(results, 'overall_accuracy'),
        _result_metric(results, 'top3_accuracy'),
        _result_metric(results, 'top5_accuracy'),
        _result_metric(results, 'precision'),
        _result_metric(results, 'recall'),
        _result_metric(results, 'f1_score'),
    ]
    min_acc = min(min(all_accs), min(all_metrics)) - 15
    max_acc = max(max(all_accs), max(all_metrics)) + 10
    y_min = max(0, min_acc)
    y_max = min(100, max_acc)
    
    # Color palette
    colors = {
        'primary': '#3498db',
        'secondary': '#e74c3c', 
        'success': '#2ecc71',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'dark': '#34495e',
        'light': '#ecf0f1'
    }
    
    # 1. Fold accuracies (top left) - Enhanced with gradient
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#ffffff')
    folds = [r['fold'] for r in fold_results]
    accs = [_fold_metric(r, 'val_acc') for r in fold_results]

    # Create gradient bars
    bars = ax1.bar(folds, accs, color=colors['primary'], edgecolor=colors['dark'],
                   linewidth=2, width=0.7, alpha=0.85)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=colors['dark'])

    # Average line with glow effect
    ax1.axhline(y=_result_metric(results, 'average_accuracy'), color=colors['secondary'],
                linestyle='--', linewidth=3, label=f"CV Average: {_result_metric(results, 'average_accuracy'):.2f}%",
                alpha=0.8)
    
    ax1.set_xlabel('Fold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Cross-Validation Accuracy by Fold', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.set_ylim((y_min, y_max))
    ax1.grid(axis='y', alpha=0.4, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Metrics (top middle) - Enhanced radar-like chart
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#ffffff')
    metrics = ['Accuracy', 'Top-3 Acc', 'Top-5 Acc', 'Precision', 'Recall', 'F1-Score']
    values = [
        _result_metric(results, 'overall_accuracy'),
        _result_metric(results, 'top3_accuracy'),
        _result_metric(results, 'top5_accuracy'),
        _result_metric(results, 'precision'),
        _result_metric(results, 'recall'),
        _result_metric(results, 'f1_score'),
    ]
    metric_colors = [
        colors['primary'],
        colors['warning'],
        colors['success'],
        colors['info'],
        colors['secondary'],
        colors['dark'],
    ]

    bars2 = ax2.bar(metrics, values, color=metric_colors,
                    edgecolor=colors['dark'], linewidth=2, width=0.65, alpha=0.85)

    for bar, v in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=colors['dark'])
    
    ax2.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylim((y_min, y_max))
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Fold details table (top right) - Enhanced with class distribution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#ffffff')
    
    # Calculate class distribution from results
    num_classes = results.get('num_classes', 51)
    num_samples = results.get('num_samples', 0)
    avg_per_class = num_samples / num_classes if num_classes > 0 else 0
    
    fold_table = (
        "╔══════════════════════════════════════════╗\n"
        "║       PER-FOLD RESULTS                    ║\n"
        "╠══════════════════════════════════════════╣\n"
    )
    for fold in fold_results:
        fold_table += f"║  Fold {fold['fold']}:      {_fold_metric(fold, 'val_acc'):>6.2f}%              ║\n"
    fold_table += (
        f"╠══════════════════════════════════════════╣\n"
        f"║  Best Fold:     Fold {np.argmax([_fold_metric(r, 'val_acc') for r in fold_results])+1:<3}             ║\n"
        f"║  Std Dev:       {_result_metric(results, 'std_accuracy'):>6.2f}%              ║\n"
        f"║  Best Acc:      {max([_fold_metric(r, 'val_acc') for r in fold_results]):>6.2f}%              ║\n"
        f"║  Worst Acc:     {min([_fold_metric(r, 'val_acc') for r in fold_results]):>6.2f}%              ║\n"
        f"╠══════════════════════════════════════════╣\n"
        f"║  Classes:       {num_classes:<3} signs              ║\n"
        f"║  Avg/Class:     {avg_per_class:>6.1f} samples        ║\n"
        f"╚══════════════════════════════════════════╝"
    )
    
    ax3.text(0.5, 0.95, fold_table, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', alpha=0.9, 
                     edgecolor='#1abc9c', linewidth=2))
    ax3.axis('off')
    
    # 4. Model Configuration (bottom left) - Enhanced with more details
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#ffffff')
    
    augment_val = args.augment if hasattr(args, 'augment') and args.augment else 1
    test_split_val = args.test_split if hasattr(args, 'test_split') else 0
    
    # Calculate model parameters: prefer authoritative value if provided in results
    input_dim = results.get('input_dim', 162)
    if 'actual_parameters' in results:
        params = float(results['actual_parameters'])
        params_note = ' (actual)'
    else:
        params = estimate_params(args.model, args.hidden, args.layers, input_dim, results.get('num_classes', 51))
        params_note = ' (estimated)'

    config_text = (
        f"╔═══════════════════════════════════════════════════════════╗\n"
        f"║            MODEL CONFIGURATION                            ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Model Type:        {args.model.upper():<15} (Bidirectional={args.model=='gru'})    ║\n"
        f"║  Hidden Dimensions: {args.hidden:<15}                       ║\n"
        f"║  Number of Layers:  {args.layers:<15}                       ║\n"
        f"║  Dropout Rate:      {args.dropout:<15}                       ║\n"
        f"║  Input Features:    {input_dim:<15}                       ║\n"
        f"║  Output Classes:    {results.get('num_classes', 51):<15}                       ║\n"
        f"║  Est. Parameters:   ~{params/1e6:.2f}M{params_note:<11}                       ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Optimizer:         AdamW                                  ║\n"
        f"║  Learning Rate:     {args.lr:<15}                       ║\n"
        f"║  Weight Decay:      1e-4                                    ║\n"
        f"║  LR Scheduler:      OneCycleLR                              ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Batch Size:        {args.batch:<15}                       ║\n"
        f"║  Max Epochs:        {args.epochs:<15}                       ║\n"
        f"║  Early Stopping:    patience={args.patience:<10}                       ║\n"
        f"║  Random Seed:       {args.seed:<15}                       ║\n"
        f"║  CV Folds:          {args.folds:<15}                       ║\n"
        f"║  Augmentation:      {augment_val}x{'':<13}                       ║\n"
        f"║  Test Split:       {test_split_val*100:.0f}%{'':<13}                       ║\n"
        f"╚═══════════════════════════════════════════════════════════╝"
    )
    
    ax4.text(0.02, 0.98, config_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7', alpha=0.9, 
                     edgecolor='#f39c12', linewidth=2))
    ax4.axis('off')
    
    # 5. Dataset Information (bottom middle) - Enhanced
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#ffffff')
    
    test_samples_line = ""
    if results.get('test_samples', 0) > 0:
        test_samples_line = f"║  Test Samples:     {results.get('test_samples', 0):<15}                       ║\n"
    
    dataset_text = (
        f"╔═══════════════════════════════════════════════════════════╗\n"
        f"║            DATASET INFORMATION                            ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Dataset:          tsl51_user_sign                        ║\n"
        f"║  Total Samples:    {results['num_samples']:<15}                       ║\n"
        f"║  Train Samples:     {results['num_samples'] - results.get('test_samples', 0):<15}                       ║\n"
        f"{test_samples_line}"
        f"║  Number of Classes: {results['num_classes']:<15}                       ║\n"
        f"║  Feature Dim:       {results['input_dim']:<15}                       ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Feature Breakdown:                                         ║\n"
        f"║    • Left Hand:   21 pts × 3 = 63 features                ║\n"
        f"║    • Right Hand:  21 pts × 3 = 63 features                ║\n"
        f"║    • Pose:       12 pts × 3 = 36 features                ║\n"
        f"║    • Total:                  = 162 features              ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Augmentation Methods:                                      ║\n"
        f"║    1. Gaussian Noise (σ=0.01)                              ║\n"
        f"║    2. Random Scale (0.95-1.05)                            ║\n"
        f"║    3. Left-Right Hand Flip                                ║\n"
        f"╚═══════════════════════════════════════════════════════════╝"
    )
    
    ax5.text(0.02, 0.98, dataset_text, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', alpha=0.9, 
                     edgecolor='#1abc9c', linewidth=2))
    ax5.axis('off')
    
    # 6. Final Results (bottom right) - Enhanced with more details
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#ffffff')
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    
    # Test results if available
    test_line = ""
    if 'test_results' in results:
        tr = results['test_results']
        test_line = (
            f"║  Test Accuracy:    {_pct(tr.get('test_accuracy', 0.0)):>6.2f}%{'':<17}║\n"
            f"║  Test Precision:   {_pct(tr.get('test_precision', 0.0)):>6.2f}%{'':<17}║\n"
            f"║  Test Recall:       {_pct(tr.get('test_recall', 0.0)):>6.2f}%{'':<17}║\n"
            f"║  Test F1-Score:     {_pct(tr.get('test_f1_score', 0.0)):>6.2f}%{'':<17}║\n"
            f"╠═══════════════════════════════════════════════════════════╣\n"
        )

    results_text = (
        f"╔═══════════════════════════════════════════════════════════╗\n"
        f"║                 FINAL RESULTS                               ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  CV Average:       {_result_metric(results, 'average_accuracy'):>6.2f}% (±{_result_metric(results, 'std_accuracy'):.2f}%{'':<10}║\n"
        f"║  Overall Accuracy: {_result_metric(results, 'overall_accuracy'):>6.2f}%{'':<17}║\n"
        f"║  Top-3 Accuracy:   {_result_metric(results, 'top3_accuracy'):>6.2f}%{'':<17}║\n"
        f"║  Top-5 Accuracy:   {_result_metric(results, 'top5_accuracy'):>6.2f}%{'':<17}║\n"
        f"║  Precision:         {_result_metric(results, 'precision'):>6.2f}%{'':<17}║\n"
        f"║  Recall:            {_result_metric(results, 'recall'):>6.2f}%{'':<17}║\n"
        f"║  F1-Score:          {_result_metric(results, 'f1_score'):>6.2f}%{'':<17}║\n"
        f"{test_line}"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║            TRAINING ENVIRONMENT                           ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Date:            {results['timestamp'][:19]:<25}║\n"
        f"║  Device:         {gpu_name:<25}║\n"
        f"║  AMP (Mixed Prec): {'ON' if results.get('use_amp', True) else 'OFF':<25}║\n"
        f"║  Class Weighting: {'Balanced' if results.get('class_weight', True) else 'None':<25}║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║                    OUTPUT FILES                             ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  Model:           tsl51_{args.model}_{results['timestamp'].replace('-','').replace(':','')[:12]}.pt     ║\n"
        f"║  Report:          cv_{results['timestamp'].replace('-','').replace(':','')[:12]}.json   ║\n"
        f"╚═══════════════════════════════════════════════════════════╝"
    )
    
    ax6.text(0.02, 0.98, results_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdedec', alpha=0.9, 
                     edgecolor='#e74c3c', linewidth=2))
    ax6.axis('off')
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(output_dir / f'results_{timestamp}.png', dpi=150, bbox_inches='tight', 
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()
    
    # Save detailed text report
    text_report_path = output_dir / f'report_{timestamp}.txt'
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TSL-51 THAI SIGN LANGUAGE TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\nMODEL PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"  Model Type: {args.model.upper()}\n")
        f.write(f"  Hidden Dimensions: {args.hidden}\n")
        f.write(f"  Number of Layers: {args.layers}\n")
        f.write(f"  Dropout Rate: {args.dropout}\n")
        f.write(f"  Input Features: {results['input_dim']}\n")
        f.write(f"  Output Classes: {results['num_classes']}\n")
        
        # Calculate estimated parameters
        params = estimate_params(args.model, args.hidden, args.layers, results['input_dim'], results['num_classes'])
        f.write(f"  Est. Parameters: ~{params/1e6:.2f}M\n")
        
        f.write("\nTRAINING CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write("  Optimizer: AdamW\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write("  Weight Decay: 1e-4\n")
        f.write("  LR Scheduler: OneCycleLR\n")
        f.write(f"  Batch Size: {args.batch}\n")
        f.write(f"  Max Epochs: {args.epochs}\n")
        f.write(f"  Early Stopping: patience={args.patience}\n")
        f.write(f"  CV Folds: {args.folds}\n")
        f.write(f"  Augmentation: {args.augment if args.augment else 1}x\n")
        f.write(f"  Test Split: {args.test_split*100:.0f}%\n")
        
        f.write("\nDATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"  Total Samples: {results['num_samples']}\n")
        f.write(f"  Number of Classes: {results['num_classes']}\n")
        f.write(f"  Feature Dimension: {results['input_dim']}\n")
        if results.get('test_samples', 0) > 0:
            f.write(f"  Train Samples: {results['num_samples'] - results['test_samples']}\n")
            f.write(f"  Test Samples: {results['test_samples']}\n")
        
        f.write("\nPER-FOLD RESULTS\n")
        f.write("-"*70 + "\n")
        for fold in fold_results:
            f.write(f"  Fold {fold['fold']}: {_fold_metric(fold, 'val_acc'):.2f}%\n")
        f.write(f"\n  Best Fold: Fold {np.argmax([_fold_metric(r, 'val_acc') for r in fold_results])+1}\n")
        f.write(f"  Std Dev: {_result_metric(results, 'std_accuracy'):.2f}%\n")

        f.write("\nOVERALL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"  CV Average: {_result_metric(results, 'average_accuracy'):.2f}% (±{_result_metric(results, 'std_accuracy'):.2f}%)\n")
        f.write(f"  Overall Accuracy: {_result_metric(results, 'overall_accuracy'):.2f}%\n")
        f.write(f"  Top-3 Accuracy:   {_result_metric(results, 'top3_accuracy'):.2f}%\n")
        f.write(f"  Top-5 Accuracy:   {_result_metric(results, 'top5_accuracy'):.2f}%\n")
        f.write(f"  Precision: {_result_metric(results, 'precision'):.2f}%\n")
        f.write(f"  Recall: {_result_metric(results, 'recall'):.2f}%\n")
        f.write(f"  F1-Score: {_result_metric(results, 'f1_score'):.2f}%\n")

        if 'test_results' in results:
            f.write("\nTEST SET RESULTS\n")
            f.write("-"*70 + "\n")
            tr = results['test_results']
            f.write(f"  Test Accuracy: {_pct(tr.get('test_accuracy', 0.0)):.2f}%\n")
            f.write(f"  Test Precision: {_pct(tr.get('test_precision', 0.0)):.2f}%\n")
            f.write(f"  Test Recall: {_pct(tr.get('test_recall', 0.0)):.2f}%\n")
            f.write(f"  Test F1-Score: {_pct(tr.get('test_f1_score', 0.0)):.2f}%\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Text Report: {text_report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='TSL-51 Thai Sign Language Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Default training with TSL-51 dataset
  python train_tsl51_v3.py

  # Use MLP model with 3-fold CV
  python train_tsl51_v3.py --model mlp --folds 3

  # Train with local dataset
  python train_tsl51_v3.py --dataset local --data-path ./data.npz

  # Custom configuration
  python train_tsl51_v3.py --hidden 128 --layers 2 --epochs 50 --lr 0.0005
        """
    )
    parser.add_argument("--dataset", type=str, default="tsl51_user_sign",
                       choices=["tsl51_user_sign", "tsl51_expert", "tsl51_expert_full", "tsl51_combined", "tsl51_full", "local"],
                       help="Dataset to use (default: tsl51_user_sign). tsl51_expert_full uses ~45k samples")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to local dataset (required for --dataset local)")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to use (default: all available)")
    parser.add_argument("--folds", type=int, default=5,
                       help="Number of CV folds (default: 5)")
    parser.add_argument("--hidden", type=int, default=256,
                       help="Hidden dimension size (default: 256)")
    parser.add_argument("--layers", type=int, default=3,
                       help="Number of layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate (default: 0.3)")
    parser.add_argument("--model", type=str, default="gru",
                       choices=["mlp", "gru", "mopgru", "hybrid", "ctc"],
                       help="Model architecture: 'mlp', 'gru', 'mopgru', 'hybrid', or 'ctc' (default: gru)")
    parser.add_argument("--feature-level", type=str, default="basic",
                       choices=["basic", "finger", "full", "face"],
                       help="Feature level: 'basic'(162), 'finger'(252), 'full'(1596), 'face'(1434)")
    parser.add_argument("--target-frames", type=int, default=30,
                       help="Target frames for sequence models (default: 30)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Maximum epochs per fold (default: 30)")
    parser.add_argument("--batch", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Do not use cached data, re-process")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download even if cache exists")
    parser.add_argument("--augment", type=int, default=0,
                       help="Data augmentation factor (e.g., 2 = 2x samples)")
    parser.add_argument("--include-augmented", action="store_true",
                       help="Include pre-augmented expert data (~45k samples). Only for --dataset tsl51_expert")
    parser.add_argument("--require-cuda", action="store_true",
                       help="If set, abort early when CUDA is not available")
    parser.add_argument("--smoke", action="store_true",
                       help="Smoke-test: epochs=1, folds=1, samples=10. Useful for CPU-only validation.")
    parser.add_argument("--test-split", type=float, default=0.0,
                       help="Train/test split ratio (e.g., 0.2 = 20%% test). If 0, use full data for CV (default: 0)")
    parser.add_argument("--export", action="store_true",
                       help="Export model to portable format after training")
    parser.add_argument("--export-name", type=str, default=None,
                       help="Export model filename (default: auto-generated)")
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("="*70)
        print("ERROR: CUDA GPU not available and --require-cuda was set!")
        print("This script requires a CUDA-capable GPU when --require-cuda is provided.")
        print("="*70)
        sys.exit(1)
    
    # Smoke-test shortcut: force minimal params for fast CPU validation.
    # folds=2 so StratifiedKFold is happy (min n_splits=2); epochs=1 so it finishes fast.
    if args.smoke:
        args.folds = 2
        args.epochs = 1
        args.batch = min(args.batch, 16)
        if args.samples is None or args.samples > 10:
            args.samples = 10
        print("[SMOKE TEST] Running with --epochs=1 --folds=2 --samples={} --batch={}".format(
            args.samples, args.batch))
    
    # Validate arguments
    if args.dataset == "local" and args.data_path is None:
        parser.error("--data-path is required when using --dataset local")
    
    if args.dataset == "local":
        args.data_path = Path(args.data_path)
        if not args.data_path.exists():
            print(f"ERROR: File not found: {args.data_path}")
            return 1
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Use AMP only when the runtime imported AMP support and CUDA is available
    use_amp = HAS_AMP and torch.cuda.is_available()
    
    print("="*70)
    print("TSL-51 THAI SIGN LANGUAGE TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model.upper()} | Folds: {args.folds} | Epochs: {args.epochs}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Load dataset based on selection
    if args.dataset == "tsl51_user_sign":
        X, y, classes = load_tsl51_user_sign(
            max_samples=args.samples,
            force_download=args.force_download
        )
    elif args.dataset == "local":
        use_cache = not args.no_cache
        X, y, classes = load_local_dataset(args.data_path, use_cache=use_cache)
    elif args.dataset == "tsl51_expert":
        X, y, classes = load_tsl51_expert(
            include_augmented=args.include_augmented,
            max_samples=args.samples,
            force_download=args.force_download
        )
    elif args.dataset == "tsl51_expert_full":
        # Import from src.data.loader for full expert dataset
        from src.data.loader import load_tsl51_expert_full
        X, y, classes = load_tsl51_expert_full(
            max_samples=args.samples,
            force_download=args.force_download
        )
    elif args.dataset == "tsl51_combined":
        X, y, classes = load_tsl51_combined(
            max_samples=args.samples,
            force_download=args.force_download
        )
    elif args.dataset == "tsl51_full":
        from src.data.loader import load_tsl51_full
        X, y, classes = load_tsl51_full(
            include_augmented=True,
            max_samples=args.samples,
            force_download=args.force_download
        )
    else:
        print(f"ERROR: Unknown dataset: {args.dataset}")
        return 1
    
    if X is None:
        return 1

    # Validate dataset quality
    from src.data.loader import validate_dataset, print_dataset_quality_report
    validation_results = validate_dataset(X, y, classes)
    print_dataset_quality_report(validation_results)
    
    if not validation_results['valid']:
        print("WARNING: Dataset validation failed. Proceeding anyway, but results may be unreliable.")

    # Ensure numpy arrays for downstream processing and consistent shapes
    try:
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        X = np.array(X, dtype=np.float32)
    try:
        y = np.asarray(y)
    except Exception:
        y = np.array(y)
    try:
        classes = np.asarray(classes)
    except Exception:
        classes = np.array(classes)

    # Determine input feature dimension defensively
    try:
        X = np.asarray(X, dtype=np.float32)
    except Exception:
        X = np.array(X, dtype=np.float32)
    input_dim = int(X.shape[1]) if getattr(X, 'ndim', 1) > 1 else 0
    
    # Apply data augmentation if requested
    if args.augment > 1:
        X, y = augment_data(X, y, classes, augmentation_factor=args.augment)
    
    # Apply train/test split if requested
    test_samples_count = 0
    has_test_set = False
    X_test = None
    y_test = None
    if args.test_split > 0:
        from sklearn.model_selection import train_test_split
        try:
            X, X_test, y, y_test = train_test_split(
                X, y, test_size=args.test_split, stratify=y, random_state=args.seed
            )
            test_samples_count = len(X_test)
            has_test_set = True
            print(f"Train/Test Split: {len(X)} train, {test_samples_count} test")
        except Exception as e:
            logger.exception("Failed to create train/test split: %s", e)
            X_test = None
            y_test = None
            has_test_set = False
    
    print(f"\nData: {len(X)} samples, {input_dim} features, {len(classes)} classes")
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    all_preds, all_targets, all_probs = [], [], []
    best_models, fold_means, fold_stds = [], [], []
    
    # Initialize GradScaler only when AMP is enabled and GradScaler is available
    try:
        # GradScaler('cuda') silences the PyTorch 2.6 deprecation warning
        # type: ignore[call-overload] - LSP resolves to old API but runtime uses torch.amp.grad_scaler
        scaler = GradScaler('cuda') if use_amp and (GradScaler is not None) else None  # type: ignore[call-overload]
    except Exception:
        scaler = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Ensure numpy operations for mean/std (X_train is np.ndarray)
        # Compute mean/std defensively and cast to native float32 arrays
        try:
            X_train = np.asarray(X_train, dtype=np.float32)
        except Exception:
            X_train = np.array(X_train, dtype=np.float32)

        try:
            mean = np.mean(X_train, axis=0).astype(np.float32)
        except Exception:
            mean = np.zeros((input_dim,), dtype=np.float32)

        try:
            std = np.std(X_train, axis=0).astype(np.float32)
        except Exception:
            std = np.ones_like(mean, dtype=np.float32)

        # Avoid zeros in std
        std = np.where(std == 0, 1.0, std) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        fold_means.append(mean)
        fold_stds.append(std)
        
        # compute_class_weight returns weights only for classes present in y_train.
        # When few samples exist (e.g., smoke test with 10 samples), some folds
        # may have only a subset of all classes. Expand to full class count with 0 weight.
        try:
            raw_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            unique_labels = np.unique(y_train)
            total_classes = len(classes)
            if len(unique_labels) == total_classes:
                weights = torch.tensor(raw_weights, dtype=torch.float32).to(DEVICE)
            else:
                # Expand: assign 0 weight to missing classes
                full_weights = torch.zeros(total_classes, dtype=torch.float32).to(DEVICE)
                for idx, label in enumerate(unique_labels):
                    full_weights[label] = raw_weights[idx]
                weights = full_weights
        except Exception:
            weights = None
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=0
        )
        
        model = MODEL_CLASSES[args.model](input_dim, len(classes), args.hidden, args.layers, args.dropout)
        model = model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=weights)  # weight=None means unweighted
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                              steps_per_epoch=len(train_loader))
        
        best_acc, best_state, patience = 0.0, None, 0
        
        # Save initial (untrained) model state so best_state is never None
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, use_amp)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            
            if patience >= args.patience:
                print(f"  Early stop @ epoch {epoch+1}")
                break
        
        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        else:
            # best_acc stayed 0 throughout; keep the randomly-initialised model
            pass
        _, final_acc, preds, targets, probs = evaluate(model, val_loader, criterion, DEVICE)

        # Compute comprehensive metrics via evaluator (percentages 0-100)
        fold_metrics = compute_metrics(
            np.array(targets),
            np.array(preds),
            classes,
            y_probs=np.array(probs),
        )

        print(f"  Fold {fold+1}: {fold_metrics['accuracy']:.2f}%")
        fold_results.append({
            'fold': fold+1,
            'val_acc': fold_metrics['accuracy'],
            'val_f1_score': fold_metrics['f1_score'],
            'val_precision': fold_metrics['precision'],
            'val_recall': fold_metrics['recall'],
            'val_top3_acc': fold_metrics['top3_accuracy'],
            'val_top5_acc': fold_metrics['top5_accuracy'],
            'per_class_metrics': fold_metrics['per_class'],
            'confusion_matrix': fold_metrics['confusion_matrix'],
            'most_confused': fold_metrics['most_confused'],
        })
        all_preds.extend(preds)
        all_targets.extend(targets)
        all_probs.extend(probs)
        best_models.append(best_state)
    
    # Aggregate fold results using evaluator helpers
    aggregated = aggregate_fold_results(fold_results)
    avg_acc = aggregated['val_acc_mean']
    std_acc = aggregated['val_acc_std']

    # Overall metrics across all predictions (percentages 0-100)
    overall_metrics = compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        classes,
        y_probs=np.array(all_probs) if all_probs else None,
    )

    # Get best fold index
    best_idx = np.argmax([r['val_acc'] for r in fold_results])
    
    # Evaluate on held-out test set if test-split was used
    test_results = None
    if args.test_split > 0 and has_test_set:
        # Normalize test set using best fold's statistics
        test_mean, test_std = fold_means[best_idx], fold_stds[best_idx]
        X_test_norm = (X_test - test_mean) / test_std
        
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test_norm), torch.tensor(y_test)),
            batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=0
        )
        
        # Load best model
        best_model = MODEL_CLASSES[args.model](input_dim, len(classes), args.hidden, args.layers, args.dropout)
        best_model = best_model.to(DEVICE)  # Move to GPU first
        best_model.load_state_dict(best_models[best_idx])  # State dict is on CPU, will be moved by load_state_dict
        best_model.eval()
        
        test_preds, test_targets, test_probs = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                out = best_model(X_batch)
                softmax_out = torch.softmax(out, dim=1)
                test_preds.extend(softmax_out.argmax(1).cpu().numpy())
                test_targets.extend(y_batch.numpy())
                test_probs.extend(softmax_out.cpu().numpy())

        test_metrics = compute_metrics(
            np.array(test_targets),
            np.array(test_preds),
            classes,
            y_probs=np.array(test_probs),
        )

        test_results = {
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1_score': test_metrics['f1_score'],
            'test_top3_accuracy': test_metrics['top3_accuracy'],
            'test_top5_accuracy': test_metrics['top5_accuracy'],
        }

        print(f"Test Set: {test_metrics['accuracy']:.2f}% | P: {test_metrics['precision']:.2f}% | R: {test_metrics['recall']:.2f}% | F1: {test_metrics['f1_score']:.2f}%")
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"CV Avg: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
    print(
        f"Overall: {overall_metrics['accuracy']:.2f}% | "
        f"P: {overall_metrics['precision']:.2f}% | "
        f"R: {overall_metrics['recall']:.2f}% | "
        f"F1: {overall_metrics['f1_score']:.2f}%"
    )

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'fold_results': fold_results,
        'average_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'overall_accuracy': float(overall_metrics['accuracy']),
        'precision': float(overall_metrics['precision']),
        'recall': float(overall_metrics['recall']),
        'f1_score': float(overall_metrics['f1_score']),
        'top3_accuracy': float(overall_metrics['top3_accuracy']),
        'top5_accuracy': float(overall_metrics['top5_accuracy']),
        'num_classes': len(classes),
        'num_samples': len(X),
        'test_samples': test_samples_count,
        'input_dim': input_dim
    }
    
    if test_results:
        results['test_results'] = test_results
    
    # Calculate actual model parameters from best model
    total_params = sum(p.numel() for p in best_models[best_idx].values())
    results['actual_parameters'] = total_params
    
    # Create unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    (PROJECT_DIR / "results").mkdir(exist_ok=True)
    json_file = PROJECT_DIR / "results" / f"cv_{timestamp}.json"

    # Make results JSON-serializable (convert Paths, numpy, torch, etc.)
    def _serialize(obj):
        # Handle pathlib.Path
        if isinstance(obj, Path):
            return str(obj)
        # numpy types
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None:
            if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        # torch tensors
        try:
            import torch as _torch
        except Exception:
            _torch = None
        if _torch is not None and isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        # dict/list/tuple
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        # fallback for objects with __dict__
        if hasattr(obj, '__dict__'):
            try:
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            except Exception:
                pass
        try:
            return str(obj)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Failed to serialize object of type %s", type(obj))
            return None

    serializable_results = _serialize(results)
    json_file.write_text(json.dumps(serializable_results, indent=2, ensure_ascii=False), encoding='utf-8')

    # Generate PNG report from the serializable payload so it matches the JSON
    save_visualizations(serializable_results, fold_results, PROJECT_DIR / "results", args)
    
    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    
    # Save model with unique name (no overwrite)
    model_filename = f"tsl51_{args.model}_{timestamp}.pt"
    model_path = PROJECT_DIR / "models" / model_filename
    
    torch.save({
        'state_dict': best_models[best_idx],
        'classes': classes.tolist(),
        'mean': fold_means[best_idx].tolist(),
        'std': fold_stds[best_idx].tolist(),
        'input_dim': input_dim,
        'num_classes': len(classes),
        'model': args.model,
        'accuracy': float(fold_results[best_idx]['val_acc']),
        'timestamp': timestamp,
        'config': vars(args)
    }, model_path)
    
    # Export to portable format if requested
    if args.export:
        print("\nExporting to portable format...")
        export_path = model_path.parent / f"{model_path.stem}_export.pt"
        
        # Create portable package
        checkpoint = torch.load(model_path, map_location='cpu')
        
        package = {
            'state_dict': checkpoint['state_dict'],
            'model_type': checkpoint['model'],
            'model_config': {
                'hidden_dim': args.hidden,
                'num_layers': args.layers,
                'dropout': args.dropout,
                'nhead': 8,
            },
            'input_dim': checkpoint['input_dim'],
            'num_classes': checkpoint['num_classes'],
            'classes': checkpoint['classes'],
            'mean': checkpoint['mean'],
            'std': checkpoint['std'],
            'feature_info': {
                'feature_level': args.feature_level,
                'feature_count': checkpoint['input_dim'],
            },
            'training_info': {
                'accuracy': checkpoint['accuracy'],
                'dataset': args.dataset,
                'epochs': args.epochs,
            },
            'version': '1.0',
        }
        
        torch.save(package, export_path)
        print(f"Exported: {export_path}")
    
    print(f"\nResults: {json_file}")
    print(f"Model: {model_path}")
    if args.export:
        print(f"Export: {model_path.parent / (model_path.stem + '_export.pt')}")
    print("DONE!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
