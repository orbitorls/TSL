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

import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

PROJECT_DIR = Path(__file__).resolve().parent

# Enforce GPU-only training
if not torch.cuda.is_available():
    print("="*70)
    print("ERROR: CUDA GPU not available!")
    print("This script requires a CUDA-capable GPU for training.")
    print("="*70)
    sys.exit(1)

DEVICE = torch.device("cuda")
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
def augment_data(X, y, classes, augmentation_factor=2, noise_level=0.01, scale_range=(0.95, 1.05)):
    """
    Augment data by applying transformations.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Label array
        classes: Class names
        augmentation_factor: How many augmented copies per sample
        noise_level: Standard deviation of Gaussian noise
        scale_range: Tuple of (min, max) scale factors
        
    Returns:
        X_aug, y_aug (augmented data)
    """
    print(f"Applying data augmentation ({augmentation_factor}x)...")
    
    X_list = [X]  # Original data
    y_list = [y]
    
    n_samples = len(X)
    
    for aug_idx in range(augmentation_factor):
        X_aug = np.zeros_like(X)
        
        for i in range(n_samples):
            # Get original features
            features = X[i].copy()
            
            # Apply random augmentation
            aug_type = np.random.choice(['noise', 'scale', 'noise_scale', 'flip'])
            
            if aug_type == 'noise':
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, features.shape)
                X_aug[i] = features + noise
                
            elif aug_type == 'scale':
                # Random scale
                scale = np.random.uniform(scale_range[0], scale_range[1])
                X_aug[i] = features * scale
                
            elif aug_type == 'noise_scale':
                # Both noise and scale
                scale = np.random.uniform(scale_range[0], scale_range[1])
                noise = np.random.normal(0, noise_level, features.shape)
                X_aug[i] = features * scale + noise
                
            else:  # flip - mirror left/right hand
                # Swap left hand (0-62) with right hand (63-125)
                left_hand = features[0:63].copy()
                right_hand = features[63:126].copy()
                new_features = np.concatenate([right_hand, left_hand, features[126:]])
                X_aug[i] = new_features
        
        X_list.append(X_aug)
        y_list.append(y)
        
        if (aug_idx + 1) % 5 == 0:
            print(f"  Augmented {aug_idx + 1}/{augmentation_factor}")
    
    # Concatenate all
    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)
    
    print(f"Augmented: {len(X)} → {len(X_final)} samples")
    return X_final, y_final


class GRUModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


MODEL_CLASSES = {'mlp': MLP, 'gru': GRUModel}


def estimate_params(model_type, hidden_dim, num_layers, input_dim, num_classes):
    """Estimate number of parameters in the model."""
    if model_type == 'gru':
        # GRU: bidirectional -> 2x hidden
        return hidden_dim * hidden_dim * 4 * 3 * num_layers + hidden_dim * input_dim * 2
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
        print(f"ERROR: Unsupported file format. Use .csv or .npz")
        return None, None, None
    
    # Handle string classes
    if classes.dtype.kind in ['U', 'S', 'O']:
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
        
        print(f"Metadata: {len(metadata)} videos, {metadata['sign_clean'].nunique()} signs")
        
        X_list, y_list = [], []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
            try:
                lm_path = row['landmark_path']
                sign = row['sign_clean']
                
                if pd.isna(sign) or pd.isna(lm_path):
                    continue
                
                # Download landmark file
                file_path = hf_hub_download(
                    repo_id='Namonpas/thai-sign-language-tsl51',
                    filename=lm_path,
                    repo_type='dataset'
                )
                
                lm_df = pd.read_csv(file_path)
                
                # Extract features (average across frames)
                features = []
                
                # Left hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'lh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(lm_df[col].mean())
                        else:
                            features.append(0.0)
                
                # Right hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'rh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(lm_df[col].mean())
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
                            features.append(lm_df[col].mean())
                        else:
                            features.append(0.0)
                
                X_list.append(np.nan_to_num(np.array(features, dtype=np.float32)))
                y_list.append(sign)
                
            except Exception as e:
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
    """Load TSL-51 from expert_metadata.csv.
    
    Args:
        include_augmented: If True, include pre-augmented data (45k+ samples)
                           If False, use only original data (~1155 samples)
        max_samples: Optional limit on number of samples
        force_download: Force re-download even if cached
    """
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
        print("Downloading TSL-51 expert data...")
        
        from huggingface_hub import hf_hub_download
        import pandas as pd
        
        # Download expert_metadata
        meta_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename='metadata/expert_metadata.csv',
            repo_type='dataset'
        )
        metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
        
        # Filter by augmentation
        if not include_augmented:
            metadata = metadata[metadata['is_augmented'] == False]
        
        print(f"Metadata: {len(metadata)} videos, {metadata['sign_clean'].nunique()} signs")
        
        X_list, y_list = [], []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
            try:
                lm_path = row['landmark_path']
                sign = row['sign_clean']
                
                if pd.isna(sign) or pd.isna(lm_path):
                    continue
                
                # Download landmark file
                file_path = hf_hub_download(
                    repo_id='Namonpas/thai-sign-language-tsl51',
                    filename=lm_path,
                    repo_type='dataset'
                )
                
                lm_df = pd.read_csv(file_path)
                
                # Extract features (average across frames)
                features = []
                
                # Left hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'lh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(lm_df[col].mean())
                        else:
                            features.append(0.0)
                
                # Right hand (21 points * 3 = 63)
                for i in range(21):
                    for c in ['x', 'y', 'z']:
                        col = f'rh_{c}{i}'
                        if col in lm_df.columns:
                            features.append(lm_df[col].mean())
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
                            features.append(lm_df[col].mean())
                        else:
                            features.append(0.0)
                
                X_list.append(np.nan_to_num(np.array(features, dtype=np.float32)))
                y_list.append(sign)
                
            except Exception as e:
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
        print(f"Cached: {len(X_full)} samples, {len(classes)} classes")
    
    # Sample if needed
    if max_samples and len(X_full) > max_samples:
        print(f"Sampling {max_samples} samples (stratified)...")
        
        from sklearn.model_selection import train_test_split
        
        try:
            _, X_sampled, _, y_sampled = train_test_split(
                X_full, y_full,
                test_size=max_samples,
                stratify=y_full,
                random_state=42
            )
        except ValueError:
            indices = np.random.choice(len(X_full), max_samples, replace=False)
            X_sampled, y_sampled = X_full[indices], y_full[indices]
        
        X, y = X_sampled, y_sampled
        
        if sample_cache:
            np.savez(sample_cache, X=X, y=y, classes=classes)
            print(f"Cached sample: {sample_cache}")
    else:
        X, y = X_full, y_full
    
    print(f"Using {len(X)} samples, {len(classes)} classes")
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
            with autocast('cuda'):
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, targets


def save_visualizations(results, fold_results, output_dir, args):
    """Save comprehensive training visualizations."""
    if not HAS_MATPLOTLIB:
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Title with gradient effect
    fig.suptitle('TSL-51 Thai Sign Language Training Report', 
                 fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)
    
    # Calculate dynamic y-axis limits
    all_accs = [r['accuracy'] * 100 for r in fold_results]
    all_metrics = [results['overall_accuracy']*100, results['precision']*100, 
                   results['recall']*100, results['f1_score']*100]
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
    accs = [r['accuracy'] * 100 for r in fold_results]
    
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
    ax1.axhline(y=results['average_accuracy'] * 100, color=colors['secondary'], 
                linestyle='--', linewidth=3, label=f"CV Average: {results['average_accuracy']*100:.2f}%",
                alpha=0.8)
    
    ax1.set_xlabel('Fold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Cross-Validation Accuracy by Fold', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.set_ylim([y_min, y_max])
    ax1.grid(axis='y', alpha=0.4, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Metrics (top middle) - Enhanced radar-like chart
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#ffffff')
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['overall_accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    metric_colors = [colors['primary'], colors['warning'], colors['success'], colors['info']]
    
    bars2 = ax2.bar(metrics, [v * 100 for v in values], color=metric_colors, 
                    edgecolor=colors['dark'], linewidth=2, width=0.65, alpha=0.85)
    
    for bar, v in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{v*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=colors['dark'])
    
    ax2.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylim([y_min, y_max])
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
        f"╔══════════════════════════════════════════╗\n"
        f"║       PER-FOLD RESULTS                    ║\n"
        f"╠══════════════════════════════════════════╣\n"
    )
    for fold in fold_results:
        fold_table += f"║  Fold {fold['fold']}:      {fold['accuracy']*100:>6.2f}%              ║\n"
    fold_table += (
        f"╠══════════════════════════════════════════╣\n"
        f"║  Best Fold:     Fold {np.argmax([r['accuracy'] for r in fold_results])+1:<3}             ║\n"
        f"║  Std Dev:       {results['std_accuracy']*100:>6.2f}%              ║\n"
        f"║  Best Acc:      {max([r['accuracy'] for r in fold_results])*100:>6.2f}%              ║\n"
        f"║  Worst Acc:     {min([r['accuracy'] for r in fold_results])*100:>6.2f}%              ║\n"
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
    
    # Calculate model parameters
    input_dim = results.get('input_dim', 162)
    params = estimate_params(args.model, args.hidden, args.layers, input_dim, results.get('num_classes', 51))
    
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
        f"║  Est. Parameters:   ~{params/1e6:.2f}M{'':<11}                       ║\n"
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
            f"║  Test Accuracy:    {tr['test_accuracy']*100:>6.2f}%{'':<17}║\n"
            f"║  Test Precision:   {tr['test_precision']*100:>6.2f}%{'':<17}║\n"
            f"║  Test Recall:       {tr['test_recall']*100:>6.2f}%{'':<17}║\n"
            f"║  Test F1-Score:     {tr['test_f1_score']*100:>6.2f}%{'':<17}║\n"
            f"╠═══════════════════════════════════════════════════════════╣\n"
        )
    
    results_text = (
        f"╔═══════════════════════════════════════════════════════════╗\n"
        f"║                 FINAL RESULTS                               ║\n"
        f"╠═══════════════════════════════════════════════════════════╣\n"
        f"║  CV Average:       {results['average_accuracy']*100:>6.2f}% (±{results['std_accuracy']*100:.2f}%{'':<10}║\n"
        f"║  Overall Accuracy: {results['overall_accuracy']*100:>6.2f}%{'':<17}║\n"
        f"║  Precision:         {results['precision']*100:>6.2f}%{'':<17}║\n"
        f"║  Recall:            {results['recall']*100:>6.2f}%{'':<17}║\n"
        f"║  F1-Score:          {results['f1_score']*100:>6.2f}%{'':<17}║\n"
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        f.write(f"  Optimizer: AdamW\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Weight Decay: 1e-4\n")
        f.write(f"  LR Scheduler: OneCycleLR\n")
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
            f.write(f"  Fold {fold['fold']}: {fold['accuracy']*100:.2f}%\n")
        f.write(f"\n  Best Fold: Fold {np.argmax([r['accuracy'] for r in fold_results])+1}\n")
        f.write(f"  Std Dev: {results['std_accuracy']*100:.2f}%\n")
        
        f.write("\nOVERALL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"  CV Average: {results['average_accuracy']*100:.2f}% (±{results['std_accuracy']*100:.2f}%)\n")
        f.write(f"  Overall Accuracy: {results['overall_accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {results['precision']*100:.2f}%\n")
        f.write(f"  Recall: {results['recall']*100:.2f}%\n")
        f.write(f"  F1-Score: {results['f1_score']*100:.2f}%\n")
        
        if 'test_results' in results:
            f.write("\nTEST SET RESULTS\n")
            f.write("-"*70 + "\n")
            tr = results['test_results']
            f.write(f"  Test Accuracy: {tr['test_accuracy']*100:.2f}%\n")
            f.write(f"  Test Precision: {tr['test_precision']*100:.2f}%\n")
            f.write(f"  Test Recall: {tr['test_recall']*100:.2f}%\n")
            f.write(f"  Test F1-Score: {tr['test_f1_score']*100:.2f}%\n")
        
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
                       choices=["tsl51_user_sign", "tsl51_expert", "tsl51_combined", "local"],
                       help="Dataset to use (default: tsl51_user_sign)")
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
                       choices=["mlp", "gru"],
                       help="Model architecture: 'mlp' or 'gru' (default: gru)")
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
    parser.add_argument("--test-split", type=float, default=0.0,
                       help="Train/test split ratio (e.g., 0.2 = 20% test). If 0, use full data for CV (default: 0)")
    args = parser.parse_args()
    
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
    
    use_amp = torch.cuda.is_available()
    
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
    elif args.dataset == "tsl51_combined":
        X, y, classes = load_tsl51_combined(
            max_samples=args.samples,
            force_download=args.force_download
        )
    else:
        print(f"ERROR: Unknown dataset: {args.dataset}")
        return 1
    
    if X is None:
        return 1
    
    # Apply data augmentation if requested
    if args.augment > 1:
        X, y = augment_data(X, y, classes, augmentation_factor=args.augment)
    
    # Apply train/test split if requested
    test_samples_count = 0
    has_test_set = False
    if args.test_split > 0:
        from sklearn.model_selection import train_test_split
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=args.test_split, stratify=y, random_state=args.seed
        )
        test_samples_count = len(X_test)
        has_test_set = True
        print(f"Train/Test Split: {len(X)} train, {test_samples_count} test")
    
    print(f"\nData: {len(X)} samples, {X.shape[1]} features, {len(classes)} classes")
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    all_preds, all_targets = [], []
    best_models, fold_means, fold_stds = [], [], []
    
    scaler = GradScaler('cuda') if use_amp else None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        mean, std = X_train.mean(0), X_train.std(0) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        fold_means.append(mean)
        fold_stds.append(std)
        
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=0
        )
        
        model = MODEL_CLASSES[args.model](X.shape[1], len(classes), args.hidden, args.layers, args.dropout)
        model = model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                              steps_per_epoch=len(train_loader))
        
        best_acc, best_state, patience = 0.0, None, 0
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, use_amp)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
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
        
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, final_acc, preds, targets = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"  Fold {fold+1}: {final_acc*100:.2f}%")
        fold_results.append({'fold': fold+1, 'accuracy': float(final_acc)})
        all_preds.extend(preds)
        all_targets.extend(targets)
        best_models.append(best_state)
    
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    overall_acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Get best fold index
    best_idx = np.argmax([r['accuracy'] for r in fold_results])
    
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
        best_model = MODEL_CLASSES[args.model](X.shape[1], len(classes), args.hidden, args.layers, args.dropout)
        best_model = best_model.to(DEVICE)  # Move to GPU first
        best_model.load_state_dict(best_models[best_idx])  # State dict is on CPU, will be moved by load_state_dict
        best_model.eval()
        
        test_preds, test_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                out = best_model(X_batch)
                test_preds.extend(out.argmax(1).cpu().numpy())
                test_targets.extend(y_batch.numpy())
        
        test_acc = accuracy_score(test_targets, test_preds)
        test_precision = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
        
        test_results = {
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1)
        }
        
        print(f"Test Set: {test_acc*100:.2f}% | P: {test_precision*100:.2f}% | R: {test_recall*100:.2f}% | F1: {test_f1*100:.2f}%")
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"CV Avg: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print(f"Overall: {overall_acc*100:.2f}% | P: {precision*100:.2f}% | R: {recall*100:.2f}% | F1: {f1*100:.2f}%")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'fold_results': fold_results,
        'average_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'overall_accuracy': float(overall_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_classes': len(classes),
        'num_samples': len(X),
        'test_samples': test_samples_count,
        'input_dim': X.shape[1]
    }
    
    if test_results:
        results['test_results'] = test_results
    
    # Create unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    (PROJECT_DIR / "results").mkdir(exist_ok=True)
    json_file = PROJECT_DIR / "results" / f"cv_{timestamp}.json"
    json_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    save_visualizations(results, fold_results, PROJECT_DIR / "results", args)
    
    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    
    # Save model with unique name (no overwrite)
    model_filename = f"tsl51_{args.model}_{timestamp}.pt"
    model_path = PROJECT_DIR / "models" / model_filename
    
    torch.save({
        'state_dict': best_models[best_idx],
        'classes': classes.tolist(),
        'mean': fold_means[best_idx].tolist(),
        'std': fold_stds[best_idx].tolist(),
        'input_dim': X.shape[1],
        'num_classes': len(classes),
        'model': args.model,
        'accuracy': float(fold_results[best_idx]['accuracy']),
        'timestamp': timestamp,
        'config': vars(args)
    }, model_path)
    
    print(f"\nResults: {json_file}")
    print(f"Model: {model_path}")
    print("DONE!")
    return 0


if __name__ == "__main__":
    sys.exit(main())