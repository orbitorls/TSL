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
--model           Model type: 'mlp', 'gru', 'mopgru', 'hybrid', 'ctc', 'cnn1d', 'temporal_attention', 'resmlp', 'lightweight' (default: gru)
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
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau as TorchReduceLROnPlateau


# ============================================================================
# CUSTOM LR SCHEDULERS
# ============================================================================

class CosineAnnealingWarmupScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Usage:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, warmup_epochs=5, total_epochs=30, min_lr=1e-6
        )
        for epoch in range(total_epochs):
            train(...)
            scheduler.step()
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.max_lr = max_lr if max_lr else self.base_lrs
        self.current_epoch = 0

    def step(self):
        """Update learning rate based on current epoch."""
        self.current_epoch += 1
        lr_list = []

        for i, group in enumerate(self.optimizer.param_groups):
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                progress = self.current_epoch / self.warmup_epochs
                lr = self.min_lr + (self.max_lr[i] - self.min_lr) * progress
            else:
                # Cosine annealing
                decay_epochs = self.total_epochs - self.warmup_epochs
                progress = (self.current_epoch - self.warmup_epochs) / decay_epochs
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                lr = self.min_lr + (self.max_lr[i] - self.min_lr) * cosine_decay

            group['lr'] = lr
            lr_list.append(lr)

        return lr_list

    def get_last_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class ReduceLROnPlateau:
    """
    Custom Reduce on Plateau scheduler with warmup support.

    Monitors a metric and reduces learning rate when no improvement is seen.
    """

    def __init__(self, optimizer, mode='max', factor=0.5, patience=3,
                 min_lr=1e-6, warmup_epochs=0, threshold=1e-4):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.threshold = threshold
        self.mode = mode

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric=None):
        """
        Update learning rate based on metric value.

        Args:
            metric: Current metric value to monitor (optional for compatibility)
        """
        self.current_epoch += 1

        # Warmup phase - no reduction
        if self.current_epoch <= self.warmup_epochs:
            return

        if self.best is None:
            self.best = metric
            return

        # Check if metric improved
        if self.mode == 'max':
            improved = metric > self.best + self.threshold
        else:
            improved = metric < self.best - self.threshold

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if no improvement for patience epochs
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        """Reduce learning rate by factor."""
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = max(group['lr'] * self.factor, self.min_lr)
            group['lr'] = new_lr

    def get_last_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
# SWA (STOCHASTIC WEIGHT AVERAGING) UTILITIES
# ============================================================================

class SWAUtility:
    """
    Stochastic Weight Averaging utility for improved generalization.

    SWA averages model weights over the last portion of training,
    which can lead to better generalization and flatter minima.
    """

    def __init__(self, model, swa_start_epoch, swa_lr, device):
        self.model = model
        self.swa_start = swa_start_epoch
        self.swa_lr = swa_lr
        self.device = device
        self.swa_count = 0
        self.swa_state = None
        self.active = False

    def update_swa(self, epoch, model_state):
        """
        Update SWA averaged weights.

        Args:
            epoch: Current training epoch
            model_state: Current model state dict
        """
        if epoch < self.swa_start:
            return

        if not self.active:
            self.active = True
            self.swa_state = {k: v.clone().to(self.device) for k, v in model_state.items()}
            self.swa_count = 1
        else:
            self.swa_count += 1
            for k, v in model_state.items():
                self.swa_state[k] = (self.swa_state[k] * (self.swa_count - 1) + v.to(self.device)) / self.swa_count

    def get_averaged_state(self):
        """Return the SWA averaged state dict."""
        return self.swa_state

    def apply_swa(self):
        """Apply SWA averaged weights to the model."""
        if self.swa_state is not None:
            self.model.load_state_dict(self.swa_state)


# ============================================================================
# DATA AUGMENTATION WITH MULTIPLE METHODS
# ============================================================================
def augment_with_method(X, y, method='basic', cutout_ratio=0.1):
    """
    Apply different augmentation methods.

    Args:
        X: Feature array (n_samples, n_features)
        y: Label array
        method: Augmentation method ('basic', 'timewarp', 'temporal_crop', 'cutout')
        cutout_ratio: Ratio of features to zero out for cutout

    Returns:
        X_aug, y_aug (augmented data)
    """
    if method == 'basic':
        return augment_data(X, y, None, augmentation_factor=1)
    elif method == 'timewarp':
        return _timewarp_augment(X, y)
    elif method == 'temporal_crop':
        return _temporal_crop_augment(X, y)
    elif method == 'cutout':
        return _cutout_augment(X, y, cutout_ratio)
    else:
        return X, y


def _timewarp_augment(X, y):
    """Apply time-warp augmentation."""
    n_samples = len(X)
    X_aug = X.copy()

    for i in range(n_samples):
        # Apply random smooth scaling to simulate temporal warping
        for j in range(X.shape[1] // 3):
            factor = np.random.uniform(1.0 - 0.05 * j / (X.shape[1] // 3),
                                       1.0 + 0.05 * j / (X.shape[1] // 3))
            start_idx = j * 3
            X_aug[i, start_idx:start_idx+3] *= factor

    return X_aug, y


def _temporal_crop_augment(X, y):
    """Apply temporal crop augmentation (random cropping with padding)."""
    n_samples = len(X)
    X_aug = X.copy()
    crop_ratio = 0.1

    for i in range(n_samples):
        crop_size = int(X.shape[1] * crop_ratio)
        crop_start = np.random.randint(0, max(1, X.shape[1] - crop_size))
        crop_end = crop_start + crop_size

        # Replace cropped region with mean of surrounding values
        if crop_start > 0:
            fill_value = X[i, :crop_start].mean()
        else:
            fill_value = X[i, crop_end:].mean() if crop_end < X.shape[1] else 0.0

        X_aug[i, crop_start:crop_end] = fill_value

    return X_aug, y


def _cutout_augment(X, y, cutout_ratio):
    """Apply cutout augmentation (randomly mask out features)."""
    n_samples = len(X)
    X_aug = X.copy()
    n_masked = int(X.shape[1] * cutout_ratio)

    for i in range(n_samples):
        # Randomly select positions to mask
        mask_indices = np.random.choice(X.shape[1], n_masked, replace=False)
        X_aug[i, mask_indices] = 0

    return X_aug, y


# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def save_confusion_matrix(y_true, y_pred, classes, output_dir, timestamp):
    """
    Generate and save confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        output_dir: Directory to save the figure
        timestamp: Timestamp string for filename
    """
    if not HAS_MATPLOTLIB or plt is None:
        return

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - TSL-51 Thai Sign Language Recognition',
                 fontsize=16, fontweight='bold', pad=20)

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{timestamp}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_dir / f'confusion_matrix_{timestamp}.png'}")


# ============================================================================
# CURVE SMOOTHING UTILITY
# ============================================================================

def smooth_curve(values, weight=0.8):
    """
    Apply exponential smoothing to training curves.

    Args:
        values: List of values to smooth
        weight: Smoothing weight (0 < weight < 1). Higher = smoother.

    Returns:
        Smoothed values list
    """
    smoothed = []
    last = values[0] if len(values) > 0 else 0

    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
from utils.dataset_utils import safe_mean

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

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


# ============================================================================
# NEW MODELS FROM ACADEMIC RESEARCH
# ============================================================================
# Import new models from src.core.models for reuse
from src.core.models import (
    CNN1DModel,
    TemporalAttentionModel,
    ResidualMLPModel,
    LightweightModel,
)

class MOPGRU(nn.Module):
    """
    Modified GRU (MOPGRU) - Multiplied Update Gate
    
    From: "An integrated mediapipe-optimized GRU model for Indian sign language recognition"
    Subramanian et al., Scientific Reports 2022
    
    Key innovation: Multiplies update gate by reset gate to discard redundant info
    This addresses hand occlusion and improves learning efficiency
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # MOPGRU layers
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, 
                        batch_first=True, bidirectional=True,
                        dropout=dropout if num_layers > 1 else 0)
        
        # Layer norm after GRU
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        out, _ = self.rnn(x)  # (batch, seq, hidden*2)
        
        # Take last timestep
        out = out[:, -1, :]  # (batch, hidden*2)
        
        # MOPGRU-style: apply norm then dropout
        out = self.norm(out)
        out = self.dropout(out)
        
        return self.fc(out)


class HybridGRUTransformer(nn.Module):
    """
    Hybrid GRU + Transformer Encoder
    
    Combines GRU (temporal) + Transformer (attention) for better sequence modeling
    Based on research: "Stack Transformer Based Spatial-Temporal Attention Model"
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3, nhead=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU for initial sequence encoding
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                       batch_first=True, bidirectional=True,
                       dropout=dropout if num_layers > 1 else 0)
        
        # Project to transformer dimension
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch, seq, hidden*2)
        
        # Project
        proj_out = self.proj(gru_out)  # (batch, seq, hidden)
        
        # Transformer attention
        trans_out = self.transformer(proj_out)  # (batch, seq, hidden)
        
        # Take last
        out = trans_out[:, -1, :]  # (batch, hidden)
        
        out = self.norm(out)
        out = self.dropout(out)
        
        return self.fc(out)


class CTCModel(nn.Module):
    """
    CTC-based model for sentence-level recognition
    
    Uses Connectionist Temporal Classification (CTC) for variable-length sequences
    Ideal for: sentences with multiple signs of varying lengths
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        # CTC uses blank token at index 0
        self.blank_idx = 0
        
        # Bidirectional GRU encoder
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                        batch_first=True, bidirectional=True,
                        dropout=dropout if num_layers > 1 else 0)
        
        # Project to num_classes (including blank)
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)  # +1 for blank
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) optional - actual sequence lengths for packing
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encode
        out, _ = self.gru(x)  # (batch, seq, hidden*2)
        
        # Project to classes + blank
        out = self.fc(out)  # (batch, seq, num_classes+1)
        
        # Log softmax for CTC
        out = torch.log_softmax(out, dim=-1)
        
        return out


# Feature extraction with extended landmarks
class FeatureExtractor:
    """
    Extended MediaPipe feature extractor
    
    Supports:
    - 162 features (legacy) - hand + pose
    - 258 features - hand + pose + finger details
    - 462 features - full hand connections
    - 1596 features - hand + pose + face (full MediaPipe)
    """
    BASIC_FEATURES = 162      # 21 hand * 3 * 2 + 12 pose * 3
    FINGER_FEATURES = 258     # + extra finger landmarks
    FULL_FEATURES = 462       # Full MediaPipe hand
    FACE_FEATURES = 1596      # + face mesh (478 landmarks)
    
    def __init__(self, feature_level='basic'):
        """
        Args:
            feature_level: 'basic' (162), 'finger' (258), 'full' (462), 'face' (1596)
        """
        self.feature_level = feature_level
        
        if feature_level == 'basic':
            self.feature_dim = self.BASIC_FEATURES
        elif feature_level == 'finger':
            self.feature_dim = self.FINGER_FEATURES
        elif feature_level == 'full':
            self.feature_dim = self.FACE_FEATURES  # Use max for full
        elif feature_level == 'face':
            self.feature_dim = self.FACE_FEATURES
        else:
            raise ValueError(f"Unknown feature_level: {feature_level}")
    
    def extract_from_dataframe(self, lm_df):
        """Extract features from landmark DataFrame."""
        features = []
        
        from utils.dataset_utils import safe_mean
        # ===== 1. BASIC: Hand + Pose (162) =====
        # Left hand (21 * 3 = 63)
        for i in range(21):
            for c in ['x', 'y', 'z']:
                col = f'lh_{c}{i}'
                if col in lm_df.columns:
                    features.append(safe_mean(lm_df[col]))
                else:
                    features.append(0.0)
        
        # Right hand (21 * 3 = 63)
        for i in range(21):
            for c in ['x', 'y', 'z']:
                col = f'rh_{c}{i}'
                if col in lm_df.columns:
                    features.append(safe_mean(lm_df[col]))
                else:
                    features.append(0.0)
        
        # Pose (12 * 3 = 36)
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
        
        # ===== 2. FINGER: Additional finger details =====
        if self.feature_level in ['finger', 'full', 'face']:
            # Additional finger-specific landmarks
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for hand_prefix in ['lh_', 'rh_']:
                for finger in finger_names:
                    for c in ['x', 'y', 'z']:
                        for joint in ['mcp', 'pip', 'dip']:
                            col = f'{hand_prefix}{finger}_{joint}_{c}'
                            if col in lm_df.columns:
                                features.append(safe_mean(lm_df[col]))
                            else:
                                features.append(0.0)
        
        # ===== 3. FACE: 478 Facial Landmarks (1434 features) =====
        if self.feature_level in ['face', 'full']:
            # MediaPipe Face Mesh has 478 landmarks
            for i in range(478):
                for c in ['x', 'y', 'z']:
                    col = f'face_{c}{i}'
                    if col in lm_df.columns:
                        features.append(safe_mean(lm_df[col]))
                    else:
                        features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)


# Frame sampling utilities
def sample_frames_uniform(landmarks, target_frames=30):
    """
    Sample frames uniformly from sequence.
    
    Args:
        landmarks: (n_frames, feature_dim) array
        target_frames: target number of frames
        
    Returns:
        sampled: (target_frames, feature_dim)
    """
    n_frames = len(landmarks)
    if n_frames == target_frames:
        return landmarks
    
    # Uniform sampling indices
    indices = np.linspace(0, n_frames - 1, target_frames).astype(int)
    return landmarks[indices]


def pad_or_truncate(sequence, target_length, pad_value=0.0):
    """
    Pad or truncate sequence to target length.
    
    Args:
        sequence: (seq_len, ...) array
        target_length: desired length
        pad_value: value for padding
        
    Returns:
        result: (target_length, ...)
    """
    seq_len = len(sequence)
    
    if seq_len == target_length:
        return sequence
    
    if seq_len < target_length:
        # Pad
        padding = np.full((target_length - seq_len,) + sequence.shape[1:], 
                        pad_value, dtype=sequence.dtype)
        return np.vstack([sequence, padding])
    else:
        # Truncate
        return sequence[:target_length]


# ============================================================================
# MODEL REGISTRY
# ============================================================================
MODEL_CLASSES = {
    'mlp': MLP,
    'gru': GRUModel,
    'mopgru': MOPGRU,
    'hybrid': HybridGRUTransformer,
    'ctc': CTCModel,
    'cnn1d': CNN1DModel,
    'temporal_attention': TemporalAttentionModel,
    'resmlp': ResidualMLPModel,
    'lightweight': LightweightModel,
}


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


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross Entropy with Label Smoothing for better generalization."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch_enhanced(
    model, loader, criterion, optimizer, scaler, device, use_amp,
    label_smoothing=0.0, mixup_alpha=0.0, gradient_clip=1.0, scheduler=None
):
    """Enhanced training with label smoothing, mixup, gradient clipping."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            X, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)

        if use_amp:
            cm: Any = autocast('cuda')
            with cm:
                out = model(X)
                if mixup_alpha > 0:
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                else:
                    loss = criterion(out, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        else:
            out = model(X)
            if mixup_alpha > 0:
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                loss = criterion(out, y)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        targets_for_acc = y_a if mixup_alpha > 0 else y
        correct += (out.argmax(1) == targets_for_acc).sum().item()
        total += y.size(0)

        if scheduler is not None:
            scheduler.step()

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
    if plt is None:
        return
    
    # Ensure all results values are native Python floats (handle serialized types)
    def _to_float(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val
        return val
    
    # Convert top-level results
    for k in ['average_accuracy', 'std_accuracy', 'overall_accuracy', 'precision', 'recall', 'f1_score', 'num_classes', 'num_samples', 'test_samples', 'input_dim']:
        if k in results:
            results[k] = _to_float(results[k])
    if 'test_results' in results and isinstance(results['test_results'], dict):
        for k in results['test_results']:
            results['test_results'][k] = _to_float(results['test_results'][k])
    # Convert fold results
    for fr in fold_results:
        if 'accuracy' in fr:
            fr['accuracy'] = _to_float(fr['accuracy'])
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Title with gradient effect
    fig.suptitle('TSL-51 Thai Sign Language Training Report', 
                 fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)
    
    # Calculate dynamic y-axis limits
    # Ensure all values are floats (handle any serialized string/numpy types)
    all_accs = [float(r['accuracy']) * 100 for r in fold_results]
    all_metrics = [float(results['overall_accuracy'])*100, float(results['precision'])*100, 
                   float(results['recall'])*100, float(results['f1_score'])*100]
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
    ax1.set_ylim((y_min, y_max))
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
                       choices=["mlp", "gru", "mopgru", "hybrid", "ctc", "cnn1d", "temporal_attention", "resmlp", "lightweight"],
                       help="Model architecture (default: gru)")
    parser.add_argument("--feature-level", type=str, default="basic",
                       choices=["basic", "finger", "full", "face"],
                       help="Feature level: 'basic'(162), 'finger'(258), 'full'(1596), 'face'(1434)")
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
    # Enhanced training arguments
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing factor (default: 0.1, 0 = disabled)")
    parser.add_argument("--mixup", type=float, default=0.2,
                       help="Mixup alpha (default: 0.2, 0 = disabled)")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping value (default: 1.0, 0 = disabled)")
    # LR Scheduling
    parser.add_argument("--lr-scheduler", type=str, default="one_cycle",
                       choices=["one_cycle", "cosine_warmup", "reduce_plateau", "warmup_cosine"],
                       help="Learning rate scheduler (default: one_cycle)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                       help="Number of warmup epochs for cosine_warmup scheduler (default: 5)")
    parser.add_argument("--plateau-factor", type=float, default=0.5,
                       help="Factor to reduce LR on plateau (default: 0.5)")
    parser.add_argument("--plateau-patience", type=int, default=3,
                       help="Epochs to wait before reducing LR on plateau (default: 3)")
    # Augmentation methods
    parser.add_argument("--augment-method", type=str, default="basic",
                       choices=["basic", "timewarp", "temporal_crop", "cutout"],
                       help="Augmentation method (default: basic)")
    parser.add_argument("--cutout-ratio", type=float, default=0.1,
                       help="Cutout ratio for cutout augmentation (default: 0.1)")
    # SWA training
    parser.add_argument("--use-swa", action="store_true",
                       help="Enable Stochastic Weight Averaging (SWA)")
    parser.add_argument("--swa-start-epoch", type=int, default=20,
                       help="Epoch to start SWA averaging (default: 20)")
    parser.add_argument("--swa-lr", type=float, default=0.0001,
                       help="SWA learning rate (default: 0.0001)")
    # Analytics
    parser.add_argument("--save-confusion-matrix", action="store_true",
                       help="Save confusion matrix visualization")
    parser.add_argument("--smooth-curves", action="store_true",
                       help="Apply smoothing to training curves")
    # Checkpointing
    parser.add_argument("--save-best-f1", action="store_true",
                       help="Save model checkpoint when F1 score improves")
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
    all_preds, all_targets = [], []
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

        # Use label smoothing loss if enabled
        if args.label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropyLoss(smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)

        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Initialize SWA if enabled
        swa_util = None
        if args.use_swa:
            swa_util = SWAUtility(model, args.swa_start_epoch, args.swa_lr, DEVICE)

        # Setup scheduler based on lr-scheduler argument
        if args.lr_scheduler == 'one_cycle':
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                  steps_per_epoch=len(train_loader))
        elif args.lr_scheduler == 'cosine_warmup':
            scheduler = CosineAnnealingWarmupScheduler(
                optimizer, warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs, min_lr=1e-6
            )
        elif args.lr_scheduler == 'reduce_plateau':
            scheduler = TorchReduceLROnPlateau(
                optimizer, mode='max', factor=args.plateau_factor,
                patience=args.plateau_patience, min_lr=1e-6
            )
        elif args.lr_scheduler == 'warmup_cosine':
            scheduler = CosineAnnealingWarmupScheduler(
                optimizer, warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs, min_lr=1e-6, max_lr=args.lr
            )

        best_acc, best_state, patience = 0.0, None, 0
        best_f1 = 0.0

        # Save initial (untrained) model state so best_state is never None
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            # Use enhanced training with label smoothing, mixup, gradient clipping
            train_loss, train_acc = train_epoch_enhanced(
                model, train_loader, criterion, optimizer, scaler, DEVICE, use_amp,
                label_smoothing=args.label_smoothing,
                mixup_alpha=args.mixup,  # Use mixup if specified, regardless of augment setting
                gradient_clip=args.gradient_clip,
                scheduler=None  # Manual step for better control
            )
            val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, DEVICE)

            # Update SWA
            if swa_util is not None:
                swa_util.update_swa(epoch, model.state_dict())

            # Step scheduler (compatible with all scheduler types)
            if args.lr_scheduler == 'one_cycle':
                scheduler.step()
            elif args.lr_scheduler in ['cosine_warmup', 'warmup_cosine']:
                scheduler.step()
            elif args.lr_scheduler == 'reduce_plateau':
                scheduler.step(val_acc)  # ReduceLROnPlateau expects metric

            # Calculate F1 for checkpointing
            val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division='warn')

            # Save based on best accuracy (or best F1 if enabled)
            if args.save_best_f1:
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

            if patience >= args.patience:
                print(f"  Early stop @ epoch {epoch+1}")
                break

        # Apply SWA if enabled
        if swa_util is not None and swa_util.active:
            swa_util.apply_swa()
            print(f"  Applied SWA (averaged over {swa_util.swa_count} updates)")

        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        else:
            # best_acc stayed 0 throughout; keep the randomly-initialised model
            pass
        _, final_acc, preds, targets = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"  Fold {fold+1}: {final_acc*100:.2f}%")
        fold_results.append({'fold': fold+1, 'accuracy': float(final_acc)})
        all_preds.extend(preds)
        all_targets.extend(targets)
        best_models.append(best_state)
    
    # Convert to floats explicitly for serialization and metrics
    acc_list = [float(r['accuracy']) for r in fold_results]
    avg_acc = float(np.mean(acc_list)) if acc_list else 0.0
    std_acc = float(np.std(acc_list)) if acc_list else 0.0
    overall_acc = accuracy_score(all_targets, all_preds)
    # sklearn typings expect zero_division to be 'warn'|'raise'|'0' in some stubs; pass 'warn' for compatibility
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division='warn')
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division='warn')
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division='warn')
    
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
        best_model = MODEL_CLASSES[args.model](input_dim, len(classes), args.hidden, args.layers, args.dropout)
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
        test_precision = precision_score(test_targets, test_preds, average='weighted', zero_division='warn')
        test_recall = recall_score(test_targets, test_preds, average='weighted', zero_division='warn')
        test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division='warn')
        
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

    # Save confusion matrix if enabled
    if args.save_confusion_matrix:
        save_confusion_matrix(all_targets, all_preds, classes.tolist(),
                             PROJECT_DIR / "results", timestamp)
    
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
        'accuracy': float(fold_results[best_idx]['accuracy']),
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
