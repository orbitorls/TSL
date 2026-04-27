"""Data loading utilities for TSL-51 datasets.

This module provides clean, modular data loading functions for TSL-51 datasets
without dependencies on the training script.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import zipfile
import logging

from utils.dataset_utils import safe_mean

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Check for tqdm
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
    tqdm = _tqdm
except ImportError:
    HAS_TQDM = False
    tqdm = None


def _extract_162_features(lm_df):
    """Extract 162 features (hand + pose) from landmark dataframe.
    
    Args:
        lm_df: Pandas DataFrame with landmark columns
        
    Returns:
        numpy array of 162 features
    """
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

    # Pose landmarks (12 points * 3 = 36)
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
    
    return np.nan_to_num(np.array(features, dtype=np.float32))


def load_tsl51_user_sign(max_samples: Optional[int] = None, force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TSL-51 user sign dataset from HuggingFace.
    
    Args:
        max_samples: Limit number of samples (None = all)
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (X, y, classes) arrays
    """
    from huggingface_hub import hf_hub_download
    
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
        
        # Download user_sign_metadata
        meta_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename='metadata/user_sign_metadata.csv',
            repo_type='dataset'
        )
        metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
        
        # Count unique signs
        try:
            raw = metadata['sign_clean']
            if hasattr(raw, 'values'):
                vals = np.asarray(raw.values)
            else:
                vals = np.asarray(raw)
            unique_signs = int(len(set(vals.tolist())))
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
                
                # Ensure we pass strings
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
                features = _extract_162_features(lm_df)
                
                X_list.append(features)
                y_list.append(sign)
                
            except Exception as e:
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
        
        try:
            _, X_sampled, _, y_sampled = train_test_split(
                X_full, y_full,
                test_size=max_samples,
                stratify=y_full,
                random_state=42
            )
        except ValueError:
            # If stratification fails, use random sampling
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


def load_tsl51_expert(include_augmented: bool = False, max_samples: Optional[int] = None, force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TSL-51 expert dataset from HuggingFace.
    
    Args:
        include_augmented: If True, include pre-augmented data (45k+ samples)
                           If False, use only original data (~1155 samples)
        max_samples: Optional limit on number of samples
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (X, y, classes) arrays
    """
    from huggingface_hub import hf_hub_download
    
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
                                features = _extract_162_features(lm_df)
                                
                                if len(features) == 162:
                                    X_list.append(features)
                                    y_list.append(str(sign))
                                    processed.add(video_id)
                                    
                        except Exception as e:
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


def load_tsl51_expert_full(max_samples: Optional[int] = None, force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ALL expert data including augmented (~45k samples).
    
    This uses the loader_expert module which extracts all files from zip files
    without metadata matching, providing the full dataset.
    
    Args:
        max_samples: Optional limit on number of samples
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (X, y, classes) arrays
    """
    from .loader_expert import load_all_expert_landmarks
    
    print("Loading full expert dataset (~45k samples)...")
    X, y, classes = load_all_expert_landmarks(
        feature_level="basic",
        max_samples=max_samples,
        force_download=force_download
    )
    
    return X, y, classes


def load_tsl51_combined(max_samples: Optional[int] = None, force_download: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_cached_dataset(cache_name: str = "user_sign_data.npz") -> Tuple[np.ndarray, np.ndarray]:
    """Load cached dataset from .cache directory.
    
    Args:
        cache_name: Name of cache file
        
    Returns:
        Tuple of (X, y) arrays
    """
    cache_path = CACHE_DIR / cache_name
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    data = np.load(cache_path)
    return data['X'], data['y']


def validate_dataset(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    """Validate dataset and return quality metrics.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
        classes: Class names array
        
    Returns:
        Dictionary containing validation results and quality metrics
    """
    results = {
        'valid': True,
        'n_samples': len(X),
        'n_features': X.shape[1] if len(X.shape) > 1 else 1,
        'n_classes': len(classes),
        'issues': []
    }
    
    # Check for NaN/Inf
    if np.isnan(X).any():
        results['valid'] = False
        results['issues'].append(f"Found {np.isnan(X).sum()} NaN values in features")
    
    if np.isinf(X).any():
        results['valid'] = False
        results['issues'].append(f"Found {np.isinf(X).sum()} Inf values in features")
    
    # Check for zero variance features
    if len(X.shape) > 1:
        feature_var = np.var(X, axis=0)
        zero_var_features = np.sum(feature_var == 0)
        if zero_var_features > 0:
            results['issues'].append(f"{zero_var_features} features have zero variance")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    results['class_distribution'] = {
        'min_samples_per_class': int(min_count),
        'max_samples_per_class': int(max_count),
        'imbalance_ratio': float(imbalance_ratio)
    }
    
    if imbalance_ratio > 10:
        results['issues'].append(f"High class imbalance: ratio {imbalance_ratio:.1f}:1")
    
    # Check feature range
    if len(X.shape) > 1:
        feature_min = X.min()
        feature_max = X.max()
        results['feature_range'] = {'min': float(feature_min), 'max': float(feature_max)}
        
        if feature_min < -10 or feature_max > 10:
            results['issues'].append(f"Feature values out of normal range: [{feature_min:.2f}, {feature_max:.2f}]")
    
    return results


def print_dataset_quality_report(validation_results: dict):
    """Print formatted dataset quality report.
    
    Args:
        validation_results: Dictionary from validate_dataset()
    """
    print("\n" + "="*70)
    print("DATASET QUALITY REPORT")
    print("="*70)
    
    print(f"Samples: {validation_results['n_samples']}")
    print(f"Features: {validation_results['n_features']}")
    print(f"Classes: {validation_results['n_classes']}")
    
    if 'class_distribution' in validation_results:
        cd = validation_results['class_distribution']
        print(f"\nClass Distribution:")
        print(f"  Min samples per class: {cd['min_samples_per_class']}")
        print(f"  Max samples per class: {cd['max_samples_per_class']}")
        print(f"  Imbalance ratio: {cd['imbalance_ratio']:.2f}:1")
    
    if 'feature_range' in validation_results:
        fr = validation_results['feature_range']
        print(f"\nFeature Range:")
        print(f"  Min: {fr['min']:.4f}")
        print(f"  Max: {fr['max']:.4f}")
    
    if validation_results['issues']:
        print(f"\n⚠️  Issues Found ({len(validation_results['issues'])}):")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    else:
        print("\n✓ No issues found")
    
    print("="*70 + "\n")


def load_local_dataset(data_path, use_cache=True):
    """Load local dataset from CSV or NumPy file.
    
    Args:
        data_path: Path to CSV or NumPy file
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (X, y, classes) or (None, None, None) on error
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
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
    else:
        print(f"ERROR: Unsupported file format: {data_path.suffix}")
        return None, None, None
    
    # Cache the loaded data
    if use_cache:
        np.savez(cache_file, X=X, y=y, classes=classes)
        print(f"Cached to: {cache_file}")
    
    return X, y, classes
