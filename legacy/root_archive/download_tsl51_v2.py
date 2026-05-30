"""
Download TSL-51 dataset efficiently (run once before training)
"""
import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import zipfile
import json
from datetime import datetime

# Windows UTF-8 fix — must be before any print/log output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Module logger
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_tsl51(max_samples=2000):
    """Download TSL-51 expert data efficiently from zip files."""
    cache_file = CACHE_DIR / f"tsl51_expert_{max_samples}.npz"
    
    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        print(f"Samples: {len(data['X'])}")
        return cache_file
    
    print("="*70)
    print("TSL-51 Expert Dataset Download")
    print("="*70)
    
    # Download both expert zip files
    zip_files = [
        ('landmarks/expert_scraped.zip', 'expert_scraped'),
        ('landmarks/expert_primary_02.zip', 'expert_primary'),
    ]
    
    X_list, y_list = [], []
    processed = set()
    
    for zip_filename, source_name in zip_files:
        print(f"\n[Processing] {source_name}...")
        
        # Download zip
        zip_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename=zip_filename,
            repo_type='dataset'
        )
        
        # Load metadata (try common encodings, robust for Thai text)
        meta_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename='metadata/expert_metadata.csv',
            repo_type='dataset'
        )
        try:
            metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
        except Exception:
            try:
                metadata = pd.read_csv(meta_path, encoding='utf-8')
            except Exception:
                metadata = pd.read_csv(meta_path, encoding='latin1')
        
        # Filter to non-augmented
        non_aug = metadata[~metadata['is_augmented']].copy()
        
        # Open zip and extract
        with zipfile.ZipFile(zip_path, 'r') as z:
            for idx, row in tqdm(non_aug.iterrows(), total=len(non_aug), desc=f"{source_name}"):
                if len(X_list) >= max_samples:
                    break
                # Extract and sanitize fields
                video_id = str(row.get('video_id', '')).strip()
                if not video_id or video_id in processed:
                    continue

                sign_raw = row.get('sign_clean', '')
                if pd.isna(sign_raw) or str(sign_raw).strip() == '':
                    continue
                sign = str(sign_raw).strip()

                # Get filename from landmark_path (robust against NaN/None)
                lm_path_raw = row.get('landmark_path', '')
                lm_path = '' if pd.isna(lm_path_raw) else str(lm_path_raw)
                lm_filename = lm_path.split('/')[-1] if ('/' in lm_path and lm_path) else lm_path
                zip_entry = f"landmarks/{lm_filename}"
                
                try:
                    with z.open(zip_entry) as f:
                        # CSVs inside zip may have different encodings; try utf-8 then latin1
                        try:
                            lm_df = pd.read_csv(f)
                        except Exception:
                            try:
                                # reopen the entry for a second read
                                with z.open(zip_entry) as f2:
                                    lm_df = pd.read_csv(f2, encoding='utf-8')
                            except Exception:
                                with z.open(zip_entry) as f3:
                                    lm_df = pd.read_csv(f3, encoding='latin1')
                    
                    # Extract 162 features
                    features = []
                    
                    from utils.dataset_utils import safe_mean
                    # Left hand (21 * 3 = 63)
                    for i in range(21):
                        for coord in ['x', 'y', 'z']:
                            col = f'lh_{coord}{i}'
                            if col in lm_df.columns:
                                features.append(safe_mean(lm_df[col]))
                            else:
                                features.append(0.0)
                    
                    # Right hand (21 * 3 = 63)
                    for i in range(21):
                        for coord in ['x', 'y', 'z']:
                            col = f'rh_{coord}{i}'
                            if col in lm_df.columns:
                                features.append(safe_mean(lm_df[col]))
                            else:
                                features.append(0.0)
                    
                    # Pose (12 * 3 = 36)
                    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                                'mouth_right', 'mouth_left']
                    for col_base in pose_cols:
                        for coord in ['x', 'y', 'z']:
                            col = f'{col_base}_{coord}'
                            if col in lm_df.columns:
                                features.append(safe_mean(lm_df[col]))
                            else:
                                features.append(0.0)
                    
                    if len(features) == 162:
                        X_list.append(np.array(features, dtype=np.float32))
                        y_list.append(str(sign))
                        processed.add(video_id)
                    else:
                        logger.debug("Skipping %s: extracted %d features (expected 162) from %s", video_id, len(features), zip_entry)
                        
                except Exception as e:
                    # Log and continue; avoid silent swallowing of exceptions
                    name = video_id if 'video_id' in locals() else lm_filename
                    try:
                        print(f"  Skipping {name}: {e}")
                    except Exception as e2:
                        logger.debug("Failed printing skip message for %s: %s", name, e2, exc_info=True)
                    # Record full debug information to logger for post-mortem
                    logger.debug("Error processing %s in %s: %s", name, source_name, e, exc_info=True)
                    continue
        
        print(f"  Total extracted so far: {len(X_list)}")
    
    if not X_list:
        print("ERROR: No valid samples!")
        return None
    
    print(f"\nExtracted: {len(X_list)} samples")
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    classes = np.array(sorted(set(y_list)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y_list], dtype=np.int64)
    
    # Save to cache
    np.savez(cache_file, X=X, y=y, classes=classes)
    # Save metadata alongside cache for quick inspection
    meta_out = cache_file.with_suffix('.json')
    class_counts = {}
    try:
        unique, counts = np.unique(y, return_counts=True)
        for idx, cnt in zip(unique.tolist(), counts.tolist()):
            cls_name = classes[int(idx)] if int(idx) < len(classes) else str(idx)
            class_counts[cls_name] = int(cnt)
    except Exception:
        logger.debug("Failed computing class counts for cache metadata", exc_info=True)

    meta_payload = {
        'cache_file': str(cache_file),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'num_samples': int(len(X)),
        'feature_dim': int(X.shape[1]) if hasattr(X, 'shape') and len(X.shape) > 1 else None,
        'num_classes': int(len(classes)),
        'class_counts': class_counts,
    }
    try:
        meta_out.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        logger.debug("Failed writing cache metadata to %s", meta_out, exc_info=True)

    print(f"\nSaved: {cache_file}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")
    
    return cache_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "--limit", type=int, default=1200, help="Number of samples")
    args = parser.parse_args()
    
    download_tsl51(args.samples)
