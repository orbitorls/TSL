"""
Download TSL-51 dataset efficiently (run once before training)
"""
import os
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_tsl51(max_samples=2000):
    """Download TSL-51 data efficiently."""
    cache_file = CACHE_DIR / f"tsl51_v2_{max_samples}.npz"
    
    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        return
    
    print("="*70)
    print("TSL-51 Dataset Download")
    print("="*70)
    
    # Download metadata first
    print("\n[1/2] Downloading metadata...")
    meta_path = hf_hub_download(
        repo_id='Namonpas/thai-sign-language-tsl51',
        filename='metadata/expert_metadata.csv',
        repo_type='dataset'
    )
    
    metadata = pd.read_csv(meta_path)
    print(f"Metadata loaded: {len(metadata)} entries, {metadata['sign_clean'].nunique()} unique signs")
    
    # Sample videos evenly across sign classes
    sign_counts = metadata['sign_clean'].value_counts()
    samples_per_sign = max(1, max_samples // len(sign_counts))
    
    selected_videos = set()
    for sign in sign_counts.index[:52]:  # 51 signs + null_act
        sign_videos = metadata[metadata['sign_clean'] == sign]['video_id'].unique()
        n = min(samples_per_sign, len(sign_videos))
        if n > 0:
            selected_videos.update(np.random.choice(sign_videos, n, replace=False).tolist())
    
    print(f"Selected {len(selected_videos)} videos")
    
    # Load landmark files
    X_list, y_list = [], []
    processed = set()
    
    print("\n[2/2] Processing landmark files...")
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
        if len(X_list) >= max_samples:
            break
        
        video_id = row['video_id']
        if video_id in processed:
            continue
        
        sign = row.get('sign_clean', None)
        if sign is None or pd.isna(sign):
            continue
        
        landmark_path = row.get('landmark_path', None)
        if landmark_path is None or pd.isna(landmark_path):
            continue
        
        try:
            lm_path = hf_hub_download(
                repo_id='Namonpas/thai-sign-language-tsl51',
                filename=landmark_path,
                repo_type='dataset'
            )
            lm_df = pd.read_csv(lm_path)
            
            # Extract features (average across frames)
            features = []
            
            # Left hand (21 points * 3 = 63)
            for i in range(21):
                for coord in ['x', 'y', 'z']:
                    col = f'lh_{coord}{i}'
                    if col in lm_df.columns:
                        features.append(lm_df[col].mean())
                    else:
                        features.append(0.0)
            
            # Right hand (21 points * 3 = 63)
            for i in range(21):
                for coord in ['x', 'y', 'z']:
                    col = f'rh_{coord}{i}'
                    if col in lm_df.columns:
                        features.append(lm_df[col].mean())
                    else:
                        features.append(0.0)
            
            # Pose landmarks (shoulder, elbow, wrist, face)
            pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                        'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                        'mouth_right', 'mouth_left']
            for col_base in pose_cols:
                for coord in ['x', 'y', 'z']:
                    col = f'{col_base}_{coord}'
                    if col in lm_df.columns:
                        features.append(lm_df[col].mean())
                    else:
                        features.append(0.0)
            
            if len(features) > 0:
                X_list.append(np.array(features, dtype=np.float32))
                y_list.append(sign)
                processed.add(video_id)
                
        except Exception as e:
            continue
    
    if not X_list:
        print("ERROR: No valid samples!")
        return
    
    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    classes = np.array(sorted(set(y_list)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y_list], dtype=np.int64)
    
    # Save cache
    np.savez(cache_file, X=X, y=y, classes=classes)
    
    print(f"\nSaved: {cache_file}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    
    download_tsl51(args.samples)