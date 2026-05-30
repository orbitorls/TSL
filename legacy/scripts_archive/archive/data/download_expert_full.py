"""
Download ALL TSL-51 expert data (including augmented) from zip files.
This extracts ~30,000+ samples from expert_scraped.zip and expert_primary_02.zip.
"""
import os
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
from utils.dataset_utils import safe_mean
from huggingface_hub import hf_hub_download
import zipfile

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_expert_full():
    """Download ALL expert data (original + augmented) from zip files."""
    cache_file = CACHE_DIR / "expert_full_data.npz"
    
    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        print(f"Samples: {len(data['X'])}, Features: {data['X'].shape[1]}, Classes: {len(data['classes'])}")
        return cache_file
    
    print("="*70)
    print("TSL-51 Expert Full Dataset Download (ALL augmented data)")
    print("="*70)
    
    # Step 1: Download metadata
    print("\n[1/3] Loading metadata...")
    meta_path = hf_hub_download(
        repo_id='Namonpas/thai-sign-language-tsl51',
        filename='metadata/expert_metadata.csv',
        repo_type='dataset'
    )
    metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
    print(f"Total metadata: {len(metadata)} entries")
    
    # Build video_id -> sign mapping (use ALL entries including augmented)
    video_to_sign = {}
    for idx, row in metadata.iterrows():
        vid = row['video_id']
        sign = row.get('sign_clean')
        if sign and not pd.isna(sign):
            video_to_sign[vid] = str(sign)
    
    print(f"Unique video IDs with signs: {len(video_to_sign)}")
    
    # Step 2: Process each zip file
    zip_files = [
        ('landmarks/expert_scraped.zip', 'expert_scraped'),
        ('landmarks/expert_primary_02.zip', 'expert_primary'),
    ]
    
    X_list, y_list = [], []
    skipped = 0
    
    for zip_filename, source_name in zip_files:
        print(f"\n[2/3] Processing {source_name}...")
        zip_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename=zip_filename,
            repo_type='dataset'
        )
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            print(f"  {len(csv_files)} CSV files in zip")
            
            for csv_file in tqdm(csv_files, desc=f"  {source_name}"):
                try:
                    # Extract video_id from filename
                    # e.g., "landmarks/vid_0193_original.csv" -> "vid_0193"
                    basename = csv_file.split('/')[-1].replace('.csv', '')
                    # Remove augmentation suffix to get base video_id
                    # vid_0193_blur_k3 -> vid_0193
                    # vid_1313_original -> vid_1313
                    base_name = basename.split('_blur')[0].split('_brightness')[0].split('_comb')[0].split('_contrast')[0].split('_noise')[0].split('_rotation')[0].split('_scale')[0]
                    
                    # Find matching sign
                    sign = None
                    # Try exact match first
                    if basename in video_to_sign:
                        sign = video_to_sign[basename]
                    elif base_name in video_to_sign:
                        sign = video_to_sign[base_name]
                    else:
                        # Try partial match
                        for vid_id, vid_sign in video_to_sign.items():
                            if vid_id.startswith(base_name) or base_name.startswith(vid_id):
                                sign = vid_sign
                                break
                    
                    if sign is None:
                        skipped += 1
                        continue
                    
                    with z.open(csv_file) as f:
                        lm_df = pd.read_csv(f)
                    
                    # Extract 162 features
                    features = []
                    
                    for i in range(21):
                        for c in ['x', 'y', 'z']:
                            col = f'lh_{c}{i}'
                            features.append(safe_mean(lm_df[col]) if col in lm_df.columns else 0.0)
                    
                    for i in range(21):
                        for c in ['x', 'y', 'z']:
                            col = f'rh_{c}{i}'
                            features.append(safe_mean(lm_df[col]) if col in lm_df.columns else 0.0)
                    
                    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                                'mouth_right', 'mouth_left']
                    for col_base in pose_cols:
                        for c in ['x', 'y', 'z']:
                            col = f'{col_base}_{c}'
                            features.append(safe_mean(lm_df[col]) if col in lm_df.columns else 0.0)
                    
                    if len(features) == 162:
                        X_list.append(np.array(features, dtype=np.float32))
                        y_list.append(sign)
                        
                except Exception as e:
                    # Log and skip problematic file; avoid silent failures
                    logger.debug("Skipping %s in %s due to processing error: %s", csv_file, source_name, e, exc_info=True)
                    # Try to print a short message for console users; if printing fails, log that too
                    try:
                        print(f"  Skipping {csv_file}: {e}")
                    except Exception as inner_e:
                        logger.debug("Failed to print skip message for %s: %s", csv_file, inner_e, exc_info=True)
                    continue
    
    if not X_list:
        print("ERROR: No samples extracted!")
        return None
    
    print(f"\nExtracted: {len(X_list)} samples (skipped: {skipped})")
    
    # Convert
    X = np.array(X_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    classes = np.array(sorted(set(y_list)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in y_list], dtype=np.int64)
    
    # Save
    np.savez(cache_file, X=X, y=y, classes=classes)
    
    print(f"\nSaved: {cache_file}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")
    print(f"Samples per class: {len(X) / len(classes):.0f}")
    
    return cache_file


if __name__ == "__main__":
    download_expert_full()
