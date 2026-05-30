# Fix the expert data download with proper Thai encoding
# This will re-download and properly encode Thai characters

import os
import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from utils.dataset_utils import safe_mean
from huggingface_hub import hf_hub_download
import zipfile

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Module logger
logger = logging.getLogger(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Force UTF-8 encoding globally
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def download_expert_fixed():
    """Download ALL expert data with PROPER Thai encoding."""
    cache_file = CACHE_DIR / "expert_full_data_fixed.npz"
    
    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        print(f"Samples: {len(data['X'])}, Features: {data['X'].shape[1]}, Classes: {len(data['classes'])}")
        # Print classes to verify Thai is readable
        print("\nClasses (checking encoding):")
        for i, c in enumerate(data['classes'][:10]):
            print(f"  {i}: {repr(c)}")
        return cache_file
    
    print("="*70)
    print("TSL-51 Expert Full Dataset Download - FIXED ENCODING")
    print("="*70)
    
    # Step 1: Download metadata with proper encoding
    print("\n[1/3] Loading metadata...")
    meta_path = hf_hub_download(
        repo_id='Namonpas/thai-sign-language-tsl51',
        filename='metadata/expert_metadata.csv',
        repo_type='dataset'
    )
    
    # Try reading with UTF-8 first, then fallback
    try:
        metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
    except Exception as e1:
        # Log and try fallback encodings
        logger.debug("utf-8-sig read failed for %s: %s", meta_path, e1, exc_info=True)
        try:
            metadata = pd.read_csv(meta_path, encoding='utf-8')
        except Exception as e2:
            logger.debug("utf-8 read failed for %s: %s", meta_path, e2, exc_info=True)
            metadata = pd.read_csv(meta_path, encoding='latin1')
    
    print(f"Total metadata: {len(metadata)} entries")
    
    # Build video_id -> sign mapping
    video_to_sign = {}
    for idx, row in metadata.iterrows():
        vid = str(row['video_id'])
        sign = str(row.get('sign_clean', ''))
        if sign and sign != 'nan':
            video_to_sign[vid] = sign
    
    print(f"Unique video IDs with signs: {len(video_to_sign)}")
    
    # Show sample to verify encoding
    print("\nSample mappings (should be readable Thai):")
    for i, (vid, sign) in enumerate(list(video_to_sign.items())[:5]):
        print(f"  {vid} -> {sign}")
    
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
                    basename = csv_file.split('/')[-1].replace('.csv', '')
                    
                    # Remove augmentation suffix
                    base_name = basename.split('_blur')[0].split('_brightness')[0].split('_comb')[0].split('_contrast')[0].split('_noise')[0].split('_rotation')[0].split('_scale')[0]
                    
                    # Find matching sign
                    sign = None
                    if basename in video_to_sign:
                        sign = video_to_sign[basename]
                    elif base_name in video_to_sign:
                        sign = video_to_sign[base_name]
                    
                    if sign is None:
                        skipped += 1
                        continue
                    
                    with z.open(csv_file) as f:
                        lm_df = pd.read_csv(f)
                    
                    # Extract 162 features
                    features = []
                    
                    # Left hand (63)
                    for i in range(21):
                        for c in ['x', 'y', 'z']:
                            col = f'lh_{c}{i}'
                            features.append(safe_mean(lm_df[col]) if col in lm_df.columns else 0.0)
                    
                    # Right hand (63)
                    for i in range(21):
                        for c in ['x', 'y', 'z']:
                            col = f'rh_{c}{i}'
                            features.append(safe_mean(lm_df[col]) if col in lm_df.columns else 0.0)
                    
                    # Pose (36)
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
                    # Log why this CSV was skipped for future debugging
                    logger.debug("Skipping %s due to processing error: %s", csv_file, e, exc_info=True)
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
    
    # Save with proper encoding
    np.savez(cache_file, X=X, y=y, classes=classes)
    
    print(f"\nSaved: {cache_file}")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")
    
    # Verify classes are readable
    print("\nVerifying saved classes (should be readable Thai):")
    for i, c in enumerate(classes[:15]):
        print(f"  {i}: {repr(c)}")
    
    return cache_file


if __name__ == "__main__":
    download_expert_fixed()
