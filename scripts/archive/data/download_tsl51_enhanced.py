"""
Download TSL-51 user_sign videos and extract enhanced (249d) features.

Usage:
    python scripts/data/download_tsl51_enhanced.py
    python scripts/data/download_tsl51_enhanced.py --max-samples 100 --target-frames 30
    python scripts/data/download_tsl51_enhanced.py --dataset user_sign  # default
    python scripts/data/download_tsl51_enhanced.py --dataset sentence

Output:
    .cache/tsl51/user_sign_data_enhanced.npz
    .cache/tsl51/sentence_data_enhanced.npz

Features extracted per frame (249 dims):
    - Base 162: left hand (63) + right hand (63) + pose (36) landmarks
    - Enhanced 87: hand spread, finger curl, orientation, scale,
      relative position, inter-hand distance, symmetry, finger crossing
"""

import argparse
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_extraction import (
    FEATURE_DIMS,
    build_enhanced_sequence_from_df,
)

CACHE_DIR = Path(".cache") / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = 'Namonpas/thai-sign-language-tsl51'


def normalize_label(filename: str) -> str | None:
    """Normalize filename to Thai word label."""
    base = filename.replace('.mp4', '')
    # Remove common suffixes
    for suffix in ['_original', '_1', '_2', '_3', '_4', '_5', '_sign', '_var']:
        if suffix in base:
            parts = base.split(suffix)
            base = parts[0]
    # Strip trailing underscores and numbers
    while base and base[-1] in '0123456789_':
        base = base[:-1]
    return base.strip() or None


def download_and_extract_user_sign(max_samples=None, target_frames=30, force=False):
    """Download user_sign landmark CSVs and extract enhanced features."""
    cache_file = CACHE_DIR / f"user_sign_data_enhanced_{target_frames}.npz"

    if cache_file.exists() and not force:
        print(f"Using cached data: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['classes']

    print("Listing files from HuggingFace...")
    files = list(list_repo_files(REPO_ID, repo_type='dataset'))
    csv_files = [f for f in files if f.startswith('landmarks/user_sign/') and f.endswith('.csv')]

    # Filter valid (non-null) samples
    valid_csvs = []
    for cf in csv_files:
        filename = cf.split('/')[-1].replace('.csv', '')
        label = normalize_label(filename)
        if label:
            valid_csvs.append((cf, label))

    print(f"Found {len(csv_files)} CSV files, {len(valid_csvs)} valid")

    if max_samples and max_samples < len(valid_csvs):
        import random
        random.seed(42)
        valid_csvs = random.sample(valid_csvs, max_samples)
        print(f"Sampled {max_samples} files")

    X_list, y_list = [], []
    errors = 0

    for csv_path_hf, label in tqdm(valid_csvs, desc="Processing user_sign"):
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=csv_path_hf,
                repo_type='dataset',
                local_dir=str(CACHE_DIR / "raw_landmarks"),
                local_dir_use_symlinks=False,
            )
            lm_df = pd.read_csv(local_path)

            seq = build_enhanced_sequence_from_df(lm_df, feature_level='enhanced', target_frames=target_frames)
            if seq is not None and seq.shape == (target_frames, FEATURE_DIMS['enhanced']):
                X_list.append(seq)
                y_list.append(label)
            else:
                errors += 1
                if errors <= 5:
                    print(f"  Warning: Bad shape for {csv_path_hf}: {seq.shape if seq is not None else None}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {csv_path_hf}: {e}")

    if not X_list:
        raise RuntimeError("No valid sequences extracted")

    X = np.array(X_list, dtype=np.float32)
    classes = np.array(sorted(set(y_list)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[label] for label in y_list], dtype=np.int64)

    print(f"\nExtracted {len(X)} sequences")
    print(f"Shape: {X.shape} (samples, frames, features)")
    print(f"Classes: {len(classes)}")
    print(f"Features: {X.shape[2]} dims (enhanced)")
    print(f"Errors: {errors}")

    np.savez(cache_file, X=X, y=y, classes=classes)
    print(f"Saved to: {cache_file}")

    return X, y, classes


def download_and_extract_sentence(max_samples=None, target_frames=30, force=False):
    """Download sentence landmark CSVs and extract enhanced features."""
    cache_file = CACHE_DIR / f"sentence_data_enhanced_{target_frames}.npz"

    if cache_file.exists() and not force:
        print(f"Using cached data: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['classes']

    print("Listing files from HuggingFace...")
    files = list(list_repo_files(REPO_ID, repo_type='dataset'))
    csv_files = [f for f in files if f.startswith('landmarks/user_sentence/') and f.endswith('.csv')]

    # Load metadata for sentence labels
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename='metadata/sentence_metadata.csv',
        repo_type='dataset',
    )
    meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')
    # Create video_id to sentence mapping
    sentence_map = {}
    if 'video_id' in meta_df.columns and 'sentence_clean' in meta_df.columns:
        for _, row in meta_df.iterrows():
            vid = str(row['video_id']).strip()
            sent = str(row['sentence_clean']).strip() if not pd.isna(row.get('sentence_clean')) else vid
            sentence_map[vid] = sent

    valid_csvs = []
    for cf in csv_files:
        filename = cf.split('/')[-1].replace('.csv', '')
        label = sentence_map.get(filename, filename)
        valid_csvs.append((cf, label))

    print(f"Found {len(csv_files)} sentence CSV files")

    if max_samples and max_samples < len(valid_csvs):
        import random
        random.seed(42)
        valid_csvs = random.sample(valid_csvs, max_samples)

    X_list, y_list = [], []
    errors = 0

    for csv_path_hf, label in tqdm(valid_csvs, desc="Processing sentence"):
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=csv_path_hf,
                repo_type='dataset',
                local_dir=str(CACHE_DIR / "raw_landmarks"),
                local_dir_use_symlinks=False,
            )
            lm_df = pd.read_csv(local_path)

            seq = build_enhanced_sequence_from_df(lm_df, feature_level='enhanced', target_frames=target_frames)
            if seq is not None and seq.shape == (target_frames, FEATURE_DIMS['enhanced']):
                X_list.append(seq)
                y_list.append(label)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {e}")

    if not X_list:
        raise RuntimeError("No valid sequences extracted")

    X = np.array(X_list, dtype=np.float32)
    classes = np.array(sorted(set(y_list)))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[label] for label in y_list], dtype=np.int64)

    print(f"\nExtracted {len(X)} sentence sequences")
    print(f"Shape: {X.shape}")
    print(f"Unique sentences: {len(classes)}")

    np.savez(cache_file, X=X, y=y, classes=classes)
    print(f"Saved to: {cache_file}")

    return X, y, classes


def main():
    parser = argparse.ArgumentParser(description="Download TSL-51 and extract enhanced features")
    parser.add_argument("--dataset", type=str, default="user_sign", choices=["user_sign", "sentence"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--target-frames", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("=" * 60)
    print("TSL-51 Enhanced Feature Extraction")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Target frames: {args.target_frames}")
    print(f"Feature dim: {FEATURE_DIMS['enhanced']} (enhanced)")
    print("=" * 60 + "\n")

    if args.dataset == "user_sign":
        X, y, classes = download_and_extract_user_sign(
            max_samples=args.max_samples,
            target_frames=args.target_frames,
            force=args.force,
        )
    else:
        X, y, classes = download_and_extract_sentence(
            max_samples=args.max_samples,
            target_frames=args.target_frames,
            force=args.force,
        )

    print("\n" + "=" * 60)
    print("Done! Use this cache for training:")
    print(f"  X shape: {X.shape}")
    print(f"  Classes: {len(classes)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
