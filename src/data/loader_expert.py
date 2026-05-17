"""Fixed expert data loader that extracts all files from zip without metadata matching."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import zipfile
import logging

from src.utils.dataset_utils import safe_mean
from .feature_extraction import FEATURE_DIMS, extract_features_from_landmark_df

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "tsl51"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_expert_landmarks(
    feature_level: str = "basic",
    max_samples: Optional[int] = None,
    force_download: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ALL expert data directly from zip files without metadata matching.

    This extracts all CSV files from the expert zip archives and uses the
    filename to determine the sign class.

    Class extraction: pdt_xxx_var_1_original.csv -> pdt_xxx
    """
    from huggingface_hub import hf_hub_download

    cache_file = CACHE_DIR / "expert_all_45k.npz"

    if cache_file.exists() and not force_download:
        data = np.load(cache_file, allow_pickle=True)
        print(f"Loaded from cache: {cache_file}")
        print(
            f"Samples: {len(data['X'])}, Features: {data['X'].shape[1]}, Classes: {len(data['classes'])}"
        )
        return data["X"], data["y"], data["classes"]

    zip_files = [
        ("landmarks/expert_scraped.zip", "scraped"),
        ("landmarks/expert_primary_02.zip", "primary_02"),
        ("landmarks/expert_primary_03.zip", "primary_03"),
    ]

    X_list = []
    y_list = []
    processed_signs = set()

    for zip_filename, source_name in zip_files:
        print(f"Processing {source_name}...")
        try:
            zip_path = hf_hub_download(
                repo_id="Namonpas/thai-sign-language-tsl51",
                filename=zip_filename,
                repo_type="dataset",
            )

            with zipfile.ZipFile(zip_path, "r") as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                print(f"  Found {len(csv_files)} CSV files")

                for i, csv_file in enumerate(csv_files):
                    if max_samples and len(X_list) >= max_samples:
                        break

                    try:
                        with z.open(csv_file) as f:
                            import pandas as pd

                            lm_df = pd.read_csv(f)

                        # Extract features
                        features = extract_features_from_landmark_df(
                            lm_df, feature_level
                        )

                        # Validate feature dimension
                        expected_dim = FEATURE_DIMS.get(feature_level, 162)
                        if len(features) != expected_dim:
                            continue

                        # Extract sign name from filename
                        # Format: landmarks/pdt_xxx_var_1_original.csv -> pdt_xxx
                        filename = csv_file.split("/")[-1].replace(".csv", "")

                        # Remove augmentation suffixes to get base class name
                        # e.g., pdt_xxx_var_1_original -> pdt_xxx
                        # e.g., kpp_noon_brightness_1.2x -> kpp_noon
                        base_name = filename
                        for suffix in [
                            "_blur",
                            "_brightness",
                            "_comb",
                            "_contrast",
                            "_noise",
                            "_rotation",
                            "_scale",
                        ]:
                            base_name = base_name.split(suffix)[0]

                        # Extract first two parts: prefix (pdt/kpp/vid) + classname (xxx)
                        parts = base_name.split("_")
                        if len(parts) >= 2:
                            # e.g., pdt_xxx or kpp_noon
                            sign_name = f"{parts[0]}_{parts[1]}"
                        else:
                            sign_name = parts[0]

                        X_list.append(features)
                        y_list.append(sign_name)

                        if len(X_list) % 1000 == 0:
                            print(f"    Processed {len(X_list)} samples...")

                    except Exception as e:
                        continue

            print(f"  Total from {source_name}: {len(X_list)}")

        except Exception as e:
            print(f"  Error processing {source_name}: {e}")
            continue

    if not X_list:
        raise RuntimeError("No expert samples could be extracted")

    # Convert to arrays
    X_full = np.array(X_list, dtype=np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Get unique classes
    unique_classes = sorted(set(y_list))
    classes = np.array(unique_classes)
    label_map = {c: i for i, c in enumerate(classes)}
    y_full = np.array([label_map[c] for c in y_list], dtype=np.int64)

    # Save to cache
    np.savez(cache_file, X=X_full, y=y_full, classes=classes)
    print(f"Saved {len(X_full)} samples to {cache_file}")

    return X_full, y_full, classes


if __name__ == "__main__":
    print("Loading all expert data...")
    X, y, classes = load_all_expert_landmarks()
    print(f"\nTotal samples: {len(X)}")
    print(f"Classes: {len(classes)}")
    print(f"Class names: {classes}")
