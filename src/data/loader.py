# src/data/loader.py - Data loading utilities
# 
# Note: This is a stub. Real implementation references 
# the original functions in train_tsl51_v3.py
# 
# To use the full data loading, import from the original module:
#   from train_tsl51_v3 import load_tsl51_user_sign, load_tsl51_expert

from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def load_tsl51_user_sign(max_samples: Optional[int] = None, force_download: bool = False):
    """Load TSL-51 user sign dataset.
    
    Args:
        max_samples: Limit number of samples (None = all)
        force_download: Force re-download even if cached
        
    Returns:
        Tuple of (X, y) arrays
    """
    # Delegate to original implementation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_tsl51_v3 import load_tsl51_user_sign as _load
    return _load(max_samples=max_samples, force_download=force_download)


def load_tsl51_expert(include_augmented: bool = False, max_samples: Optional[int] = None, force_download: bool = False):
    """Load TSL-51 expert dataset.
    
    Args:
        include_augmented: Include augmented data
        max_samples: Limit number of samples
        force_download: Force re-download
        
    Returns:
        Tuple of (X, y) arrays
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_tsl51_v3 import load_tsl51_expert as _load
    return _load(include_augmented=include_augmented, max_samples=max_samples, force_download=force_download)


def load_cached_dataset(cache_name: str = "user_sign_data.npz") -> Tuple[np.ndarray, np.ndarray]:
    """Load cached dataset from .cache directory.
    
    Args:
        cache_name: Name of cache file
        
    Returns:
        Tuple of (X, y) arrays
    """
    cache_dir = Path(__file__).parent.parent / ".cache" / "tsl51"
    cache_path = cache_dir / cache_name
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    data = np.load(cache_path)
    return data['X'], data['y']