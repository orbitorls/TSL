"""Compatibility shims for optional dependencies (AMP, tqdm, matplotlib) and platform-specific fixes."""

import os
import sys


def setup_windows_encoding():
    """Fix Windows console encoding for Thai characters."""
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def setup_mkl_threads():
    """Set MKL environment variable to avoid duplicate library errors."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class _noop_context:
    def __init__(self, enabled=False):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


HAS_AMP = False
GradScaler = None  # type: ignore[misc]
autocast = _noop_context  # type: ignore[assignment]

# Prefer torch.amp API (PyTorch >= 2.0) for mixed precision.
# torch.cuda.amp is deprecated in PyTorch 2.6+ in favour of torch.amp.
try:
    from torch.amp.grad_scaler import GradScaler
    from torch.amp.autocast_mode import autocast

    HAS_AMP = True
except Exception:
    try:
        from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]

        HAS_AMP = True
    except Exception:
        GradScaler = None
        autocast = _noop_context  # type: ignore[assignment]
        HAS_AMP = False

try:
    from tqdm import tqdm as _tqdm

    HAS_TQDM = True
    tqdm = _tqdm
except ImportError:
    HAS_TQDM = False
    tqdm = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]
