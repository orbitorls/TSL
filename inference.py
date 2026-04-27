"""Legacy shim for src.inference.runner."""

from src.inference.runner import *  # noqa: F401,F403
from src.inference.runner import main, TSLPredictor  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
