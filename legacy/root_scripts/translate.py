"""Legacy shim for src.inference.translate."""

from src.inference.translate import *  # noqa: F401,F403
from src.inference.translate import main  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
