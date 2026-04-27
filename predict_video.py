"""Legacy shim for src.inference.predict_video."""

from src.inference.predict_video import *  # noqa: F401,F403
from src.inference.predict_video import main  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
