"""Legacy shim for src.inference.camera_translate."""

from src.inference.camera_translate import *  # noqa: F401,F403
from src.inference.camera_translate import main, ThaiSignTranslator  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
