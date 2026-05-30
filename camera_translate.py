"""Backward-compatible re-export for legacy imports.

Historically the project exposed top-level modules such as ``camera_translate.py``
for interactive scripts. The canonical implementation now lives under
``src.inference.camera_translate`` but several existing workflows (and tests)
still import the legacy module path.

This shim keeps those imports working without duplicating logic.
"""

from src.inference.camera_translate import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
