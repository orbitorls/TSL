"""Backward-compatible re-export for legacy imports.

The project data extraction helpers were migrated to ``src.data.extractor``.
This shim preserves the historical top-level module name used by older scripts.
"""

from src.data.extractor import *  # noqa: F401,F403

