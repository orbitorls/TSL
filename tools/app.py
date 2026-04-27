#!/usr/bin/env python3
"""Compatibility import for the Textual TUI application.

The full implementation lives in ``tools.tui``. Keeping this module as a
thin re-export preserves the documented ``from tools.app import TSLApp`` path
without maintaining a second, incomplete app class.
"""

from .tui import TSLApp

__all__ = ["TSLApp"]
