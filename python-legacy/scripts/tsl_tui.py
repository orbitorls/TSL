#!/usr/bin/env python3
"""Thin launcher for the Textual training TUI (same as python -m tsl_tui)."""

from __future__ import annotations

import sys
from pathlib import Path

_LEGACY = Path(__file__).resolve().parents[1]
_SRC = _LEGACY / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tsl_tui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
