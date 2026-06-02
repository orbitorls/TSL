#!/usr/bin/env python3
"""Redirect to the repo-root golden exporter (canonical for CI)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO_ROOT / "scripts" / "export_golden.py"

if __name__ == "__main__":
    if not _CANONICAL.is_file():
        print(f"[error] missing canonical exporter: {_CANONICAL}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[redirect] python-legacy/scripts/export_golden.py -> {_CANONICAL}")
    runpy.run_path(str(_CANONICAL), run_name="__main__")
