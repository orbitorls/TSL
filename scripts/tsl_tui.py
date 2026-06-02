#!/usr/bin/env python3
"""Thin launcher for the training TUI (repo root)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "python-legacy"
SRC = LEGACY / "src"

env = {**dict(__import__("os").environ), "PYTHONPATH": str(SRC)}
raise SystemExit(
    subprocess.call(
        [sys.executable, "-m", "tsl_tui", *sys.argv[1:]],
        cwd=str(LEGACY),
        env=env,
    )
)
