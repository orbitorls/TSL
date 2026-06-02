#!/usr/bin/env python3
"""Thin launcher for Streamlit web inference app (repo root)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "python-legacy" / "web_infer_app.py"

raise SystemExit(
    subprocess.call([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
    ], cwd=str(REPO))
)
