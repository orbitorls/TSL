#!/usr/bin/env python3
"""Launch TSL translate FastAPI backend."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "python-legacy"


def main() -> int:
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "translate_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ],
        cwd=str(LEGACY),
    )


if __name__ == "__main__":
    raise SystemExit(main())
