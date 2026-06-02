"""Entry point: python -m tsl_tui (from python-legacy)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_LEGACY = _SRC.parent
for _entry in (_SRC, _LEGACY):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from tsl_tui.app import TslTuiApp  # noqa: E402


def main() -> None:
    TslTuiApp().run()


if __name__ == "__main__":
    main()
