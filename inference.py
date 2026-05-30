"""Backward-compatible re-export for legacy ``inference.py`` scripts."""

from src.inference.runner import *  # noqa: F401,F403


def main():
    from src.inference.runner import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

