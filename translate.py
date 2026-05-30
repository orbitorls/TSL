"""Backward-compatible re-export for legacy ``translate.py`` scripts."""

from src.inference.translate import *  # noqa: F401,F403


def main():
    from src.inference.translate import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

