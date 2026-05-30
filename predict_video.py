"""Backward-compatible re-export for legacy ``predict_video.py`` scripts."""

from src.inference.predict_video import *  # noqa: F401,F403


def main():
    from src.inference.predict_video import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

