"""Backward-compatible launcher for the legacy training CLI."""


def main():
    from legacy.root_scripts.train_tsl51_v3 import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
