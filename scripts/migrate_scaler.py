#!/usr/bin/env python3
"""Convert sklearn StandardScaler joblib (.pkl) to scaler.json for Rust (tsl-core)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export StandardScaler to JSON for tsl-core::StandardScaler"
    )
    parser.add_argument("pkl", type=Path, help="Path to scaler.pkl (joblib)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: same stem as input, .json)",
    )
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("joblib is required: pip install joblib scikit-learn", file=sys.stderr)
        return 1

    pkl_path = args.pkl
    if not pkl_path.is_file():
        print(f"[ERROR] not found: {pkl_path}", file=sys.stderr)
        return 1

    out_path = args.output or pkl_path.with_suffix(".json")
    scaler = joblib.load(pkl_path)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    n_features = getattr(scaler, "n_features_in_", None)
    if mean is None or scale is None:
        print("[ERROR] object does not look like sklearn StandardScaler", file=sys.stderr)
        return 1
    if n_features is None:
        n_features = len(mean)

    payload = {
        "mean": [float(x) for x in mean],
        "scale": [float(x) for x in scale],
        "n_features": int(n_features),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({payload['n_features']} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
