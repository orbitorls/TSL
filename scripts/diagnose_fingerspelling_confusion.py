from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_LEGACY = REPO_ROOT / "python-legacy"
sys.path.insert(0, str(PY_LEGACY))

DEFAULT_ARTIFACT = (
    REPO_ROOT
    / ".tools"
    / "train_runs_25690531-131843"
    / "fs_e80_b64"
    / "artifacts"
    / "fingerspelling"
)
DEFAULT_CACHE = (
    REPO_ROOT
    / ".tools"
    / "train_runs_25690531-131843"
    / "fs_e80_b64"
    / "work"
    / "features"
    / "keypoints_cache.npz"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline fingerspelling confusion report (top pairs, margin buckets)."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE,
        help="NPZ with X_test, y_test, class_names from train_local_all",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "fingerspelling_confusion")
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in raw.items()}


def top_confusion_pairs(cm: np.ndarray, names: list[str], limit: int = 10) -> list[dict[str, object]]:
    pairs: list[tuple[int, str, str]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), names[i], names[j]))
    pairs.sort(reverse=True)
    return [
        {"count": count, "true": true_name, "pred": pred_name}
        for count, true_name, pred_name in pairs[:limit]
    ]


def margin_bucket_report(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    pred = probs.argmax(axis=1)
    margins = []
    for row, idx in zip(probs, pred):
        top2 = np.partition(row, -2)[-2:]
        margins.append(float(np.max(top2) - np.min(top2)))
    margins_arr = np.asarray(margins)
    low = margins_arr < threshold
    high = ~low
    out: dict[str, object] = {
        "margin_threshold": threshold,
        "low_margin_count": int(low.sum()),
        "high_margin_count": int(high.sum()),
    }
    if low.any():
        out["accuracy_margin_low"] = float(accuracy_score(y_true[low], pred[low]))
    else:
        out["accuracy_margin_low"] = None
    if high.any():
        out["accuracy_margin_high"] = float(accuracy_score(y_true[high], pred[high]))
    else:
        out["accuracy_margin_high"] = None
    return out


def pair_count(cm: np.ndarray, names: list[str], a: str, b: str) -> dict[str, int]:
    if a not in names or b not in names:
        return {"a_to_b": 0, "b_to_a": 0, "total": 0}
    ia, ib = names.index(a), names.index(b)
    a_to_b = int(cm[ia, ib])
    b_to_a = int(cm[ib, ia])
    return {"a_to_b": a_to_b, "b_to_a": b_to_a, "total": a_to_b + b_to_a}


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir
    cache_path = args.cache_path
    if not (artifact / "model.keras").exists():
        raise SystemExit(f"Missing model: {artifact / 'model.keras'}")
    if not cache_path.exists():
        raise SystemExit(f"Missing feature cache: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = [str(x) for x in data["class_names"]]

    labels = load_labels(artifact / "labels.json")
    scaler = joblib.load(artifact / "scaler.pkl")
    model = tf.keras.models.load_model(artifact / "model.keras")

    X_s = scaler.transform(X_test).astype(np.float32)
    probs = model.predict(X_s, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    names = class_names
    top_pairs = top_confusion_pairs(cm, names)
    margin_report = margin_bucket_report(probs, y_test, threshold=args.margin_threshold)
    ko_bor = pair_count(cm, names, "KO_KAI", "BOR_BAI_MAI")

    summary = {
        "artifact_dir": str(artifact.resolve()),
        "cache_path": str(cache_path.resolve()),
        "test_samples": int(len(y_test)),
        "test_accuracy": acc,
        "class_names": names,
        "top_confusion_pairs": top_pairs,
        "ko_kai_vs_bor_bai_mai": ko_bor,
        "margin_buckets": margin_report,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[fs-diagnose] test accuracy: {acc:.4f} ({len(y_test)} samples)")
    print("[fs-diagnose] top confusion pairs (true -> pred):")
    for entry in top_pairs:
        print(f"  {entry['count']:4d}  {entry['true']} -> {entry['pred']}")
    print(
        f"[fs-diagnose] KO_KAI <-> BOR_BAI_MAI: "
        f"{ko_bor['a_to_b']} (ก->บ) + {ko_bor['b_to_a']} (บ->ก) = {ko_bor['total']}"
    )
    print(
        f"[fs-diagnose] margin < {args.margin_threshold}: "
        f"acc={margin_report.get('accuracy_margin_low')} "
        f"(n={margin_report['low_margin_count']})"
    )
    print(
        f"[fs-diagnose] margin >= {args.margin_threshold}: "
        f"acc={margin_report.get('accuracy_margin_high')} "
        f"(n={margin_report['high_margin_count']})"
    )
    print(f"[fs-diagnose] wrote {out_json}")


if __name__ == "__main__":
    main()
