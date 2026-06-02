from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or prepare augmented models from base + external NPZ caches."
    )
    parser.add_argument("--track", required=True, choices=("fingerspelling", "tsl51"))
    parser.add_argument("--base-cache", required=True, type=Path)
    parser.add_argument("--external-cache", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-artifact-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--combine-only", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_class_names(value: np.ndarray) -> list[str]:
    return [str(v) for v in list(value)]


def align_external_labels(
    external: dict[str, np.ndarray], base_classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    external_classes = normalize_class_names(external["class_names"])
    base_index = {label: idx for idx, label in enumerate(base_classes)}
    missing = [label for label in external_classes if label not in base_index]
    if missing:
        raise ValueError(f"external cache has labels outside base classes: {missing}")
    remap = {idx: base_index[label] for idx, label in enumerate(external_classes)}
    y = np.asarray([remap[int(v)] for v in external["y"]], dtype=np.int32)
    return np.asarray(external["X"], dtype=np.float32), y


def base_xy(base: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "X" in base and "y" in base:
        return np.asarray(base["X"], dtype=np.float32), np.asarray(base["y"], dtype=np.int32)
    if {"X_train", "y_train", "X_test", "y_test"}.issubset(base):
        X = np.concatenate(
            [
                np.asarray(base["X_train"], dtype=np.float32),
                np.asarray(base["X_test"], dtype=np.float32),
            ],
            axis=0,
        )
        y = np.concatenate(
            [
                np.asarray(base["y_train"], dtype=np.int32),
                np.asarray(base["y_test"], dtype=np.int32),
            ],
            axis=0,
        )
        return X, y
    raise KeyError("base cache must contain X/y or X_train/y_train/X_test/y_test")


def merged_cache(base: dict[str, np.ndarray], external: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    base_classes = normalize_class_names(base["class_names"])
    X_base, y_base = base_xy(base)
    X_ext, y_ext = align_external_labels(external, base_classes)
    X = np.concatenate([X_base, X_ext], axis=0)
    y = np.concatenate([y_base, y_ext], axis=0)
    merged = dict(base)
    merged["X"] = X
    merged["y"] = y
    merged["class_names"] = np.asarray(base_classes, dtype=object)
    return merged


def train_fingerspelling(cache: dict[str, np.ndarray], out_dir: Path, epochs: int, batch_size: int) -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(cache["X"], dtype=np.float32)
    y = np.asarray(cache["y"], dtype=np.int32)
    classes = normalize_class_names(cache["class_names"])
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(classes), activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_tr_s, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    (out_dir / "labels.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "track": "thai_fingerspelling",
                "external_augmented": True,
                "num_classes": len(classes),
                "test_accuracy": float(acc),
                "test_loss": float(loss),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def train_tsl51(
    cache: dict[str, np.ndarray],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    base_artifact_dir: Path | None,
) -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(cache["X"], dtype=np.float32)
    y = np.asarray(cache["y"], dtype=np.int32)
    classes = normalize_class_names(cache["class_names"])
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, 162))

    def scale(values: np.ndarray) -> np.ndarray:
        return scaler.transform(values.reshape(-1, 162)).reshape(-1, 60, 162).astype(np.float32)

    X_tr_s = scale(X_tr)
    X_test_s = scale(X_test)
    if base_artifact_dir is not None and (base_artifact_dir / "tsl51_model.keras").exists():
        model = tf.keras.models.load_model(base_artifact_dir / "tsl51_model.keras")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(60, 162)),
                tf.keras.layers.Masking(mask_value=0.0),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.20),
                tf.keras.layers.Dense(len(classes), activation="softmax", dtype="float32"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    model.fit(X_tr_s, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "tsl51_model.keras")
    joblib.dump(scaler, out_dir / "tsl51_scaler.pkl")
    (out_dir / "tsl51_labels.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "tsl51_model_manifest.json").write_text(
        json.dumps(
            {
                "track": "tsl51_word_signs",
                "external_augmented": True,
                "fine_tuned_from": str(base_artifact_dir) if base_artifact_dir else None,
                "num_classes": len(classes),
                "test_accuracy": float(acc),
                "test_loss": float(loss),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    base = load_npz(args.base_cache)
    external = load_npz(args.external_cache)
    merged = merged_cache(base, external)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / f"{args.track}_combined_external_augmented.npz"
    np.savez_compressed(combined_path, **merged)
    summary = {
        "track": args.track,
        "base_cache": str(args.base_cache),
        "external_cache": str(args.external_cache),
        "combined_cache": str(combined_path),
        "combined_shape": list(merged["X"].shape),
        "num_classes": len(merged["class_names"]),
        "trained": False,
    }
    if not args.combine_only and not args.dry_run:
        if args.track == "fingerspelling":
            train_fingerspelling(merged, args.out_dir / "artifacts" / "fingerspelling", args.epochs, args.batch_size)
        else:
            train_tsl51(
                merged,
                args.out_dir / "artifacts" / "tsl51",
                args.epochs,
                args.batch_size,
                args.base_artifact_dir,
            )
        summary["trained"] = True
    summary_path = args.out_dir / "external_augmented_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
