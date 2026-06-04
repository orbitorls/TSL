from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np

TSL51_SEQ_LEN = 60
TSL51_FEATURE_DIM = 162


def augment_external_sequences(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    n_copies: int = 5,
    noise_sigma: float = 0.015,
    scale_range: tuple[float, float] = (0.88, 1.12),
    speed_range: tuple[float, float] = (0.82, 1.18),
) -> tuple[np.ndarray, np.ndarray]:
    """Return original + n_copies augmented copies of webcam sequences.

    Augmentations applied independently per copy:
      - Gaussian coordinate noise  (simulates signing style variation)
      - Uniform scale jitter        (simulates different arm length / camera distance)
      - Temporal speed jitter       (simulates different signing speed)

    Output shape: (N * (n_copies + 1), SEQ_LEN, FEATURE_DIM).
    Labels are repeated to match.
    """
    if len(X) == 0:
        return X, y
    batches_X = [X]
    batches_y = [y]
    for _ in range(n_copies):
        X_c = X.copy()
        # Gaussian noise
        X_c = X_c + rng.standard_normal(X_c.shape).astype(np.float32) * noise_sigma
        # Uniform scale jitter per sequence
        scales = rng.uniform(scale_range[0], scale_range[1], size=(len(X_c), 1, 1)).astype(np.float32)
        X_c = X_c * scales
        # Temporal speed jitter: stretch/compress each sequence then resample to SEQ_LEN
        speed_factors = rng.uniform(speed_range[0], speed_range[1], size=len(X_c))
        X_jittered = np.empty_like(X_c)
        for i, (seq, factor) in enumerate(zip(X_c, speed_factors)):
            n_mid = max(1, int(TSL51_SEQ_LEN * factor))
            # Resample to n_mid frames
            idx_mid = np.linspace(0, TSL51_SEQ_LEN - 1, n_mid).round().astype(int)
            seq_mid = seq[idx_mid]
            # Resample back to TSL51_SEQ_LEN
            idx_final = np.linspace(0, len(seq_mid) - 1, TSL51_SEQ_LEN).round().astype(int)
            X_jittered[i] = seq_mid[idx_final]
        batches_X.append(X_jittered)
        batches_y.append(y)
    return np.concatenate(batches_X, axis=0), np.concatenate(batches_y, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or prepare augmented models from base + external NPZ caches."
    )
    parser.add_argument("--track", required=True, choices=("fingerspelling", "tsl51"))
    parser.add_argument("--base-cache", required=True, type=Path)
    parser.add_argument("--external-cache", required=True, action="append", type=Path)
    parser.add_argument("--external-cache-split", action="append", choices=("train", "val"), default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-artifact-dir", type=Path, default=None)
    parser.add_argument("--base-labels", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--external-sample-weight", type=float, default=1.0)
    parser.add_argument("--scaler-mode", choices=("reuse", "refit"), default="reuse")
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--combine-only", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--augment-external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-copies", type=int, default=5)
    parser.add_argument(
        "--require-external-val",
        action="store_true",
        default=False,
        help="Fail TSL51 training unless at least one external validation sample is available.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def normalize_class_names(value: np.ndarray | Sequence[object]) -> list[str]:
    return [str(v) for v in list(value)]


def load_label_list(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[str(i)]) for i in range(len(raw))]
    raise ValueError(f"{path} must contain a label list or dict")


def validate_tsl51_classes(classes: Sequence[str], *, expected_count: int = 51) -> None:
    if len(classes) != expected_count:
        raise ValueError(f"TSL51 label map must contain {expected_count} classes, got {len(classes)}")
    duplicates = sorted({label for label in classes if classes.count(label) > 1})
    if duplicates:
        raise ValueError("TSL51 label map contains duplicate labels: " + ", ".join(duplicates[:5]))


def artifact_labels(base_artifact_dir: Path | None, base_labels: Path | None) -> list[str] | None:
    if base_labels is not None:
        return load_label_list(base_labels)
    if base_artifact_dir is not None:
        labels_path = base_artifact_dir / "tsl51_labels.json"
        if labels_path.exists():
            return load_label_list(labels_path)
    return None


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


def split_values(cache: dict[str, np.ndarray], fallback_split: str | None = None) -> np.ndarray:
    if "split" in cache:
        return np.asarray([str(value) for value in cache["split"]], dtype=object)
    if fallback_split is None:
        raise ValueError("external cache missing split metadata; pass --external-cache-split")
    return np.asarray([fallback_split] * len(cache["y"]), dtype=object)


def split_external_cache(
    external: dict[str, np.ndarray],
    base_classes: list[str],
    *,
    fallback_split: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    X, y = align_external_labels(external, base_classes)
    splits = split_values(external, fallback_split=fallback_split)
    if len(splits) != len(y):
        raise ValueError("external split metadata length does not match y")
    if np.any(splits == "external_test"):
        raise ValueError("external_test rows cannot be used for training/fine-tuning")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val"):
        mask = splits == split
        result[split] = (X[mask], y[mask])
    unknown = sorted({str(value) for value in splits if str(value) not in {"train", "val"}})
    if unknown:
        raise ValueError("unsupported external cache split(s): " + ", ".join(unknown))
    return result


def collect_external_splits(
    caches: Iterable[dict[str, np.ndarray]],
    base_classes: list[str],
    *,
    fallback_splits: Sequence[str | None] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_X: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    fallback_list = list(fallback_splits or [])
    for idx, cache in enumerate(caches):
        fallback = fallback_list[idx] if idx < len(fallback_list) else None
        split = split_external_cache(cache, base_classes, fallback_split=fallback)
        if len(split["train"][1]):
            train_X.append(split["train"][0])
            train_y.append(split["train"][1])
        if len(split["val"][1]):
            val_X.append(split["val"][0])
            val_y.append(split["val"][1])

    def concat(parts: list[np.ndarray], shape_tail: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0, *shape_tail), dtype=dtype)

    return {
        "train": (
            concat(train_X, (TSL51_SEQ_LEN, TSL51_FEATURE_DIM), np.dtype("float32")),
            concat(train_y, (), np.dtype("int32")),
        ),
        "val": (
            concat(val_X, (TSL51_SEQ_LEN, TSL51_FEATURE_DIM), np.dtype("float32")),
            concat(val_y, (), np.dtype("int32")),
        ),
    }


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


def validate_tsl51_training_args(scaler_mode: str, base_artifact_dir: Path | None) -> None:
    if scaler_mode == "reuse" and base_artifact_dir is None:
        raise ValueError("--scaler-mode reuse requires --base-artifact-dir")


def validate_external_validation_requirement(require_external_val: bool, external_val_count: int) -> None:
    if require_external_val and external_val_count == 0:
        raise ValueError("--require-external-val requires at least one external val sample")


def scale_tsl51(values: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(values.reshape(-1, TSL51_FEATURE_DIM)).reshape(
        -1, TSL51_SEQ_LEN, TSL51_FEATURE_DIM
    ).astype(np.float32)


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
    base: dict[str, np.ndarray],
    external_caches: list[dict[str, np.ndarray]],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    base_artifact_dir: Path | None,
    *,
    base_labels: Path | None = None,
    scaler_mode: str = "reuse",
    learning_rate: float = 1e-5,
    external_sample_weight: float = 1.0,
    freeze_base: bool = True,
    unfreeze_last_n: int = 0,
    external_cache_splits: Sequence[str | None] | None = None,
    require_external_val: bool = False,
    augment_external: bool = False,
    augment_copies: int = 5,
) -> dict[str, object]:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    validate_tsl51_training_args(scaler_mode, base_artifact_dir)
    base_classes = artifact_labels(base_artifact_dir, base_labels) or normalize_class_names(base["class_names"])
    validate_tsl51_classes(base_classes)
    cache_classes = normalize_class_names(base["class_names"])
    if cache_classes != base_classes:
        raise ValueError("base cache class_names do not match base artifact labels")

    X_base, y_base = base_xy(base)
    external = collect_external_splits(
        external_caches,
        base_classes,
        fallback_splits=external_cache_splits,
    )
    X_ext_train, y_ext_train = external["train"]
    X_ext_val, y_ext_val = external["val"]
    if len(y_ext_train) == 0:
        raise ValueError("at least one external train sample is required")
    if augment_external and len(X_ext_train) > 0:
        rng = np.random.default_rng(42)
        X_ext_train, y_ext_train = augment_external_sequences(
            X_ext_train, y_ext_train, rng, n_copies=augment_copies
        )
    validate_external_validation_requirement(require_external_val, len(y_ext_val))

    if len(np.unique(y_base)) > 1 and min(np.bincount(y_base)) >= 2:
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
            X_base, y_base, test_size=0.15, stratify=y_base, random_state=42
        )
    else:
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
            X_base, y_base, test_size=0.15, random_state=42
        )

    X_train = np.concatenate([X_base_train, X_ext_train], axis=0)
    y_train = np.concatenate([y_base_train, y_ext_train], axis=0)
    sample_weight = np.concatenate(
        [
            np.ones(len(y_base_train), dtype=np.float32),
            np.full(len(y_ext_train), float(external_sample_weight), dtype=np.float32),
        ],
        axis=0,
    )
    if len(y_ext_val):
        X_val = X_ext_val
        y_val = y_ext_val
        validation_data = None
    else:
        X_val = X_base_test
        y_val = y_base_test
        validation_data = None

    if scaler_mode == "reuse":
        assert base_artifact_dir is not None
        scaler = joblib.load(base_artifact_dir / "tsl51_scaler.pkl")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, TSL51_FEATURE_DIM))
    X_train_s = scale_tsl51(X_train, scaler)
    X_val_s = scale_tsl51(X_val, scaler)
    X_base_test_s = scale_tsl51(X_base_test, scaler)

    if base_artifact_dir is not None and (base_artifact_dir / "tsl51_model.keras").exists():
        model = tf.keras.models.load_model(base_artifact_dir / "tsl51_model.keras")
        output_shape = model.output_shape[-1]
        if int(output_shape) != len(base_classes):
            raise ValueError(f"base model output size {output_shape} does not match {len(base_classes)} labels")
        if freeze_base:
            for layer in model.layers:
                layer.trainable = False
            for layer in model.layers[-max(unfreeze_last_n, 1):]:
                layer.trainable = True
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(TSL51_SEQ_LEN, TSL51_FEATURE_DIM)),
                tf.keras.layers.Masking(mask_value=0.0),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.20),
                tf.keras.layers.Dense(len(base_classes), activation="softmax", dtype="float32"),
            ]
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-7),
    ]
    history = model.fit(
        X_train_s,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_s, y_val),
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=1,
    )
    base_loss, base_acc = model.evaluate(X_base_test_s, y_base_test, verbose=0)
    if len(y_ext_val):
        external_val_loss, external_val_acc = model.evaluate(X_val_s, y_ext_val, verbose=0)
    else:
        external_val_loss = None
        external_val_acc = None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "tsl51_model.keras")
    # joblib is required here because existing runtime artifacts store sklearn StandardScaler as .pkl.
    joblib.dump(scaler, out_dir / "tsl51_scaler.pkl")
    (out_dir / "tsl51_labels.json").write_text(
        json.dumps({str(i): label for i, label in enumerate(base_classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "track": "tsl51_word_signs",
        "external_augmented": True,
        "fine_tuned_from": str(base_artifact_dir) if base_artifact_dir else None,
        "num_classes": len(base_classes),
        "test_accuracy": float(base_acc),
        "test_loss": float(base_loss),
        "base_internal_accuracy": float(base_acc),
        "base_internal_loss": float(base_loss),
        "external_val_accuracy": float(external_val_acc) if external_val_acc is not None else None,
        "external_val_loss": float(external_val_loss) if external_val_loss is not None else None,
        "scaler_mode": scaler_mode,
        "learning_rate": learning_rate,
        "external_sample_weight": float(external_sample_weight),
        "freeze_base": freeze_base,
        "unfreeze_last_n": unfreeze_last_n,
        "base_train_samples": int(len(y_base_train)),
        "base_test_samples": int(len(y_base_test)),
        "external_train_samples": int(len(y_ext_train)),
        "external_val_samples": int(len(y_ext_val)),
        "epochs_ran": int(len(history.history.get("loss", []))),
    }
    (out_dir / "tsl51_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def combine_external_for_summary(
    base: dict[str, np.ndarray], external_caches: list[dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    merged = dict(base)
    current = merged
    for external in external_caches:
        current = merged_cache(current, external)
    return current


def main() -> None:
    args = parse_args()
    base = load_npz(args.base_cache)
    external_caches = [load_npz(path) for path in args.external_cache]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined = combine_external_for_summary(base, external_caches)
    combined_path = args.out_dir / f"{args.track}_combined_external_augmented.npz"
    np.savez_compressed(combined_path, **combined)
    summary = {
        "track": args.track,
        "base_cache": str(args.base_cache),
        "external_cache": [str(path) for path in args.external_cache],
        "combined_cache": str(combined_path),
        "combined_shape": list(combined["X"].shape),
        "num_classes": len(combined["class_names"]),
        "trained": False,
    }
    if not args.combine_only and not args.dry_run:
        if args.track == "fingerspelling":
            if len(external_caches) != 1:
                raise ValueError("fingerspelling currently supports one external cache")
            train_fingerspelling(combined, args.out_dir / "artifacts" / "fingerspelling", args.epochs, args.batch_size)
        else:
            manifest = train_tsl51(
                base,
                external_caches,
                args.out_dir / "artifacts" / "tsl51",
                args.epochs,
                args.batch_size,
                args.base_artifact_dir,
                base_labels=args.base_labels,
                scaler_mode=args.scaler_mode,
                learning_rate=args.learning_rate,
                external_sample_weight=args.external_sample_weight,
                freeze_base=args.freeze_base,
                unfreeze_last_n=args.unfreeze_last_n,
                external_cache_splits=args.external_cache_split,
                require_external_val=args.require_external_val,
                augment_external=args.augment_external,
                augment_copies=args.augment_copies,
            )
            summary.update({"model_manifest": manifest})
        summary["trained"] = True
    summary_path = args.out_dir / "external_augmented_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
