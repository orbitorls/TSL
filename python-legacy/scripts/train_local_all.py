"""Train both TSL tracks locally with cache-friendly defaults.

Recommended runtime for GPU training:
  WSL2 Ubuntu + Python 3.11 + TensorFlow GPU

Examples:
  python scripts/train_local_all.py --preflight
  python scripts/train_local_all.py --tracks fingerspelling
  python scripts/train_local_all.py --tracks tsl51 --max-tsl51-samples 512
  python scripts/train_local_all.py --tracks both
  python scripts/train_local_all.py --fs-only-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.keypoints import FEATURE_SIZE, extract_and_normalize
from src.sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT, csv_to_sequence
from src.train_paths import discover_fs_zip, resolve_fs_dataset_root, resolve_tsl51_metadata_dir


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
TSL51_REPO_ID = "Namonpas/thai-sign-language-tsl51"
TSL51_METADATA_FILES = (
    "metadata/expert_metadata.csv",
    "metadata/user_sign_metadata.csv",
)
LOCAL_CONFIG_PATH = ROOT / "config.local.json"


@dataclass(frozen=True)
class RuntimeConfig:
    artifact_dir: Path
    work_root: Path
    fs_zip: Path | None
    fs_dataset_root: Path | None
    tsl51_metadata_dir: Path | None
    tsl51_file_cache: Path
    force_feature_cache: bool
    fs_only_cache: bool
    tsl51_only_cache: bool
    mixed_precision: bool
    fs_epochs: int
    tsl51_epochs: int
    fs_batch_size: int
    tsl51_batch_size: int
    max_tsl51_samples: int | None
    fs_workers: int
    tsl51_download_workers: int
    tsl51_parse_workers: int
    skip_tflite: bool


def _default_workers() -> int:
    return min(os.cpu_count() or 4, 8)


def _load_local_config() -> dict[str, Any]:
    if not LOCAL_CONFIG_PATH.is_file():
        return {}
    try:
        raw = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[warn] ignoring invalid {LOCAL_CONFIG_PATH}: {exc}")
        return {}
    if not isinstance(raw, dict):
        return {}
    print(f"[config] loaded {LOCAL_CONFIG_PATH}")
    return raw


def _config_defaults_for_argparse() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in _load_local_config().items():
        if value is None:
            continue
        dest = key.replace("-", "_")
        out[dest] = value
    return out


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_to_work_if_needed(path: Path, work_root: Path, label: str) -> Path:
    """Copy datasets away from /mnt/* when running under WSL for faster I/O."""
    resolved = path.resolve()
    text = str(resolved).replace("\\", "/")
    if not text.startswith("/mnt/"):
        return resolved

    target = work_root / "inputs" / label / resolved.name
    if resolved.is_dir():
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[copy] {resolved} -> {target}")
        shutil.copytree(resolved, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == resolved.stat().st_size:
        return target
    print(f"[copy] {resolved} -> {target}")
    shutil.copy2(resolved, target)
    return target


def _is_wsl_mounted_path(path: Path) -> bool:
    return str(path.resolve()).replace("\\", "/").startswith("/mnt/")


def _find_training_root(dataset_root: Path) -> Path:
    for p in dataset_root.rglob("Training set"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f'Could not find "Training set" under {dataset_root}')


def _image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS]


def _read_metadata_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def _require_imports(module_names: list[str]) -> dict[str, bool]:
    import importlib.util

    result: dict[str, bool] = {}
    for name in module_names:
        result[name] = importlib.util.find_spec(name) is not None
    return result


def configure_tensorflow(mixed_precision: bool):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"[warn] could not enable memory growth for {gpu}: {exc}")

    if mixed_precision and gpus:
        try:
            from tensorflow.keras import mixed_precision as mp_policy

            mp_policy.set_global_policy("mixed_float16")
            print("[tf] mixed precision enabled")
        except Exception as exc:
            print(f"[warn] mixed precision disabled: {exc}")

    print("[tf] GPUs:", gpus)
    if not gpus and sys.platform == "win32":
        print(
            "[warn] no TensorFlow GPU on native Windows; use WSL2 + "
            "requirements-local-gpu.txt for RTX training"
        )
    return tf


def preflight() -> int:
    print("[env] Python:", sys.version)
    print("[env] platform:", sys.platform)
    if shutil.which("nvidia-smi"):
        os.system("nvidia-smi")
    else:
        print("[warn] nvidia-smi not found on PATH")

    modules = _require_imports(
        [
            "tensorflow",
            "mediapipe",
            "cv2",
            "sklearn",
            "joblib",
            "huggingface_hub",
            "tqdm",
        ]
    )
    for name, ok in modules.items():
        print(f"[dep] {name}: {'ok' if ok else 'missing'}")

    if modules.get("tensorflow"):
        try:
            tf = configure_tensorflow(mixed_precision=False)
            print("[tf] version:", tf.__version__)
        except Exception as exc:
            print("[tf] import/config failed:", exc)

    missing = [name for name, ok in modules.items() if not ok]
    if missing:
        print("[fail] missing deps:", ", ".join(missing))
        venv_train = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"
        if sys.platform == "win32" and venv_train.is_file():
            print(
                "[hint] Windows: activate training deps with:\n"
                f"  {venv_train} scripts/train_local_all.py --preflight"
            )
        elif sys.platform.startswith("linux"):
            print(
                "[hint] WSL/Linux: source ~/venvs/tsl/bin/activate && "
                "pip install -r requirements-local-gpu.txt"
            )
        return 2
    return 0


def prepare_fingerspelling_dataset(cfg: RuntimeConfig) -> tuple[Path, Path | None]:
    dataset_root = cfg.fs_dataset_root
    if dataset_root is not None and dataset_root.exists():
        if _is_wsl_mounted_path(dataset_root) and cfg.fs_zip is not None and cfg.fs_zip.exists():
            print("[fs] using zip path for WSL-local extraction instead of copying directory")
        else:
            dataset_root = _copy_to_work_if_needed(dataset_root, cfg.work_root, "one_stage_tfs")
            training_root = _find_training_root(dataset_root)
            test_root = training_root.parent / "Test set"
            return training_root, test_root if test_root.exists() else None

    if cfg.fs_zip is not None and cfg.fs_zip.exists():
        zip_path = _copy_to_work_if_needed(cfg.fs_zip, cfg.work_root, "one_stage_tfs_zip")
        extract_root = cfg.work_root / "datasets" / "one_stage_tfs"
        training_root = None if not extract_root.exists() else next(
            (p for p in extract_root.rglob("Training set") if p.is_dir()), None
        )
        if training_root is None:
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"[fs] extracting {zip_path} -> {extract_root}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
            training_root = _find_training_root(extract_root)

        test_root = training_root.parent / "Test set"
        return training_root, test_root if test_root.exists() else None

    if dataset_root is not None and dataset_root.exists():
        dataset_root = _copy_to_work_if_needed(dataset_root, cfg.work_root, "one_stage_tfs")
        training_root = _find_training_root(dataset_root)
        test_root = training_root.parent / "Test set"
        return training_root, test_root if test_root.exists() else None

    raise FileNotFoundError(
        "No fingerspelling dataset found. Provide --fs-dataset-root or --fs-zip."
    )


def _fs_extract_class_folder(payload: tuple[int, str]) -> tuple[list[np.ndarray], list[int], int, str]:
    """ProcessPool worker: one MediaPipe Hands instance per class folder."""
    class_idx, class_dir_str = payload
    import cv2
    import mediapipe as mp

    class_dir = Path(class_dir_str)
    X: list[np.ndarray] = []
    y: list[int] = []
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for image_path in _image_files(class_dir):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = extract_and_normalize(hands.process(rgb))
            if feat is None:
                continue
            X.append(feat)
            y.append(class_idx)
    return X, y, class_idx, class_dir.name


def _extract_split_parallel(root: Path, workers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders under {root}")

    payloads = [(idx, str(class_dir)) for idx, class_dir in enumerate(class_dirs)]
    class_names = [p.name for p in class_dirs]
    X_all: list[np.ndarray] = []
    y_all: list[int] = []

    worker_count = max(1, min(workers, len(payloads)))
    print(f"[fs] extracting {root.name} with {worker_count} workers")
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_fs_extract_class_folder, p) for p in payloads]
        for fut in as_completed(futures):
            X_part, y_part, _, _ = fut.result()
            X_all.extend(X_part)
            y_all.extend(y_part)

    if not X_all:
        raise RuntimeError(f"No valid MediaPipe hands features under {root}")
    return (
        np.asarray(X_all, dtype=np.float32),
        np.asarray(y_all, dtype=np.int32),
        class_names,
    )


def extract_fingerspelling_features(
    training_root: Path,
    test_root: Path | None,
    cache_path: Path,
    force: bool,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        if int(data["feature_size"]) == FEATURE_SIZE:
            print("[fs] loaded feature cache:", cache_path)
            return (
                data["X_train"],
                data["y_train"],
                data["X_test"] if len(data["X_test"]) else None,
                data["y_test"] if len(data["y_test"]) else None,
                list(data["class_names"]),
            )

    X_train, y_train, class_names = _extract_split_parallel(training_root, workers)
    if test_root is not None:
        X_test, y_test, _ = _extract_split_parallel(test_root, workers)
    else:
        X_test, y_test = None, None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        feature_size=FEATURE_SIZE,
        X_train=X_train,
        y_train=y_train,
        X_test=np.asarray(X_test if X_test is not None else [], dtype=np.float32),
        y_test=np.asarray(y_test if y_test is not None else [], dtype=np.int32),
        class_names=np.asarray(class_names, dtype=object),
    )
    print("[fs] saved feature cache:", cache_path)
    return X_train, y_train, X_test, y_test, class_names


def train_fingerspelling(cfg: RuntimeConfig, tf) -> None:
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    training_root, test_root = prepare_fingerspelling_dataset(cfg)
    cache_path = cfg.work_root / "features" / "keypoints_cache.npz"
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, class_names = (
        extract_fingerspelling_features(
            training_root,
            test_root,
            cache_path,
            cfg.force_feature_cache,
            cfg.fs_workers,
        )
    )
    if cfg.fs_only_cache:
        print("[fs] --fs-only-cache: skipping fit")
        return

    if X_test_raw is not None and y_test_raw is not None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=0.12,
            stratify=y_train_raw,
            random_state=42,
        )
        X_test, y_test = X_test_raw, y_test_raw
    else:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=0.10,
            stratify=y_train_raw,
            random_state=42,
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp,
            y_tmp,
            test_size=0.111,
            stratify=y_tmp,
            random_state=42,
        )

    def mirror_x(X: np.ndarray) -> np.ndarray:
        Xm = X.reshape(-1, 21, 3).copy()
        Xm[:, :, 0] *= -1.0
        return Xm.reshape(-1, FEATURE_SIZE)

    X_tr_aug = np.concatenate([X_tr, mirror_x(X_tr)]).astype(np.float32)
    y_tr_aug = np.concatenate([y_tr, y_tr]).astype(np.int32)

    scaler = StandardScaler()
    scaler.fit(X_tr_aug)
    X_tr_s = scaler.transform(X_tr_aug).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr_s, y_tr_aug))
        .shuffle(min(len(X_tr_s), 8192), seed=42, reshuffle_each_iteration=True)
        .batch(cfg.fs_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_s, y_val))
        .batch(cfg.fs_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_SIZE,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(class_names), activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.fs_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    print("[fs] test accuracy:", acc)
    try:
        print(classification_report(y_test, y_pred, target_names=class_names))
    except UnicodeEncodeError:
        print("[fingerspelling] classification report omitted (console encoding)")

    out = cfg.artifact_dir / "fingerspelling"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    joblib.dump(scaler, out / "scaler.pkl")
    _json_dump(out / "labels.json", {str(i): name for i, name in enumerate(class_names)})
    _json_dump(
        out / "model_manifest.json",
        {
            "track": "thai_fingerspelling",
            "feature_size": FEATURE_SIZE,
            "num_classes": len(class_names),
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "artifacts": ["model.keras", "model.tflite", "labels.json", "scaler.pkl"],
        },
    )
    if not cfg.skip_tflite:
        _try_export_tflite(tf, model, out / "model.tflite", "fingerspelling")
    else:
        print("[fingerspelling] TFLite export skipped (--skip-tflite)")
    print("[fs] artifacts:", out)


def _prepare_tsl51_metadata(cfg: RuntimeConfig) -> list[Path]:
    if cfg.tsl51_metadata_dir is not None and cfg.tsl51_metadata_dir.exists():
        meta_dir = cfg.tsl51_metadata_dir
        candidates = [
            meta_dir / "expert_metadata.csv",
            meta_dir / "user_sign_metadata.csv",
        ]
        nested = [
            meta_dir / "metadata" / "expert_metadata.csv",
            meta_dir / "metadata" / "user_sign_metadata.csv",
        ]
        if all(p.exists() for p in candidates):
            return candidates
        if all(p.exists() for p in nested):
            return nested

    from huggingface_hub import hf_hub_download

    out = cfg.work_root / "tsl51_metadata"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rel in TSL51_METADATA_FILES:
        local = Path(hf_hub_download(TSL51_REPO_ID, rel, repo_type="dataset"))
        target = out / Path(rel).name
        if not target.exists() or target.stat().st_size != local.stat().st_size:
            shutil.copy2(local, target)
        paths.append(target)
    return paths


def _ensure_tsl51_landmark(rel_path: str, cache_dir: Path) -> Path | None:
    rel = rel_path.replace("\\", "/").strip()
    cached = cache_dir / rel
    if cached.exists():
        return cached

    # Some snapshots package landmarks in zip files without subfolders.
    # Try a basename fallback in common local folders before remote download.
    fname = Path(rel).name
    fallback_dirs = (
        cache_dir / "landmarks",
        cache_dir / "landmarks" / "expert_scraped",
        cache_dir / "landmarks" / "expert_primary_01",
        cache_dir / "landmarks" / "expert_primary_02",
        cache_dir / "landmarks" / "user_sign",
    )
    for d in fallback_dirs:
        p = d / fname
        if p.exists():
            return p

    if os.environ.get("TSL51_DISABLE_HF_DOWNLOAD", "").strip() == "1":
        return None

    from huggingface_hub import hf_hub_download

    try:
        local = Path(hf_hub_download(TSL51_REPO_ID, rel, repo_type="dataset"))
    except Exception as exc:
        print(f"[tsl51] missing landmark skipped: {rel} ({exc})")
        return None
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, cached)
    return cached


def _tsl51_parse_cached_csv(payload: tuple[str, int]) -> tuple[np.ndarray, int] | None:
    csv_path, class_idx = payload
    try:
        return csv_to_sequence(Path(csv_path), seq_len=SEQ_LEN_DEFAULT), class_idx
    except Exception:
        return None


def extract_tsl51_features(
    cfg: RuntimeConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_path = cfg.work_root / "features" / "tsl51_features.npz"
    metadata_paths = _prepare_tsl51_metadata(cfg)
    rows = _read_metadata_rows(metadata_paths)
    rows = [
        r
        for r in rows
        if str(r.get("sign_id", "")).strip()
        and str(r.get("landmark_path", "")).strip()
        and str(r.get("sign_id", "")).strip() != "null_act"
    ]
    if not rows:
        raise FileNotFoundError(
            "No TSL-51 metadata rows after filtering. "
            f"Check --tsl51-metadata-dir (loaded from {metadata_paths})."
        )

    signature = f"{len(rows)}|{rows[0]['landmark_path']}|{rows[-1]['landmark_path']}"
    if cache_path.exists() and not cfg.force_feature_cache:
        data = np.load(cache_path, allow_pickle=True)
        if (
            str(data["signature"]) == signature
            and int(data["feature_dim"]) == FEATURE_DIM
            and int(data["seq_len"]) == SEQ_LEN_DEFAULT
        ):
            print("[tsl51] loaded feature cache:", cache_path)
            return data["X"], data["y"], list(data["class_names"])

    cfg.tsl51_file_cache.mkdir(parents=True, exist_ok=True)
    download_workers = max(1, min(cfg.tsl51_download_workers, len(rows)))
    print(f"[tsl51] downloading landmarks with {download_workers} threads")
    local_csvs: list[Path | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {
            pool.submit(
                _ensure_tsl51_landmark,
                str(row["landmark_path"]),
                cfg.tsl51_file_cache,
            ): idx
            for idx, row in enumerate(rows)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                local_csvs[idx] = fut.result()
            except Exception as exc:
                local_csvs[idx] = None
                print(f"[tsl51] download failed, row skipped: {exc}")

    available_rows = [rows[i] for i in range(len(rows)) if local_csvs[i] is not None]
    if cfg.max_tsl51_samples is not None:
        available_rows = available_rows[: cfg.max_tsl51_samples]

    if not available_rows:
        raise RuntimeError("No TSL-51 landmark CSV files available after download filtering")

    class_names = sorted({str(r["sign_id"]).strip() for r in available_rows})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    parse_payloads = []
    for row in available_rows:
        local = _ensure_tsl51_landmark(str(row["landmark_path"]), cfg.tsl51_file_cache)
        if local is None:
            continue
        parse_payloads.append((str(local), class_to_idx[str(row["sign_id"]).strip()]))
    if not parse_payloads:
        raise RuntimeError("No TSL-51 landmark CSV files available after download filtering")
    parse_workers = max(1, min(cfg.tsl51_parse_workers, len(parse_payloads)))
    print(f"[tsl51] parsing sequences with {parse_workers} workers")

    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=parse_workers) as pool:
        futures = [pool.submit(_tsl51_parse_cached_csv, p) for p in parse_payloads]
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                skipped += 1
                continue
            seq, class_idx = result
            X.append(seq)
            y.append(class_idx)

    if not X:
        raise RuntimeError("No valid TSL-51 sequences loaded")
    if skipped:
        print("[tsl51] skipped rows:", skipped)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=signature,
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN_DEFAULT,
        X=X_arr,
        y=y_arr,
        class_names=np.asarray(class_names, dtype=object),
    )
    print("[tsl51] saved feature cache:", cache_path)
    return X_arr, y_arr, class_names


def train_tsl51(cfg: RuntimeConfig, tf) -> None:
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_all, y_all, class_names = extract_tsl51_features(cfg)
    # Stratified splits require at least 2 examples per class.
    cls_ids, cls_counts = np.unique(y_all, return_counts=True)
    keep_ids = {int(i) for i, c in zip(cls_ids, cls_counts) if int(c) >= 2}
    if len(keep_ids) < len(cls_ids):
        mask = np.asarray([int(v) in keep_ids for v in y_all], dtype=bool)
        dropped = int((~mask).sum())
        X_all = X_all[mask]
        y_all = y_all[mask]
        remap = {old: new for new, old in enumerate(sorted(keep_ids))}
        y_all = np.asarray([remap[int(v)] for v in y_all], dtype=np.int32)
        class_names = [class_names[i] for i in sorted(keep_ids)]
        print(f"[tsl51] dropped {dropped} samples from rare classes (<2)")
    if cfg.tsl51_only_cache:
        print("[tsl51] --tsl51-only-cache: skipping fit")
        return

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.10, stratify=y_all, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.111, stratify=y_tmp, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, FEATURE_DIM))

    def scale(X: np.ndarray) -> np.ndarray:
        return scaler.transform(X.reshape(-1, FEATURE_DIM)).reshape(
            -1, SEQ_LEN_DEFAULT, FEATURE_DIM
        ).astype(np.float32)

    X_tr_s = scale(X_tr)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr_s, y_tr))
        .shuffle(min(len(X_tr_s), 8192), seed=42, reshuffle_each_iteration=True)
        .batch(cfg.tsl51_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_s, y_val))
        .batch(cfg.tsl51_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQ_LEN_DEFAULT, FEATURE_DIM)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(96, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(len(class_names), activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.tsl51_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    print("[tsl51] test accuracy:", acc)
    try:
        print(classification_report(y_test, y_pred, target_names=class_names))
    except UnicodeEncodeError:
        print("[tsl51] classification report omitted (console encoding)")

    out = cfg.artifact_dir / "tsl51"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "tsl51_model.keras")
    joblib.dump(scaler, out / "tsl51_scaler.pkl")
    _json_dump(out / "tsl51_labels.json", {str(i): name for i, name in enumerate(class_names)})
    _json_dump(
        out / "tsl51_model_manifest.json",
        {
            "track": "tsl51_word_signs",
            "dataset": TSL51_REPO_ID,
            "feature_dim": FEATURE_DIM,
            "seq_len": SEQ_LEN_DEFAULT,
            "num_classes": len(class_names),
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "artifacts": [
                "tsl51_model.keras",
                "tsl51_model.tflite",
                "tsl51_labels.json",
                "tsl51_scaler.pkl",
            ],
        },
    )
    if not cfg.skip_tflite:
        _try_export_tflite(tf, model, out / "tsl51_model.tflite", "tsl51")
    else:
        print("[tsl51] TFLite export skipped (--skip-tflite)")
    print("[tsl51] artifacts:", out)


def _try_export_tflite(tf, model, output_path: Path, label: str) -> None:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_path.write_bytes(converter.convert())
        print(f"[{label}] saved TFLite:", output_path)
    except Exception as exc:
        print(f"[{label}] TFLite export skipped:", exc)


def default_work_root() -> Path:
    if sys.platform.startswith("linux"):
        return Path.home() / "tsl_training"
    return ROOT / "_runtime" / "local_training"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train both TSL tracks locally.")
    p.set_defaults(**_config_defaults_for_argparse())
    p.add_argument("--tracks", choices=("both", "fingerspelling", "tsl51"), default="both")
    p.add_argument("--preflight", action="store_true", help="Check deps/GPU and exit")
    p.add_argument("--work-root", default=str(default_work_root()))
    p.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--fs-zip", default=None, help="Optional One-Stage-TFS zip path")
    p.add_argument(
        "--fs-dataset-root",
        default=str(REPO_ROOT / "data" / "fingerspelling"),
    )
    p.add_argument(
        "--tsl51-metadata-dir",
        default=str(REPO_ROOT / "data" / "tsl51" / "metadata"),
    )
    p.add_argument("--tsl51-file-cache", default=str(default_work_root() / "tsl51_files"))
    p.add_argument("--force-feature-cache", action="store_true")
    p.add_argument(
        "--fs-only-cache",
        action="store_true",
        help="Build fingerspelling NPZ cache only (no fit)",
    )
    p.add_argument(
        "--tsl51-only-cache",
        action="store_true",
        help="Build TSL-51 NPZ cache only (no fit)",
    )
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--fs-epochs", type=int, default=120)
    p.add_argument("--tsl51-epochs", type=int, default=120)
    p.add_argument("--fs-batch-size", type=int, default=128)
    p.add_argument("--tsl51-batch-size", type=int, default=64)
    p.add_argument("--max-tsl51-samples", type=int, default=None)
    p.add_argument(
        "--fs-workers",
        type=int,
        default=_default_workers(),
        help="Parallel MediaPipe workers for fingerspelling extraction",
    )
    p.add_argument(
        "--tsl51-download-workers",
        type=int,
        default=_default_workers(),
        help="Parallel HF download threads for TSL-51 landmarks",
    )
    p.add_argument(
        "--tsl51-parse-workers",
        type=int,
        default=_default_workers(),
        help="Parallel CSV parse workers for TSL-51 sequences",
    )
    p.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip TFLite export when local converter is unstable",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        return preflight()

    work_root = Path(args.work_root).expanduser().resolve()
    fs_zip = discover_fs_zip(
        REPO_ROOT,
        Path(args.fs_zip).expanduser().resolve() if args.fs_zip else None,
    )
    fs_dataset_root = (
        resolve_fs_dataset_root(
            REPO_ROOT,
            Path(args.fs_dataset_root).expanduser().resolve(),
        )
        if args.fs_dataset_root
        else None
    )
    tsl51_metadata_dir = (
        resolve_tsl51_metadata_dir(
            REPO_ROOT,
            Path(args.tsl51_metadata_dir).expanduser().resolve(),
        )
        if args.tsl51_metadata_dir
        else None
    )
    cfg = RuntimeConfig(
        artifact_dir=Path(args.artifact_dir).expanduser().resolve(),
        work_root=work_root,
        fs_zip=fs_zip,
        fs_dataset_root=fs_dataset_root,
        tsl51_metadata_dir=tsl51_metadata_dir,
        tsl51_file_cache=Path(args.tsl51_file_cache).expanduser().resolve(),
        force_feature_cache=args.force_feature_cache,
        fs_only_cache=args.fs_only_cache,
        tsl51_only_cache=args.tsl51_only_cache,
        mixed_precision=not args.no_mixed_precision,
        fs_epochs=args.fs_epochs,
        tsl51_epochs=args.tsl51_epochs,
        fs_batch_size=args.fs_batch_size,
        tsl51_batch_size=args.tsl51_batch_size,
        max_tsl51_samples=args.max_tsl51_samples,
        fs_workers=max(1, args.fs_workers),
        tsl51_download_workers=max(1, args.tsl51_download_workers),
        tsl51_parse_workers=max(1, args.tsl51_parse_workers),
        skip_tflite=args.skip_tflite,
    )
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    run_fs = args.tracks in ("both", "fingerspelling")
    run_tsl51 = args.tracks in ("both", "tsl51")
    need_tf = (run_fs and not cfg.fs_only_cache) or (run_tsl51 and not cfg.tsl51_only_cache)
    tf = configure_tensorflow(mixed_precision=cfg.mixed_precision) if need_tf else None

    if run_fs:
        train_fingerspelling(cfg, tf)
    if run_tsl51:
        train_tsl51(cfg, tf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
