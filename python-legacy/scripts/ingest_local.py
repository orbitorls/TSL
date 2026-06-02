"""Local dataset ingestion helper for both training tracks.

Usage examples:
  python scripts/ingest_local.py fingerspelling --zip "D:/datasets/One-Stage-TFS.zip"
  python scripts/ingest_local.py fingerspelling --dataset-root "D:/datasets/One-Stage-TFS"
  python scripts/ingest_local.py tsl51
  python scripts/ingest_local.py tsl51 --source "./data/tsl51_raw"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import shutil
import stat
import zipfile
from pathlib import Path

# Allow this script to be executed from repo root (python scripts/ingest_local.py ...).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from src.keypoints import FEATURE_SIZE
from src.sequence_keypoints import FEATURE_DIM, SEQ_LEN_DEFAULT


def _collect_source_metadata_files(source_root: Path) -> list[Path]:
    direct_candidates = [
        source_root / "expert_metadata.csv",
        source_root / "user_sign_metadata.csv",
    ]
    if all(path.exists() for path in direct_candidates):
        return direct_candidates

    nested = source_root / "metadata"
    nested_candidates = [
        nested / "expert_metadata.csv",
        nested / "user_sign_metadata.csv",
    ]
    if all(path.exists() for path in nested_candidates):
        return nested_candidates

    raise FileNotFoundError(
        "Source must contain expert_metadata.csv and user_sign_metadata.csv either at root "
        f"or under {nested}"
    )


def _ensure_dir(path: Path, force: bool) -> None:
    def _on_rm_error(func, target, exc_info):  # noqa: ANN001
        try:
            Path(target).chmod(stat.S_IWRITE)
            func(target)
        except Exception:
            raise

    if path.exists():
        if force:
            shutil.rmtree(path, onexc=_on_rm_error)
        else:
            raise FileExistsError(f"{path} already exists. Use --force to replace it.")
    path.mkdir(parents=True, exist_ok=True)


def _find_training_root(dataset_root: Path) -> Path:
    for p in dataset_root.rglob("Training set"):
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f'Could not find "Training set" folder under {dataset_root}. '
        "Unzip and point to the parent folder of the extracted dataset."
    )


def _count_files(root: Path, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")) -> int:
    total = 0
    for ext in exts:
        total += len(list(root.rglob(f"*{ext}")))
    return total


def _extract_zip_tolerant(zip_path: Path, out_root: Path) -> tuple[int, int]:
    """
    Extract ZIP members one-by-one and continue when a member is corrupted.
    Returns (ok_members, failed_members).
    """
    ok_members = 0
    failed_members = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            try:
                zf.extract(member, out_root)
                ok_members += 1
            except Exception as exc:
                failed_members += 1
                print(f"[warn] skipped corrupted member: {member.filename} ({exc})")
    return ok_members, failed_members


def _concat_metadata_files(source_files: list[Path], out_file: Path) -> tuple[int, int, int, int]:
    """
    Merge metadata CSVs into one and return:
    (total_rows, train_rows, null_rows, num_classes)
    """
    required_cols = {"sign_id", "landmark_path"}
    header: list[str] | None = None
    total_rows = 0
    train_rows = 0
    null_rows = 0
    class_ids: set[str] = set()

    with open(out_file, "w", newline="", encoding="utf-8") as out:
        writer: csv.DictWriter | None = None
        for path in source_files:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = list(reader.fieldnames)
                    writer = csv.DictWriter(
                        out, fieldnames=header, extrasaction="ignore"
                    )
                    writer.writeheader()
                missing = required_cols.difference(reader.fieldnames)
                if missing:
                    raise ValueError(
                        f"Metadata file {path} missing required columns: {sorted(missing)}"
                    )
                for row in reader:
                    sid = str(row.get("sign_id", "")).strip()
                    if sid == "null_act":
                        null_rows += 1
                    elif sid:
                        train_rows += 1
                        class_ids.add(sid)
                    total_rows += 1
                    if writer is not None:
                        writer.writerow(row)

    if header is None:
        raise ValueError("No rows found in provided metadata files")

    return total_rows, train_rows, null_rows, len(class_ids)


def ingest_fingerspelling(args: argparse.Namespace) -> None:
    out_root = Path(args.output)

    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        if out_root != dataset_root:
            out_root = dataset_root
            if out_root.exists() and not args.force:
                print(
                    f"[warn] output differs from dataset_root, keeping existing dataset at {out_root} "
                    f"and writing manifest there."
                )
    else:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP not found: {zip_path}")
        _ensure_dir(out_root, force=args.force)
        extracted_ok, extracted_failed = _extract_zip_tolerant(zip_path, out_root)
        dataset_root = out_root

    training_root = _find_training_root(dataset_root)
    test_root = training_root.parent / "Test set"

    train_classes = [p for p in training_root.iterdir() if p.is_dir()]
    train_images = _count_files(training_root)
    test_images = _count_files(test_root) if test_root.exists() else 0

    print(f"[ok] Prepared fingerspelling dataset at: {out_root}")
    print(f"     training_root : {training_root}")
    print(f"     classes       : {len(train_classes)}")
    print(f"     train images  : {train_images}")
    print(f"     test images   : {test_images if test_root.exists() else 'n/a'}")

    manifest = {
        "track": "thai_fingerspelling",
        "source": str(dataset_root),
        "dataset_root": str(out_root),
        "training_root": str(training_root),
        "test_root": str(test_root) if test_root.exists() else "",
        "num_classes": int(len(train_classes)),
        "num_training_images": int(train_images),
        "num_test_images": int(test_images if test_root.exists() else 0),
        "feature_size": FEATURE_SIZE,
        "extracted_members_ok": int(extracted_ok if args.dataset_root is None else 0),
        "extracted_members_failed": int(extracted_failed if args.dataset_root is None else 0),
    }
    (out_root / "ingest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _require_dependency(name: str) -> None:
    raise ImportError(
        f"Missing optional dependency: {name}. "
        "Install the missing package and retry."
    )


def ingest_tsl51(args: argparse.Namespace) -> None:
    out_root = Path(args.output)
    _ensure_dir(out_root, force=args.force)

    metadata_out = out_root / "metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)

    if args.source:
        source_root = Path(args.source)
        if not source_root.exists():
            raise FileNotFoundError(f"Source path not found: {source_root}")

        meta_files = _collect_source_metadata_files(source_root)

        for src in meta_files:
            shutil.copy(src, metadata_out / src.name)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            _require_dependency("huggingface_hub")
            return

        REPO_ID = "Namonpas/thai-sign-language-tsl51"
        METADATA_FILES = (
            "metadata/expert_metadata.csv",
            "metadata/user_sign_metadata.csv",
        )

        for rel in METADATA_FILES:
            local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
            local_path = Path(local)
            target = metadata_out / Path(rel).name
            target.write_bytes(local_path.read_bytes())
            print(f"downloaded {rel}")
    # summarize metadata by merging the two known files
    total_rows, train_rows, null_rows, class_count = _concat_metadata_files(
        source_files=[metadata_out / "expert_metadata.csv", metadata_out / "user_sign_metadata.csv"],
        out_file=metadata_out / "combined_metadata.csv",
    )

    manifest = {
        "track": "tsl51_word_signs",
        "dataset": "Namonpas/thai-sign-language-tsl51",
        "metadata_dir": str(metadata_out),
        "meta_rows": int(total_rows),
        "num_classes": class_count,
        "num_training_rows": int(train_rows),
        "num_null_rows": int(null_rows),
        "seq_len": SEQ_LEN_DEFAULT,
        "feature_dim": FEATURE_DIM,
    }
    (out_root / "ingest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[ok] TSL-51 metadata prepared in:", metadata_out)
    print(f"     total rows   : {manifest['meta_rows']}")
    print(f"     train rows   : {manifest['num_training_rows']}")
    print(f"     classes      : {manifest['num_classes']}")
    print(f"     null_act rows: {manifest['num_null_rows']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets locally before training.")
    sub = parser.add_subparsers(dest="track", required=True)

    p1 = sub.add_parser("fingerspelling", help="Prepare One-Stage-TFS from local zip")
    src_group = p1.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--zip", help="Path to One-Stage-TFS zip")
    src_group.add_argument(
        "--dataset-root",
        help="Already-extracted One-Stage-TFS root folder containing 'Training set'.",
    )
    p1.add_argument(
        "--output",
        default="data/one_stage_tfs",
        help="Folder to write manifest to (default: data/one_stage_tfs).",
    )
    p1.add_argument("--force", action="store_true", help="Overwrite output folder if exists")
    p1.set_defaults(func=ingest_fingerspelling)

    p2 = sub.add_parser("tsl51", help="Prepare TSL-51 metadata cache")
    p2.add_argument(
        "--source",
        help="Optional local metadata folder containing metadata/expert_metadata.csv and metadata/user_sign_metadata.csv",
    )
    p2.add_argument(
        "--output",
        default="data/tsl51_raw",
        help="Folder to store local metadata snapshot",
    )
    p2.add_argument("--force", action="store_true", help="Overwrite output folder if exists")
    p2.set_defaults(func=ingest_tsl51)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
