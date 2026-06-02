from __future__ import annotations

from pathlib import Path

from translate_api.store import SessionStore

_store: SessionStore | None = None
_repo_root: Path | None = None


def init_dependencies(store: SessionStore, repo_root: Path) -> None:
    global _store, _repo_root
    _store = store
    _repo_root = repo_root


def get_store() -> SessionStore:
    if _store is None:
        raise RuntimeError("SessionStore not initialized")
    return _store


def get_repo_root() -> Path:
    if _repo_root is None:
        raise RuntimeError("repo root not initialized")
    return _repo_root
