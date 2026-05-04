"""Security utilities for TSL-51."""

import os
from pathlib import Path
from typing import Union


def validate_file_path(file_path: Union[str, Path], base_dir: Union[str, Path, None] = None) -> Path:
    """Validate file path to prevent path traversal attacks.

    Args:
        file_path: Path to validate
        base_dir: Base directory to restrict access to (optional)

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or outside base_dir
    """
    file_path = Path(file_path).resolve()

    if base_dir is not None:
        base_dir = Path(base_dir).resolve()
        try:
            file_path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path '{file_path}' is outside base directory '{base_dir}'")

    if not file_path.exists():
        raise ValueError(f"File '{file_path}' does not exist")

    if not file_path.is_file():
        raise ValueError(f"Path '{file_path}' is not a file")

    return file_path


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove path separators and parent directory references
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = filename.replace('..', '_')

    # Remove any null bytes
    filename = filename.replace('\0', '')

    # Strip leading/trailing whitespace and dots
    filename = filename.strip('. ')

    # Ensure not empty
    if not filename:
        filename = 'unnamed'

    return filename


def safe_path_join(base: Union[str, Path], *parts: str) -> Path:
    """Safely join path parts, preventing traversal outside base.

    Args:
        base: Base directory
        *parts: Path parts to join

    Returns:
        Resolved Path within base directory

    Raises:
        ValueError: If resulting path is outside base
    """
    base = Path(base).resolve()
    result = base.joinpath(*parts).resolve()

    try:
        result.relative_to(base)
    except ValueError:
        raise ValueError(f"Path '{result}' is outside base directory '{base}'")

    return result