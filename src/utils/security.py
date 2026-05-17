"""Security utilities for TSL-51."""

from pathlib import Path


def validate_file_path(
    file_path: Path,
    allowed_extensions: set[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Validate a file path for security.

    Args:
        file_path: Path to validate
        allowed_extensions: Set of allowed extensions (e.g., {'.pt', '.npz'})
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or has disallowed extension
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    path = Path(file_path)

    # Check for path traversal attempts
    try:
        # Resolve to absolute path
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {file_path}") from e

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if allowed_extensions and resolved.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Disallowed file extension: {resolved.suffix}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    return resolved
