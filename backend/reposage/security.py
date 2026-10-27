"""Validation helpers for repository identifiers and repository-relative paths."""

import re
from pathlib import Path, PurePosixPath, PureWindowsPath


_REPO_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_repo_name(repo_name: str) -> str:
    """Return a safe single path-component repository name."""
    if not isinstance(repo_name, str) or not _REPO_NAME_PATTERN.fullmatch(repo_name):
        raise ValueError("Invalid repository name")
    if repo_name.endswith((".", " ")):
        raise ValueError("Invalid repository name")
    if repo_name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("Invalid repository name")
    return repo_name


def normalize_repo_relative_path(file_path: str) -> str:
    """Normalize a relative path while rejecting POSIX and Windows traversal."""
    if not isinstance(file_path, str) or not file_path or "\0" in file_path:
        raise ValueError("Invalid repository-relative path")

    windows_path = PureWindowsPath(file_path)
    normalized = file_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError("Invalid repository-relative path")

    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid repository-relative path")
    return "/".join(parts)


def resolve_repo_file(clone_dir: str | Path, repo_name: str, file_path: str) -> Path:
    """Resolve a repository-relative file and reject symlink/path escapes."""
    safe_repo_name = validate_repo_name(repo_name)
    relative_path = normalize_repo_relative_path(file_path)
    clone_root = Path(clone_dir).resolve()
    repo_root = (clone_root / safe_repo_name).resolve()
    if not repo_root.is_relative_to(clone_root):
        raise ValueError("Repository path escapes the clone directory")

    resolved_path = (repo_root / relative_path).resolve()
    if not resolved_path.is_relative_to(repo_root):
        raise ValueError("File path escapes the repository")
    return resolved_path