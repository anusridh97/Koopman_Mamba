"""Controller-owned hashing for Phase 2a checkpoint bundles."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any


def _stable_regular_file(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Checkpoint bundle must not contain symlinks: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint entry is not a regular file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if before_identity != after_identity or before_identity != current_identity:
        raise ValueError(f"Checkpoint entry changed while hashing: {path}")
    if before.st_size <= 0:
        raise ValueError(f"Checkpoint entry must not be empty: {path}")
    return {
        "byte_size": int(before.st_size),
        "sha256": digest.hexdigest(),
    }


def checkpoint_bundle_identity(
    run_dir: str | Path,
    declared_path: Any,
) -> dict[str, Any]:
    """Hash a file or directory bundle rooted strictly inside one trial."""

    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError("Checkpoint bundle path must be a nonempty relative path")
    relative = Path(declared_path)
    if (
        relative.is_absolute()
        or "\\" in declared_path
        or relative.as_posix() != declared_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Checkpoint bundle path must be normalized and relative")

    root = Path(run_dir).expanduser().resolve()
    unresolved = root / relative
    for parent in (unresolved, *unresolved.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(
                f"Checkpoint bundle path must not traverse symlinks: {declared_path}"
            )
        if parent == root:
            break
    try:
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint bundle escapes or is missing from its run: {declared_path}"
        ) from exc

    if candidate.is_file():
        entries = [(candidate.name, candidate)]
    elif candidate.is_dir():
        entries = []
        for child in sorted(candidate.rglob("*")):
            if child.is_symlink():
                raise ValueError(
                    f"Checkpoint bundle must not contain symlinks: {child}"
                )
            if child.is_dir():
                continue
            if not child.is_file():
                raise ValueError(
                    f"Checkpoint bundle contains a non-file entry: {child}"
                )
            entries.append((child.relative_to(candidate).as_posix(), child))
    else:
        raise ValueError(f"Checkpoint bundle is not a file or directory: {candidate}")
    if not entries:
        raise ValueError("Checkpoint bundle must contain at least one nonempty file")

    bundle_digest = hashlib.sha256()
    total_bytes = 0
    for entry_name, entry_path in entries:
        identity = _stable_regular_file(entry_path)
        total_bytes += int(identity["byte_size"])
        bundle_digest.update(entry_name.encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(str(identity["byte_size"]).encode("ascii"))
        bundle_digest.update(b"\0")
        bundle_digest.update(str(identity["sha256"]).encode("ascii"))
        bundle_digest.update(b"\n")
    return {
        "path": relative.as_posix(),
        "byte_size": total_bytes,
        "file_count": len(entries),
        "sha256": bundle_digest.hexdigest(),
        "digest_algorithm": "sha256(path_nul_size_nul_content_sha256_newline_v1)",
    }
