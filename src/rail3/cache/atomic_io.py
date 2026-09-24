"""Same-directory atomic creation without overwriting immutable objects."""

from __future__ import annotations

import os
import tempfile
import hashlib
from collections.abc import Iterable
from pathlib import Path


def atomic_create_bytes(destination: Path, payload: bytes) -> None:
    """Atomically create ``destination`` and fail if it already exists."""

    workspace = Path.cwd().resolve()
    resolved = destination.resolve()
    try:
        destination = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("atomic cache destination escapes the repository workspace") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_create_chunks(destination: Path, chunks: Iterable[bytes]) -> dict[str, int | str]:
    """Atomically create a potentially large file from deterministic byte chunks."""

    workspace = Path.cwd().resolve()
    resolved = destination.resolve()
    try:
        destination = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("atomic cache destination escapes the repository workspace") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("atomic file chunks must be bytes")
                handle.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    verified_digest = hashlib.sha256()
    verified_bytes = 0
    with destination.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            verified_digest.update(chunk)
            verified_bytes += len(chunk)
    if verified_bytes != byte_count or verified_digest.hexdigest() != digest.hexdigest():
        raise RuntimeError("atomic chunk file corruption detected after creation")
    return {"bytes": byte_count, "sha256": digest.hexdigest()}
