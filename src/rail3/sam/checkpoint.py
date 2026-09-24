"""Fail-closed identity and file validation for the pinned SAM 3.1 checkpoint."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "facebook/sam3.1"
OFFICIAL_FILENAME = "sam3.1_multiplex.pt"
OFFICIAL_ACCESS = "official_gated_hugging_face"
PINNED_UPSTREAM_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
MAX_CHECKPOINT_BYTES = 15 * 1024**3
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Sam31CheckpointIdentity:
    """Portable identity for the one approved official checkpoint."""

    schema_version: str
    checkpoint_id: str
    repository: str
    filename: str
    bytes: int
    sha256: str
    upstream_commit: str
    role: str
    access: str
    license_audit: str
    storage_root: str
    tracked: bool

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> "Sam31CheckpointIdentity":
        checkpoint = _mapping(root, "checkpoint")
        storage = _mapping(root, "storage")
        identity = cls(
            schema_version=_string(root, "schema_version"),
            checkpoint_id=_string(checkpoint, "id"),
            repository=_string(checkpoint, "repository"),
            filename=_string(checkpoint, "filename"),
            bytes=_integer(checkpoint, "bytes"),
            sha256=_string(checkpoint, "sha256"),
            upstream_commit=_string(checkpoint, "upstream_commit"),
            role=_string(checkpoint, "role"),
            access=_string(checkpoint, "access"),
            license_audit=_string(checkpoint, "license_audit"),
            storage_root=_string(storage, "root"),
            tracked=_boolean(storage, "tracked"),
        )
        identity.validate()
        return identity

    def validate(self) -> None:
        if self.schema_version != "rail3.checkpoint.v1":
            raise ValueError("unsupported checkpoint identity schema")
        if self.checkpoint_id != "sam31_official_multiplex":
            raise ValueError("unapproved checkpoint logical ID")
        if self.repository != OFFICIAL_REPOSITORY:
            raise ValueError("unapproved checkpoint repository")
        if self.filename != OFFICIAL_FILENAME:
            raise ValueError("unapproved checkpoint filename")
        if not 0 < self.bytes <= MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint byte count violates the bounded policy")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("checkpoint SHA-256 must be lowercase hexadecimal")
        if self.upstream_commit != PINNED_UPSTREAM_COMMIT:
            raise ValueError("checkpoint identity is not bound to the pinned source")
        if self.role != "SAM 3.1 Object Multiplex":
            raise ValueError("unapproved checkpoint role")
        if self.access != OFFICIAL_ACCESS:
            raise ValueError("unapproved checkpoint access class")
        if self.license_audit != "docs/SAM31_CHECKPOINT_AUDIT.md":
            raise ValueError("checkpoint identity must bind the reviewed license audit")
        if self.storage_root != "artifacts/checkpoints/sam3":
            raise ValueError("checkpoint storage root is not the reviewed ignored root")
        if self.tracked is not False:
            raise ValueError("checkpoint storage must be explicitly untracked")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.checkpoint_id,
            "repository": self.repository,
            "filename": self.filename,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "upstream_commit": self.upstream_commit,
            "role": self.role,
            "access": self.access,
            "license_audit": self.license_audit,
            "storage_root": self.storage_root,
            "tracked": self.tracked,
        }


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_file(
    identity: Sam31CheckpointIdentity,
    path: Path,
) -> dict[str, Any]:
    identity.validate()
    if path.name != identity.filename:
        raise ValueError("checkpoint path filename differs from the frozen identity")
    if not path.is_file():
        raise ValueError("frozen checkpoint file is absent")
    observed_bytes = path.stat().st_size
    if observed_bytes != identity.bytes:
        raise ValueError("checkpoint byte count mismatch")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != identity.sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    return {
        "status": "PASS",
        "filename": path.name,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string(root: Mapping[str, Any], key: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(root: Mapping[str, Any], key: str) -> int:
    value = root.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _boolean(root: Mapping[str, Any], key: str) -> bool:
    value = root.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
