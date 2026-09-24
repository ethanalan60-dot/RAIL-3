"""Root-dirfd, no-follow, immutable caches for formal V9B1 trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from rail3.cache import (
    CacheCorruptionError,
    CacheIdentity,
    CandidateCache,
    CandidateGenerationResult,
)
from rail3.contracts import canonical_json_bytes
from rail3.sam.voc_actions import (
    ACTION_CACHE_SCHEMA,
    ActionCacheIdentity,
    ActionOutcome,
    TrajectoryCache,
    TrajectoryCacheError,
)


class V9B1SecureIOError(RuntimeError):
    pass


def _parts(relative: Path) -> tuple[str, ...]:
    if relative.is_absolute():
        try:
            relative = relative.relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise V9B1SecureIOError(
                f"V9B1_SECURE_PATH_ESCAPES_WORKSPACE:{relative}"
            ) from exc
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise V9B1SecureIOError(f"V9B1_SECURE_PATH_NOT_RELATIVE:{relative}")
    return tuple(pure.parts)


def _workspace_fd() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(Path.cwd().resolve(), flags)


def _parent_fd(relative: Path, *, create: bool) -> tuple[int, str]:
    parts = _parts(relative)
    current = _workspace_fd()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
                next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current, parts[-1]
    except Exception:
        os.close(current)
        raise


def secure_read_bytes(relative: Path) -> bytes:
    parent_fd, leaf = _parent_fd(relative, create=False)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise V9B1SecureIOError(f"V9B1_SECURE_FILE_NOT_SINGLE_LINK:{relative}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise V9B1SecureIOError(f"V9B1_SECURE_FILE_CHANGED_DURING_READ:{relative}")
    return b"".join(chunks)


def secure_create_once(relative: Path, encoded: bytes) -> None:
    parent_fd, leaf = _parent_fd(relative, create=True)
    temporary = f".{leaf}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise V9B1SecureIOError(f"V9B1_SECURE_WRITE_FAILED:{relative}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            leaf,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def secure_create_or_verify(relative: Path, encoded: bytes) -> None:
    try:
        secure_create_once(relative, encoded)
    except FileExistsError:
        if secure_read_bytes(relative) != encoded:
            raise V9B1SecureIOError(f"V9B1_IMMUTABLE_FILE_CONFLICT:{relative}")


def secure_open_lock(relative: Path, mode: int = 0o600) -> int:
    parent_fd, leaf = _parent_fd(relative, create=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(leaf, flags, mode, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise V9B1SecureIOError(f"V9B1_LOCK_NOT_SINGLE_LINK_REGULAR:{relative}")
    return descriptor


class SecureCandidateCache(CandidateCache):
    def store(self, identity: CacheIdentity, result: CandidateGenerationResult) -> Path:
        self._validate_result_identity(identity, result)
        payload = result.to_dict()
        envelope = {
            "schema_version": "rail3.candidate-cache.v1",
            "cache_key": identity.cache_key,
            "identity": {
                "model_spec_id": identity.model_spec_id,
                "asset_id": identity.asset_id,
                "image_sha256": identity.image_sha256,
                "prompt_id": identity.prompt_id,
                "state_id": identity.state_id,
                "adapter_schema": identity.adapter_schema,
            },
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        }
        encoded = canonical_json_bytes(envelope) + b"\n"
        try:
            secure_create_or_verify(self.path_for(identity), encoded)
        except V9B1SecureIOError as exc:
            raise CacheCorruptionError("IMMUTABLE_CACHE_CONFLICT", str(exc)) from exc
        return self.path_for(identity)

    def load(self, identity: CacheIdentity) -> CandidateGenerationResult:
        try:
            envelope = json.loads(secure_read_bytes(self.path_for(identity)))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, V9B1SecureIOError) as exc:
            raise CacheCorruptionError("CACHE_JSON_INVALID", "cache object is unreadable") from exc
        expected_identity = {
            "model_spec_id": identity.model_spec_id,
            "asset_id": identity.asset_id,
            "image_sha256": identity.image_sha256,
            "prompt_id": identity.prompt_id,
            "state_id": identity.state_id,
            "adapter_schema": identity.adapter_schema,
        }
        payload = envelope.get("payload")
        if not (
            envelope.get("schema_version") == "rail3.candidate-cache.v1"
            and envelope.get("cache_key") == identity.cache_key
            and envelope.get("identity") == expected_identity
            and isinstance(payload, dict)
            and envelope.get("payload_sha256")
            == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        ):
            raise CacheCorruptionError("CACHE_ENVELOPE_MISMATCH", "cache envelope drift")
        try:
            result = CandidateGenerationResult.from_dict(payload)
            self._validate_result_identity(identity, result)
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorruptionError("CACHE_PAYLOAD_INVALID", "candidate payload drift") from exc
        return result


class SecureTrajectoryCache(TrajectoryCache):
    def store(self, identity: ActionCacheIdentity, outcome: ActionOutcome) -> Path:
        self._validate_identity(identity, outcome)
        payload = outcome.to_dict()
        envelope = {
            "schema_version": ACTION_CACHE_SCHEMA,
            "cache_key": identity.cache_key,
            "identity": identity.to_dict(),
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            "semantic_sha256": outcome.semantic_sha256,
        }
        encoded = canonical_json_bytes(envelope) + b"\n"
        try:
            secure_create_or_verify(self.path_for(identity), encoded)
        except V9B1SecureIOError as exc:
            raise TrajectoryCacheError("IMMUTABLE_TRAJECTORY_CACHE_CONFLICT", str(exc)) from exc
        return self.path_for(identity)

    def load(self, identity: ActionCacheIdentity) -> ActionOutcome:
        try:
            envelope = json.loads(secure_read_bytes(self.path_for(identity)))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, V9B1SecureIOError) as exc:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_JSON_INVALID", "cache unreadable") from exc
        payload = envelope.get("payload")
        if not (
            envelope.get("schema_version") == ACTION_CACHE_SCHEMA
            and envelope.get("cache_key") == identity.cache_key
            and envelope.get("identity") == identity.to_dict()
            and isinstance(payload, dict)
            and envelope.get("payload_sha256")
            == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        ):
            raise TrajectoryCacheError("TRAJECTORY_CACHE_ENVELOPE_MISMATCH", "cache drift")
        try:
            outcome = ActionOutcome.from_dict(payload)
            self._validate_identity(identity, outcome)
        except (KeyError, TypeError, ValueError) as exc:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_PAYLOAD_INVALID", "payload drift") from exc
        if envelope.get("semantic_sha256") != outcome.semantic_sha256:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_SEMANTIC_HASH_MISMATCH", "hash drift")
        return outcome
