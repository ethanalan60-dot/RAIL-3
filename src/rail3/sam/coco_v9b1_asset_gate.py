"""Fail-closed V9B1 SAM source/checkpoint preflight with no model import.

The formal worker calls :func:`validate_sam_assets_before_model` immediately
before constructing the real backend.  Importing this module does not import
torch or the upstream SAM package.  Tests exercise the low-level validators
only with tiny temporary files and independent temporary Git repositories.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from rail3.sam.coco_v9b1_trajectory import V9B1ContractError


ASSET_LOCK_PATH = Path("artifacts/paper/source_data/tmlr_v9b0_r2/sam_frozen_asset_lock.json")
ASSET_LOCK_SHA256 = "f3ecf7d6e044bff58711e51c8f1bf79523cbe45f49f9fbc2cd22a676587af257"
MATERIALIZATION_RECEIPT_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b0_r2/sam_asset_materialization_receipt.json"
)
MATERIALIZATION_RECEIPT_SHA256 = (
    "b9419df3b9c77b83b1f895d263dbba825e35c273f7e178ae8e0373f74aacda71"
)
MATERIALIZATION_RECEIPT_BYTES = 2_979
LOAD_SMOKE_PATH = Path("artifacts/paper/source_data/tmlr_v9b0_r2/sam_asset_load_smoke.json")
LOAD_SMOKE_SHA256 = "33281cd854aee4bb6e7f375d66b9e2280e4edcbc5c5e262807f274608fc87167"
LOAD_SMOKE_BYTES = 1_884
SOURCE_PATH = Path("artifacts/source/sam3")
SOURCE_REVISION = "96914d2425f90a64f45ca977c2b5165418099543"
SOURCE_TREE = "573deb167702e014829a5b830de8ae62abe891d5"
SOURCE_REPOSITORY = "https://github.com/facebookresearch/sam3.git"
CHECKPOINT_PATH = Path("artifacts/checkpoints/sam3/sam3.1_multiplex.pt")
CHECKPOINT_BYTES = 3_502_755_717
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def _relative_parts(relative: Path) -> tuple[str, ...]:
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "." in pure.parts:
        raise V9B1ContractError("V9B1_ASSET_PATH_NOT_EXACT", str(relative))
    return tuple(pure.parts)


def _open_repository_root(repo_root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(repo_root, flags)
    except OSError as exc:
        raise V9B1ContractError("V9B1_REPOSITORY_ROOT_OPEN_FAILED", str(repo_root)) from exc


def _open_relative(repo_fd: int, relative: Path, *, directory: bool) -> int:
    parts = _relative_parts(relative)
    current = os.dup(repo_fd)
    try:
        for part in parts[:-1]:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        flags = os.O_RDONLY | os.O_CLOEXEC
        if directory:
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(parts[-1], flags, dir_fd=current)
    except OSError as exc:
        raise V9B1ContractError("V9B1_ASSET_PATH_OPEN_FAILED", str(relative)) from exc
    finally:
        os.close(current)


def _stable_read(descriptor: int, *, expected_bytes: int | None = None) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise V9B1ContractError("V9B1_ASSET_NOT_SINGLE_LINK_REGULAR", "file descriptor")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise V9B1ContractError("V9B1_ASSET_BYTE_COUNT_MISMATCH", str(before.st_size))
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    )
    if identity_before != identity_after:
        raise V9B1ContractError("V9B1_ASSET_CHANGED_DURING_READ", "file descriptor")
    return b"".join(chunks)


def read_small_json_identity(
    repo_root: Path,
    relative: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    repo_fd = _open_repository_root(repo_root)
    try:
        descriptor = _open_relative(repo_fd, relative, directory=False)
        try:
            encoded = _stable_read(descriptor, expected_bytes=expected_bytes)
        finally:
            os.close(descriptor)
    finally:
        os.close(repo_fd)
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise V9B1ContractError("V9B1_ASSET_RECEIPT_HASH_DRIFT", str(relative))
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError("V9B1_ASSET_RECEIPT_JSON_INVALID", str(relative)) from exc
    if not isinstance(payload, dict):
        raise V9B1ContractError("V9B1_ASSET_RECEIPT_SCHEMA_INVALID", str(relative))
    return payload


def hash_checkpoint_identity(
    repo_root: Path,
    relative: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Hash one exact no-follow file; callers must never pass a discovered path."""

    repo_fd = _open_repository_root(repo_root)
    try:
        descriptor = _open_relative(repo_fd, relative, directory=False)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_bytes
            ):
                raise V9B1ContractError(
                    "V9B1_CHECKPOINT_STAT_MISMATCH", str(relative)
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(repo_fd)
    before_identity = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    )
    if before_identity != after_identity:
        raise V9B1ContractError("V9B1_CHECKPOINT_CHANGED_DURING_HASH", str(relative))
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise V9B1ContractError("V9B1_CHECKPOINT_HASH_MISMATCH", str(relative))
    return {
        "path": str(relative),
        "bytes": expected_bytes,
        "sha256": actual,
        "regular_single_link": True,
        "stable_during_hash": True,
    }


def stat_checkpoint_identity(
    repo_root: Path,
    relative: Path,
    *,
    expected_bytes: int,
) -> dict[str, Any]:
    repo_fd = _open_repository_root(repo_root)
    try:
        descriptor = _open_relative(repo_fd, relative, directory=False)
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(repo_fd)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != expected_bytes
    ):
        raise V9B1ContractError("V9B1_CHECKPOINT_STAT_MISMATCH", str(relative))
    return {
        "path": str(relative),
        "bytes": expected_bytes,
        "regular_single_link": True,
        "sha256_not_recomputed": CHECKPOINT_SHA256,
    }


def _git(source_fd: int, *arguments: str, expected_codes: tuple[int, ...] = (0,)) -> str:
    command = ["git", "-C", f"/proc/self/fd/{source_fd}", *arguments]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(source_fd,),
    )
    if completed.returncode not in expected_codes:
        raise V9B1ContractError(
            "V9B1_SAM_SOURCE_GIT_COMMAND_FAILED", " ".join(arguments)
        )
    return completed.stdout.strip()


def validate_source_checkout(
    repo_root: Path,
    relative: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_repository: str,
) -> dict[str, Any]:
    repo_fd = _open_repository_root(repo_root)
    try:
        source_fd = _open_relative(repo_fd, relative, directory=True)
    finally:
        os.close(repo_fd)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        try:
            git_fd = os.open(".git", directory_flags, dir_fd=source_fd)
        except OSError as exc:
            raise V9B1ContractError("V9B1_SAM_SOURCE_GITDIR_NOT_REAL_DIRECTORY", str(relative)) from exc
        if not stat.S_ISDIR(os.fstat(git_fd).st_mode):
            raise V9B1ContractError("V9B1_SAM_SOURCE_GITDIR_NOT_REAL_DIRECTORY", str(relative))
        revision = _git(source_fd, "rev-parse", "HEAD")
        tree = _git(source_fd, "rev-parse", "HEAD^{tree}")
        status = _git(source_fd, "status", "--porcelain=v1", "--untracked-files=all")
        git_dir = Path(_git(source_fd, "rev-parse", "--absolute-git-dir"))
        common_dir = Path(
            _git(source_fd, "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        remote = _git(source_fd, "remote", "get-url", "origin")
        symbolic = _git(source_fd, "symbolic-ref", "-q", "HEAD", expected_codes=(0, 1))
        source_stat_before = os.fstat(source_fd)
        git_metadata_before = os.fstat(git_fd)
        expected_git_dir = (repo_root / relative / ".git").absolute()
        if (
            revision != expected_revision
            or tree != expected_tree
            or status
            or symbolic
            or remote != expected_repository
            or git_dir != expected_git_dir
            or common_dir != expected_git_dir
        ):
            raise V9B1ContractError("V9B1_SAM_SOURCE_IDENTITY_DRIFT", str(relative))
        try:
            objects_fd = os.open("objects", directory_flags, dir_fd=git_fd)
            try:
                info_fd = os.open("info", directory_flags, dir_fd=objects_fd)
                try:
                    try:
                        os.stat("alternates", dir_fd=info_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise V9B1ContractError(
                            "V9B1_SAM_SOURCE_EXTERNAL_OBJECTS",
                            str(expected_git_dir / "objects/info/alternates"),
                        )
                finally:
                    os.close(info_fd)
            finally:
                os.close(objects_fd)
        except OSError as exc:
            raise V9B1ContractError("V9B1_SAM_SOURCE_OBJECT_STORE_INVALID", str(relative)) from exc
        final_identity = (
            _git(source_fd, "rev-parse", "HEAD"),
            _git(source_fd, "rev-parse", "HEAD^{tree}"),
            _git(source_fd, "status", "--porcelain=v1", "--untracked-files=all"),
            _git(source_fd, "remote", "get-url", "origin"),
        )
        source_stat_after = os.fstat(source_fd)
        git_metadata_after = os.fstat(git_fd)
        if (
            source_stat_before.st_dev,
            source_stat_before.st_ino,
            source_stat_before.st_mtime_ns,
            source_stat_before.st_ctime_ns,
        ) != (
            source_stat_after.st_dev,
            source_stat_after.st_ino,
            source_stat_after.st_mtime_ns,
            source_stat_after.st_ctime_ns,
        ) or (
            git_metadata_before.st_dev,
            git_metadata_before.st_ino,
        ) != (
            git_metadata_after.st_dev,
            git_metadata_after.st_ino,
        ) or final_identity != (revision, tree, status, remote):
            raise V9B1ContractError("V9B1_SAM_SOURCE_CHANGED_DURING_AUDIT", str(relative))
    finally:
        if "git_fd" in locals():
            os.close(git_fd)
        os.close(source_fd)
    return {
        "path": str(relative),
        "revision": revision,
        "tree": tree,
        "clean": True,
        "detached": True,
        "repository": remote,
        "independent_git_objects": True,
    }


def _validate_receipt_closure(
    lock: dict[str, Any], materialization: dict[str, Any], smoke: dict[str, Any]
) -> None:
    if not (
        lock.get("schema_version") == "rail3.tmlr-v9b0-r2-sam-frozen-asset-lock.v1"
        and lock.get("status") == "SAM_FROZEN_ASSET_LOCK_PASS"
        and lock.get("source_revision") == SOURCE_REVISION
        and lock.get("source_tree") == SOURCE_TREE
        and lock.get("source_repo") == SOURCE_REPOSITORY
        and lock.get("checkpoint_destination_path") == str(CHECKPOINT_PATH)
        and lock.get("checkpoint_bytes") == CHECKPOINT_BYTES
        and lock.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and lock.get("materialization_receipt") == {
            "path": str(MATERIALIZATION_RECEIPT_PATH),
            "bytes": MATERIALIZATION_RECEIPT_BYTES,
            "sha256": MATERIALIZATION_RECEIPT_SHA256,
        }
        and lock.get("load_smoke_receipt") == {
            "path": str(LOAD_SMOKE_PATH),
            "bytes": LOAD_SMOKE_BYTES,
            "sha256": LOAD_SMOKE_SHA256,
        }
        and all(lock.get("checks", {}).get(key) is True for key in (
            "checkpoint_bytes_exact", "checkpoint_regular_single_link",
            "checkpoint_sha256_exact", "fallback_download_zero",
            "real_image_inference_zero", "sam_load_smoke_pass", "source_clean",
            "source_revision_exact", "source_tree_exact",
        ))
    ):
        raise V9B1ContractError("V9B1_SAM_ASSET_LOCK_DRIFT", str(ASSET_LOCK_PATH))
    destination = materialization.get("source", {}).get("destination", {})
    checkpoint = materialization.get("checkpoint", {})
    if not (
        materialization.get("schema_version")
        == "rail3.tmlr-v9b0-r2-sam-materialization-receipt.v1"
        and materialization.get("status") == "SAM31_EXACT_LOCAL_MATERIALIZATION_PASS"
        and destination.get("revision") == SOURCE_REVISION
        and destination.get("tree") == SOURCE_TREE
        and destination.get("clean") is True
        and destination.get("detached") is True
        and destination.get("alternates_absent") is True
        and checkpoint.get("bytes") == CHECKPOINT_BYTES
        and checkpoint.get("sha256") == CHECKPOINT_SHA256
        and checkpoint.get("independent_inode") is True
        and all(materialization.get("checks", {}).get(key) is value for key, value in {
            "checkpoint_bytes_exact": True,
            "checkpoint_independent_copy": True,
            "checkpoint_sha256_exact": True,
            "fallback_model_used": False,
            "network_retrieval_used": False,
            "real_image_inference": False,
            "source_clean": True,
            "source_revision_exact": True,
            "source_tree_exact": True,
        }.items())
    ):
        raise V9B1ContractError("V9B1_SAM_MATERIALIZATION_RECEIPT_DRIFT", str(MATERIALIZATION_RECEIPT_PATH))
    if not (
        smoke.get("schema_version") == "rail3.tmlr-v9b0-r2-sam-load-smoke.v1"
        and smoke.get("status") == "SAM_ASSET_LOAD_PASS"
        and smoke.get("source_identity", {}).get("revision") == SOURCE_REVISION
        and smoke.get("source_identity", {}).get("tree") == SOURCE_TREE
        and smoke.get("source_identity", {}).get("clean") is True
        and smoke.get("checkpoint_identity") == {
            "bytes": CHECKPOINT_BYTES,
            "path": str(CHECKPOINT_PATH),
            "sha256": CHECKPOINT_SHA256,
        }
        and smoke.get("real_image_forward_calls") == 0
        and smoke.get("sam_inference_calls") == 0
        and smoke.get("active_image_states") == 0
        and smoke.get("trainable_parameter_count") == 0
        and all(smoke.get("checks", {}).get(key) is True for key in (
            "checkpoint_bytes_exact", "checkpoint_sha256_exact",
            "cleanroom_import_precedence_exact", "parameters_frozen",
            "real_image_forward_absent", "source_revision_exact", "source_tree_exact",
        ))
    ):
        raise V9B1ContractError("V9B1_SAM_LOAD_SMOKE_RECEIPT_DRIFT", str(LOAD_SMOKE_PATH))


def validate_sam_assets_before_model(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    sam = config.get("sam", {})
    exact_config = {
        "source_path": str(SOURCE_PATH),
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fallback_allowed": False,
        "runtime_download_allowed": False,
        "real_backend_only": True,
    }
    if any(sam.get(key) != value for key, value in exact_config.items()):
        raise V9B1ContractError("V9B1_SAM_CONFIG_IDENTITY_DRIFT", "SAM authority drift")
    lock = read_small_json_identity(
        repo_root, ASSET_LOCK_PATH, expected_sha256=ASSET_LOCK_SHA256
    )
    materialization = read_small_json_identity(
        repo_root,
        MATERIALIZATION_RECEIPT_PATH,
        expected_sha256=MATERIALIZATION_RECEIPT_SHA256,
        expected_bytes=MATERIALIZATION_RECEIPT_BYTES,
    )
    smoke = read_small_json_identity(
        repo_root,
        LOAD_SMOKE_PATH,
        expected_sha256=LOAD_SMOKE_SHA256,
        expected_bytes=LOAD_SMOKE_BYTES,
    )
    _validate_receipt_closure(lock, materialization, smoke)
    source = validate_source_checkout(
        repo_root,
        SOURCE_PATH,
        expected_revision=SOURCE_REVISION,
        expected_tree=SOURCE_TREE,
        expected_repository=SOURCE_REPOSITORY,
    )
    checkpoint = hash_checkpoint_identity(
        repo_root,
        CHECKPOINT_PATH,
        expected_bytes=CHECKPOINT_BYTES,
        expected_sha256=CHECKPOINT_SHA256,
    )
    return {
        "receipt_closure": "PASS",
        "source": source,
        "checkpoint": checkpoint,
        "fallback_downloads": 0,
        "model_builds": 0,
        "checkpoint_loads": 0,
        "real_image_forwards": 0,
        "sam_prompt_calls": 0,
    }


def validate_sam_asset_metadata_before_launch(
    repo_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate small receipts/source and checkpoint stat, but do not hash/load it."""

    sam = config.get("sam", {})
    if not (
        sam.get("source_path") == str(SOURCE_PATH)
        and sam.get("source_revision") == SOURCE_REVISION
        and sam.get("source_tree") == SOURCE_TREE
        and sam.get("checkpoint_path") == str(CHECKPOINT_PATH)
        and sam.get("checkpoint_bytes") == CHECKPOINT_BYTES
        and sam.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and sam.get("fallback_allowed") is False
        and sam.get("runtime_download_allowed") is False
        and sam.get("real_backend_only") is True
    ):
        raise V9B1ContractError("V9B1_SAM_CONFIG_IDENTITY_DRIFT", "SAM authority drift")
    lock = read_small_json_identity(
        repo_root, ASSET_LOCK_PATH, expected_sha256=ASSET_LOCK_SHA256
    )
    materialization = read_small_json_identity(
        repo_root,
        MATERIALIZATION_RECEIPT_PATH,
        expected_sha256=MATERIALIZATION_RECEIPT_SHA256,
        expected_bytes=MATERIALIZATION_RECEIPT_BYTES,
    )
    smoke = read_small_json_identity(
        repo_root,
        LOAD_SMOKE_PATH,
        expected_sha256=LOAD_SMOKE_SHA256,
        expected_bytes=LOAD_SMOKE_BYTES,
    )
    _validate_receipt_closure(lock, materialization, smoke)
    return {
        "receipt_closure": "PASS",
        "source": validate_source_checkout(
            repo_root,
            SOURCE_PATH,
            expected_revision=SOURCE_REVISION,
            expected_tree=SOURCE_TREE,
            expected_repository=SOURCE_REPOSITORY,
        ),
        "checkpoint": stat_checkpoint_identity(
            repo_root, CHECKPOINT_PATH, expected_bytes=CHECKPOINT_BYTES
        ),
        "checkpoint_hash_recomputed": False,
        "model_builds": 0,
        "checkpoint_loads": 0,
        "real_image_forwards": 0,
        "sam_prompt_calls": 0,
    }
