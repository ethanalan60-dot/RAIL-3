"""CUDA-import-free admission gate for one V9B1 SAM worker process."""

from __future__ import annotations

from collections.abc import Mapping, Set
import fcntl
import os
from pathlib import Path
import stat
import subprocess


REQUIRED_BASE_ENVIRONMENT = {
    "RAIL3_STAGE": "AUTO_ANNOTATION",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}
GPU_UUIDS = (
    "GPU-0c4fc6e5-d153-8780-ed6c-4c612f8e3de6",
    "GPU-18a76220-b743-4a44-dcc1-14d69b505593",
)


def expected_worker_environment(shard: int) -> dict[str, str]:
    if shard not in {0, 1}:
        raise ValueError("V9B1 shard must be 0 or 1")
    return {
        **REQUIRED_BASE_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": str(shard),
        "RAIL3_PHYSICAL_GPU_UUID": GPU_UUIDS[shard],
        "RAIL3_GPU_LEASE_PATH": str(
            Path.cwd().resolve()
            / ".cache/tmlr-v9b1-gpu-leases"
            / f"{GPU_UUIDS[shard]}.lock"
        ),
    }


def query_gpu_inventory() -> tuple[tuple[int, str], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("V9B1_GPU_INVENTORY_QUERY_FAILED")
    rows: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise RuntimeError("V9B1_GPU_INVENTORY_FORMAT_INVALID")
        rows.append((int(parts[0]), parts[1]))
    return tuple(rows)


def validate_full_gpu_inventory(inventory: tuple[tuple[int, str], ...]) -> None:
    if inventory != tuple(enumerate(GPU_UUIDS)):
        raise RuntimeError(f"V9B1_GPU_UUID_INVENTORY_DRIFT:{inventory}")


def validate_visible_gpu_uuid(physical_index: int, expected_uuid: str) -> None:
    inventory = query_gpu_inventory()
    if (physical_index, expected_uuid) not in inventory:
        raise RuntimeError(f"V9B1_VISIBLE_GPU_UUID_DRIFT:{inventory}")


def validate_inherited_gpu_lease(shard: int, environment: Mapping[str, str]) -> int:
    value = environment.get("RAIL3_GPU_LEASE_FD")
    try:
        descriptor = int(value or "")
        metadata = os.fstat(descriptor)
    except (ValueError, OSError) as exc:
        raise RuntimeError("V9B1_GPU_LEASE_MISSING") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("V9B1_GPU_LEASE_NOT_SINGLE_LINK_REGULAR")
    try:
        target = os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError as exc:
        raise RuntimeError("V9B1_GPU_LEASE_IDENTITY_UNREADABLE") from exc
    expected_path = environment.get("RAIL3_GPU_LEASE_PATH")
    if target != expected_path:
        raise RuntimeError("V9B1_GPU_LEASE_IDENTITY_DRIFT")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError("V9B1_GPU_LEASE_NOT_HELD") from exc
    return descriptor


def validate_worker_environment(
    shard: int,
    environment: Mapping[str, str],
    imported_modules: Set[str],
    *,
    require_lease: bool = False,
    query_visible_uuid: bool = False,
) -> int:
    """Fail before any CUDA/SAM import if worker isolation is not exact."""

    required = expected_worker_environment(shard)
    drift = {
        key: {"expected": value, "actual": environment.get(key)}
        for key, value in required.items()
        if environment.get(key) != value
    }
    forbidden_imports = sorted(
        name for name in imported_modules
        if name == "torch" or name.startswith("torch.") or name == "sam3" or name.startswith("sam3.")
    )
    if drift:
        raise RuntimeError(f"V9B1_WORKER_ENVIRONMENT_DRIFT:{drift}")
    if forbidden_imports:
        raise RuntimeError(f"V9B1_CUDA_OR_SAM_IMPORTED_BEFORE_GATE:{forbidden_imports}")
    if require_lease:
        validate_inherited_gpu_lease(shard, environment)
    if query_visible_uuid:
        validate_visible_gpu_uuid(shard, GPU_UUIDS[shard])
    return shard
