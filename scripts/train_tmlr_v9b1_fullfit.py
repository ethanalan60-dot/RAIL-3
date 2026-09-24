#!/usr/bin/env python3
"""Thin, fail-closed launcher for dedicated TMLR V9B1 full-fit jobs.

The launcher admits two distinct roots, opens only config-allowlisted inputs,
holds one physical-GPU lease, and publishes checkpoint/manifest pairs with a
kernel-enforced no-replace transaction.  It has no SAM, COCO, trajectory,
prediction, or evaluation path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterator, Mapping


CONFIG_PATH = Path("configs/experiments/tmlr_v9b1_fullfit_v1.json")
EXPECTED_INPUT_ROOT = Path("release-inputs/voc-fit")
EXPECTED_OUTPUT_ROOT = Path("release-runs/fullfit")
FORMAL_ACK = "TMLR_V9B1_FULLFIT_AUTHORIZED"
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/rail3/models/tmlr_v9b1/__init__.py"),
    Path("src/rail3/models/tmlr_v9b1/fullfit.py"),
    Path("scripts/train_tmlr_v9b1_fullfit.py"),
)
RUN_ROOT = Path("artifacts/voc2012/tmlr-v9b1-fullfit/S1364")
REGISTRY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/"
    "fullfit_checkpoint_lock.json"
)
GPU_LOCK_ROOT = Path(".cache/tmlr-v9b1-gpu-leases")
EXPECTED_FAMILIES = (
    "R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P", "R3_P",
)
EXPECTED_SEEDS = (13, 37, 71)
EXPECTED_EPOCHS = {
    "R0_SMALL_P": 120,
    "R0_CM_P": 120,
    "R1_P": 120,
    "RECT_P": 120,
    "UNION_P": 120,
    "R3_P": 108,
}
EXPECTED_BUNDLES = {
    "R0_SMALL_P": "F0_P",
    "R0_CM_P": "F0_P",
    "R1_P": "F0_P",
    "RECT_P": "RECT_FULL_RASTER_P",
    "UNION_P": "UNION_FULL_RASTER_P",
    "R3_P": "F0_P",
}
EXPECTED_PARAMETER_COUNTS = {
    "R0_SMALL_P": 7_809,
    "R0_CM_P": 103_369,
    "R1_P": 103_521,
    "RECT_P": 103_521,
    "UNION_P": 103_521,
    "R3_P": 103_521,
}
EXPECTED_MODEL_CLASSES = {
    "R0_SMALL_P": "rail3.models.m06e.models.GlobalResidual",
    "R0_CM_P": "rail3.models.tmlr_v6.models.CapacityMatchedGlobalResidual",
    "R1_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "RECT_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "UNION_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "R3_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
}
EXPECTED_OBJECTIVES = {
    family: (
        "R3_FROZEN_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK_NO_COST"
        if family == "R3_P" else "STATE_HUBER_DELTA_0_05"
    )
    for family in EXPECTED_FAMILIES
}
EXPECTED_OPTIMIZER_RECIPE = {
    "class": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "scheduler": None,
    "amp": False,
    "ddp": False,
    "early_stopping": False,
    "warm_start": False,
}
EXPECTED_GPU = {
    "R0_SMALL_P": "0", "RECT_P": "0", "R3_P": "0",
    "R0_CM_P": "1", "R1_P": "1", "UNION_P": "1",
}
EXPECTED_SOURCE_PATHS = frozenset({
    "src/rail3/models/m06e/features.py",
    "src/rail3/models/m06e/losses.py",
    "src/rail3/models/m06e/models.py",
    "src/rail3/models/m06g/losses.py",
    "src/rail3/models/tmlr_v6/models.py",
    "src/rail3/models/tmlr_v6/schema.py",
    "src/rail3/models/tmlr_v6/training.py",
    "src/rail3/models/tmlr_v6/phase2_models.py",
    "src/rail3/models/tmlr_v6/phase2_objectives.py",
})
EXPECTED_BUNDLE_ALLOWLIST_SHA256 = (
    "14281d1a200cb7283eb10bce76ec37282873e999569af4f8fd313df10e8bc316"
)
EXPECTED_SOURCE_ALLOWLIST_SHA256 = (
    "e9dd8ddc03db18cc46a70240c4bf865c42316f8f1bc81997dc2310e6c4c19a30"
)
EXPECTED_AUTHORITY_ALLOWLIST_SHA256 = (
    "2623cf6da6209cb98ce339dd5f7c0a60ada12590c2a0bcc98085faa9a3f5a564"
)
EXPECTED_FORBIDDEN_PATH_FRAGMENTS = (
    "instances_train2017.json",
    "instances_val2017.json",
    "artifacts/checkpoints/sam3",
    "artifacts/source/sam3",
    "artifacts/protected",
    "validation40",
    "calibration30",
    "pilot_test30",
    "pilot-test30",
    "official_voc",
    "segppd",
    "uav",
    "handoff",
    "trajectory",
    "predictions.npz",
)
EXPECTED_RUNTIME_ENVIRONMENT = {
    "interpreter": "RELEASE_INTERPRETER_UNBOUND",
    "python": "3.12.3",
    "numpy": "1.26.4",
    "torch": "2.10.0+cu128",
    "cuda_runtime": "12.8",
}
EXPECTED_ZERO_COUNTERS = {
    "coco_annotation_reads": 0,
    "coco_per_image_gt_materialized": 0,
    "sam_calls": 0,
    "trajectory_reads": 0,
    "protected_reads": 0,
    "uav_reads": 0,
    "predictions": 0,
    "performance_results": 0,
}
RUN_MANIFEST_KEYS = frozenset({
    "schema_version",
    "run_id",
    "family",
    "seed",
    "bundle_role",
    "fullfit_epoch",
    "epochs_completed",
    "epoch_rule_id",
    "epoch_lock_sha256",
    "config_sha256",
    "model_class",
    "parameter_count",
    "objective_id",
    "optimizer_recipe",
    "prediction_semantics",
    "ensemble_id",
    "implementation_commit",
    "implementation_sha256",
    "bundle_identities",
    "training_loss_finite",
    "wall_seconds",
    "peak_vram_bytes",
    "physical_gpu",
    "zero_counters",
    "created_utc",
    "checkpoint",
    "terminal_state",
})
SHA256_HEX = frozenset("0123456789abcdef")
RENAME_NOREPLACE = 1


class FullFitLaunchError(RuntimeError):
    """A typed fail-closed V9B1 admission or publication failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullFitLaunchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise FullFitLaunchError(f"invalid {description} JSON") from error
    if not isinstance(value, dict):
        raise FullFitLaunchError(f"{description} must be a JSON object")
    return value


def _validate_sha256(value: Any, *, description: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in SHA256_HEX for char in text):
        raise FullFitLaunchError(f"invalid {description} SHA-256")
    return text


def _validate_relative_path(value: Any, *, description: str) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text or pure.is_absolute() or ".." in pure.parts
        or "." in pure.parts or "\\" in text
    ):
        raise FullFitLaunchError(f"invalid {description} relative path")
    return Path(*pure.parts)


def _validate_identity(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) < {"path", "bytes", "sha256"}:
        raise FullFitLaunchError(f"invalid {description} identity schema")
    relative = _validate_relative_path(value["path"], description=description)
    size = value["bytes"]
    if type(size) is not int or size <= 0:
        raise FullFitLaunchError(f"invalid {description} byte count")
    return {
        "path": relative.as_posix(),
        "bytes": size,
        "sha256": _validate_sha256(value["sha256"], description=description),
    }


def _open_root(path: Path, *, expected: Path, description: str) -> int:
    lexical = path.absolute()
    if lexical != expected or lexical.is_symlink() or lexical.resolve() != lexical:
        raise FullFitLaunchError(f"{description} is not the exact frozen root")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise FullFitLaunchError(f"{description} is not a directory")
    return descriptor


def _open_directory_at(root_fd: int, relative: Path, *, create: bool = False) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            child = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise FullFitLaunchError("path component is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_file_at(root_fd: int, relative: Path) -> int:
    parent_fd = _open_directory_at(root_fd, relative.parent)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(relative.name, flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise FullFitLaunchError(f"input is not regular single-link: {relative}")
    return descriptor


def _read_all_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _verify_fd(descriptor: int, identity: Mapping[str, Any], *, description: str) -> bytes:
    metadata = os.fstat(descriptor)
    data = _read_all_fd(descriptor)
    observed = hashlib.sha256(data).hexdigest()
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or metadata.st_size != int(identity["bytes"])
        or len(data) != int(identity["bytes"])
        or observed != str(identity["sha256"])
    ):
        raise FullFitLaunchError(f"{description} identity drift")
    return data


def _read_verified_identity(
    root_fd: int, identity_value: Any, *, description: str,
) -> bytes:
    identity = _validate_identity(identity_value, description=description)
    descriptor = _open_file_at(root_fd, Path(identity["path"]))
    try:
        return _verify_fd(descriptor, identity, description=description)
    finally:
        os.close(descriptor)


def _file_identity_at(root_fd: int, relative: Path) -> dict[str, Any]:
    descriptor = _open_file_at(root_fd, relative)
    try:
        data = _read_all_fd(descriptor)
        return {
            "path": relative.as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    finally:
        os.close(descriptor)


@contextmanager
def _open_verified_stream(
    root_fd: int, identity_value: Any, *, description: str,
) -> Iterator[BinaryIO]:
    identity = _validate_identity(identity_value, description=description)
    descriptor = _open_file_at(root_fd, Path(identity["path"]))
    try:
        _verify_fd(descriptor, identity, description=description)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            yield stream
    finally:
        os.close(descriptor)


def _read_config(output_fd: int) -> dict[str, Any]:
    descriptor = _open_file_at(output_fd, CONFIG_PATH)
    try:
        data = _read_all_fd(descriptor)
    finally:
        os.close(descriptor)
    return _load_json_bytes(data, description="TMLR V9B1 config")


def _expected_job_rows() -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "seed": seed,
            "epochs": EXPECTED_EPOCHS[family],
            "bundle_role": EXPECTED_BUNDLES[family],
            "gpu": EXPECTED_GPU[family],
        }
        for family in EXPECTED_FAMILIES
        for seed in EXPECTED_SEEDS
    ]


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "rail3.tmlr-v9b1-fullfit.v1"
        or config.get("protocol_id") != "TMLR-V9B1-FULLFIT-V1"
        or config.get("status")
        != "TMLR_V9B1_FULLFIT_PROTOCOL_FROZEN_BEFORE_TRAINING"
        or config.get("result_blind") is not True
        or config.get("formal_training_requires_committed_clean_implementation") is not True
        or config.get("formal_training_requires_explicit_cli_ack") != FORMAL_ACK
    ):
        raise FullFitLaunchError("TMLR V9B1 top-level config contract drift")
    roots = config.get("dual_root_contract")
    if not isinstance(roots, Mapping) or (
        roots.get("input_root") != str(EXPECTED_INPUT_ROOT)
        or roots.get("output_root") != str(EXPECTED_OUTPUT_ROOT)
        or roots.get("input_mode") != "READ_ONLY_EXACT_ALLOWLIST"
        or roots.get("output_mode") != "NEW_FILES_TRANSACTIONAL_NO_REPLACE"
        or roots.get("roots_must_be_distinct") is not True
        or roots.get("unknown_input_path") != "DENY"
    ):
        raise FullFitLaunchError("TMLR V9B1 dual-root contract drift")
    jobs = config.get("jobs")
    if jobs != _expected_job_rows() or len(jobs) != 18:
        raise FullFitLaunchError("TMLR V9B1 exact 18-job grid drift")
    population = config.get("fullfit_population")
    if not isinstance(population, Mapping) or (
        population.get("scale") != "S1364"
        or population.get("groups") != 1_364
        or population.get("states") != 27_280
        or population.get("states_per_group") != 20
        or population.get("atoms_by_bundle_role") != {
            "F0_P": 126_377,
            "RECT_FULL_RASTER_P": 126_377,
            "UNION_FULL_RASTER_P": 53_928,
        }
        or population.get("sorted_group_ids_canonical_json_newline_sha256")
        != "a420f6e744f4984a60b91c9db22347456e57e0957601119a330cc52c3777ee1d"
    ):
        raise FullFitLaunchError("TMLR V9B1 frozen S1364 population drift")
    if config.get("runtime_environment") != EXPECTED_RUNTIME_ENVIRONMENT:
        raise FullFitLaunchError("TMLR V9B1 runtime environment contract drift")
    if config.get("gpu_workers") != {
        "0": {
            "physical_uuid": "GPU-0c4fc6e5-d153-8780-ed6c-4c612f8e3de6",
            "families_in_order": ["R0_SMALL_P", "RECT_P", "R3_P"],
        },
        "1": {
            "physical_uuid": "GPU-18a76220-b743-4a44-dcc1-14d69b505593",
            "families_in_order": ["R0_CM_P", "R1_P", "UNION_P"],
        },
        "seeds_in_order": [13, 37, 71],
        "visible_devices_per_worker": 1,
        "concurrent_jobs_per_gpu_max": 1,
        "cublas_workspace_config": ":4096:8",
        "minimum_free_vram_bytes": 8_589_934_592,
        "watchdog_seconds": 900,
    }:
        raise FullFitLaunchError("TMLR V9B1 GPU worker contract drift")
    epoch_rule = config.get("epoch_rule")
    if not isinstance(epoch_rule, Mapping) or (
        epoch_rule.get("id") != "V9_FULLFIT_EPOCH_RULE_V1"
        or epoch_rule.get("epochs") != EXPECTED_EPOCHS
        or epoch_rule.get("same_rule_for_all_families") is not True
        or epoch_rule.get("family_specific_manual_adjustment") is not False
    ):
        raise FullFitLaunchError("TMLR V9B1 fixed-epoch rule drift")
    bundles = config.get("bundle_input_allowlist")
    if not isinstance(bundles, Mapping) or set(bundles) != {
        "F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P",
    }:
        raise FullFitLaunchError("TMLR V9B1 bundle role allowlist drift")
    if hashlib.sha256(_canonical_json_bytes(bundles)).hexdigest() != (
        EXPECTED_BUNDLE_ALLOWLIST_SHA256
    ):
        raise FullFitLaunchError("TMLR V9B1 exact bundle identity allowlist drift")
    expected_parts = {
        "feature_manifest", "feature_array", "target_manifest", "target_array",
    }
    input_paths: list[str] = []
    for role, parts in bundles.items():
        if not isinstance(parts, Mapping) or set(parts) != expected_parts:
            raise FullFitLaunchError(f"TMLR V9B1 {role} part allowlist drift")
        for name, value in parts.items():
            identity = _validate_identity(value, description=f"{role}:{name}")
            input_paths.append(identity["path"])
    if len(input_paths) != 12 or len(set(input_paths)) != 12:
        raise FullFitLaunchError("TMLR V9B1 exact 12-file bundle allowlist drift")
    sources = config.get("frozen_scientific_source_allowlist")
    if not isinstance(sources, Mapping) or set(sources) != EXPECTED_SOURCE_PATHS:
        raise FullFitLaunchError("TMLR V9B1 scientific source allowlist drift")
    if hashlib.sha256(_canonical_json_bytes(sources)).hexdigest() != (
        EXPECTED_SOURCE_ALLOWLIST_SHA256
    ):
        raise FullFitLaunchError("TMLR V9B1 exact source identity allowlist drift")
    for path, value in sources.items():
        identity = _validate_identity(
            {"path": path, **dict(value)}, description=f"source:{path}",
        )
        if identity["path"] != path:
            raise FullFitLaunchError("TMLR V9B1 scientific source path drift")
    if config.get("zero_counters") != EXPECTED_ZERO_COUNTERS:
        raise FullFitLaunchError("TMLR V9B1 zero-counter contract drift")
    if tuple(config.get("forbidden_input_path_fragments", ())) != (
        EXPECTED_FORBIDDEN_PATH_FRAGMENTS
    ):
        raise FullFitLaunchError("TMLR V9B1 forbidden-path contract drift")
    if config.get("training_contract") != {
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "scheduler": None,
        "amp": False,
        "ddp": False,
        "early_stopping": False,
        "warm_start": False,
        "checkpoint_epoch": "FINAL_FIXED_EPOCH",
        "scaler_fit": (
            "ALL_S1364_FIT_ROWS_FLOAT64_MEAN_STD_"
            "SCALE_LT_1E_8_TO_ONE_CAST_FLOAT32"
        ),
        "first_five_objective": "STATE_HUBER_DELTA_0_05",
        "r3_objective": "FROZEN_R3_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK",
        "r3_cost_input": False,
        "r3_internal_type_placeholder": "ZEROS_LENGTH_5_NOT_READ_NOT_SAVED",
        "v6_checkpoint_read": False,
        "optimizer_state_saved": False,
    }:
        raise FullFitLaunchError("TMLR V9B1 training recipe drift")
    if config.get("ensemble_contract") != {
        "id": "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION",
        "seed_order": [13, 37, 71],
        "per_seed_surface_dtype_before_stack": "float64",
        "aggregation": "arithmetic_mean_axis_0",
        "selection_after_mean": True,
        "model_weight_averaging": False,
        "voting": False,
        "cross_family_ensemble": False,
    }:
        raise FullFitLaunchError("TMLR V9B1 ensemble contract drift")
    if config.get("frozen_v6_protocol_projection") != {
        "capacity_match": {
            "global_input_dimension": 57,
            "target_parameter_count": 103_521,
            "search_domain": [1, 4_096],
        },
        "models": {
            "first_batch": {
                "R0_SMALL_P": {
                    "bundle": "F0_P", "kind": "global_small",
                    "target": "absolute residual", "parameters": 7_809,
                },
                "R0_CM_P": {
                    "bundle": "F0_P", "kind": "global_capacity_matched",
                    "target": "absolute residual", "parameters": 103_369,
                },
                "R1_P": {
                    "bundle": "F0_P", "kind": "shared_atomic",
                    "target": "absolute residual", "parameters": 103_521,
                },
                "RECT_P": {
                    "bundle": "RECT_FULL_RASTER_P", "kind": "shared_atomic",
                    "target": "absolute residual", "parameters": 103_521,
                },
                "UNION_P": {
                    "bundle": "UNION_FULL_RASTER_P", "kind": "shared_atomic",
                    "target": "absolute residual", "parameters": 103_521,
                },
            },
        },
    }:
        raise FullFitLaunchError("TMLR V9B1 frozen V6 model projection drift")
    output = config.get("output_contract")
    if not isinstance(output, Mapping) or (
        output.get("run_root") != RUN_ROOT.as_posix()
        or output.get("run_path_template") != "{family}/seed_{seed}"
        or output.get("exact_run_files") != ["checkpoint.pt", "run-manifest.json"]
        or output.get("staging_suffix") != ".incomplete"
        or output.get("publication") != "ATOMIC_NOREPLACE"
        or output.get("stale_state_policy")
        != "FAIL_CLOSED_NO_AUTOMATIC_DELETE"
        or output.get("predictions_written") is not False
        or output.get("optimizer_state_written") is not False
        or output.get("final_registry") != REGISTRY_PATH.as_posix()
    ):
        raise FullFitLaunchError("TMLR V9B1 output contract drift")


def _validate_authorities(output_fd: int, config: Mapping[str, Any]) -> None:
    authorities = config.get("recovery_authorities")
    if not isinstance(authorities, Mapping) or set(authorities) != {
        "safe_summary", "epoch_lock", "epoch_source_registry", "readiness",
    }:
        raise FullFitLaunchError("TMLR V9B1 recovery authority set drift")
    if hashlib.sha256(_canonical_json_bytes(authorities)).hexdigest() != (
        EXPECTED_AUTHORITY_ALLOWLIST_SHA256
    ):
        raise FullFitLaunchError("TMLR V9B1 exact recovery authority identity drift")
    payloads: dict[str, dict[str, Any]] = {}
    for name, value in authorities.items():
        if not isinstance(value, Mapping) or "required_status" not in value:
            raise FullFitLaunchError(f"TMLR V9B1 {name} authority schema drift")
        payloads[name] = _load_json_bytes(
            _read_verified_identity(output_fd, value, description=f"authority:{name}"),
            description=f"authority:{name}",
        )
        if payloads[name].get("status") != value["required_status"]:
            raise FullFitLaunchError(f"TMLR V9B1 {name} authority status drift")
    safe = payloads["safe_summary"]
    readiness = payloads["readiness"]
    epoch_lock = payloads["epoch_lock"]
    if (
        safe.get("terminal_state") != "TMLR_V9B0_R2_RECOVERY_LOCK_READY"
        or safe.get("performance_results") != 0
        or safe.get("model_predictions") != 0
        or safe.get("training_jobs") != 0
        or safe.get("coco_per_image_gt_materialized") != 0
        or readiness.get("performance_results") != 0
        or readiness.get("model_predictions") != 0
        or readiness.get("training_jobs") != 0
        or readiness.get("automatic_v9b1_transition_allowed") is not False
        or epoch_lock.get("rule_id") != "V9_FULLFIT_EPOCH_RULE_V1"
        or epoch_lock.get("status") != "V9_FULLFIT_EPOCH_LOCK_PASS"
    ):
        raise FullFitLaunchError("TMLR V9B1 recovery readiness closure drift")
    rows = epoch_lock.get("seed_specific_checkpoint_epochs")
    observed = [
        {"family": row.get("family"), "seed": row.get("seed"), "epoch": row.get("epoch")}
        for row in rows
    ] if isinstance(rows, list) else None
    expected = [
        {"family": family, "seed": seed, "epoch": EXPECTED_EPOCHS[family]}
        for family in EXPECTED_FAMILIES for seed in EXPECTED_SEEDS
    ]
    if observed != expected:
        raise FullFitLaunchError("TMLR V9B1 epoch-lock 18-job projection drift")


def _validate_allowlisted_inputs(
    input_fd: int, output_fd: int, config: Mapping[str, Any],
) -> None:
    forbidden = tuple(str(value).lower() for value in config["forbidden_input_path_fragments"])
    allowed_paths: list[str] = []
    for parts in config["bundle_input_allowlist"].values():
        for identity in parts.values():
            normalized = _validate_identity(identity, description="bundle input")
            allowed_paths.append(normalized["path"])
            if any(token in normalized["path"].lower() for token in forbidden):
                raise FullFitLaunchError("TMLR V9B1 allowlisted input is forbidden")
            _read_verified_identity(input_fd, identity, description="bundle input")
    if len(allowed_paths) != 12 or len(set(allowed_paths)) != 12:
        raise FullFitLaunchError("TMLR V9B1 input allowlist cardinality drift")
    for path, value in config["frozen_scientific_source_allowlist"].items():
        identity = {"path": path, **dict(value)}
        _read_verified_identity(input_fd, identity, description=f"V6 source:{path}")
        _read_verified_identity(output_fd, identity, description=f"cleanroom source:{path}")


def preflight(input_root: Path, output_root: Path) -> dict[str, Any]:
    if input_root.absolute() == output_root.absolute():
        raise FullFitLaunchError("TMLR V9B1 input and output roots must be distinct")
    input_fd = _open_root(input_root, expected=EXPECTED_INPUT_ROOT, description="input root")
    output_fd = _open_root(output_root, expected=EXPECTED_OUTPUT_ROOT, description="output root")
    try:
        config = _read_config(output_fd)
        _validate_config(config)
        _validate_authorities(output_fd, config)
        _validate_allowlisted_inputs(input_fd, output_fd, config)
        return config
    finally:
        os.close(input_fd)
        os.close(output_fd)


def _implementation_identity(output_root: Path) -> tuple[str, dict[str, str]]:
    paths = [path.as_posix() for path in IMPLEMENTATION_PATHS]
    for path in paths:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path], cwd=output_root,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise FullFitLaunchError(f"TMLR V9B1 implementation is not committed: {path}")
    status_text = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=output_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status_text:
        raise FullFitLaunchError("TMLR V9B1 implementation paths are not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=output_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        data = (output_root / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"],
            cwd=output_root, check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != digest:
            raise FullFitLaunchError(
                f"TMLR V9B1 implementation/commit drift: {relative}"
            )
        hashes[relative.as_posix()] = digest
    return commit, hashes


def _assert_clean_worktree(output_root: Path) -> None:
    descriptor = _open_root(
        output_root, expected=EXPECTED_OUTPUT_ROOT, description="output root",
    )
    os.close(descriptor)
    status_text = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=output_root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status_text:
        raise FullFitLaunchError(
            "TMLR V9B1 formal worker/finalize requires a clean worktree"
        )


def _prepare_cleanroom_import_path(output_root: Path) -> None:
    if any(name == "rail3" or name.startswith("rail3.") for name in sys.modules):
        raise FullFitLaunchError("TMLR V9B1 rail3 was imported before admission")
    source_root = output_root / "src"
    if source_root.is_symlink() or source_root.resolve() != source_root.absolute():
        raise FullFitLaunchError("TMLR V9B1 cleanroom src root drift")
    sys.path.insert(0, str(source_root))


def _assert_import_origins(
    output_root: Path, config: Mapping[str, Any], fullfit_module: Any,
    implementation_hashes: Mapping[str, str],
) -> None:
    expected_fullfit = (output_root / "src/rail3/models/tmlr_v9b1/fullfit.py").resolve()
    package = sys.modules.get("rail3.models.tmlr_v9b1")
    expected_package = (output_root / "src/rail3/models/tmlr_v9b1/__init__.py").resolve()
    if (
        Path(str(fullfit_module.__file__)).resolve() != expected_fullfit
        or package is None
        or Path(str(getattr(package, "__file__", ""))).resolve() != expected_package
        or hashlib.sha256(expected_fullfit.read_bytes()).hexdigest()
        != implementation_hashes.get("src/rail3/models/tmlr_v9b1/fullfit.py")
        or hashlib.sha256(expected_package.read_bytes()).hexdigest()
        != implementation_hashes.get("src/rail3/models/tmlr_v9b1/__init__.py")
    ):
        raise FullFitLaunchError("TMLR V9B1 fullfit imported outside cleanroom src")
    for relative, identity in config["frozen_scientific_source_allowlist"].items():
        module_name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = sys.modules.get(module_name)
        expected = (output_root / relative).resolve()
        if module is None or Path(str(getattr(module, "__file__", ""))).resolve() != expected:
            raise FullFitLaunchError(
                f"TMLR V9B1 scientific module origin drift: {module_name}"
            )
        if hashlib.sha256(expected.read_bytes()).hexdigest() != identity["sha256"]:
            raise FullFitLaunchError(
                f"TMLR V9B1 imported scientific module identity drift: {module_name}"
            )


class _GpuLease:
    def __init__(self, descriptor: int, kernel_socket: socket.socket) -> None:
        self.descriptor = descriptor
        self.kernel_socket = kernel_socket


@contextmanager
def _exclusive_gpu_lease(
    output_root: Path, physical_gpu: str, physical_uuid: str,
) -> Iterator[_GpuLease]:
    if not physical_uuid.startswith("GPU-") or any(
        character not in "0123456789abcdef-" for character in physical_uuid[4:].lower()
    ):
        raise FullFitLaunchError("TMLR V9B1 physical GPU UUID is malformed")
    root_fd = _open_root(
        output_root, expected=EXPECTED_OUTPUT_ROOT, description="output root",
    )
    lock_fd = _open_directory_at(root_fd, GPU_LOCK_ROOT, create=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        f"{physical_uuid}.lock", flags, 0o600, dir_fd=lock_fd,
    )
    kernel_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    address = f"\0rail3-v9b1-{physical_uuid}"
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FullFitLaunchError("TMLR V9B1 GPU lock is not regular single-link")
        try:
            kernel_socket.bind(address)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise FullFitLaunchError(
                f"TMLR V9B1 physical GPU {physical_gpu} already has a worker"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield _GpuLease(descriptor, kernel_socket)
    finally:
        os.close(descriptor)
        os.close(lock_fd)
        os.close(root_fd)
        kernel_socket.close()


def _assert_cuda_assignment(config: Mapping[str, Any], physical_gpu: str) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise FullFitLaunchError("TMLR V9B1 deterministic cuBLAS environment drift")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu:
        raise FullFitLaunchError("TMLR V9B1 physical GPU assignment drift")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    observed: dict[str, str] = {}
    for line in query:
        index, separator, uuid = line.partition(",")
        if not separator:
            raise FullFitLaunchError("TMLR V9B1 GPU identity query drift")
        observed[index.strip()] = uuid.strip()
    expected_uuid = config["gpu_workers"][physical_gpu]["physical_uuid"]
    if observed.get(physical_gpu) != expected_uuid:
        raise FullFitLaunchError("TMLR V9B1 physical GPU UUID drift")


def _assert_runtime_environment(
    config: Mapping[str, Any], *, numpy_module: Any, torch_module: Any,
) -> None:
    expected = config["runtime_environment"]
    observed = {
        "interpreter": sys.executable,
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": str(numpy_module.__version__),
        "torch": str(torch_module.__version__),
        "cuda_runtime": str(torch_module.version.cuda),
    }
    if observed != expected:
        raise FullFitLaunchError(
            f"TMLR V9B1 runtime environment drift: {observed}"
        )


def _rename_noreplace_at(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise FullFitLaunchError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FullFitLaunchError("TMLR V9B1 no-replace destination already exists")
        raise OSError(code, os.strerror(code))


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _write_bytes_exclusive(directory_fd: int, name: str, data: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FullFitLaunchError("TMLR V9B1 output is not regular single-link")
    finally:
        os.close(descriptor)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _checkpoint_to_exclusive_file(
    directory_fd: int, name: str, payload: Mapping[str, Any], torch_module: Any,
) -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
    try:
        with os.fdopen(os.dup(descriptor), "wb", closefd=True) as stream:
            torch_module.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        data = _read_all_fd(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FullFitLaunchError("TMLR V9B1 checkpoint is not regular single-link")
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    finally:
        os.close(descriptor)


def _run_parent_fd(output_fd: int, family: str) -> int:
    if family not in EXPECTED_FAMILIES:
        raise FullFitLaunchError("unregistered TMLR V9B1 output family")
    return _open_directory_at(output_fd, RUN_ROOT / family, create=True)


def _publish_one_run(
    *, output_fd: int, family: str, seed: int,
    checkpoint_payload: Mapping[str, Any], manifest_base: Mapping[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    parent_fd = _run_parent_fd(output_fd, family)
    final_name = f"seed_{seed}"
    stage_name = f"{final_name}.incomplete"
    claim_name = f"{final_name}.claim"
    try:
        names = set(os.listdir(parent_fd))
        if final_name in names:
            raise FullFitLaunchError("TMLR V9B1 final run already exists")
        if stage_name in names or claim_name in names:
            raise FullFitLaunchError("TMLR V9B1 stale transaction requires human resolution")
        try:
            os.mkdir(claim_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FullFitLaunchError("TMLR V9B1 job already has a claimant") from error
        os.fsync(parent_fd)
        os.mkdir(stage_name, mode=0o755, dir_fd=parent_fd)
        stage_fd = _open_directory_at(parent_fd, Path(stage_name))
        try:
            checkpoint_identity = _checkpoint_to_exclusive_file(
                stage_fd, "checkpoint.pt", checkpoint_payload, torch_module,
            )
            checkpoint_identity["path"] = (
                RUN_ROOT / family / final_name / "checkpoint.pt"
            ).as_posix()
            manifest = {
                **dict(manifest_base),
                "checkpoint": checkpoint_identity,
                "terminal_state": "TMLR_V9B1_FULLFIT_JOB_PASS",
            }
            manifest_data = _canonical_json_bytes(manifest) + b"\n"
            _write_bytes_exclusive(stage_fd, "run-manifest.json", manifest_data)
            if set(os.listdir(stage_fd)) != {"checkpoint.pt", "run-manifest.json"}:
                raise FullFitLaunchError("TMLR V9B1 staged run file set drift")
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        _rename_noreplace_at(parent_fd, stage_name, final_name)
        os.fsync(parent_fd)
        os.rmdir(claim_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return manifest
    finally:
        # On failure, deliberately retain claim/staging state as forensic
        # evidence.  The launcher never guesses whether it is safe to delete.
        os.close(parent_fd)


def _run_one_job(
    *, input_fd: int, output_fd: int, config: Mapping[str, Any],
    job_row: Mapping[str, Any], implementation_commit: str,
    implementation_hashes: Mapping[str, str], device: Any,
    torch_module: Any, fullfit_module: Any,
) -> dict[str, Any]:
    existing = _existing_run_or_none(
        output_fd=output_fd,
        family=str(job_row["family"]),
        seed=int(job_row["seed"]),
        config=config,
        implementation_commit=implementation_commit,
        implementation_hashes=implementation_hashes,
    )
    if existing is not None:
        return existing
    role = str(job_row["bundle_role"])
    parts = config["bundle_input_allowlist"][role]
    # Manifests are identity-only authorities.  Their content is not parsed;
    # the same pinned NPZ descriptors are hashed and then consumed by np.load.
    for name in ("feature_manifest", "target_manifest"):
        _read_verified_identity(input_fd, parts[name], description=f"{role}:{name}")
    with (
        _open_verified_stream(
            input_fd, parts["feature_array"], description=f"{role}:feature_array",
        ) as feature_stream,
        _open_verified_stream(
            input_fd, parts["target_array"], description=f"{role}:target_array",
        ) as target_stream,
    ):
        bundle = fullfit_module.load_fullfit_bundle(
            role_id=role,
            feature_stream=feature_stream,
            target_stream=target_stream,
        )
    job = fullfit_module.FullFitJob(
        family=str(job_row["family"]), seed=int(job_row["seed"]),
        epochs=int(job_row["epochs"]), bundle_role=role,
    )
    started = time.monotonic()
    torch_module.cuda.reset_peak_memory_stats(0)
    result = fullfit_module.train_fullfit_job(
        job=job,
        bundle=bundle,
        protocol=config["frozen_v6_protocol_projection"],
        device=device,
    )
    torch_module.cuda.synchronize(0)
    wall_seconds = time.monotonic() - started
    peak_vram = int(torch_module.cuda.max_memory_allocated(0))
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds <= 0.0
        or wall_seconds > float(config["gpu_workers"]["watchdog_seconds"])
        or peak_vram < 0
    ):
        raise FullFitLaunchError("TMLR V9B1 job exceeded its resource contract")
    checkpoint = fullfit_module.checkpoint_payload(
        result,
        epoch_lock_sha256=config["recovery_authorities"]["epoch_lock"]["sha256"],
        bundle_identities=parts,
        implementation_commit=implementation_commit,
        implementation_sha256=implementation_hashes,
        config_sha256=implementation_hashes[CONFIG_PATH.as_posix()],
    )
    manifest_base = {
        "schema_version": "rail3.tmlr-v9b1-fullfit-run.v1",
        "run_id": job.run_id,
        "family": job.family,
        "seed": job.seed,
        "bundle_role": job.bundle_role,
        "fullfit_epoch": job.epochs,
        "epochs_completed": result.epochs_completed,
        "epoch_rule_id": fullfit_module.EPOCH_RULE_ID,
        "epoch_lock_sha256": config["recovery_authorities"]["epoch_lock"]["sha256"],
        "config_sha256": implementation_hashes[CONFIG_PATH.as_posix()],
        "model_class": result.model_class,
        "parameter_count": result.parameter_count,
        "objective_id": result.objective_id,
        "optimizer_recipe": dict(fullfit_module.OPTIMIZER_RECIPE),
        "prediction_semantics": fullfit_module.PREDICTION_SEMANTICS,
        "ensemble_id": fullfit_module.ENSEMBLE_ID,
        "implementation_commit": implementation_commit,
        "implementation_sha256": dict(implementation_hashes),
        "bundle_identities": parts,
        "training_loss_finite": result.training_loss_finite,
        "wall_seconds": wall_seconds,
        "peak_vram_bytes": peak_vram,
        "physical_gpu": str(job_row["gpu"]),
        "zero_counters": dict(EXPECTED_ZERO_COUNTERS),
        "created_utc": _utc_now(),
    }
    _publish_one_run(
        output_fd=output_fd, family=job.family, seed=job.seed,
        checkpoint_payload=checkpoint, manifest_base=manifest_base,
        torch_module=torch_module,
    )
    published, _ = _verify_run(
        output_fd, job.family, job.seed, config=config,
        implementation_commit=implementation_commit,
        implementation_hashes=implementation_hashes,
    )
    return published


def run_worker(
    *, input_root: Path, output_root: Path, physical_gpu: str,
    formal_ack: str,
) -> list[dict[str, Any]]:
    if formal_ack != FORMAL_ACK:
        raise FullFitLaunchError("formal training acknowledgement is absent")
    _assert_clean_worktree(output_root)
    config = preflight(input_root, output_root)
    implementation_commit, implementation_hashes = _implementation_identity(output_root)
    _assert_clean_worktree(output_root)
    if physical_gpu not in {"0", "1"}:
        raise FullFitLaunchError("unregistered TMLR V9B1 physical GPU")
    _assert_cuda_assignment(config, physical_gpu)
    # Heavy libraries and the training module are imported only after every
    # path, authority, implementation, environment, and explicit-ack gate.
    _prepare_cleanroom_import_path(output_root)
    import numpy
    import torch
    from rail3.models.tmlr_v9b1 import fullfit

    _assert_runtime_environment(config, numpy_module=numpy, torch_module=torch)
    _assert_import_origins(
        output_root, config, fullfit, implementation_hashes,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise FullFitLaunchError("TMLR V9B1 worker requires one visible CUDA device")
    device = torch.device("cuda:0")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < int(config["gpu_workers"]["minimum_free_vram_bytes"]):
        raise FullFitLaunchError("TMLR V9B1 GPU has insufficient free VRAM")
    rows = [row for row in config["jobs"] if row["gpu"] == physical_gpu]
    input_fd = _open_root(input_root, expected=EXPECTED_INPUT_ROOT, description="input root")
    output_fd = _open_root(output_root, expected=EXPECTED_OUTPUT_ROOT, description="output root")
    try:
        physical_uuid = config["gpu_workers"][physical_gpu]["physical_uuid"]
        with _exclusive_gpu_lease(output_root, physical_gpu, physical_uuid):
            return [
                _run_one_job(
                    input_fd=input_fd, output_fd=output_fd, config=config,
                    job_row=row, implementation_commit=implementation_commit,
                    implementation_hashes=implementation_hashes,
                    device=device, torch_module=torch, fullfit_module=fullfit,
                )
                for row in rows
            ]
    finally:
        os.close(input_fd)
        os.close(output_fd)


def _verify_run(
    output_fd: int, family: str, seed: int, *, config: Mapping[str, Any],
    implementation_commit: str, implementation_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = RUN_ROOT / family / f"seed_{seed}"
    run_fd = _open_directory_at(output_fd, relative)
    try:
        if set(os.listdir(run_fd)) != {"checkpoint.pt", "run-manifest.json"}:
            raise FullFitLaunchError("TMLR V9B1 completed run file set drift")
        for name in ("checkpoint.pt", "run-manifest.json"):
            descriptor = os.open(
                name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=run_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise FullFitLaunchError("TMLR V9B1 completed artifact link drift")
                if name == "run-manifest.json":
                    manifest_data = _read_all_fd(descriptor)
                    manifest = _load_json_bytes(
                        manifest_data, description="run manifest",
                    )
                    run_manifest_identity = {
                        "path": (relative / "run-manifest.json").as_posix(),
                        "bytes": len(manifest_data),
                        "sha256": hashlib.sha256(manifest_data).hexdigest(),
                    }
                else:
                    checkpoint_identity = {
                        "bytes": metadata.st_size,
                        "sha256": hashlib.sha256(_read_all_fd(descriptor)).hexdigest(),
                    }
            finally:
                os.close(descriptor)
    finally:
        os.close(run_fd)
    expected = {
        "schema_version": "rail3.tmlr-v9b1-fullfit-run.v1",
        "run_id": (
            f"S1364__{family}__seed_{seed}__epoch_{EXPECTED_EPOCHS[family]}"
        ),
        "family": family,
        "seed": seed,
        "fullfit_epoch": EXPECTED_EPOCHS[family],
        "epochs_completed": EXPECTED_EPOCHS[family],
        "bundle_role": EXPECTED_BUNDLES[family],
        "terminal_state": "TMLR_V9B1_FULLFIT_JOB_PASS",
        "zero_counters": EXPECTED_ZERO_COUNTERS,
        "epoch_rule_id": "V9_FULLFIT_EPOCH_RULE_V1",
        "epoch_lock_sha256": config["recovery_authorities"]["epoch_lock"]["sha256"],
        "config_sha256": implementation_hashes[CONFIG_PATH.as_posix()],
        "ensemble_id": config["ensemble_contract"]["id"],
        "implementation_commit": implementation_commit,
        "implementation_sha256": dict(implementation_hashes),
        "bundle_identities": config["bundle_input_allowlist"][EXPECTED_BUNDLES[family]],
        "physical_gpu": EXPECTED_GPU[family],
        "training_loss_finite": True,
        "model_class": EXPECTED_MODEL_CLASSES[family],
        "parameter_count": EXPECTED_PARAMETER_COUNTS[family],
        "objective_id": EXPECTED_OBJECTIVES[family],
        "optimizer_recipe": EXPECTED_OPTIMIZER_RECIPE,
        "prediction_semantics": "ABSOLUTE_RESIDUAL",
    }
    if set(manifest) != RUN_MANIFEST_KEYS:
        raise FullFitLaunchError("TMLR V9B1 completed run manifest key-set drift")
    drift = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if drift:
        raise FullFitLaunchError(f"TMLR V9B1 completed run identity drift: {drift}")
    if (
        not math.isfinite(float(manifest.get("wall_seconds", math.nan)))
        or float(manifest["wall_seconds"]) <= 0.0
        or float(manifest["wall_seconds"])
        > float(config["gpu_workers"]["watchdog_seconds"])
        or type(manifest.get("peak_vram_bytes")) is not int
        or int(manifest["peak_vram_bytes"]) < 0
    ):
        raise FullFitLaunchError("TMLR V9B1 completed run resource drift")
    declared = manifest.get("checkpoint")
    if not isinstance(declared, Mapping) or (
        declared.get("bytes") != checkpoint_identity["bytes"]
        or declared.get("sha256") != checkpoint_identity["sha256"]
        or declared.get("path") != (relative / "checkpoint.pt").as_posix()
    ):
        raise FullFitLaunchError("TMLR V9B1 checkpoint identity drift")
    return manifest, run_manifest_identity


def _existing_run_or_none(
    *, output_fd: int, family: str, seed: int, config: Mapping[str, Any],
    implementation_commit: str, implementation_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    parent_fd = _run_parent_fd(output_fd, family)
    final_name = f"seed_{seed}"
    try:
        names = set(os.listdir(parent_fd))
        if f"{final_name}.incomplete" in names or f"{final_name}.claim" in names:
            raise FullFitLaunchError(
                "TMLR V9B1 stale transaction requires human resolution"
            )
        exists = final_name in names
    finally:
        os.close(parent_fd)
    if not exists:
        return None
    manifest, _ = _verify_run(
        output_fd, family, seed, config=config,
        implementation_commit=implementation_commit,
        implementation_hashes=implementation_hashes,
    )
    return manifest


def _assert_exact_run_tree(output_fd: int) -> None:
    root_fd = _open_directory_at(output_fd, RUN_ROOT)
    try:
        if set(os.listdir(root_fd)) != set(EXPECTED_FAMILIES):
            raise FullFitLaunchError("TMLR V9B1 run-root family set drift")
        for family in EXPECTED_FAMILIES:
            family_fd = _open_directory_at(root_fd, Path(family))
            try:
                expected = {f"seed_{seed}" for seed in EXPECTED_SEEDS}
                if set(os.listdir(family_fd)) != expected:
                    raise FullFitLaunchError(
                        f"TMLR V9B1 {family} exact seed-directory set drift"
                    )
            finally:
                os.close(family_fd)
    finally:
        os.close(root_fd)


def _publish_registry(output_fd: int, registry: Mapping[str, Any]) -> None:
    parent_fd = _open_directory_at(output_fd, REGISTRY_PATH.parent, create=True)
    temporary = f".{REGISTRY_PATH.name}.incomplete"
    try:
        names = set(os.listdir(parent_fd))
        if REGISTRY_PATH.name in names or temporary in names:
            raise FullFitLaunchError("TMLR V9B1 registry destination already exists")
        _write_bytes_exclusive(
            parent_fd, temporary, _canonical_json_bytes(registry) + b"\n",
        )
        _rename_noreplace_at(parent_fd, temporary, REGISTRY_PATH.name)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _existing_registry_or_none(output_fd: int) -> dict[str, Any] | None:
    parent_fd = _open_directory_at(output_fd, REGISTRY_PATH.parent, create=True)
    try:
        names = set(os.listdir(parent_fd))
        temporary = f".{REGISTRY_PATH.name}.incomplete"
        if temporary in names:
            raise FullFitLaunchError(
                "TMLR V9B1 stale registry transaction requires human resolution"
            )
        if REGISTRY_PATH.name not in names:
            return None
        descriptor = os.open(
            REGISTRY_PATH.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise FullFitLaunchError(
                    "TMLR V9B1 existing registry is not regular single-link"
                )
            return _load_json_bytes(
                _read_all_fd(descriptor), description="checkpoint registry",
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _checkpoint_lock_payload(
    *, config: Mapping[str, Any], config_identity: Mapping[str, Any],
    implementation_commit: str, implementation_hashes: Mapping[str, str],
    verified_runs: list[tuple[dict[str, Any], dict[str, Any]]],
    created_utc: str,
) -> dict[str, Any]:
    expected_pairs = [
        (family, seed) for family in EXPECTED_FAMILIES for seed in EXPECTED_SEEDS
    ]
    observed_pairs = [
        (report.get("family"), report.get("seed")) for report, _ in verified_runs
    ]
    if observed_pairs != expected_pairs or len(verified_runs) != 18:
        raise FullFitLaunchError("TMLR V9B1 checkpoint lock requires exact 18 jobs")
    normalized_config = _validate_identity(
        config_identity, description="TMLR V9B1 checkpoint-lock config",
    )
    if (
        normalized_config["path"] != CONFIG_PATH.as_posix()
        or normalized_config["sha256"]
        != implementation_hashes.get(CONFIG_PATH.as_posix())
    ):
        raise FullFitLaunchError("TMLR V9B1 checkpoint lock config identity drift")
    for report, run_manifest_identity in verified_runs:
        family = str(report["family"])
        seed = int(report["seed"])
        run_root = RUN_ROOT / family / f"seed_{seed}"
        normalized_manifest = _validate_identity(
            run_manifest_identity, description="TMLR V9B1 run manifest",
        )
        normalized_checkpoint = _validate_identity(
            report.get("checkpoint"), description="TMLR V9B1 checkpoint",
        )
        if (
            normalized_manifest["path"]
            != (run_root / "run-manifest.json").as_posix()
            or normalized_checkpoint["path"]
            != (run_root / "checkpoint.pt").as_posix()
        ):
            raise FullFitLaunchError("TMLR V9B1 checkpoint-lock run path drift")
    return {
        "schema_version": "rail3.tmlr-v9b1-fullfit-checkpoint-lock.v1",
        "status": "TMLR_V9B1_FULLFIT_18_OF_18_PASS",
        "job_count": 18,
        "family_count": 6,
        "seeds": list(EXPECTED_SEEDS),
        "epoch_rule_id": "V9_FULLFIT_EPOCH_RULE_V1",
        "epoch_lock_sha256": config["recovery_authorities"]["epoch_lock"]["sha256"],
        "ensemble_id": config["ensemble_contract"]["id"],
        "config": normalized_config,
        "prospective_schema_id": "PROSPECTIVE_V1",
        "checkpoint_schema_version": "rail3.tmlr-v9b1-fullfit-checkpoint.v1",
        "run_manifest_schema_version": "rail3.tmlr-v9b1-fullfit-run.v1",
        "implementation_commit": implementation_commit,
        "implementation_sha256": dict(implementation_hashes),
        "runs": [
            {
                "family": report["family"],
                "seed": report["seed"],
                "fullfit_epoch": report["fullfit_epoch"],
                "epochs_completed": report["epochs_completed"],
                "bundle_role": report["bundle_role"],
                "bundle_identities": report["bundle_identities"],
                "prospective_schema_id": "PROSPECTIVE_V1",
                "parameter_count": report["parameter_count"],
                "model_class": report["model_class"],
                "objective_id": report["objective_id"],
                "config_sha256": report["config_sha256"],
                "implementation_commit": report["implementation_commit"],
                "implementation_sha256": report["implementation_sha256"],
                "checkpoint": report["checkpoint"],
                "run_manifest": run_manifest_identity,
            }
            for report, run_manifest_identity in verified_runs
        ],
        "zero_counters": dict(EXPECTED_ZERO_COUNTERS),
        "created_utc": created_utc,
        "terminal_state": "TMLR_V9B1_FULLFIT_CHECKPOINTS_READY",
    }


def finalize(input_root: Path, output_root: Path) -> dict[str, Any]:
    _assert_clean_worktree(output_root)
    config = preflight(input_root, output_root)
    implementation_commit, implementation_hashes = _implementation_identity(output_root)
    _assert_clean_worktree(output_root)
    output_fd = _open_root(output_root, expected=EXPECTED_OUTPUT_ROOT, description="output root")
    try:
        _assert_exact_run_tree(output_fd)
        verified_runs = [
            _verify_run(
                output_fd, family, seed, config=config,
                implementation_commit=implementation_commit,
                implementation_hashes=implementation_hashes,
            )
            for family in EXPECTED_FAMILIES for seed in EXPECTED_SEEDS
        ]
        config_identity = _file_identity_at(output_fd, CONFIG_PATH)
        if config_identity["sha256"] != implementation_hashes[CONFIG_PATH.as_posix()]:
            raise FullFitLaunchError("TMLR V9B1 config/implementation identity drift")
        registry = _checkpoint_lock_payload(
            config=config,
            config_identity=config_identity,
            implementation_commit=implementation_commit,
            implementation_hashes=implementation_hashes,
            verified_runs=verified_runs,
            created_utc=_utc_now(),
        )
        existing = _existing_registry_or_none(output_fd)
        if existing is not None:
            expected_without_timestamp = {
                key: value for key, value in registry.items() if key != "created_utc"
            }
            observed_without_timestamp = {
                key: value for key, value in existing.items() if key != "created_utc"
            }
            if (
                expected_without_timestamp != observed_without_timestamp
                or not isinstance(existing.get("created_utc"), str)
            ):
                raise FullFitLaunchError("TMLR V9B1 existing registry drift")
            return existing
        _publish_registry(output_fd, registry)
        return registry
    finally:
        os.close(output_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--worker", choices=("0", "1"))
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--input-root", type=Path, default=EXPECTED_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--formal-training-ack", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.worker is not None and arguments.formal_training_ack != FORMAL_ACK:
        # This gate intentionally precedes all filesystem and heavy-library work.
        raise FullFitLaunchError("formal training acknowledgement is absent")
    if arguments.check:
        config = preflight(arguments.input_root, arguments.output_root)
        print(json.dumps({
            "status": "TMLR_V9B1_FULLFIT_PREFLIGHT_PASS",
            "jobs": len(config["jobs"]),
            "training_jobs_started": 0,
        }, sort_keys=True))
        return 0
    if arguments.worker is not None:
        reports = run_worker(
            input_root=arguments.input_root,
            output_root=arguments.output_root,
            physical_gpu=arguments.worker,
            formal_ack=arguments.formal_training_ack,
        )
        print(json.dumps({"status": "PASS", "jobs": len(reports)}, sort_keys=True))
        return 0
    registry = finalize(arguments.input_root, arguments.output_root)
    print(json.dumps({"status": registry["status"], "jobs": 18}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
