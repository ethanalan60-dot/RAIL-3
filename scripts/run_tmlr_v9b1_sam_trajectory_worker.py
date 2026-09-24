#!/usr/bin/env python3
"""Run or preflight one statically assigned V9B1 label-free SAM shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from rail3.sam.coco_v9b1_worker_gate import validate_worker_environment


EXECUTION_TOKEN = "TMLR_V9B1_LABEL_FREE_REAL_SAM"


def _require_import_origin(module_name: str, relative: str) -> None:
    module = sys.modules.get(module_name)
    actual = None if module is None else Path(module.__file__).resolve()
    expected = (REPOSITORY / relative).resolve()
    if actual != expected:
        raise RuntimeError(f"V9B1_IMPORT_ORIGIN_DRIFT:{module_name}:{actual}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-real")
    return parser.parse_args()


def _create_once(path: Path, encoded: bytes, reason_code: str) -> None:
    from rail3.sam.coco_v9b1_cache import secure_create_once

    try:
        secure_create_once(path, encoded)
    except FileExistsError:
        try:
            metadata = path.lstat()
            from rail3.sam.coco_v9b1_cache import secure_read_bytes

            existing = secure_read_bytes(path)
        except OSError as exc:
            raise RuntimeError(f"{reason_code}:{path}") from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or existing != encoded:
            raise RuntimeError(f"{reason_code}:{path}")


def _failure(shard: int, stage: str, exc: Exception, *, write_ledger: bool) -> int:
    reason_code = str(getattr(exc, "reason_code", exc.args[0] if exc.args else type(exc).__name__))
    payload = {
        "schema_version": "rail3.tmlr-v9b1-label-free-failure.v1",
        "status": "TMLR_V9B1_SAM_TRAJECTORY_BLOCKED",
        "shard": shard,
        "stage": stage,
        "reason_code": reason_code,
        "message": str(exc),
        "coco_annotation_rows_materialized": 0,
        "coco_image_label_rows_materialized": 0,
        "model_predictions": 0,
        "performance_results": 0,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if write_ledger:
        destination = (
            REPOSITORY
            / "artifacts/paper/source_data/tmlr_v9b1"
            / f"coco_external_trajectory_failure_shard{shard}.json"
        )
        try:
            _create_once(destination, encoded, "V9B1_FAILURE_LEDGER_CONFLICT")
        except Exception as ledger_exc:
            payload["failure_ledger_error"] = str(ledger_exc)
            encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    sys.stderr.buffer.write(encoded)
    return 2


def _augment_inventory(
    inventory: dict[str, Any],
    *,
    shard: int,
    physical_gpu: int,
    gpu_uuid: str,
    state_bytes: bytes,
    shard_bytes: bytes,
    implementation_commit: str,
    telemetry_path: Path,
    telemetry_sha256: str,
    model_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        **inventory,
        "shard": shard,
        "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid,
        "state_manifest_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "shard_manifest_sha256": hashlib.sha256(shard_bytes).hexdigest(),
        "implementation_commit": implementation_commit,
        "run_telemetry": {
            "path": str(telemetry_path.relative_to(REPOSITORY)),
            "sha256": telemetry_sha256,
        },
        "sam_identity": model_identity,
    }


def _run(arguments: argparse.Namespace) -> int:
    worker_started = time.monotonic()
    if Path.cwd().resolve() != REPOSITORY:
        raise RuntimeError("V9B1_WORKER_CWD_DRIFT")
    _require_import_origin(
        "rail3.sam.coco_v9b1_worker_gate", "src/rail3/sam/coco_v9b1_worker_gate.py"
    )
    physical_gpu = validate_worker_environment(
        arguments.shard,
        os.environ,
        set(sys.modules),
        require_lease=True,
        query_visible_uuid=True,
    )
    failure_path = (
        REPOSITORY / "artifacts/paper/source_data/tmlr_v9b1"
        / f"coco_external_trajectory_failure_shard{arguments.shard}.json"
    )
    try:
        from rail3.sam.coco_v9b1_cache import secure_read_bytes

        secure_read_bytes(failure_path)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError(f"V9B1_UNRESOLVED_SHARD_FAILURE_LEDGER:{arguments.shard}")

    # All imports that can reach the real backend remain below GPU admission.
    from rail3.contracts import canonical_json_bytes
    from rail3.sam.coco_v9b1_cache import SecureCandidateCache, SecureTrajectoryCache
    from rail3.sam.coco_v9b1_asset_gate import validate_sam_asset_metadata_before_launch
    from rail3.sam.coco_v9b1_trajectory import (
        EXPECTED_GPU_UUIDS,
        SHARD_CACHE_ROOT,
        V9B1ContractError,
        V9B1TrajectoryDriver,
        build_semantic_states,
        derive_bundles_from_a0_cache,
        load_config,
        load_frozen_panel_images,
        load_taxonomy,
        read_stable_regular_bytes,
        validate_frozen_execution_manifests,
        validate_inventory,
        validate_run_telemetry_receipt,
        validate_e4_execution_registry,
        validate_small_authorities,
    )
    from rail3.sam.sam31_backend import official_sam31_model_spec, real_backend_config_hash

    config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, config)
    images = load_frozen_panel_images(REPOSITORY)
    states = build_semantic_states(images, load_taxonomy(REPOSITORY))
    all_technical_missingness = validate_e4_execution_registry(REPOSITORY, config)
    manifest = validate_frozen_execution_manifests(
        REPOSITORY,
        config,
        states,
        require_global_clean=arguments.preflight_only,
    )
    output = config["outputs"]
    source_root = REPOSITORY / output["source_data_root"]
    state_path = source_root / output["state_manifest"]
    shard_path = source_root / output["shard_manifest"]
    state_bytes = read_stable_regular_bytes(state_path)
    shard_bytes = read_stable_regular_bytes(shard_path)
    shard_states = tuple(state for state in states if state.shard == arguments.shard)
    shard_state_ids = {state.semantic_state_id for state in shard_states}
    technical_missingness = tuple(
        entry for entry in all_technical_missingness
        if entry.record.semantic_state_id in shard_state_ids
    )
    expected_shard = config["trajectory"]["shards"][arguments.shard]
    manifest_shard = manifest["shards"][arguments.shard]
    config_projection = {key: manifest_shard[key] for key in expected_shard}
    if not (
        expected_shard == config_projection
        and len(shard_states) == expected_shard["states"]
        and len({state.image.asset_id for state in shard_states}) == expected_shard["images"]
        and physical_gpu == expected_shard["physical_gpu"]
        and os.environ["RAIL3_PHYSICAL_GPU_UUID"] == expected_shard["gpu_uuid"]
        and expected_shard["gpu_uuid"] == EXPECTED_GPU_UUIDS[arguments.shard]
    ):
        raise V9B1ContractError("V9B1_WORKER_SHARD_GRID_DRIFT", str(arguments.shard))

    cache_root = REPOSITORY / SHARD_CACHE_ROOT / f"shard-{arguments.shard}"
    expected_roots = manifest_shard["cache_roots"]
    if expected_roots != {
        "a0": str(SHARD_CACHE_ROOT / f"shard-{arguments.shard}" / "a0"),
        "actions": str(SHARD_CACHE_ROOT / f"shard-{arguments.shard}" / "actions"),
        "plans": str(SHARD_CACHE_ROOT / f"shard-{arguments.shard}" / "plans"),
    }:
        raise V9B1ContractError("V9B1_WORKER_CACHE_ROOT_DRIFT", str(arguments.shard))
    a0_cache = SecureCandidateCache(cache_root / "a0")
    action_cache = SecureTrajectoryCache(cache_root / "actions")
    plan_root = cache_root / "plans"
    inventory_path = source_root / f"coco_external_trajectory_inventory_shard{arguments.shard}.json"
    telemetry_path = source_root / f"coco_external_trajectory_run_telemetry_shard{arguments.shard}.json"
    model_spec = official_sam31_model_spec()
    model_identity = {
        "model_spec_id": model_spec.model_spec_id,
        "model_source_commit": model_spec.source_commit,
        "checkpoint_sha256": model_spec.checkpoint_sha256,
        "sam_config_hash": real_backend_config_hash(),
    }

    try:
        frozen_inventory_bytes = read_stable_regular_bytes(inventory_path)
    except FileNotFoundError:
        frozen_inventory_bytes = None
    if frozen_inventory_bytes is not None:
        bundles = derive_bundles_from_a0_cache(
            shard_states,
            model_spec_id=model_spec.model_spec_id,
            a0_cache=a0_cache,
            technical_missingness=technical_missingness,
        )
        regenerated = validate_inventory(
            bundles,
            a0_cache=a0_cache,
            action_cache=action_cache,
            plan_lock_root=plan_root,
            expected_model_identity=model_identity,
            technical_missingness=technical_missingness,
        )
        telemetry_encoded = read_stable_regular_bytes(telemetry_path)
        try:
            frozen_telemetry = json.loads(telemetry_encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V9B1ContractError(
                "V9B1_RUN_TELEMETRY_JSON_INVALID", str(telemetry_path)
            ) from exc
        validate_run_telemetry_receipt(
            frozen_telemetry,
            shard=arguments.shard,
            gpu_uuid=expected_shard["gpu_uuid"],
            model_identity=model_identity,
        )
        regenerated = _augment_inventory(
            regenerated,
            shard=arguments.shard,
            physical_gpu=physical_gpu,
            gpu_uuid=expected_shard["gpu_uuid"],
            state_bytes=state_bytes,
            shard_bytes=shard_bytes,
            implementation_commit=manifest["implementation_commit"],
            telemetry_path=telemetry_path,
            telemetry_sha256=hashlib.sha256(telemetry_encoded).hexdigest(),
            model_identity=model_identity,
        )
        expected_encoded = canonical_json_bytes(regenerated) + b"\n"
        if frozen_inventory_bytes != expected_encoded or not regenerated["ready"]:
            raise V9B1ContractError("V9B1_COMPLETED_SHARD_RESUME_DRIFT", str(arguments.shard))
        print(json.dumps({
            "status": "V9B1_SHARD_ALREADY_COMPLETE_NO_MODEL_LOAD",
            "shard": arguments.shard,
            "semantic_records": regenerated["semantic_record_count"],
        }, sort_keys=True, separators=(",", ":")))
        return 0

    metadata = validate_sam_asset_metadata_before_launch(REPOSITORY, config)
    if arguments.preflight_only:
        print(json.dumps({
            "status": "V9B1_SHARD_PREFLIGHT_PASS",
            "shard": arguments.shard,
            "gpu_uuid": expected_shard["gpu_uuid"],
            "checkpoint_hash_recomputed": False,
            "model_builds": metadata["model_builds"],
            "sam_prompt_calls": metadata["sam_prompt_calls"],
        }, sort_keys=True, separators=(",", ":")))
        return 0

    from rail3.sam.coco_v9b1_real import RealSam31TrajectoryBackend

    with RealSam31TrajectoryBackend.build(
        repo_root=REPOSITORY, config=config, physical_gpu_index=physical_gpu
    ) as backend:
        bundles = V9B1TrajectoryDriver(
            backend=backend,
            a0_cache=a0_cache,
            action_cache=action_cache,
            plan_lock_root=plan_root,
            technical_missingness=technical_missingness,
        ).run(shard_states)
        inventory = validate_inventory(
            bundles,
            a0_cache=a0_cache,
            action_cache=action_cache,
            plan_lock_root=plan_root,
            expected_model_identity=model_identity,
            technical_missingness=technical_missingness,
        )
        execution_telemetry = {
            "schema_version": "rail3.tmlr-v9b1-label-free-run-telemetry.v1",
            "shard": arguments.shard,
            "physical_gpu": physical_gpu,
            "gpu_uuid": expected_shard["gpu_uuid"],
            "backend_counters": backend.counters(),
            "sam_identity": backend.evidence(),
            "real_backend": True,
            "coco_annotation_rows_materialized": 0,
            "coco_image_label_rows_materialized": 0,
            "model_predictions": 0,
            "performance_results": 0,
            "technical_missingness_count": len(technical_missingness),
            "technical_missingness_semantic_state_ids": sorted(
                entry.record.semantic_state_id for entry in technical_missingness
            ),
        }
    execution_telemetry["wall_seconds"] = time.monotonic() - worker_started
    telemetry_encoded = canonical_json_bytes(execution_telemetry) + b"\n"
    validate_run_telemetry_receipt(
        execution_telemetry,
        shard=arguments.shard,
        gpu_uuid=expected_shard["gpu_uuid"],
        model_identity=model_identity,
    )
    _create_once(telemetry_path, telemetry_encoded, "V9B1_RUN_TELEMETRY_CONFLICT")
    inventory = _augment_inventory(
        inventory,
        shard=arguments.shard,
        physical_gpu=physical_gpu,
        gpu_uuid=expected_shard["gpu_uuid"],
        state_bytes=state_bytes,
        shard_bytes=shard_bytes,
        implementation_commit=manifest["implementation_commit"],
        telemetry_path=telemetry_path,
        telemetry_sha256=hashlib.sha256(telemetry_encoded).hexdigest(),
        model_identity=model_identity,
    )
    if not inventory["ready"] or inventory["semantic_record_count"] != expected_shard["semantic_records"]:
        raise V9B1ContractError("V9B1_WORKER_INVENTORY_NOT_READY", str(arguments.shard))
    encoded = canonical_json_bytes(inventory) + b"\n"
    _create_once(inventory_path, encoded, "V9B1_WORKER_INVENTORY_CONFLICT")
    sys.stdout.buffer.write(encoded)
    return 0


def main() -> int:
    arguments = _arguments()
    stage = "PREFLIGHT" if arguments.preflight_only else "REAL_EXECUTION"
    try:
        if not arguments.preflight_only and arguments.execute_real != EXECUTION_TOKEN:
            raise RuntimeError("V9B1_REAL_EXECUTION_TOKEN_MISMATCH")
        return _run(arguments)
    except Exception as exc:
        return _failure(
            arguments.shard,
            stage,
            exc,
            write_ledger=not arguments.preflight_only,
        )


if __name__ == "__main__":
    raise SystemExit(main())
