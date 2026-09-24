#!/usr/bin/env python3
"""Freeze the three V9B1 pre-action feature roles from A0/A1/A2 only.

This producer has no annotation, target, evaluation, or future-action outcome
reader.  Its normal mode is create-once; ``--check`` revalidates only the
observable A0/A1/A2 cache boundary and the published feature bank.
"""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

FORMAL_TOKEN = "TMLR_V9B1_LABEL_FREE_FEATURES_AUTHORIZED"
CONFIG_PATH = Path("configs/experiments/tmlr_v9b1_coco_label_free_zeroshot.json")
INVENTORY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_feature_inventory.json"
)
TRAJECTORY_INVENTORY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_trajectory_inventory.json"
)
FEATURE_PATHS = {
    "F0_P": Path("artifacts/features/tmlr-v9b1/f0/features.npz"),
    "RECT_FULL_RASTER_P": Path("artifacts/features/tmlr-v9b1/rect/features.npz"),
    "UNION_FULL_RASTER_P": Path("artifacts/features/tmlr-v9b1/union/features.npz"),
}
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/rail3/models/tmlr_v9b1/inference.py"),
    Path("src/rail3/sam/coco_v9b1_trajectory.py"),
    Path("src/rail3/diagnostic/representations.py"),
    Path("src/rail3/regions/m06e_causal.py"),
    Path("scripts/build_tmlr_v9b1_label_free_features.py"),
)
EXPECTED_ARRAY_KEYS = frozenset({
    "atom_x", "state_x", "action_x16", "state_lengths", "state_ids",
    "trajectory_state_ids", "image_group_ids", "atom_ids",
})
INVENTORY_KEYS = frozenset({
    "schema_version", "status", "panel_lock_id", "panel_sha256",
    "state_manifest", "shard_manifest", "trajectory_inventory", "sam_identity",
    "schema_id", "observable_actions", "observable_cache_snapshot", "runtime_environment",
    "future_action_outcome_reads", "future_runtime_features", "coco_scaler_fit_rows",
    "coco_per_image_gt_materialized", "performance_results", "state_count", "roles",
    "implementation_commit", "implementation_sha256", "checks",
})
ROLE_KEYS = frozenset({
    "path", "bytes", "sha256", "role", "states", "atoms", "atom_dimension",
    "state_dimension", "action_dimension", "state_ids_sha256", "payload_sha256",
})
CHECK_KEYS = frozenset({
    "three_roles_exact", "all_state_ids_unique", "all_arrays_finite",
    "a0_a1_a2_only", "no_future_runtime", "no_coco_scaler_fit", "no_ground_truth",
})


class FeatureBuildError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_bytes(path: Path) -> bytes:
    from rail3.sam.coco_v9b1_trajectory import read_stable_regular_bytes

    return read_stable_regular_bytes(path)


def _identity(path: Path) -> dict[str, Any]:
    encoded = _stable_bytes(path)
    return {
        "path": path.relative_to(REPOSITORY).as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _clean_committed_identity(frozen_commit: str | None = None) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(REPOSITORY), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if status:
        raise FeatureBuildError("V9B1_FEATURE_WORKTREE_NOT_CLEAN")
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if frozen_commit is not None and subprocess.run(
        ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", frozen_commit, head],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise FeatureBuildError("V9B1_FEATURE_IMPLEMENTATION_NOT_ANCESTOR")
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        tracked = subprocess.run(
            ["git", "-C", str(REPOSITORY), "ls-files", "--error-unmatch", relative.as_posix()],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            raise FeatureBuildError(f"V9B1_FEATURE_IMPLEMENTATION_UNTRACKED:{relative}")
        hashes[relative.as_posix()] = _identity(REPOSITORY / relative)["sha256"]
    return {"commit": frozen_commit or head, "sha256": hashes}


def _validate_runtime_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = config.get("environment", {})
    if not (
        Path(sys.executable).resolve() == Path(str(expected.get("interpreter", ""))).resolve()
        and platform.python_version() == expected.get("python")
        and np.__version__ == expected.get("numpy")
    ):
        raise FeatureBuildError("V9B1_FEATURE_RUNTIME_ENVIRONMENT_DRIFT")
    module_paths = {
        "rail3.models.tmlr_v9b1.inference": "src/rail3/models/tmlr_v9b1/inference.py",
        "rail3.sam.coco_v9b1_trajectory": "src/rail3/sam/coco_v9b1_trajectory.py",
        "rail3.diagnostic.representations": "src/rail3/diagnostic/representations.py",
        "rail3.regions.m06e_causal": "src/rail3/regions/m06e_causal.py",
    }
    origins: dict[str, str] = {}
    for module_name, relative in module_paths.items():
        imported = importlib.import_module(module_name)
        observed = Path(str(getattr(imported, "__file__", ""))).resolve()
        if observed != (REPOSITORY / relative).resolve():
            raise FeatureBuildError("V9B1_FEATURE_IMPORT_ORIGIN_DRIFT")
        origins[module_name] = relative
    return {
        "interpreter": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "import_origins": origins,
    }


def feature_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    from rail3.models.tmlr_v9b1.inference import feature_payload_sha256 as core_identity

    try:
        return core_identity(payload)
    except ValueError as error:
        raise FeatureBuildError("V9B1_FEATURE_ARRAY_KEYSET_DRIFT") from error


def _npz_bytes(payload: Mapping[str, np.ndarray]) -> bytes:
    feature_payload_sha256(payload)
    stream = BytesIO()
    np.savez_compressed(stream, **{name: payload[name] for name in sorted(payload)})
    return stream.getvalue()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(BytesIO(_stable_bytes(path)), allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARRAY_KEYS:
            raise FeatureBuildError("V9B1_FEATURE_NPZ_KEYSET_DRIFT")
        return {name: np.asarray(archive[name]) for name in sorted(archive.files)}


def build_feature_arrays_from_bundles(
    bundles: Sequence[Any], action_cache: Any, *, model_spec_id: str,
) -> dict[str, Any]:
    """Pure cache consumer whose only action reads are the frozen A1 and A2."""

    from rail3.models.tmlr_v9b1.inference import (
        build_state_role_features, concatenate_role_features,
    )

    rows = []
    for bundle in bundles:
        by_code = {
            plan.action_code: (plan, identity)
            for plan, identity in zip(bundle.plans, bundle.action_identities)
        }
        if set(by_code) != {"A1", "A2", "A3", "A4", "A5", "A6"}:
            raise FeatureBuildError("V9B1_FEATURE_PLAN_GRID_DRIFT")
        # Do not load, inspect, hash, or summarize A3--A6 outcomes here.
        a1 = action_cache.load(by_code["A1"][1])
        a2 = action_cache.load(by_code["A2"][1])
        rows.append(build_state_role_features(
            state=bundle.state,
            canonical=bundle.a0_result,
            a1=a1,
            a2=a2,
            model_spec_id=model_spec_id,
        ))
    return concatenate_role_features(rows, expected_states=len(bundles))


def _observable_cache_snapshot(
    bundles: Sequence[Any], a0_cache: Any, action_cache: Any,
) -> dict[str, Any]:
    """Bind exactly the A0/A1/A2 objects consumed by the feature graph."""

    from rail3.sam.coco_v9b1_trajectory import (
        V9B1ContractError, observable_cache_snapshot,
    )

    try:
        return observable_cache_snapshot(bundles, a0_cache, action_cache)
    except (V9B1ContractError, AttributeError, TypeError) as error:
        raise FeatureBuildError("V9B1_OBSERVABLE_CACHE_DRIFT") from error


def _validate_trajectory_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _stable_bytes(REPOSITORY / TRAJECTORY_INVENTORY_PATH)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureBuildError("V9B1_TRAJECTORY_INVENTORY_INVALID") from error
    if encoded != _canonical_json_bytes(value) + b"\n":
        raise FeatureBuildError("V9B1_TRAJECTORY_INVENTORY_NOT_CANONICAL")
    output = config["outputs"]
    source_root = Path(output["source_data_root"])
    state_identity = _identity(REPOSITORY / source_root / output["state_manifest"])
    shard_identity = _identity(REPOSITORY / source_root / output["shard_manifest"])
    observable = value.get("observable_cache_snapshot")
    if not (
        value.get("status") == "TMLR_V9B1_LABEL_FREE_TRAJECTORY_INVENTORY_READY"
        and value.get("ready") is True
        and value.get("state_count") == 20_000
        and value.get("semantic_record_count") == 140_000
        and value.get("failure_count") == 0
        and value.get("state_manifest_sha256") == state_identity["sha256"]
        and value.get("shard_manifest_sha256") == shard_identity["sha256"]
        and value.get("coco_per_image_gt_materialized") == 0
        and value.get("coco_annotation_rows_materialized") == 0
        and value.get("coco_image_label_rows_materialized") == 0
        and value.get("model_predictions") == 0
        and value.get("performance_results") == 0
        and isinstance(observable, Mapping)
        and set(observable) == {"records", "actions", "aggregate_sha256"}
        and observable.get("records") == 60_000
        and observable.get("actions") == ["A0", "A1", "A2"]
        and isinstance(observable.get("aggregate_sha256"), str)
        and len(observable["aggregate_sha256"]) == 64
        and set(observable["aggregate_sha256"]) <= frozenset("0123456789abcdef")
        and value.get("final_cache_roots") == {
            "a0": "artifacts/candidates/tmlr-v9b1/final/a0",
            "actions": "artifacts/candidates/tmlr-v9b1/final/actions",
            "plans": "artifacts/candidates/tmlr-v9b1/final/plans",
        }
    ):
        raise FeatureBuildError("V9B1_TRAJECTORY_INVENTORY_NOT_READY")
    return {
        "payload": value,
        "identity": {
            "path": TRAJECTORY_INVENTORY_PATH.as_posix(),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "state_identity": state_identity,
        "shard_identity": shard_identity,
    }


def _feature_entry(role: str, path: Path, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from rail3.models.tmlr_v9b1.inference import RoleFeatureArrays, state_id_order_sha256

    arrays = RoleFeatureArrays(role_id=role, **payload)
    arrays.validate(expected_states=20_000)
    identity = _identity(path)
    return {
        **identity,
        "role": role,
        "states": len(arrays.state_ids),
        "atoms": len(arrays.atom_ids),
        "atom_dimension": 27,
        "state_dimension": 41,
        "action_dimension": 16,
        "state_ids_sha256": state_id_order_sha256(arrays.state_ids),
        "payload_sha256": feature_payload_sha256(payload),
    }


def _inventory_payload(
    *, config: Mapping[str, Any], trajectory: Mapping[str, Any],
    features: Mapping[str, Mapping[str, Any]], implementation: Mapping[str, Any],
    observable_snapshot: Mapping[str, Any], runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "rail3.tmlr-v9b1-label-free-feature-inventory.v1",
        "status": "TMLR_V9B1_LABEL_FREE_FEATURE_INVENTORY_PASS",
        "panel_lock_id": config["authorities"]["panel_manifest"]["lock_id"],
        "panel_sha256": config["authorities"]["panel_manifest"]["sha256"],
        "state_manifest": trajectory["state_identity"],
        "shard_manifest": trajectory["shard_identity"],
        "trajectory_inventory": trajectory["identity"],
        "sam_identity": trajectory["payload"]["sam_identity"],
        "schema_id": "PROSPECTIVE_V1",
        "observable_actions": ["A0", "A1", "A2"],
        "observable_cache_snapshot": dict(observable_snapshot),
        "runtime_environment": dict(runtime_environment),
        "future_action_outcome_reads": 0,
        "future_runtime_features": 0,
        "coco_scaler_fit_rows": 0,
        "coco_per_image_gt_materialized": 0,
        "performance_results": 0,
        "state_count": 20_000,
        "roles": [features[role] for role in (
            "F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P",
        )],
        "implementation_commit": implementation["commit"],
        "implementation_sha256": implementation["sha256"],
        "checks": {
            "three_roles_exact": True,
            "all_state_ids_unique": True,
            "all_arrays_finite": True,
            "a0_a1_a2_only": True,
            "no_future_runtime": True,
            "no_coco_scaler_fit": True,
            "no_ground_truth": True,
        },
    }


def _state_manifest_ids(identity: Mapping[str, Any]) -> np.ndarray:
    path = REPOSITORY / str(identity.get("path", ""))
    if _identity(path) != dict(identity):
        raise FeatureBuildError("V9B1_FEATURE_STATE_MANIFEST_IDENTITY_DRIFT")
    encoded = _stable_bytes(path)
    stream = csv.DictReader(encoded.decode("utf-8").splitlines())
    from rail3.sam.coco_v9b1_trajectory import STATE_MANIFEST_FIELDS

    if tuple(stream.fieldnames or ()) != STATE_MANIFEST_FIELDS:
        raise FeatureBuildError("V9B1_FEATURE_STATE_MANIFEST_HEADER_DRIFT")
    values = np.asarray([row["semantic_state_id"] for row in stream])
    if values.shape != (20_000,) or len(set(values.astype(str))) != 20_000:
        raise FeatureBuildError("V9B1_FEATURE_STATE_MANIFEST_GRID_DRIFT")
    return values


def _validate_inventory_contract(
    inventory: Mapping[str, Any], *, config: Mapping[str, Any],
    trajectory: Mapping[str, Any], implementation: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> None:
    expected_roles = ["F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P"]
    rows = inventory.get("roles")
    snapshot = inventory.get("observable_cache_snapshot")
    if not (
        set(inventory) == INVENTORY_KEYS
        and inventory.get("schema_version")
        == "rail3.tmlr-v9b1-label-free-feature-inventory.v1"
        and inventory.get("status") == "TMLR_V9B1_LABEL_FREE_FEATURE_INVENTORY_PASS"
        and inventory.get("panel_lock_id")
        == config["authorities"]["panel_manifest"]["lock_id"]
        and inventory.get("panel_sha256")
        == config["authorities"]["panel_manifest"]["sha256"]
        and inventory.get("state_manifest") == trajectory["state_identity"]
        and inventory.get("shard_manifest") == trajectory["shard_identity"]
        and inventory.get("trajectory_inventory") == trajectory["identity"]
        and inventory.get("sam_identity") == trajectory["payload"]["sam_identity"]
        and inventory.get("schema_id") == "PROSPECTIVE_V1"
        and inventory.get("observable_actions") == ["A0", "A1", "A2"]
        and snapshot == trajectory["payload"].get("observable_cache_snapshot")
        and isinstance(snapshot, Mapping)
        and set(snapshot) == {"records", "actions", "aggregate_sha256"}
        and snapshot.get("records") == 60_000
        and snapshot.get("actions") == ["A0", "A1", "A2"]
        and isinstance(snapshot.get("aggregate_sha256"), str)
        and len(snapshot["aggregate_sha256"]) == 64
        and set(snapshot["aggregate_sha256"]) <= frozenset("0123456789abcdef")
        and inventory.get("runtime_environment") == dict(runtime_environment)
        and inventory.get("future_action_outcome_reads") == 0
        and inventory.get("future_runtime_features") == 0
        and inventory.get("coco_scaler_fit_rows") == 0
        and inventory.get("coco_per_image_gt_materialized") == 0
        and inventory.get("performance_results") == 0
        and inventory.get("state_count") == 20_000
        and isinstance(rows, list)
        and [row.get("role") for row in rows] == expected_roles
        and all(isinstance(row, Mapping) and set(row) == ROLE_KEYS for row in rows)
        and inventory.get("implementation_commit") == implementation["commit"]
        and inventory.get("implementation_sha256") == implementation["sha256"]
        and isinstance(inventory.get("checks"), Mapping)
        and set(inventory["checks"]) == CHECK_KEYS
        and all(type(value) is bool and value is True for value in inventory["checks"].values())
    ):
        raise FeatureBuildError("V9B1_FEATURE_INVENTORY_METADATA_DRIFT")


def _publish_or_validate_feature(path: Path, encoded: bytes) -> None:
    from rail3.cache.atomic_io import atomic_create_bytes

    try:
        atomic_create_bytes(REPOSITORY / path, encoded)
    except FileExistsError:
        if _stable_bytes(REPOSITORY / path) != encoded:
            raise FeatureBuildError(f"V9B1_FEATURE_PARTIAL_RESUME_CONFLICT:{path}")


def build() -> dict[str, Any]:
    from rail3.cache.atomic_io import atomic_create_bytes
    from rail3.sam.coco_v9b1_cache import SecureCandidateCache, SecureTrajectoryCache
    from rail3.sam.coco_v9b1_trajectory import (
        FINAL_A0_CACHE_ROOT, FINAL_ACTION_CACHE_ROOT,
        build_semantic_states, derive_bundles_from_a0_cache, load_config,
        load_frozen_panel_images, load_taxonomy,
        validate_frozen_execution_manifests, validate_small_authorities,
    )
    from rail3.sam.sam31_backend import official_sam31_model_spec

    if (REPOSITORY / INVENTORY_PATH).exists():
        raise FeatureBuildError("V9B1_FEATURE_INVENTORY_EXISTS_USE_CHECK")
    implementation = _clean_committed_identity()
    config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, config)
    runtime_environment = _validate_runtime_environment(config)
    states = build_semantic_states(load_frozen_panel_images(REPOSITORY), load_taxonomy(REPOSITORY))
    validate_frozen_execution_manifests(
        REPOSITORY, config, states, require_global_clean=True,
    )
    trajectory = _validate_trajectory_authority(config)
    model_spec = official_sam31_model_spec()
    a0_cache = SecureCandidateCache(REPOSITORY / FINAL_A0_CACHE_ROOT)
    action_cache = SecureTrajectoryCache(REPOSITORY / FINAL_ACTION_CACHE_ROOT)
    bundles = derive_bundles_from_a0_cache(
        states, model_spec_id=model_spec.model_spec_id, a0_cache=a0_cache,
    )
    observable_snapshot = _observable_cache_snapshot(bundles, a0_cache, action_cache)
    if observable_snapshot != trajectory["payload"]["observable_cache_snapshot"]:
        raise FeatureBuildError("V9B1_OBSERVABLE_CACHE_NOT_FROZEN_BY_TRAJECTORY")
    arrays_by_role = build_feature_arrays_from_bundles(
        bundles, action_cache, model_spec_id=model_spec.model_spec_id,
    )
    if _observable_cache_snapshot(bundles, a0_cache, action_cache) != observable_snapshot:
        raise FeatureBuildError("V9B1_OBSERVABLE_CACHE_CHANGED_DURING_BUILD")
    # Independently reopen every frozen authority before any final feature write.
    post_config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, post_config)
    post_states = build_semantic_states(
        load_frozen_panel_images(REPOSITORY), load_taxonomy(REPOSITORY),
    )
    validate_frozen_execution_manifests(
        REPOSITORY, post_config, post_states, require_global_clean=True,
    )
    if post_config != config or _validate_trajectory_authority(post_config) != trajectory:
        raise FeatureBuildError("V9B1_FEATURE_AUTHORITY_CHANGED_DURING_BUILD")
    encoded_by_role = {
        role: _npz_bytes(arrays_by_role[role].npz_payload()) for role in FEATURE_PATHS
    }
    entries: dict[str, dict[str, Any]] = {}
    for role, path in FEATURE_PATHS.items():
        payload = arrays_by_role[role].npz_payload()
        _publish_or_validate_feature(path, encoded_by_role[role])
        entries[role] = _feature_entry(role, REPOSITORY / path, payload)
    inventory = _inventory_payload(
        config=config, trajectory=trajectory, features=entries,
        implementation=implementation, observable_snapshot=observable_snapshot,
        runtime_environment=runtime_environment,
    )
    # A crash before inventory publication is exactly resumable: every existing
    # feature must byte-match the deterministic frozen payload above.
    if _validate_trajectory_authority(load_config(REPOSITORY)) != trajectory:
        raise FeatureBuildError("V9B1_FEATURE_AUTHORITY_CHANGED_BEFORE_INVENTORY")
    if _clean_committed_identity(implementation["commit"]) != implementation:
        raise FeatureBuildError("V9B1_FEATURE_IMPLEMENTATION_CHANGED_DURING_BUILD")
    _validate_inventory_contract(
        inventory, config=config, trajectory=trajectory, implementation=implementation,
        runtime_environment=runtime_environment,
    )
    atomic_create_bytes(REPOSITORY / INVENTORY_PATH, _canonical_json_bytes(inventory) + b"\n")
    return inventory


def check() -> dict[str, Any]:
    encoded = _stable_bytes(REPOSITORY / INVENTORY_PATH)
    try:
        inventory = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureBuildError("V9B1_FEATURE_INVENTORY_INVALID") from error
    if encoded != _canonical_json_bytes(inventory) + b"\n":
        raise FeatureBuildError("V9B1_FEATURE_INVENTORY_NOT_CANONICAL")
    from rail3.sam.coco_v9b1_cache import SecureCandidateCache, SecureTrajectoryCache
    from rail3.sam.coco_v9b1_trajectory import (
        FINAL_A0_CACHE_ROOT, FINAL_ACTION_CACHE_ROOT,
        build_semantic_states, derive_bundles_from_a0_cache, load_config,
        load_frozen_panel_images, load_taxonomy, validate_frozen_execution_manifests,
        validate_small_authorities,
    )
    from rail3.sam.sam31_backend import official_sam31_model_spec

    config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, config)
    runtime_environment = _validate_runtime_environment(config)
    trajectory = _validate_trajectory_authority(config)
    implementation = _clean_committed_identity(str(inventory.get("implementation_commit", "")))
    _validate_inventory_contract(
        inventory, config=config, trajectory=trajectory, implementation=implementation,
        runtime_environment=runtime_environment,
    )
    states = build_semantic_states(
        load_frozen_panel_images(REPOSITORY), load_taxonomy(REPOSITORY),
    )
    validate_frozen_execution_manifests(
        REPOSITORY, config, states, require_global_clean=True,
    )
    bundles = derive_bundles_from_a0_cache(
        states,
        model_spec_id=official_sam31_model_spec().model_spec_id,
        a0_cache=SecureCandidateCache(REPOSITORY / FINAL_A0_CACHE_ROOT),
    )
    current_observable = _observable_cache_snapshot(
        bundles,
        SecureCandidateCache(REPOSITORY / FINAL_A0_CACHE_ROOT),
        SecureTrajectoryCache(REPOSITORY / FINAL_ACTION_CACHE_ROOT),
    )
    if current_observable != inventory["observable_cache_snapshot"]:
        raise FeatureBuildError("V9B1_OBSERVABLE_CACHE_CHANGED_AFTER_FEATURE_BUILD")
    expected_roles = ["F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P"]
    rows = inventory.get("roles")
    state_ids = _state_manifest_ids(inventory["state_manifest"])
    for row in rows:
        role = row["role"]
        path = REPOSITORY / FEATURE_PATHS[role]
        payload = _load_npz(path)
        expected = _feature_entry(role, path, payload)
        if row != expected:
            raise FeatureBuildError("V9B1_FEATURE_FILE_IDENTITY_DRIFT")
        if not np.array_equal(payload["state_ids"].astype(str), state_ids.astype(str)):
            raise FeatureBuildError("V9B1_FEATURE_STATE_ORDER_DRIFT")
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", metavar="TOKEN")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            result = check()
        else:
            if arguments.build != FORMAL_TOKEN:
                raise FeatureBuildError("V9B1_FEATURE_FORMAL_TOKEN_MISMATCH")
            result = build()
    except Exception as error:
        print(f"STOP-BLOCKED_TMLR_V9B1_FEATURES: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "states": 20_000}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
