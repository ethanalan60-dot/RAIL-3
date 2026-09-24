#!/usr/bin/env python3
"""Freeze 18 continuous V9B1 predictions and six mean-before-selection ensembles."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO, StringIO
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

FORMAL_TOKEN = "TMLR_V9B1_LABEL_FREE_PREDICTIONS_AUTHORIZED"
CONFIG_PATH = Path("configs/experiments/tmlr_v9b1_coco_label_free_zeroshot.json")
FULLFIT_CONFIG_PATH = Path("configs/experiments/tmlr_v9b1_fullfit_v1.json")
CHECKPOINT_LOCK_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/fullfit_checkpoint_lock.json"
)
FEATURE_INVENTORY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_feature_inventory.json"
)
PREDICTION_INVENTORY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_prediction_inventory.csv"
)
PREDICTION_MANIFEST_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_prediction_manifest.json"
)
PREDICTION_ROOT = Path("artifacts/predictions/tmlr-v9b1")
FEATURE_PATHS = {
    "F0_P": Path("artifacts/features/tmlr-v9b1/f0/features.npz"),
    "RECT_FULL_RASTER_P": Path("artifacts/features/tmlr-v9b1/rect/features.npz"),
    "UNION_FULL_RASTER_P": Path("artifacts/features/tmlr-v9b1/union/features.npz"),
}
FAMILIES = ("R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P", "R3_P")
SEEDS = (13, 37, 71)
FAMILY_ROLE = {
    "R0_SMALL_P": "F0_P", "R0_CM_P": "F0_P", "R1_P": "F0_P",
    "RECT_P": "RECT_FULL_RASTER_P", "UNION_P": "UNION_FULL_RASTER_P",
    "R3_P": "F0_P",
}
SEED_KEYS = frozenset({
    "family", "bundle_role", "seed", "checkpoint_sha256",
    "feature_payload_sha256", "producer_commit", "inference_sha256",
    "producer_sha256", "state_ids", "prediction",
})
ENSEMBLE_KEYS = frozenset({
    "family", "bundle_role", "ensemble_id", "feature_payload_sha256",
    "checkpoint_sha256_by_seed", "producer_commit", "inference_sha256",
    "producer_sha256", "state_ids", "prediction", "feasibility", "selected_action",
    "selected_prediction",
})
INVENTORY_FIELDS = (
    "artifact_kind", "family", "seed", "bundle_role", "rows", "actions",
    "selected_action_rows", "path", "bytes", "sha256", "payload_sha256",
    "checkpoint_sha256", "feature_payload_sha256",
)
IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    FULLFIT_CONFIG_PATH,
    Path("src/rail3/models/tmlr_v9b1/fullfit.py"),
    Path("src/rail3/models/tmlr_v9b1/inference.py"),
    Path("scripts/predict_tmlr_v9b1_label_free.py"),
)
FULLFIT_IMPLEMENTATION_PATHS = (
    FULLFIT_CONFIG_PATH,
    Path("src/rail3/models/tmlr_v9b1/__init__.py"),
    Path("src/rail3/models/tmlr_v9b1/fullfit.py"),
    Path("scripts/train_tmlr_v9b1_fullfit.py"),
)
FEATURE_IMPLEMENTATION_PATHS = (
    CONFIG_PATH,
    Path("src/rail3/models/tmlr_v9b1/inference.py"),
    Path("src/rail3/sam/coco_v9b1_trajectory.py"),
    Path("src/rail3/diagnostic/representations.py"),
    Path("src/rail3/regions/m06e_causal.py"),
    Path("scripts/build_tmlr_v9b1_label_free_features.py"),
)
FULLFIT_ZERO_COUNTERS = {
    "coco_annotation_reads": 0,
    "coco_per_image_gt_materialized": 0,
    "sam_calls": 0,
    "trajectory_reads": 0,
    "protected_reads": 0,
    "uav_reads": 0,
    "predictions": 0,
    "performance_results": 0,
}
FULLFIT_LOCK_KEYS = frozenset({
    "schema_version", "status", "job_count", "family_count", "seeds",
    "epoch_rule_id", "epoch_lock_sha256", "ensemble_id", "config",
    "prospective_schema_id", "checkpoint_schema_version", "run_manifest_schema_version",
    "implementation_commit", "implementation_sha256", "runs", "zero_counters",
    "created_utc", "terminal_state",
})
FULLFIT_LOCK_RUN_KEYS = frozenset({
    "family", "seed", "fullfit_epoch", "epochs_completed", "bundle_role",
    "bundle_identities", "prospective_schema_id", "parameter_count", "model_class",
    "objective_id", "config_sha256", "implementation_commit", "implementation_sha256",
    "checkpoint", "run_manifest",
})
RUN_MANIFEST_KEYS = frozenset({
    "schema_version", "run_id", "family", "seed", "bundle_role", "fullfit_epoch",
    "epochs_completed", "epoch_rule_id", "epoch_lock_sha256", "config_sha256",
    "model_class", "parameter_count", "objective_id", "optimizer_recipe",
    "prediction_semantics", "ensemble_id", "implementation_commit",
    "implementation_sha256", "bundle_identities", "training_loss_finite",
    "wall_seconds", "peak_vram_bytes", "physical_gpu", "zero_counters",
    "created_utc", "checkpoint", "terminal_state",
})
FEATURE_INVENTORY_KEYS = frozenset({
    "schema_version", "status", "panel_lock_id", "panel_sha256", "state_manifest",
    "shard_manifest", "trajectory_inventory", "sam_identity", "schema_id",
    "observable_actions", "observable_cache_snapshot", "runtime_environment",
    "future_action_outcome_reads",
    "future_runtime_features", "coco_scaler_fit_rows", "coco_per_image_gt_materialized",
    "performance_results", "state_count", "roles", "implementation_commit",
    "implementation_sha256", "checks",
})
FEATURE_ROLE_KEYS = frozenset({
    "path", "bytes", "sha256", "role", "states", "atoms", "atom_dimension",
    "state_dimension", "action_dimension", "state_ids_sha256", "payload_sha256",
})
FEATURE_CHECK_KEYS = frozenset({
    "three_roles_exact", "all_state_ids_unique", "all_arrays_finite",
    "a0_a1_a2_only", "no_future_runtime", "no_coco_scaler_fit", "no_ground_truth",
})
EXPECTED_EPOCHS = {
    "R0_SMALL_P": 120, "R0_CM_P": 120, "R1_P": 120,
    "RECT_P": 120, "UNION_P": 120, "R3_P": 108,
}
EXPECTED_PARAMETER_COUNTS = {
    "R0_SMALL_P": 7_809, "R0_CM_P": 103_369, "R1_P": 103_521,
    "RECT_P": 103_521, "UNION_P": 103_521, "R3_P": 103_521,
}
EXPECTED_MODEL_CLASSES = {
    "R0_SMALL_P": "rail3.models.m06e.models.GlobalResidual",
    "R0_CM_P": "rail3.models.tmlr_v6.models.CapacityMatchedGlobalResidual",
    "R1_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "RECT_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "UNION_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "R3_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
}
EXPECTED_OPTIMIZER_RECIPE = {
    "class": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.0001,
    "scheduler": None, "amp": False, "ddp": False, "early_stopping": False,
    "warm_start": False,
}
EXPECTED_GPU = {
    "R0_SMALL_P": "0", "RECT_P": "0", "R3_P": "0",
    "R0_CM_P": "1", "R1_P": "1", "UNION_P": "1",
}


class PredictionBuildError(RuntimeError):
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


def _load_canonical_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    encoded = _stable_bytes(path)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionBuildError(f"V9B1_{description}_JSON_INVALID") from error
    if not isinstance(value, dict) or encoded != _canonical_json_bytes(value) + b"\n":
        raise PredictionBuildError(f"V9B1_{description}_NOT_CANONICAL")
    return value, encoded


def _load_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    encoded = _stable_bytes(path)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictionBuildError(f"V9B1_{description}_JSON_INVALID") from error
    if not isinstance(value, dict):
        raise PredictionBuildError(f"V9B1_{description}_ROOT_INVALID")
    return value, encoded


def _clean_committed_identity(frozen_commit: str | None = None) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(REPOSITORY), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if status:
        raise PredictionBuildError("V9B1_PREDICTION_WORKTREE_NOT_CLEAN")
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if frozen_commit is not None and subprocess.run(
        ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", frozen_commit, head],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise PredictionBuildError("V9B1_PREDICTION_IMPLEMENTATION_NOT_ANCESTOR")
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        if subprocess.run(
            ["git", "-C", str(REPOSITORY), "ls-files", "--error-unmatch", relative.as_posix()],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            raise PredictionBuildError(f"V9B1_PREDICTION_IMPLEMENTATION_UNTRACKED:{relative}")
        hashes[relative.as_posix()] = _identity(REPOSITORY / relative)["sha256"]
    return {"commit": frozen_commit or head, "sha256": hashes}


def _validate_runtime_and_scientific_sources(fullfit_config: Mapping[str, Any]) -> None:
    runtime = fullfit_config.get("runtime_environment", {})
    if not (
        Path(sys.executable).resolve() == Path(str(runtime.get("interpreter", ""))).resolve()
        and platform.python_version() == runtime.get("python")
        and np.__version__ == runtime.get("numpy")
        and torch.__version__ == runtime.get("torch")
        and torch.version.cuda == runtime.get("cuda_runtime")
    ):
        raise PredictionBuildError("V9B1_PREDICTION_RUNTIME_ENVIRONMENT_DRIFT")
    allowlist = fullfit_config.get("frozen_scientific_source_allowlist")
    if not isinstance(allowlist, Mapping) or len(allowlist) != 9:
        raise PredictionBuildError("V9B1_SCIENTIFIC_SOURCE_ALLOWLIST_DRIFT")
    for relative_text, declared in allowlist.items():
        relative = Path(relative_text)
        actual = _identity(REPOSITORY / relative)
        if not isinstance(declared, Mapping) or (
            declared.get("bytes") != actual["bytes"]
            or declared.get("sha256") != actual["sha256"]
        ):
            raise PredictionBuildError("V9B1_SCIENTIFIC_SOURCE_IDENTITY_DRIFT")
        module_name = relative.as_posix()[4:-3].replace("/", ".")
        imported = importlib.import_module(module_name)
        if Path(str(getattr(imported, "__file__", ""))).resolve() != (REPOSITORY / relative).resolve():
            raise PredictionBuildError("V9B1_SCIENTIFIC_SOURCE_IMPORT_ORIGIN_DRIFT")


def _trajectory_authority_for_features(config: Mapping[str, Any]) -> dict[str, Any]:
    trajectory_path = REPOSITORY / Path(
        config["outputs"]["source_data_root"]
    ) / config["outputs"]["trajectory_inventory"]
    trajectory, encoded = _load_canonical_json(trajectory_path, "TRAJECTORY_INVENTORY")
    state_path = REPOSITORY / Path(config["outputs"]["source_data_root"]) / config["outputs"]["state_manifest"]
    shard_path = REPOSITORY / Path(config["outputs"]["source_data_root"]) / config["outputs"]["shard_manifest"]
    identity = {
        "path": trajectory_path.relative_to(REPOSITORY).as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    state_identity = _identity(state_path)
    shard_identity = _identity(shard_path)
    if not (
        trajectory.get("status") == "TMLR_V9B1_LABEL_FREE_TRAJECTORY_INVENTORY_READY"
        and trajectory.get("ready") is True
        and trajectory.get("state_count") == 20_000
        and trajectory.get("semantic_record_count") == 140_000
        and trajectory.get("failure_count") == 0
        and trajectory.get("state_manifest_sha256") == state_identity["sha256"]
        and trajectory.get("shard_manifest_sha256") == shard_identity["sha256"]
        and trajectory.get("coco_per_image_gt_materialized") == 0
        and trajectory.get("performance_results") == 0
    ):
        raise PredictionBuildError("V9B1_TRAJECTORY_AUTHORITY_DRIFT")
    return {
        "payload": trajectory,
        "identity": identity,
        "state_identity": state_identity,
        "shard_identity": shard_identity,
    }


def _state_manifest_ids(identity: Mapping[str, Any]) -> np.ndarray:
    path = REPOSITORY / str(identity.get("path", ""))
    if _identity(path) != dict(identity):
        raise PredictionBuildError("V9B1_STATE_MANIFEST_IDENTITY_DRIFT")
    encoded = _stable_bytes(path)
    from rail3.sam.coco_v9b1_trajectory import STATE_MANIFEST_FIELDS

    reader = csv.DictReader(StringIO(encoded.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != STATE_MANIFEST_FIELDS:
        raise PredictionBuildError("V9B1_STATE_MANIFEST_HEADER_DRIFT")
    state_ids = np.asarray([row["semantic_state_id"] for row in reader])
    if state_ids.shape != (20_000,) or len(set(state_ids.astype(str))) != 20_000:
        raise PredictionBuildError("V9B1_STATE_MANIFEST_GRID_DRIFT")
    return state_ids


def _validate_feature_inventory_metadata(
    inventory: Mapping[str, Any], *, config: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> None:
    rows = inventory.get("roles")
    snapshot = inventory.get("observable_cache_snapshot")
    expected_feature_runtime = {
        "interpreter": str(Path(config["environment"]["interpreter"]).resolve()),
        "python": config["environment"]["python"],
        "numpy": config["environment"]["numpy"],
        "import_origins": {
            "rail3.models.tmlr_v9b1.inference": "src/rail3/models/tmlr_v9b1/inference.py",
            "rail3.sam.coco_v9b1_trajectory": "src/rail3/sam/coco_v9b1_trajectory.py",
            "rail3.diagnostic.representations": "src/rail3/diagnostic/representations.py",
            "rail3.regions.m06e_causal": "src/rail3/regions/m06e_causal.py",
        },
    }
    sha_hex = frozenset("0123456789abcdef")
    if not (
        set(inventory) == FEATURE_INVENTORY_KEYS
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
        and isinstance(snapshot, Mapping)
        and set(snapshot) == {"records", "actions", "aggregate_sha256"}
        and snapshot.get("records") == 60_000
        and snapshot.get("actions") == ["A0", "A1", "A2"]
        and isinstance(snapshot.get("aggregate_sha256"), str)
        and len(snapshot["aggregate_sha256"]) == 64
        and set(snapshot["aggregate_sha256"]) <= sha_hex
        and snapshot == trajectory["payload"].get("observable_cache_snapshot")
        and inventory.get("runtime_environment") == expected_feature_runtime
        and inventory.get("future_action_outcome_reads") == 0
        and inventory.get("future_runtime_features") == 0
        and inventory.get("coco_scaler_fit_rows") == 0
        and inventory.get("coco_per_image_gt_materialized") == 0
        and inventory.get("performance_results") == 0
        and inventory.get("state_count") == 20_000
        and isinstance(rows, list)
        and [row.get("role") for row in rows] == list(FEATURE_PATHS)
        and all(isinstance(row, Mapping) and set(row) == FEATURE_ROLE_KEYS for row in rows)
        and isinstance(inventory.get("checks"), Mapping)
        and set(inventory["checks"]) == FEATURE_CHECK_KEYS
        and all(type(value) is bool and value is True for value in inventory["checks"].values())
    ):
        raise PredictionBuildError("V9B1_FEATURE_INVENTORY_METADATA_DRIFT")
    for role, row in zip(FEATURE_PATHS, rows):
        if not (
            row.get("role") == role
            and row.get("path") == FEATURE_PATHS[role].as_posix()
            and type(row.get("bytes")) is int
            and row["bytes"] > 0
            and isinstance(row.get("sha256"), str)
            and len(row["sha256"]) == 64
            and set(row["sha256"]) <= sha_hex
            and row.get("states") == 20_000
            and type(row.get("atoms")) is int
            and row["atoms"] > 0
            and row.get("atom_dimension") == 27
            and row.get("state_dimension") == 41
            and row.get("action_dimension") == 16
            and isinstance(row.get("state_ids_sha256"), str)
            and len(row["state_ids_sha256"]) == 64
            and set(row["state_ids_sha256"]) <= sha_hex
            and isinstance(row.get("payload_sha256"), str)
            and len(row["payload_sha256"]) == 64
            and set(row["payload_sha256"]) <= sha_hex
        ):
            raise PredictionBuildError("V9B1_FEATURE_ROLE_METADATA_DRIFT")


def _load_feature_bank() -> tuple[dict[str, Any], dict[str, Any]]:
    from rail3.models.tmlr_v9b1.inference import (
        RoleFeatureArrays, feature_payload_sha256, state_id_order_sha256,
    )

    inventory, _ = _load_canonical_json(
        REPOSITORY / FEATURE_INVENTORY_PATH, "FEATURE_INVENTORY",
    )
    config, _ = _load_json(REPOSITORY / CONFIG_PATH, "CONFIG")
    trajectory = _trajectory_authority_for_features(config)
    _validate_feature_inventory_metadata(
        inventory, config=config, trajectory=trajectory,
    )
    frozen_hashes = inventory.get("implementation_sha256")
    frozen_commit = str(inventory.get("implementation_commit", ""))
    current_head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if (
        not isinstance(frozen_hashes, dict)
        or set(frozen_hashes) != {path.as_posix() for path in FEATURE_IMPLEMENTATION_PATHS}
        or subprocess.run(
            ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", frozen_commit, current_head],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        or any(
            _identity(REPOSITORY / path)["sha256"] != frozen_hashes[path.as_posix()]
            for path in FEATURE_IMPLEMENTATION_PATHS
        )
    ):
        raise PredictionBuildError("V9B1_FEATURE_IMPLEMENTATION_IDENTITY_DRIFT")
    rows = inventory.get("roles")
    frozen_state_ids = _state_manifest_ids(inventory["state_manifest"])
    arrays: dict[str, Any] = {}
    for row in rows:
        role = row["role"]
        path = REPOSITORY / FEATURE_PATHS[role]
        identity = _identity(path)
        if row.get("path") != FEATURE_PATHS[role].as_posix() or any(
            row.get(key) != value for key, value in identity.items()
        ):
            raise PredictionBuildError("V9B1_FEATURE_FILE_IDENTITY_DRIFT")
        with np.load(BytesIO(_stable_bytes(path)), allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        values = RoleFeatureArrays(role_id=role, **payload)
        values.validate(expected_states=20_000)
        if (
            feature_payload_sha256(payload) != row.get("payload_sha256")
            or state_id_order_sha256(values.state_ids) != row.get("state_ids_sha256")
            or row.get("states") != 20_000
            or row.get("atoms") != len(values.atom_ids)
            or not np.array_equal(values.state_ids.astype(str), frozen_state_ids.astype(str))
        ):
            raise PredictionBuildError("V9B1_FEATURE_PAYLOAD_IDENTITY_DRIFT")
        arrays[role] = values
    reference = arrays["F0_P"].state_ids.astype(str)
    if any(
        not np.array_equal(reference, arrays[role].state_ids.astype(str))
        for role in FEATURE_PATHS
    ):
        raise PredictionBuildError("V9B1_FEATURE_CROSS_ROLE_STATE_ORDER_DRIFT")
    return inventory, arrays


def _load_checkpoint_authority() -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, int], Any]]:
    from rail3.models.tmlr_v9b1.inference import validate_fullfit_checkpoint_for_inference

    lock, _ = _load_canonical_json(REPOSITORY / CHECKPOINT_LOCK_PATH, "CHECKPOINT_LOCK")
    fullfit_config, config_encoded = _load_json(
        REPOSITORY / FULLFIT_CONFIG_PATH, "FULLFIT_CONFIG",
    )
    if not (
        fullfit_config.get("schema_version") == "rail3.tmlr-v9b1-fullfit.v1"
        and fullfit_config.get("status")
        == "TMLR_V9B1_FULLFIT_PROTOCOL_FROZEN_BEFORE_TRAINING"
        and fullfit_config.get("result_blind") is True
        and fullfit_config.get("zero_counters") == FULLFIT_ZERO_COUNTERS
    ):
        raise PredictionBuildError("V9B1_FULLFIT_CONFIG_CONTRACT_DRIFT")
    _validate_runtime_and_scientific_sources(fullfit_config)
    config_sha256 = hashlib.sha256(config_encoded).hexdigest()
    expected_pairs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    rows = lock.get("runs")
    if not (
        set(lock) == FULLFIT_LOCK_KEYS
        and lock.get("schema_version") == "rail3.tmlr-v9b1-fullfit-checkpoint-lock.v1"
        and lock.get("status") == "TMLR_V9B1_FULLFIT_18_OF_18_PASS"
        and lock.get("job_count") == 18
        and lock.get("family_count") == 6
        and lock.get("seeds") == list(SEEDS)
        and lock.get("epoch_rule_id") == "V9_FULLFIT_EPOCH_RULE_V1"
        and lock.get("ensemble_id") == "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"
        and lock.get("prospective_schema_id") == "PROSPECTIVE_V1"
        and lock.get("checkpoint_schema_version")
        == "rail3.tmlr-v9b1-fullfit-checkpoint.v1"
        and lock.get("run_manifest_schema_version")
        == "rail3.tmlr-v9b1-fullfit-run.v1"
        and lock.get("terminal_state") == "TMLR_V9B1_FULLFIT_CHECKPOINTS_READY"
        and isinstance(lock.get("created_utc"), str)
        and isinstance(rows, list)
        and [(row.get("family"), row.get("seed")) for row in rows] == expected_pairs
        and lock.get("config") == {
            "path": FULLFIT_CONFIG_PATH.as_posix(),
            "sha256": config_sha256,
            "bytes": len(config_encoded),
        }
        and lock.get("epoch_lock_sha256")
        == fullfit_config.get("recovery_authorities", {}).get("epoch_lock", {}).get("sha256")
        and lock.get("zero_counters") == FULLFIT_ZERO_COUNTERS
    ):
        raise PredictionBuildError("V9B1_CHECKPOINT_LOCK_NOT_READY")
    frozen_implementation = lock.get("implementation_sha256")
    frozen_commit = str(lock.get("implementation_commit", ""))
    current_head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if (
        not isinstance(frozen_implementation, dict)
        or set(frozen_implementation) != {path.as_posix() for path in FULLFIT_IMPLEMENTATION_PATHS}
        or subprocess.run(
            ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", frozen_commit, current_head],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        or any(
            _identity(REPOSITORY / path)["sha256"]
            != frozen_implementation[path.as_posix()]
            for path in FULLFIT_IMPLEMENTATION_PATHS
        )
    ):
        raise PredictionBuildError("V9B1_CHECKPOINT_IMPLEMENTATION_IDENTITY_DRIFT")
    models: dict[tuple[str, int], Any] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != FULLFIT_LOCK_RUN_KEYS:
            raise PredictionBuildError("V9B1_CHECKPOINT_LOCK_RUN_SCHEMA_DRIFT")
        family, seed = str(row["family"]), int(row["seed"])
        checkpoint_identity = row.get("checkpoint", {})
        expected_run_root = Path(
            f"artifacts/voc2012/tmlr-v9b1-fullfit/S1364/{family}/seed_{seed}"
        )
        expected_checkpoint_path = expected_run_root / "checkpoint.pt"
        expected_manifest_path = expected_run_root / "run-manifest.json"
        expected_objective = (
            "R3_FROZEN_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK_NO_COST"
            if family == "R3_P" else "STATE_HUBER_DELTA_0_05"
        )
        if (
            checkpoint_identity.get("path") != expected_checkpoint_path.as_posix()
            or set(checkpoint_identity) != {"path", "bytes", "sha256"}
            or row.get("run_manifest", {}).get("path") != expected_manifest_path.as_posix()
            or set(row.get("run_manifest", {})) != {"path", "bytes", "sha256"}
            or row.get("bundle_role") != FAMILY_ROLE[family]
            or row.get("fullfit_epoch") != EXPECTED_EPOCHS[family]
            or row.get("epochs_completed") != EXPECTED_EPOCHS[family]
            or row.get("prospective_schema_id") != "PROSPECTIVE_V1"
            or row.get("parameter_count") != EXPECTED_PARAMETER_COUNTS[family]
            or row.get("model_class") != EXPECTED_MODEL_CLASSES[family]
            or row.get("objective_id") != expected_objective
            or row.get("bundle_identities")
            != fullfit_config["bundle_input_allowlist"][FAMILY_ROLE[family]]
            or row.get("config_sha256") != config_sha256
            or row.get("implementation_commit") != lock.get("implementation_commit")
            or row.get("implementation_sha256") != lock.get("implementation_sha256")
        ):
            raise PredictionBuildError("V9B1_CHECKPOINT_LOCK_RUN_AUTHORITY_DRIFT")
        checkpoint_path = REPOSITORY / expected_checkpoint_path
        encoded = _stable_bytes(checkpoint_path)
        if (
            len(encoded) != checkpoint_identity.get("bytes")
            or hashlib.sha256(encoded).hexdigest() != checkpoint_identity.get("sha256")
        ):
            raise PredictionBuildError("V9B1_CHECKPOINT_PHYSICAL_IDENTITY_DRIFT")
        manifest_identity = row.get("run_manifest", {})
        manifest_path = REPOSITORY / expected_manifest_path
        if any(
            manifest_identity.get(key) != value
            for key, value in _identity(manifest_path).items()
        ):
            raise PredictionBuildError("V9B1_RUN_MANIFEST_PHYSICAL_IDENTITY_DRIFT")
        manifest, _ = _load_canonical_json(manifest_path, "RUN_MANIFEST")
        if not (
            set(manifest) == RUN_MANIFEST_KEYS
            and manifest.get("schema_version") == "rail3.tmlr-v9b1-fullfit-run.v1"
            and manifest.get("run_id")
            == f"S1364__{family}__seed_{seed}__epoch_{EXPECTED_EPOCHS[family]}"
            and manifest.get("family") == family
            and manifest.get("seed") == seed
            and manifest.get("bundle_role") == FAMILY_ROLE[family]
            and manifest.get("fullfit_epoch") == EXPECTED_EPOCHS[family]
            and manifest.get("epochs_completed") == EXPECTED_EPOCHS[family]
            and manifest.get("epoch_rule_id") == "V9_FULLFIT_EPOCH_RULE_V1"
            and manifest.get("epoch_lock_sha256") == lock["epoch_lock_sha256"]
            and manifest.get("config_sha256") == config_sha256
            and manifest.get("model_class") == EXPECTED_MODEL_CLASSES[family]
            and manifest.get("parameter_count") == EXPECTED_PARAMETER_COUNTS[family]
            and manifest.get("objective_id") == expected_objective
            and manifest.get("optimizer_recipe") == EXPECTED_OPTIMIZER_RECIPE
            and manifest.get("prediction_semantics") == "ABSOLUTE_RESIDUAL"
            and manifest.get("ensemble_id") == lock["ensemble_id"]
            and manifest.get("implementation_commit") == lock["implementation_commit"]
            and manifest.get("implementation_sha256") == lock["implementation_sha256"]
            and manifest.get("bundle_identities") == row["bundle_identities"]
            and manifest.get("training_loss_finite") is True
            and type(manifest.get("wall_seconds")) in {int, float}
            and math.isfinite(float(manifest["wall_seconds"]))
            and 0.0 < float(manifest["wall_seconds"])
            <= float(fullfit_config["gpu_workers"]["watchdog_seconds"])
            and type(manifest.get("peak_vram_bytes")) is int
            and manifest["peak_vram_bytes"] >= 0
            and manifest.get("physical_gpu") == EXPECTED_GPU[family]
            and isinstance(manifest.get("created_utc"), str)
            and bool(manifest["created_utc"])
            and manifest.get("zero_counters") == FULLFIT_ZERO_COUNTERS
            and manifest.get("checkpoint") == dict(checkpoint_identity)
            and manifest.get("terminal_state") == "TMLR_V9B1_FULLFIT_JOB_PASS"
        ):
            raise PredictionBuildError("V9B1_RUN_MANIFEST_METADATA_DRIFT")
        try:
            payload = torch.load(
                BytesIO(encoded), map_location="cpu", weights_only=True,
            )
        except Exception as error:
            raise PredictionBuildError("V9B1_CHECKPOINT_SAFE_LOAD_FAILED") from error
        models[(family, seed)] = validate_fullfit_checkpoint_for_inference(
            payload,
            family=family,
            seed=seed,
            protocol=fullfit_config["frozen_v6_protocol_projection"],
            expected_epoch_lock_sha256=lock["epoch_lock_sha256"],
            expected_config_sha256=row["config_sha256"],
            expected_bundle_identities=row["bundle_identities"],
            expected_implementation_commit=row["implementation_commit"],
            expected_implementation_sha256=row["implementation_sha256"],
            expected_checkpoint_sha256=checkpoint_identity["sha256"],
        )
    return lock, fullfit_config, models


def _seed_path(family: str, seed: int) -> Path:
    return PREDICTION_ROOT / family / f"seed_{seed}.npz"


def _ensemble_path(family: str) -> Path:
    return PREDICTION_ROOT / family / "ensemble.npz"


def _assert_exact_prediction_tree() -> None:
    root = REPOSITORY / PREDICTION_ROOT
    if root.is_symlink() or not root.is_dir() or set(os.listdir(root)) != set(FAMILIES):
        raise PredictionBuildError("V9B1_PREDICTION_ROOT_SET_DRIFT")
    expected = {"ensemble.npz", *(f"seed_{seed}.npz" for seed in SEEDS)}
    for family in FAMILIES:
        directory = root / family
        if directory.is_symlink() or not directory.is_dir() or set(os.listdir(directory)) != expected:
            raise PredictionBuildError("V9B1_PREDICTION_FAMILY_FILE_SET_DRIFT")


def _authority_snapshot(
    feature_inventory: Mapping[str, Any], checkpoint_lock: Mapping[str, Any],
    fullfit_config: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    paths = {
        CONFIG_PATH,
        FULLFIT_CONFIG_PATH,
        FEATURE_INVENTORY_PATH,
        CHECKPOINT_LOCK_PATH,
        Path(str(feature_inventory["state_manifest"]["path"])),
        Path(str(feature_inventory["shard_manifest"]["path"])),
        Path(str(feature_inventory["trajectory_inventory"]["path"])),
    }
    paths.update(Path(str(row["path"])) for row in feature_inventory["roles"])
    for row in checkpoint_lock["runs"]:
        paths.add(Path(str(row["checkpoint"]["path"])))
        paths.add(Path(str(row["run_manifest"]["path"])))
    paths.update(Path(path) for path in fullfit_config["frozen_scientific_source_allowlist"])
    return tuple(_identity(REPOSITORY / path) for path in sorted(paths, key=lambda value: value.as_posix()))


def _assert_authority_snapshot(expected: tuple[dict[str, Any], ...]) -> None:
    observed = tuple(
        _identity(REPOSITORY / Path(row["path"])) for row in expected
    )
    if observed != expected:
        raise PredictionBuildError("V9B1_PREDICTION_INPUT_CHANGED_DURING_RUN")


def _load_frozen_inputs_fixed_point() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[tuple[str, int], Any],
    tuple[dict[str, Any], ...],
]:
    first_feature_inventory, _ = _load_feature_bank()
    first_checkpoint_lock, first_fullfit_config, _ = _load_checkpoint_authority()
    snapshot = _authority_snapshot(
        first_feature_inventory, first_checkpoint_lock, first_fullfit_config,
    )
    feature_inventory, features = _load_feature_bank()
    checkpoint_lock, fullfit_config, models = _load_checkpoint_authority()
    repeated = _authority_snapshot(feature_inventory, checkpoint_lock, fullfit_config)
    if not (
        first_feature_inventory == feature_inventory
        and first_checkpoint_lock == checkpoint_lock
        and first_fullfit_config == fullfit_config
        and snapshot == repeated
    ):
        raise PredictionBuildError("V9B1_PREDICTION_INPUT_FIXED_POINT_DRIFT")
    return (
        feature_inventory, features, checkpoint_lock, fullfit_config, models, repeated,
    )


def _npz_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{name: payload[name] for name in sorted(payload)})
    return stream.getvalue()


def _load_npz(path: Path, expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(BytesIO(_stable_bytes(path)), allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise PredictionBuildError("V9B1_PREDICTION_NPZ_KEYSET_DRIFT")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scalar_text(value: np.ndarray) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise PredictionBuildError("V9B1_PREDICTION_STRING_SCALAR_DRIFT")
    result = str(value.item())
    if not result:
        raise PredictionBuildError("V9B1_PREDICTION_STRING_SCALAR_EMPTY")
    return result


def _validate_seed_payload(
    payload: Mapping[str, np.ndarray], *, family: str, seed: int,
    state_ids: np.ndarray, checkpoint_sha256: str, feature_sha256: str,
    implementation: Mapping[str, Any],
) -> Any:
    from rail3.models.tmlr_v9b1.inference import (
        SeedContinuousPrediction, prediction_payload_sha256,
    )

    if set(payload) != SEED_KEYS:
        raise PredictionBuildError("V9B1_SEED_PREDICTION_KEYSET_DRIFT")
    raw_seed = np.asarray(payload["seed"])
    if raw_seed.shape != () or raw_seed.dtype.kind not in {"i", "u"}:
        raise PredictionBuildError("V9B1_SEED_PREDICTION_SEED_DRIFT")
    record = SeedContinuousPrediction(
        family=_scalar_text(np.asarray(payload["family"])),
        bundle_role=_scalar_text(np.asarray(payload["bundle_role"])),
        seed=int(raw_seed.item()),
        checkpoint_sha256=_scalar_text(np.asarray(payload["checkpoint_sha256"])),
        feature_payload_sha256=_scalar_text(np.asarray(payload["feature_payload_sha256"])),
        state_ids=np.asarray(payload["state_ids"]),
        values=np.asarray(payload["prediction"]),
    )
    payload_hash = prediction_payload_sha256(
        state_ids=record.state_ids, prediction=record.values,
    )
    if not (
        record.family == family
        and record.seed == seed
        and record.bundle_role == FAMILY_ROLE[family]
        and record.checkpoint_sha256 == checkpoint_sha256
        and record.feature_payload_sha256 == feature_sha256
        and _scalar_text(np.asarray(payload["producer_commit"])) == implementation["commit"]
        and _scalar_text(np.asarray(payload["inference_sha256"]))
        == implementation["sha256"]["src/rail3/models/tmlr_v9b1/inference.py"]
        and _scalar_text(np.asarray(payload["producer_sha256"]))
        == implementation["sha256"]["scripts/predict_tmlr_v9b1_label_free.py"]
        and np.array_equal(record.state_ids.astype(str), state_ids.astype(str))
        and len(payload_hash) == 64
    ):
        raise PredictionBuildError("V9B1_SEED_PREDICTION_IDENTITY_DRIFT")
    return record


def _seed_payload(record: Any, implementation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "family": np.asarray(record.family),
        "bundle_role": np.asarray(record.bundle_role),
        "seed": np.asarray(record.seed, dtype=np.int64),
        "checkpoint_sha256": np.asarray(record.checkpoint_sha256),
        "feature_payload_sha256": np.asarray(record.feature_payload_sha256),
        "producer_commit": np.asarray(implementation["commit"]),
        "inference_sha256": np.asarray(
            implementation["sha256"]["src/rail3/models/tmlr_v9b1/inference.py"]
        ),
        "producer_sha256": np.asarray(
            implementation["sha256"]["scripts/predict_tmlr_v9b1_label_free.py"]
        ),
        "state_ids": np.asarray(record.state_ids),
        "prediction": np.asarray(record.values, dtype=np.float32),
    }


def _selected_payload_sha256(
    state_ids: np.ndarray, actions: np.ndarray, values: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"TMLR_V9B1_LAMBDA_ZERO_SELECTION_V1\0")
    for state_id, action, value in zip(state_ids.astype(str), actions.astype(str), values):
        for text in (state_id, action):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_ensemble_payload(
    payload: Mapping[str, np.ndarray], *, family: str, state_ids: np.ndarray,
    feature_sha256: str, checkpoint_hashes: tuple[str, ...],
    implementation: Mapping[str, Any],
) -> Any:
    from rail3.models.tmlr_v9b1.inference import (
        EnsembleContinuousPrediction, select_lambda_zero_actions,
    )

    if set(payload) != ENSEMBLE_KEYS:
        raise PredictionBuildError("V9B1_ENSEMBLE_KEYSET_DRIFT")
    if _scalar_text(payload["ensemble_id"]) != "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION":
        raise PredictionBuildError("V9B1_ENSEMBLE_RULE_DRIFT")
    raw_hashes = np.asarray(payload["checkpoint_sha256_by_seed"])
    if raw_hashes.shape != (3,) or tuple(raw_hashes.astype(str)) != checkpoint_hashes:
        raise PredictionBuildError("V9B1_ENSEMBLE_CHECKPOINT_ORDER_DRIFT")
    ensemble = EnsembleContinuousPrediction(
        family=_scalar_text(payload["family"]),
        bundle_role=_scalar_text(payload["bundle_role"]),
        feature_payload_sha256=_scalar_text(payload["feature_payload_sha256"]),
        checkpoint_sha256_by_seed=MappingProxyType(dict(zip(SEEDS, checkpoint_hashes))),
        state_ids=np.asarray(payload["state_ids"]),
        values=np.asarray(payload["prediction"]),
    )
    feasibility = np.asarray(payload["feasibility"])
    actions, selected = select_lambda_zero_actions(
        ensemble, feasibility, state_ids=np.asarray(payload["state_ids"]),
    )
    if not (
        ensemble.family == family
        and ensemble.bundle_role == FAMILY_ROLE[family]
        and ensemble.feature_payload_sha256 == feature_sha256
        and _scalar_text(np.asarray(payload["producer_commit"])) == implementation["commit"]
        and _scalar_text(np.asarray(payload["inference_sha256"]))
        == implementation["sha256"]["src/rail3/models/tmlr_v9b1/inference.py"]
        and _scalar_text(np.asarray(payload["producer_sha256"]))
        == implementation["sha256"]["scripts/predict_tmlr_v9b1_label_free.py"]
        and np.array_equal(ensemble.state_ids.astype(str), state_ids.astype(str))
        and np.array_equal(actions.astype(str), np.asarray(payload["selected_action"]).astype(str))
        and np.array_equal(selected, np.asarray(payload["selected_prediction"]))
    ):
        raise PredictionBuildError("V9B1_ENSEMBLE_SELECTION_IDENTITY_DRIFT")
    return ensemble


def _publish_or_validate(path: Path, payload: Mapping[str, Any], expected_keys: frozenset[str]) -> None:
    from rail3.cache.atomic_io import atomic_create_bytes

    encoded = _npz_bytes(payload)
    try:
        atomic_create_bytes(REPOSITORY / path, encoded)
    except FileExistsError:
        existing = _load_npz(REPOSITORY / path, expected_keys)
        if set(existing) != set(payload) or any(
            not np.array_equal(np.asarray(existing[name]), np.asarray(payload[name]), equal_nan=False)
            for name in payload
        ):
            raise PredictionBuildError("V9B1_EXISTING_PREDICTION_CONFLICT")


def _prediction_inventory_rows(
    feature_inventory: Mapping[str, Any], checkpoint_lock: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> list[dict[str, str]]:
    from rail3.models.tmlr_v9b1.inference import (
        prediction_payload_sha256, state_id_order_sha256,
    )
    _assert_exact_prediction_tree()

    feature_by_role = {row["role"]: row for row in feature_inventory["roles"]}
    checkpoint_by_pair = {
        (row["family"], row["seed"]): row["checkpoint"]["sha256"]
        for row in checkpoint_lock["runs"]
    }
    rows: list[dict[str, str]] = []
    for family in FAMILIES:
        role = FAMILY_ROLE[family]
        feature_sha = feature_by_role[role]["payload_sha256"]
        for seed in SEEDS:
            path = _seed_path(family, seed)
            payload = _load_npz(REPOSITORY / path, SEED_KEYS)
            record = _validate_seed_payload(
                payload, family=family, seed=seed,
                state_ids=payload["state_ids"],
                checkpoint_sha256=checkpoint_by_pair[(family, seed)],
                feature_sha256=feature_sha,
                implementation=implementation,
            )
            if state_id_order_sha256(record.state_ids) != feature_by_role[role]["state_ids_sha256"]:
                raise PredictionBuildError("V9B1_SEED_STATE_IDS_NOT_IN_FEATURE_BANK")
            identity = _identity(REPOSITORY / path)
            rows.append({
                "artifact_kind": "SEED_CONTINUOUS",
                "family": family,
                "seed": str(seed),
                "bundle_role": role,
                "rows": str(len(record.state_ids)),
                "actions": "5",
                "selected_action_rows": "0",
                "path": path.as_posix(),
                "bytes": str(identity["bytes"]),
                "sha256": identity["sha256"],
                "payload_sha256": prediction_payload_sha256(
                    state_ids=record.state_ids, prediction=record.values,
                ),
                "checkpoint_sha256": record.checkpoint_sha256,
                "feature_payload_sha256": feature_sha,
            })
        path = _ensemble_path(family)
        payload = _load_npz(REPOSITORY / path, ENSEMBLE_KEYS)
        checkpoint_hashes = tuple(checkpoint_by_pair[(family, seed)] for seed in SEEDS)
        ensemble = _validate_ensemble_payload(
            payload, family=family, state_ids=payload["state_ids"],
            feature_sha256=feature_sha, checkpoint_hashes=checkpoint_hashes,
            implementation=implementation,
        )
        if state_id_order_sha256(ensemble.state_ids) != feature_by_role[role]["state_ids_sha256"]:
            raise PredictionBuildError("V9B1_ENSEMBLE_STATE_IDS_NOT_IN_FEATURE_BANK")
        identity = _identity(REPOSITORY / path)
        rows.append({
            "artifact_kind": "ENSEMBLE_CONTINUOUS_AND_LAMBDA0_SELECTION",
            "family": family,
            "seed": "13+37+71",
            "bundle_role": role,
            "rows": str(len(ensemble.state_ids)),
            "actions": "5",
            "selected_action_rows": str(len(ensemble.state_ids)),
            "path": path.as_posix(),
            "bytes": str(identity["bytes"]),
            "sha256": identity["sha256"],
            "payload_sha256": hashlib.sha256((
                prediction_payload_sha256(
                    state_ids=ensemble.state_ids, prediction=ensemble.values,
                ) + _selected_payload_sha256(
                    ensemble.state_ids, payload["selected_action"], payload["selected_prediction"],
                )
            ).encode("ascii")).hexdigest(),
            "checkpoint_sha256": ";".join(checkpoint_hashes),
            "feature_payload_sha256": feature_sha,
        })
    return rows


def _inventory_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _read_inventory_csv() -> tuple[list[dict[str, str]], bytes]:
    encoded = _stable_bytes(REPOSITORY / PREDICTION_INVENTORY_PATH)
    stream = StringIO(encoded.decode("utf-8"), newline="")
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
        raise PredictionBuildError("V9B1_PREDICTION_INVENTORY_HEADER_DRIFT")
    rows = list(reader)
    if len(rows) != 24 or encoded != _inventory_csv_bytes(rows):
        raise PredictionBuildError("V9B1_PREDICTION_INVENTORY_ROW_DRIFT")
    return rows, encoded


def build_manifest_from_frozen_outputs(
    *, config: Mapping[str, Any], feature_inventory: Mapping[str, Any],
    checkpoint_lock: Mapping[str, Any], implementation: Mapping[str, Any],
) -> dict[str, Any]:
    rows, inventory_encoded = _read_inventory_csv()
    regenerated = _prediction_inventory_rows(
        feature_inventory, checkpoint_lock, implementation,
    )
    if rows != regenerated:
        raise PredictionBuildError("V9B1_PREDICTION_INVENTORY_CONTENT_DRIFT")
    feature_identity = _identity(REPOSITORY / FEATURE_INVENTORY_PATH)
    checkpoint_identity = _identity(REPOSITORY / CHECKPOINT_LOCK_PATH)
    state_manifest = feature_inventory["state_manifest"]
    trajectory_inventory = feature_inventory["trajectory_inventory"]
    return {
        "schema_version": "rail3.tmlr-v9b1-label-free-prediction-manifest.v1",
        "status": "TMLR_V9B1_COCO_PREDICTIONS_FROZEN",
        "panel_lock_id": config["authorities"]["panel_manifest"]["lock_id"],
        "panel_sha256": config["authorities"]["panel_manifest"]["sha256"],
        "state_manifest": state_manifest,
        "trajectory_inventory": trajectory_inventory,
        "sam_identity": feature_inventory["sam_identity"],
        "feature_inventory": feature_identity,
        "feature_schema_id": "PROSPECTIVE_V1",
        "feature_schema_sha256": implementation["sha256"][
            "src/rail3/models/tmlr_v9b1/inference.py"
        ],
        "checkpoint_lock": checkpoint_identity,
        "model_families": list(FAMILIES),
        "seed_order": list(SEEDS),
        "ensemble_rule": "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION",
        "selection_lambda": 0.0,
        "selection_action_order": ["STOP", "A3", "A4", "A5", "A6"],
        "seed_prediction_artifacts": 18,
        "ensemble_prediction_artifacts": 6,
        "continuous_prediction_rows": 18 * 20_000,
        "ensemble_rows": 6 * 20_000,
        "selected_action_rows": 6 * 20_000,
        "prediction_inventory": {
            "path": PREDICTION_INVENTORY_PATH.as_posix(),
            "bytes": len(inventory_encoded),
            "sha256": hashlib.sha256(inventory_encoded).hexdigest(),
            "rows": 24,
        },
        "prediction_artifacts": rows,
        "implementation_commit": implementation["commit"],
        "implementation_sha256": implementation["sha256"],
        "coco_public_schema_read_count_frozen": 2,
        "coco_schema_reads_added": 0,
        "coco_per_image_gt_materialized": 0,
        "performance_results": 0,
        "distribution_analysis_outputs": 0,
        "checks": {
            "zero_missing_expected_prediction": True,
            "zero_duplicate_key": True,
            "zero_nonfinite": True,
            "all_state_ids_known": True,
            "all_actions_legal": True,
            "all_model_ids_frozen": True,
            "all_checkpoint_hashes_match": True,
            "continuous_float64_mean_before_selection": True,
            "manifest_rebuild_stable": True,
        },
    }


def predict() -> dict[str, Any]:
    from rail3.cache.atomic_io import atomic_create_bytes
    from rail3.models.tmlr_v9b1.inference import (
        ensemble_continuous_predictions, predict_continuous_residuals,
        select_lambda_zero_actions,
    )

    if (REPOSITORY / PREDICTION_MANIFEST_PATH).exists():
        raise PredictionBuildError("V9B1_PREDICTION_MANIFEST_ALREADY_EXISTS_USE_CHECK")
    implementation = _clean_committed_identity()
    config, _ = _load_json(REPOSITORY / CONFIG_PATH, "CONFIG")
    if not (
        config.get("schema_version") == "rail3.tmlr-v9b1-coco-label-free-zeroshot.v1"
        and config.get("read_boundary", {}).get("coco_public_schema_read_count_frozen") == 2
        and config.get("read_boundary", {}).get("coco_annotation_source_reads_authorized") == 0
        and config.get("hard_zero_counters", {}).get("performance_results") == 0
    ):
        raise PredictionBuildError("V9B1_PREDICTION_CONFIG_DRIFT")
    (
        feature_inventory, features, checkpoint_lock, fullfit_config, models,
        input_snapshot,
    ) = _load_frozen_inputs_fixed_point()
    feature_by_role = {row["role"]: row for row in feature_inventory["roles"]}
    seed_records: dict[str, dict[int, Any]] = {}
    for family in FAMILIES:
        role = FAMILY_ROLE[family]
        feature_sha = feature_by_role[role]["payload_sha256"]
        seed_records[family] = {}
        for seed in SEEDS:
            path = _seed_path(family, seed)
            if (REPOSITORY / path).exists():
                payload = _load_npz(REPOSITORY / path, SEED_KEYS)
                record = _validate_seed_payload(
                    payload, family=family, seed=seed,
                    state_ids=features[role].state_ids,
                    checkpoint_sha256=models[(family, seed)].checkpoint_sha256,
                    feature_sha256=feature_sha,
                    implementation=implementation,
                )
            else:
                record = predict_continuous_residuals(
                    checkpoint=models[(family, seed)],
                    features=features[role],
                    feature_payload_sha256=feature_sha,
                    device=torch.device("cpu"),
                )
                _publish_or_validate(
                    path, _seed_payload(record, implementation), SEED_KEYS,
                )
            seed_records[family][seed] = record
        ensemble = ensemble_continuous_predictions(seed_records[family])
        feasibility = np.asarray(features[role].action_x16[:, :, 0])
        actions, selected = select_lambda_zero_actions(
            ensemble, feasibility, state_ids=features[role].state_ids,
        )
        ensemble_payload = {
            "family": np.asarray(family),
            "bundle_role": np.asarray(role),
            "ensemble_id": np.asarray("CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"),
            "feature_payload_sha256": np.asarray(feature_sha),
            "checkpoint_sha256_by_seed": np.asarray([
                seed_records[family][seed].checkpoint_sha256 for seed in SEEDS
            ]),
            "producer_commit": np.asarray(implementation["commit"]),
            "inference_sha256": np.asarray(
                implementation["sha256"]["src/rail3/models/tmlr_v9b1/inference.py"]
            ),
            "producer_sha256": np.asarray(
                implementation["sha256"]["scripts/predict_tmlr_v9b1_label_free.py"]
            ),
            "state_ids": np.asarray(ensemble.state_ids),
            "prediction": np.asarray(ensemble.values, dtype=np.float64),
            "feasibility": feasibility.astype(bool),
            "selected_action": actions,
            "selected_prediction": selected.astype(np.float64),
        }
        _publish_or_validate(_ensemble_path(family), ensemble_payload, ENSEMBLE_KEYS)
    if _clean_committed_identity(implementation["commit"]) != implementation:
        raise PredictionBuildError("V9B1_PREDICTION_IMPLEMENTATION_CHANGED_DURING_RUN")
    _assert_authority_snapshot(input_snapshot)
    rows = _prediction_inventory_rows(
        feature_inventory, checkpoint_lock, implementation,
    )
    inventory_encoded = _inventory_csv_bytes(rows)
    try:
        atomic_create_bytes(REPOSITORY / PREDICTION_INVENTORY_PATH, inventory_encoded)
    except FileExistsError:
        if _stable_bytes(REPOSITORY / PREDICTION_INVENTORY_PATH) != inventory_encoded:
            raise PredictionBuildError("V9B1_PREDICTION_INVENTORY_CONFLICT")
    _assert_authority_snapshot(input_snapshot)
    first = build_manifest_from_frozen_outputs(
        config=config, feature_inventory=feature_inventory,
        checkpoint_lock=checkpoint_lock, implementation=implementation,
    )
    second = build_manifest_from_frozen_outputs(
        config=config, feature_inventory=feature_inventory,
        checkpoint_lock=checkpoint_lock, implementation=implementation,
    )
    first_encoded = _canonical_json_bytes(first) + b"\n"
    second_encoded = _canonical_json_bytes(second) + b"\n"
    if first_encoded != second_encoded or hashlib.sha256(first_encoded).digest() != hashlib.sha256(second_encoded).digest():
        raise PredictionBuildError("STOP-BLOCKED_TMLR_V9B1_MANIFEST_DRIFT")
    if _clean_committed_identity(implementation["commit"]) != implementation:
        raise PredictionBuildError("V9B1_PREDICTION_IMPLEMENTATION_CHANGED_BEFORE_PUBLISH")
    _assert_authority_snapshot(input_snapshot)
    atomic_create_bytes(REPOSITORY / PREDICTION_MANIFEST_PATH, first_encoded)
    return first


def check() -> dict[str, Any]:
    published, encoded = _load_canonical_json(
        REPOSITORY / PREDICTION_MANIFEST_PATH, "PREDICTION_MANIFEST",
    )
    implementation = _clean_committed_identity(
        str(published.get("implementation_commit", ""))
    )
    config, _ = _load_json(REPOSITORY / CONFIG_PATH, "CONFIG")
    (
        feature_inventory, _, checkpoint_lock, fullfit_config, _, input_snapshot,
    ) = _load_frozen_inputs_fixed_point()
    first = build_manifest_from_frozen_outputs(
        config=config, feature_inventory=feature_inventory,
        checkpoint_lock=checkpoint_lock, implementation=implementation,
    )
    second = build_manifest_from_frozen_outputs(
        config=config, feature_inventory=feature_inventory,
        checkpoint_lock=checkpoint_lock, implementation=implementation,
    )
    rebuilt = _canonical_json_bytes(first) + b"\n"
    if first != second or encoded != rebuilt or published != first:
        raise PredictionBuildError("STOP-BLOCKED_TMLR_V9B1_MANIFEST_DRIFT")
    _assert_authority_snapshot(input_snapshot)
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", metavar="TOKEN")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            result = check()
        else:
            if arguments.predict != FORMAL_TOKEN:
                raise PredictionBuildError("V9B1_PREDICTION_FORMAL_TOKEN_MISMATCH")
            result = predict()
    except Exception as error:
        print(f"STOP-BLOCKED_TMLR_V9B1_PREDICTIONS: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "manifest_sha256": hashlib.sha256(_canonical_json_bytes(result) + b"\n").hexdigest(),
        "performance_results": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
