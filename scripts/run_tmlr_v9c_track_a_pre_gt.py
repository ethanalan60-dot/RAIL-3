#!/usr/bin/env python3
"""Freeze V9C Track-A features and predictions before the first COCO-GT read.

The entry point accepts only the final label-free trajectory lock, A0/A1/A2
cache objects, and the exact 18 VOC-FIT full-fit checkpoints.  It deliberately
has no annotation, target, or evaluation reader.  The one state with no
scientific A0 remains in the panel missingness ledger and is never imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

TOKEN = "TMLR_V9C_TRACK_A_PRE_GT_PREDICTIONS"
V9B1_CONFIG = Path("configs/experiments/tmlr_v9b1_coco_label_free_zeroshot.json")
V9_PROTOCOL_CONFIG = Path("configs/experiments/tmlr_v9_coco_external_v2.json")
V9C_ADDENDUM = Path("configs/experiments/tmlr_v9c_utility_aligned_addendum.json")
EXECUTION_PLAN = Path("docs/exec-plans/012-tmlr-v9c-utility-aligned-pre-gt.md")
TRACK_B_NO_GO_RECORD = Path("docs/experiments/TMLR_V9C_TRACK_B_NO_GO.md")
FINAL_TRAJECTORY_LOCK = Path(
    "artifacts/paper/source_data/tmlr_v9c/final_trajectory_lock.json"
)
TECHNICAL_REPORT = Path(
    "artifacts/paper/source_data/tmlr_v9c/technical_missingness_report.json"
)
FINAL_TRAJECTORY_INVENTORY = Path(
    "artifacts/paper/source_data/tmlr_v9b1/coco_external_trajectory_inventory.json"
)
CHECKPOINT_LOCK = Path(
    "artifacts/paper/source_data/tmlr_v9b1/fullfit_checkpoint_lock.json"
)
SOURCE_ROOT = Path("artifacts/paper/source_data/tmlr_v9c")
FEATURE_ROOT = Path("artifacts/features/tmlr-v9c/track-a")
PREDICTION_ROOT = Path("artifacts/predictions/tmlr-v9c/track-a")
FEATURE_INVENTORY = SOURCE_ROOT / "track_a_feature_inventory.json"
PREDICTION_INVENTORY = SOURCE_ROOT / "track_a_coco_prediction_inventory.csv"
PREDICTION_MANIFEST_BUILD1 = SOURCE_ROOT / "track_a_coco_prediction_manifest_build1.json"
PREDICTION_MANIFEST_BUILD2 = SOURCE_ROOT / "track_a_coco_prediction_manifest_build2.json"
PREDICTION_LOCK = SOURCE_ROOT / "track_a_coco_prediction_lock.json"
LEGACY_OUTPUTS = (
    SOURCE_ROOT / "track_a_prediction_inventory.csv",
    SOURCE_ROOT / "track_a_prediction_manifest.json",
    SOURCE_ROOT / "track_a_prediction_manifest_double_verification.json",
    SOURCE_ROOT / "prediction_before_gt_lock.json",
    SOURCE_ROOT / "track_a_coco_prediction_manifest.json",
)

FINAL_TRAJECTORY_LOCK_SHA256 = (
    "178311abce739304665e6fa37aba203308bc56995b20cdeea391293e5e0ee5b1"
)
FINAL_TRAJECTORY_INVENTORY_SHA256 = (
    "8a16df8045d6c4ab5b35f962553c6a20fa75f8448ac6f56e7a09fc92ccfb91ef"
)
CHECKPOINT_LOCK_SHA256 = (
    "8bc23e80e1be701f3db3a5549e7df6678edd5016caa8847b7ee474da9e1e19d2"
)
V9C_ADDENDUM_SHA256 = (
    "e1c3bbdcdc16c2348cc861d85919fb12f591bcfd37b62b5527a356adb6fc53f8"
)
TRACK_B_NO_GO_RECORD_SHA256 = (
    "d7fcb2e17d70d5d7e35b7171576f9231c1ed11be8d9e7c7ba7e36272010ae8f6"
)
TRACK_B_NO_GO_COMMIT = "3d62385a39ad268ee0f2ac6b2a719d33d726dd33"
PANEL_SHA256 = (
    "a0feb5f180d473d791cd9163ce062717b602ce3b1a049c3c3cc25132e06b2529"
)
PANEL_LOCK_ID = (
    "tmlr_v9a_r_coco_panel_"
    "9e80da655198b353e1f0e224282710acf214b96ea6f685f2d43c613f8d418886"
)
FAMILIES = ("R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P", "R3_P")
SEEDS = (13, 37, 71)
ROLES = ("F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P")
FAMILY_ROLE = {
    "R0_SMALL_P": "F0_P",
    "R0_CM_P": "F0_P",
    "R1_P": "F0_P",
    "RECT_P": "RECT_FULL_RASTER_P",
    "UNION_P": "UNION_FULL_RASTER_P",
    "R3_P": "F0_P",
}
FEATURE_PATHS = {
    "F0_P": FEATURE_ROOT / "f0" / "features.npz",
    "RECT_FULL_RASTER_P": FEATURE_ROOT / "rect" / "features.npz",
    "UNION_FULL_RASTER_P": FEATURE_ROOT / "union" / "features.npz",
}
ACTION_ORDER = ("STOP", "A3", "A4", "A5", "A6")
IMPLEMENTATION_PATHS = (
    V9B1_CONFIG,
    V9_PROTOCOL_CONFIG,
    V9C_ADDENDUM,
    EXECUTION_PLAN,
    TRACK_B_NO_GO_RECORD,
    Path("src/rail3/models/tmlr_v9b1/inference.py"),
    Path("src/rail3/sam/coco_v9b1_trajectory.py"),
    Path("src/rail3/diagnostic/representations.py"),
    Path("src/rail3/regions/m06e_causal.py"),
    Path("scripts/build_tmlr_v9b1_label_free_features.py"),
    Path("scripts/predict_tmlr_v9b1_label_free.py"),
    Path("scripts/run_tmlr_v9c_track_a_pre_gt.py"),
)
INVENTORY_FIELDS = (
    "artifact_kind", "family", "seed", "bundle_role", "rows", "actions",
    "path", "bytes", "sha256", "payload_sha256", "checkpoint_sha256",
    "scaler_sha256", "feature_payload_sha256",
)
FEATURE_KEYS = frozenset({
    "atom_x", "state_x", "action_x16", "state_lengths", "state_ids",
    "trajectory_state_ids", "image_group_ids", "atom_ids",
})
SEED_KEYS = frozenset({
    "family", "bundle_role", "seed", "checkpoint_sha256",
    "scaler_sha256", "feature_payload_sha256", "producer_commit",
    "producer_sha256", "state_ids", "prediction",
})
ENSEMBLE_KEYS = frozenset({
    "family", "bundle_role", "ensemble_id", "checkpoint_sha256_by_seed",
    "scaler_sha256_by_seed", "feature_payload_sha256", "producer_commit",
    "producer_sha256", "state_ids", "prediction", "feasibility",
    "selected_action", "selected_prediction",
})


class TrackAPreGTError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable(path: Path) -> bytes:
    from rail3.sam.coco_v9b1_trajectory import read_stable_regular_bytes

    return read_stable_regular_bytes(path)


def _identity(relative: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    encoded = _stable(REPOSITORY / relative)
    digest = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise TrackAPreGTError(f"V9C_IDENTITY_DRIFT:{relative}")
    return {"path": relative.as_posix(), "bytes": len(encoded), "sha256": digest}


def _load_json(relative: Path, *, canonical: bool = True) -> dict[str, Any]:
    encoded = _stable(REPOSITORY / relative)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrackAPreGTError(f"V9C_JSON_INVALID:{relative}") from error
    if not isinstance(value, dict) or (canonical and encoded != _canonical(value) + b"\n"):
        raise TrackAPreGTError(f"V9C_JSON_CONTRACT_DRIFT:{relative}")
    return value


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise TrackAPreGTError(f"V9C_GIT_FAILED:{' '.join(arguments)}")
    return result.stdout.strip()


def _validate_track_b_no_go() -> dict[str, Any]:
    identity = _identity(TRACK_B_NO_GO_RECORD, TRACK_B_NO_GO_RECORD_SHA256)
    encoded = _stable(REPOSITORY / TRACK_B_NO_GO_RECORD)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TrackAPreGTError("V9C_TRACK_B_NO_GO_RECORD_INVALID") from error
    required_markers = (
        "TMLR_V9C_TRACK_B_NO_GO_TRACK_A_ENABLED",
        "TRACK_B_STATUS = NO_GO",
        "TRACK_B_EVALUATED = FALSE",
        "TRACK_B_SCIENTIFIC_FAILURE = FALSE",
        "TRACK_A_STATUS = READY",
        "COCO_GT_READS = 0",
        "MODEL_PREDICTIONS = 0",
        "PERFORMANCE_RESULTS = 0",
        "NEXT_SAFE_STAGE = TRACK_A_PREDICTION_LOCK",
        "Authenticated VOC FIT cross-fit assets required for Track B are unavailable",
    )
    if any(marker not in text for marker in required_markers):
        raise TrackAPreGTError("V9C_TRACK_B_NO_GO_SEMANTICS_DRIFT")
    if (
        _git("rev-list", "-1", "HEAD", "--", TRACK_B_NO_GO_RECORD.as_posix())
        != TRACK_B_NO_GO_COMMIT
    ):
        raise TrackAPreGTError("V9C_TRACK_B_NO_GO_COMMIT_DRIFT")
    if subprocess.run(
        [
            "git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor",
            TRACK_B_NO_GO_COMMIT, _git("rev-parse", "HEAD"),
        ],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise TrackAPreGTError("V9C_TRACK_B_NO_GO_NOT_COMMITTED_ANCESTOR")
    return {
        "record": identity,
        "disposition_commit": TRACK_B_NO_GO_COMMIT,
        "status": "NO_GO",
        "evaluated": False,
        "scientific_failure": False,
        "reason_code": "AUTHENTICATED_VOC_FIT_CROSSFIT_ASSETS_UNAVAILABLE",
        "track_a_status": "READY",
        "excluded_from_v9c_evaluation": True,
    }


def _implementation(frozen_commit: str | None = None) -> dict[str, Any]:
    if Path.cwd().resolve() != REPOSITORY:
        raise TrackAPreGTError("V9C_CWD_DRIFT")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise TrackAPreGTError("V9C_WORKTREE_NOT_CLEAN")
    head = _git("rev-parse", "HEAD")
    if frozen_commit is not None:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", frozen_commit, head],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            raise TrackAPreGTError("V9C_IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    hashes: dict[str, str] = {}
    for path in IMPLEMENTATION_PATHS:
        if _git("ls-files", "--error-unmatch", path.as_posix()) != path.as_posix():
            raise TrackAPreGTError(f"V9C_IMPLEMENTATION_NOT_TRACKED:{path}")
        hashes[path.as_posix()] = _identity(path)["sha256"]
    return {"commit": frozen_commit or head, "sha256": hashes}


def _create_once(relative: Path, encoded: bytes) -> dict[str, Any]:
    from rail3.cache.atomic_io import atomic_create_bytes

    try:
        atomic_create_bytes(REPOSITORY / relative, encoded)
    except FileExistsError:
        if _stable(REPOSITORY / relative) != encoded:
            raise TrackAPreGTError(f"V9C_CREATE_ONCE_CONFLICT:{relative}")
    return _identity(relative)


def _npz_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **{key: payload[key] for key in sorted(payload)})
    return stream.getvalue()


def _load_npz(relative: Path, expected: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(BytesIO(_stable(REPOSITORY / relative)), allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise TrackAPreGTError(f"V9C_NPZ_SCHEMA_DRIFT:{relative}")
        return {key: np.asarray(archive[key]) for key in archive.files}


def _feature_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    schema = config.get("feature_schema")
    expected = {
        "id": "PROSPECTIVE_V1",
        "atom_dimension": 27,
        "state_dimension": 41,
        "action_dimension": 16,
        "state_class_one_hot_indices": [21, 41],
        "class_one_hot_order": "VOC class id 1 through 20",
        "dataset_identity_feature": False,
        "future_runtime_feature": False,
        "ground_truth_inference_feature": False,
    }
    if schema != expected:
        raise TrackAPreGTError("V9C_FEATURE_SCHEMA_DRIFT")
    return {"contract": expected, "sha256": hashlib.sha256(_canonical(expected)).hexdigest()}


def _validate_authorities() -> dict[str, Any]:
    track_b_no_go = _validate_track_b_no_go()
    lock_identity = _identity(FINAL_TRAJECTORY_LOCK, FINAL_TRAJECTORY_LOCK_SHA256)
    inventory_identity = _identity(
        FINAL_TRAJECTORY_INVENTORY, FINAL_TRAJECTORY_INVENTORY_SHA256,
    )
    checkpoint_identity = _identity(CHECKPOINT_LOCK, CHECKPOINT_LOCK_SHA256)
    _identity(V9C_ADDENDUM, V9C_ADDENDUM_SHA256)
    lock = _load_json(FINAL_TRAJECTORY_LOCK)
    inventory = _load_json(FINAL_TRAJECTORY_INVENTORY)
    addendum = _load_json(V9C_ADDENDUM, canonical=False)
    report = _load_json(TECHNICAL_REPORT)
    if not (
        lock.get("status") == "TMLR_V9C_FINAL_TRAJECTORIES_READY"
        and lock.get("states") == 20_000
        and lock.get("final_inventory") == inventory_identity
        and lock.get("panel", {}).get("sha256") == PANEL_SHA256
        and inventory.get("status") == "TMLR_V9C_FINAL_TRAJECTORIES_ACCOUNTED"
        and inventory.get("ready") is True
        and inventory.get("state_count") == 20_000
        and inventory.get("trajectory_state_count") == 19_999
        and inventory.get("trajectory_record_count") == 139_993
        and inventory.get("observable_cache_snapshot", {}).get("records") == 59_997
        and inventory.get("coco_gt_reads") == 0
        and inventory.get("model_predictions") == 0
        and inventory.get("performance_results") == 0
        and report.get("status") == "INFERENCE_COMPLETENESS_PASS"
        and report.get("combined", {}).get("trajectory_incomplete_state_count") == 4
        and report.get("state_level", {}).get("prior_e4_missing_state_count") == 1
        and report.get("action_level", {}).get("missing_action_count") == 3
        and addendum.get("result_blind") is True
        and addendum.get("prediction_lock", {}).get("lock_id")
        == "PREDICTION_BEFORE_GT_LOCK"
        and addendum.get("prediction_lock", {}).get("builds") == 2
        and addendum.get("track_a", {}).get("families") == list(FAMILIES)
        and addendum.get("track_a", {}).get("superseded_by_track_b") is False
        and addendum.get("zero_counters_at_freeze") == {
            "COCO_GT_READS": 0,
            "MODEL_PREDICTIONS": 0,
            "PERFORMANCE_RESULTS": 0,
        }
    ):
        raise TrackAPreGTError("V9C_PRE_GT_AUTHORITY_DRIFT")
    if lock.get("technical_missingness_report") != _identity(TECHNICAL_REPORT):
        raise TrackAPreGTError("V9C_TECHNICAL_MISSINGNESS_IDENTITY_DRIFT")
    config = _load_json(V9B1_CONFIG, canonical=False)
    if (
        config.get("authorities", {}).get("panel_manifest", {}).get("sha256")
        != PANEL_SHA256
        or config.get("authorities", {}).get("panel_manifest", {}).get("lock_id")
        != PANEL_LOCK_ID
        or config.get("hard_zero_counters", {}).get("performance_results") != 0
    ):
        raise TrackAPreGTError("V9C_PANEL_OR_ZERO_BOUNDARY_DRIFT")
    return {
        "lock": lock,
        "lock_identity": lock_identity,
        "inventory": inventory,
        "inventory_identity": inventory_identity,
        "checkpoint_identity": checkpoint_identity,
        "config": config,
        "track_b_no_go": track_b_no_go,
        "feature_schema": _feature_schema(_load_json(
            V9_PROTOCOL_CONFIG, canonical=False,
        )),
        "report": report,
    }


def _load_bundles(authority: Mapping[str, Any]) -> tuple[Any, ...]:
    from rail3.sam.coco_v9b1_cache import SecureCandidateCache, SecureTrajectoryCache
    from rail3.sam.coco_v9b1_trajectory import (
        FINAL_A0_CACHE_ROOT,
        build_semantic_states,
        derive_bundles_from_a0_cache,
        load_config,
        load_frozen_panel_images,
        load_taxonomy,
        observable_cache_snapshot,
        validate_e4_execution_registry,
        validate_frozen_execution_manifests,
        validate_small_authorities,
    )
    from rail3.sam.sam31_backend import official_sam31_model_spec

    config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, config)
    states = build_semantic_states(load_frozen_panel_images(REPOSITORY), load_taxonomy(REPOSITORY))
    validate_frozen_execution_manifests(
        REPOSITORY, config, states, require_global_clean=True,
    )
    missing = validate_e4_execution_registry(REPOSITORY, config)
    bundles = derive_bundles_from_a0_cache(
        states,
        model_spec_id=official_sam31_model_spec().model_spec_id,
        a0_cache=SecureCandidateCache(REPOSITORY / FINAL_A0_CACHE_ROOT),
        technical_missingness=missing,
    )
    if len(states) != 20_000 or len(bundles) != 19_999 or len(missing) != 1:
        raise TrackAPreGTError("V9C_EXECUTABLE_STATE_PARTITION_DRIFT")
    snapshot = observable_cache_snapshot(
        bundles,
        SecureCandidateCache(REPOSITORY / FINAL_A0_CACHE_ROOT),
        SecureTrajectoryCache(
            REPOSITORY / "artifacts/candidates/tmlr-v9b1/final/actions"
        ),
    )
    if snapshot != authority["inventory"]["observable_cache_snapshot"]:
        raise TrackAPreGTError("V9C_OBSERVABLE_TRAJECTORY_DRIFT")
    return bundles


def _feature_entry(role: str, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from rail3.models.tmlr_v9b1.inference import (
        RoleFeatureArrays, feature_payload_sha256, state_id_order_sha256,
    )

    arrays = RoleFeatureArrays(role_id=role, **payload)
    arrays.validate(expected_states=19_999)
    identity = _identity(FEATURE_PATHS[role])
    return {
        **identity,
        "role": role,
        "states": len(arrays.state_ids),
        "atoms": len(arrays.atom_ids),
        "state_ids_sha256": state_id_order_sha256(arrays.state_ids),
        "payload_sha256": feature_payload_sha256(payload),
    }


def _build_features(
    authority: Mapping[str, Any], implementation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from rail3.sam.coco_v9b1_cache import SecureTrajectoryCache
    from rail3.sam.sam31_backend import official_sam31_model_spec
    from scripts.build_tmlr_v9b1_label_free_features import (
        build_feature_arrays_from_bundles,
    )

    bundles = _load_bundles(authority)
    arrays = build_feature_arrays_from_bundles(
        bundles,
        SecureTrajectoryCache(
            REPOSITORY / "artifacts/candidates/tmlr-v9b1/final/actions"
        ),
        model_spec_id=official_sam31_model_spec().model_spec_id,
    )
    entries = []
    for role in ROLES:
        payload = arrays[role].npz_payload()
        _create_once(FEATURE_PATHS[role], _npz_bytes(payload))
        entries.append(_feature_entry(role, payload))
    missing_state_ids = authority["report"]["state_level"]["semantic_state_ids"]
    inventory = {
        "schema_version": "rail3.tmlr-v9c-track-a-feature-inventory.v1",
        "status": "TMLR_V9C_TRACK_A_FEATURES_FROZEN",
        "panel_lock_id": PANEL_LOCK_ID,
        "panel_sha256": PANEL_SHA256,
        "panel_state_count": 20_000,
        "predictable_state_count": 19_999,
        "technical_missing_prediction_state_count": 1,
        "technical_missing_prediction_state_ids": missing_state_ids,
        "trajectory_lock": authority["lock_identity"],
        "trajectory_inventory": authority["inventory_identity"],
        "track_b_no_go": authority["track_b_no_go"],
        "observable_cache_snapshot": authority["inventory"]["observable_cache_snapshot"],
        "feature_schema": authority["feature_schema"],
        "roles": entries,
        "implementation": dict(implementation),
        "input_boundary": {
            "observable_actions": ["A0", "A1", "A2"],
            "future_action_outcome_reads": 0,
            "coco_scaler_fit_rows": 0,
            "coco_gt_reads": 0,
            "performance_results": 0,
        },
    }
    _create_once(FEATURE_INVENTORY, _canonical(inventory) + b"\n")
    return inventory, arrays


def _load_features(authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from rail3.models.tmlr_v9b1.inference import RoleFeatureArrays, feature_payload_sha256

    inventory = _load_json(FEATURE_INVENTORY)
    expected_missing_state_ids = authority["report"]["state_level"][
        "semantic_state_ids"
    ]
    expected_input_boundary = {
        "observable_actions": ["A0", "A1", "A2"],
        "future_action_outcome_reads": 0,
        "coco_scaler_fit_rows": 0,
        "coco_gt_reads": 0,
        "performance_results": 0,
    }
    if not (
        inventory.get("schema_version")
        == "rail3.tmlr-v9c-track-a-feature-inventory.v1"
        and inventory.get("status") == "TMLR_V9C_TRACK_A_FEATURES_FROZEN"
        and inventory.get("panel_lock_id") == PANEL_LOCK_ID
        and inventory.get("panel_sha256") == PANEL_SHA256
        and inventory.get("panel_state_count") == 20_000
        and inventory.get("predictable_state_count") == 19_999
        and inventory.get("technical_missing_prediction_state_count") == 1
        and inventory.get("technical_missing_prediction_state_ids")
        == expected_missing_state_ids
        and inventory.get("trajectory_lock") == authority["lock_identity"]
        and inventory.get("trajectory_inventory") == authority["inventory_identity"]
        and inventory.get("track_b_no_go") == authority["track_b_no_go"]
        and inventory.get("observable_cache_snapshot")
        == authority["inventory"]["observable_cache_snapshot"]
        and inventory.get("feature_schema") == authority["feature_schema"]
        and [row.get("role") for row in inventory.get("roles", [])] == list(ROLES)
        and inventory.get("input_boundary") == expected_input_boundary
    ):
        raise TrackAPreGTError("V9C_FEATURE_INVENTORY_DRIFT")
    arrays: dict[str, Any] = {}
    for row in inventory["roles"]:
        role = row["role"]
        if _identity(FEATURE_PATHS[role]) != {
            key: row[key] for key in ("path", "bytes", "sha256")
        }:
            raise TrackAPreGTError("V9C_FEATURE_FILE_IDENTITY_DRIFT")
        payload = _load_npz(FEATURE_PATHS[role], FEATURE_KEYS)
        values = RoleFeatureArrays(role_id=role, **payload)
        values.validate(expected_states=19_999)
        if (
            feature_payload_sha256(payload) != row["payload_sha256"]
            or row != _feature_entry(role, payload)
        ):
            raise TrackAPreGTError("V9C_FEATURE_PAYLOAD_IDENTITY_DRIFT")
        arrays[role] = values
    reference = arrays["F0_P"].state_ids.astype(str)
    if any(not np.array_equal(reference, arrays[role].state_ids.astype(str)) for role in ROLES):
        raise TrackAPreGTError("V9C_FEATURE_ROLE_ORDER_DRIFT")
    return inventory, arrays


def _mask_action_technical_missingness(
    state_ids: np.ndarray, feasibility: np.ndarray, rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    result = np.asarray(feasibility).astype(bool, copy=True)
    ids = np.asarray(state_ids).astype(str)
    if result.shape != (len(ids), len(ACTION_ORDER)) or len(set(ids.tolist())) != len(ids):
        raise TrackAPreGTError("V9C_FEASIBILITY_GEOMETRY_DRIFT")
    index = {state_id: offset for offset, state_id in enumerate(ids)}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        state_id = str(row.get("semantic_state_id", ""))
        action = str(row.get("action_code", ""))
        key = (state_id, action)
        if action != "A4" or state_id not in index or key in seen:
            raise TrackAPreGTError("V9C_ACTION_MISSINGNESS_LEDGER_DRIFT")
        if not result[index[state_id], ACTION_ORDER.index(action)]:
            raise TrackAPreGTError("V9C_ACTION_MISSINGNESS_NOT_STRUCTURALLY_FEASIBLE")
        result[index[state_id], ACTION_ORDER.index(action)] = False
        seen.add(key)
    if len(seen) != 3 or not np.all(result[:, 0]) or np.any(~result.any(axis=1)):
        raise TrackAPreGTError("V9C_ACTION_MISSINGNESS_APPLICATION_DRIFT")
    return result


def _seed_path(family: str, seed: int) -> Path:
    return PREDICTION_ROOT / family / f"seed_{seed}.npz"


def _ensemble_path(family: str) -> Path:
    return PREDICTION_ROOT / family / "ensemble.npz"


def _text_scalar(value: np.ndarray) -> str:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in {"U", "S"}:
        raise TrackAPreGTError("V9C_STRING_SCALAR_DRIFT")
    return str(raw.item())


def _selected_hash(state_ids: np.ndarray, actions: np.ndarray, values: np.ndarray) -> str:
    digest = hashlib.sha256(b"TMLR_V9C_TRACK_A_LAMBDA0_SELECTION_V1\0")
    for state_id, action, value in zip(
        np.asarray(state_ids).astype(str), np.asarray(actions).astype(str), np.asarray(values),
    ):
        digest.update(state_id.encode("utf-8") + b"\0" + action.encode("ascii") + b"\0")
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def _hash_field(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _scaler_profile(scalers: Mapping[str, Any]) -> str:
    base_keys = {"state_mean", "state_scale", "action_mean", "action_scale"}
    atomic_keys = base_keys | {"atom_mean", "atom_scale"}
    keys = set(scalers)
    if keys == base_keys:
        return "GLOBAL_STATE_ACTION"
    if keys == atomic_keys:
        return "ATOMIC_STATE_ACTION_ATOM"
    raise TrackAPreGTError("V9C_SCALER_KEY_SET_DRIFT")


def _scaler_payload_sha256(model: Any) -> str:
    scalers = model.scalers
    if not isinstance(scalers, Mapping) or model.bundle_role not in ROLES:
        raise TrackAPreGTError("V9C_SCALER_ROOT_DRIFT")
    profile = _scaler_profile(scalers)
    expected_dimensions = {"state": 41, "action": 16, "atom": 27}
    digest = hashlib.sha256(b"RAIL3_TMLR_V9B1_FROZEN_VOC_SCALER_V1\0")
    for field in ("PROSPECTIVE_V1", model.bundle_role, profile):
        _hash_field(digest, field)
    digest.update(len(scalers).to_bytes(4, byteorder="big", signed=False))
    for name in sorted(scalers):
        value = np.asarray(scalers[name])
        dimension = expected_dimensions[name.rsplit("_", 1)[0]]
        if (
            value.shape != (dimension,)
            or value.dtype != np.dtype(np.float32)
            or not np.isfinite(value).all()
            or (name.endswith("_scale") and np.any(value <= 0))
        ):
            raise TrackAPreGTError("V9C_SCALER_PAYLOAD_DRIFT")
        canonical = np.asarray(value, dtype="<f4", order="C")
        _hash_field(digest, name)
        _hash_field(digest, "<f4")
        digest.update(canonical.ndim.to_bytes(4, byteorder="big", signed=False))
        for size in canonical.shape:
            digest.update(int(size).to_bytes(8, byteorder="big", signed=False))
        encoded = canonical.tobytes(order="C")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _scaler_hashes(
    models: Mapping[tuple[str, int], Any], checkpoint_lock: Mapping[str, Any],
) -> dict[str, Any]:
    from rail3.models.tmlr_v9b1.inference import FrozenCheckpointModel

    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if set(models) != expected:
        raise TrackAPreGTError("V9C_SCALER_MODEL_SET_DRIFT")
    checkpoint_by_pair = {
        (row["family"], row["seed"]): row["checkpoint"]["sha256"]
        for row in _checkpoint_identities(checkpoint_lock)
    }
    by_family: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    for family in FAMILIES:
        by_seed: dict[str, str] = {}
        for seed in SEEDS:
            model = models[(family, seed)]
            if (
                not isinstance(model, FrozenCheckpointModel)
                or model.family != family
                or model.seed != seed
                or model.bundle_role != FAMILY_ROLE[family]
                or model.checkpoint_sha256 != checkpoint_by_pair[(family, seed)]
            ):
                raise TrackAPreGTError("V9C_SCALER_MODEL_IDENTITY_DRIFT")
            by_seed[str(seed)] = _scaler_payload_sha256(model)
            bindings.append({
                "family": family,
                "seed": seed,
                "bundle_role": model.bundle_role,
                "checkpoint_sha256": model.checkpoint_sha256,
                "scaler_sha256": by_seed[str(seed)],
            })
        consensus = set(by_seed.values())
        if len(consensus) != 1:
            raise TrackAPreGTError("V9C_SCALER_SEED_CONSENSUS_DRIFT")
        by_family[family] = {
            "bundle_role": FAMILY_ROLE[family],
            "scaler_profile": _scaler_profile(models[(family, SEEDS[0])].scalers),
            "scaler_sha256_by_seed": by_seed,
            "seed_consensus_scaler_sha256": next(iter(consensus)),
        }
    unique_scaler_count = len({row["scaler_sha256"] for row in bindings})
    if unique_scaler_count != 4:
        raise TrackAPreGTError("V9C_SCALER_UNIQUE_IDENTITY_COUNT_DRIFT")
    return {
        "schema_version": "rail3.tmlr-v9b1-frozen-voc-scaler.v1",
        "hash_schema_version": "RAIL3_TMLR_V9B1_FROZEN_VOC_SCALER_V1",
        "hash_serialization": (
            "DOMAIN_PREFIX;UINT64_BE_LENGTH_PREFIXED_UTF8_FIELDS;"
            "UINT32_BE_NDIM;UINT64_BE_DIMS_AND_NBYTES;LITTLE_ENDIAN_F32_C_BYTES"
        ),
        "source": "AUTHENTICATED_VOC_FIT_CHECKPOINT_PAYLOAD",
        "prospective_schema_id": "PROSPECTIVE_V1",
        "unique_scaler_count": unique_scaler_count,
        "by_family": by_family,
        "checkpoint_scaler_bindings": bindings,
        "checkpoint_scaler_bindings_sha256": hashlib.sha256(
            _canonical(bindings)
        ).hexdigest(),
    }


def _assert_prediction_tree(*, complete: bool) -> None:
    root = REPOSITORY / PREDICTION_ROOT
    if not root.exists():
        if complete:
            raise TrackAPreGTError("V9C_PREDICTION_ROOT_MISSING")
        return
    if root.is_symlink() or not root.is_dir():
        raise TrackAPreGTError("V9C_PREDICTION_ROOT_INVALID")
    observed_families = set(os.listdir(root))
    expected_families = set(FAMILIES)
    if not observed_families <= expected_families or (
        complete and observed_families != expected_families
    ):
        raise TrackAPreGTError("V9C_PREDICTION_FAMILY_SET_DRIFT")
    expected_files = {"ensemble.npz", *(f"seed_{seed}.npz" for seed in SEEDS)}
    for family in observed_families:
        directory = root / family
        if directory.is_symlink() or not directory.is_dir():
            raise TrackAPreGTError("V9C_PREDICTION_FAMILY_PATH_INVALID")
        observed_files = set(os.listdir(directory))
        if not observed_files <= expected_files or (
            complete and observed_files != expected_files
        ):
            raise TrackAPreGTError("V9C_PREDICTION_FILE_SET_DRIFT")
        if any(
            (directory / name).is_symlink() or not (directory / name).is_file()
            for name in observed_files
        ):
            raise TrackAPreGTError("V9C_PREDICTION_FILE_INVALID")


def _load_seed_record(
    *, family: str, seed: int, feature_row: Mapping[str, Any],
    features: Any, checkpoint_sha256: str, scaler_sha256: str,
    implementation: Mapping[str, Any],
) -> Any:
    from rail3.models.tmlr_v9b1.inference import SeedContinuousPrediction

    payload = _load_npz(_seed_path(family, seed), SEED_KEYS)
    values = np.asarray(payload["prediction"])
    stored_seed = np.asarray(payload["seed"])
    stored_state_ids = np.asarray(payload["state_ids"])
    if not (
        _text_scalar(payload["family"]) == family
        and _text_scalar(payload["bundle_role"]) == FAMILY_ROLE[family]
        and stored_seed.shape == ()
        and stored_seed.dtype == np.dtype(np.int64)
        and int(stored_seed.item()) == seed
        and _text_scalar(payload["checkpoint_sha256"]) == checkpoint_sha256
        and _text_scalar(payload["scaler_sha256"]) == scaler_sha256
        and _text_scalar(payload["feature_payload_sha256"])
        == feature_row["payload_sha256"]
        and _text_scalar(payload["producer_commit"]) == implementation["commit"]
        and _text_scalar(payload["producer_sha256"])
        == implementation["sha256"]["scripts/run_tmlr_v9c_track_a_pre_gt.py"]
        and np.array_equal(
            stored_state_ids.astype(str),
            features.state_ids.astype(str),
        )
        and stored_state_ids.ndim == 1
        and stored_state_ids.dtype.kind in {"U", "S"}
        and values.dtype == np.dtype(np.float32)
        and values.shape == (19_999, 5)
        and np.isfinite(values).all()
    ):
        raise TrackAPreGTError("V9C_SEED_PREDICTION_DRIFT")
    return SeedContinuousPrediction(
        family=family,
        bundle_role=FAMILY_ROLE[family],
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        feature_payload_sha256=str(feature_row["payload_sha256"]),
        state_ids=stored_state_ids.astype(str),
        values=values,
    )


def _prediction_rows(
    feature_inventory: Mapping[str, Any], checkpoint_lock: Mapping[str, Any],
    implementation: Mapping[str, Any], features: Mapping[str, Any],
    action_missingness_rows: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[str, int], Any],
) -> list[dict[str, str]]:
    from rail3.models.tmlr_v9b1.inference import (
        prediction_payload_sha256,
    )

    by_role = {row["role"]: row for row in feature_inventory["roles"]}
    by_pair = {
        (row["family"], int(row["seed"])): row["checkpoint"]["sha256"]
        for row in checkpoint_lock["runs"]
    }
    scaler_registry = _scaler_hashes(models, checkpoint_lock)
    scaler_by_pair = {
        (family, seed): scaler_registry["by_family"][family][
            "scaler_sha256_by_seed"
        ][str(seed)]
        for family in FAMILIES
        for seed in SEEDS
    }
    result: list[dict[str, str]] = []
    for family in FAMILIES:
        role = FAMILY_ROLE[family]
        seed_values: list[np.ndarray] = []
        for seed in SEEDS:
            path = _seed_path(family, seed)
            record = _load_seed_record(
                family=family,
                seed=seed,
                feature_row=by_role[role],
                features=features[role],
                checkpoint_sha256=by_pair[(family, seed)],
                scaler_sha256=scaler_by_pair[(family, seed)],
                implementation=implementation,
            )
            seed_values.append(np.asarray(record.values, dtype=np.float64))
            payload = _load_npz(path, SEED_KEYS)
            identity = _identity(path)
            result.append({
                "artifact_kind": "SEED_CONTINUOUS",
                "family": family,
                "seed": str(seed),
                "bundle_role": role,
                "rows": "19999",
                "actions": "5",
                "path": path.as_posix(),
                "bytes": str(identity["bytes"]),
                "sha256": identity["sha256"],
                "payload_sha256": prediction_payload_sha256(
                    state_ids=payload["state_ids"], prediction=payload["prediction"],
                ),
                "checkpoint_sha256": by_pair[(family, seed)],
                "scaler_sha256": scaler_by_pair[(family, seed)],
                "feature_payload_sha256": by_role[role]["payload_sha256"],
            })
        path = _ensemble_path(family)
        payload = _load_npz(path, ENSEMBLE_KEYS)
        stored_checkpoint_hashes = np.asarray(payload["checkpoint_sha256_by_seed"])
        stored_scaler_hashes = np.asarray(payload["scaler_sha256_by_seed"])
        stored_state_ids = np.asarray(payload["state_ids"])
        stored_selected_actions = np.asarray(payload["selected_action"])
        expected_feasibility = _mask_action_technical_missingness(
            features[role].state_ids,
            features[role].action_x16[:, :, 0],
            action_missingness_rows,
        )
        expected_prediction = np.stack(seed_values, axis=0).mean(
            axis=0, dtype=np.float64,
        )
        selected_indices = np.argmin(
            np.where(expected_feasibility, expected_prediction, np.inf), axis=1,
        )
        expected_actions = np.asarray(ACTION_ORDER)[selected_indices]
        expected_selected = expected_prediction[
            np.arange(len(expected_prediction)), selected_indices
        ]
        if not (
            _text_scalar(payload["family"]) == family
            and _text_scalar(payload["bundle_role"]) == role
            and _text_scalar(payload["ensemble_id"])
            == "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"
            and stored_checkpoint_hashes.shape == (3,)
            and stored_checkpoint_hashes.dtype.kind in {"U", "S"}
            and tuple(stored_checkpoint_hashes.astype(str))
            == tuple(by_pair[(family, seed)] for seed in SEEDS)
            and stored_scaler_hashes.shape == (3,)
            and stored_scaler_hashes.dtype.kind in {"U", "S"}
            and tuple(stored_scaler_hashes.astype(str))
            == tuple(scaler_by_pair[(family, seed)] for seed in SEEDS)
            and _text_scalar(payload["feature_payload_sha256"])
            == by_role[role]["payload_sha256"]
            and _text_scalar(payload["producer_commit"]) == implementation["commit"]
            and _text_scalar(payload["producer_sha256"])
            == implementation["sha256"]["scripts/run_tmlr_v9c_track_a_pre_gt.py"]
            and np.array_equal(
                stored_state_ids.astype(str),
                features[role].state_ids.astype(str),
            )
            and stored_state_ids.ndim == 1
            and stored_state_ids.dtype.kind in {"U", "S"}
            and np.asarray(payload["prediction"]).dtype == np.dtype(np.float64)
            and np.asarray(payload["prediction"]).shape == (19_999, 5)
            and stored_selected_actions.shape == (19_999,)
            and stored_selected_actions.dtype.kind in {"U", "S"}
            and np.asarray(payload["selected_prediction"]).dtype
            == np.dtype(np.float64)
            and np.asarray(payload["selected_prediction"]).shape == (19_999,)
            and np.asarray(payload["feasibility"]).shape == (19_999, 5)
            and np.asarray(payload["feasibility"]).dtype == np.dtype(bool)
            and np.isfinite(np.asarray(payload["prediction"])).all()
            and np.isfinite(np.asarray(payload["selected_prediction"])).all()
            and np.array_equal(payload["prediction"], expected_prediction)
            and np.array_equal(payload["feasibility"], expected_feasibility)
            and np.array_equal(stored_selected_actions.astype(str), expected_actions)
            and np.array_equal(payload["selected_prediction"], expected_selected)
        ):
            raise TrackAPreGTError("V9C_ENSEMBLE_PREDICTION_DRIFT")
        identity = _identity(path)
        result.append({
            "artifact_kind": "ENSEMBLE_CONTINUOUS_AND_LAMBDA0_SELECTION",
            "family": family,
            "seed": "13+37+71",
            "bundle_role": role,
            "rows": "19999",
            "actions": "5",
            "path": path.as_posix(),
            "bytes": str(identity["bytes"]),
            "sha256": identity["sha256"],
            "payload_sha256": hashlib.sha256((
                prediction_payload_sha256(
                    state_ids=payload["state_ids"], prediction=payload["prediction"],
                ) + _selected_hash(
                    payload["state_ids"], payload["selected_action"],
                    payload["selected_prediction"],
                )
            ).encode("ascii")).hexdigest(),
            "checkpoint_sha256": ";".join(by_pair[(family, seed)] for seed in SEEDS),
            "scaler_sha256": scaler_registry["by_family"][family][
                "seed_consensus_scaler_sha256"
            ],
            "feature_payload_sha256": by_role[role]["payload_sha256"],
        })
    keys = {
        (row["artifact_kind"], row["family"], row["seed"])
        for row in result
    }
    paths = {row["path"] for row in result}
    if len(result) != 24 or len(keys) != 24 or len(paths) != 24:
        raise TrackAPreGTError("V9C_PREDICTION_ARTIFACT_COUNT_DRIFT")
    return result


def _inventory_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _checkpoint_identities(checkpoint_lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_pairs = [(family, seed) for family in FAMILIES for seed in SEEDS]
    runs = checkpoint_lock.get("runs")
    if not isinstance(runs, list) or [
        (row.get("family"), row.get("seed"))
        for row in runs
        if isinstance(row, Mapping)
    ] != expected_pairs:
        raise TrackAPreGTError("V9C_CHECKPOINT_PAIR_SET_DRIFT")
    result: list[dict[str, Any]] = []
    for row, (family, seed) in zip(runs, expected_pairs):
        checkpoint = row.get("checkpoint")
        if (
            row.get("bundle_role") != FAMILY_ROLE[family]
            or not isinstance(checkpoint, Mapping)
            or set(checkpoint) != {"path", "bytes", "sha256"}
        ):
            raise TrackAPreGTError("V9C_CHECKPOINT_IDENTITY_DRIFT")
        result.append({
            "family": family,
            "seed": seed,
            "bundle_role": FAMILY_ROLE[family],
            "checkpoint": dict(checkpoint),
        })
    return result


def _model_hashes(checkpoint_lock: Mapping[str, Any]) -> dict[str, Any]:
    identities = _checkpoint_identities(checkpoint_lock)
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        checkpoints = {
            str(seed): next(
                row["checkpoint"]["sha256"] for row in identities
                if row["family"] == family and int(row["seed"]) == seed
            )
            for seed in SEEDS
        }
        by_family[family] = {
            "checkpoint_sha256_by_seed": checkpoints,
            "model_set_sha256": hashlib.sha256(_canonical(checkpoints)).hexdigest(),
        }
    return {
        "checkpoint_identities": identities,
        "by_family": by_family,
        "panel_model_sha256": hashlib.sha256(_canonical(by_family)).hexdigest(),
    }


def _prediction_count_contract() -> dict[str, int]:
    return {
        "panel_states": 20_000,
        "predictable_states": 19_999,
        "missing_states": 1,
        "families": 6,
        "seed_models": 18,
        "prediction_artifacts": 24,
        "rows_per_prediction_artifact": 19_999,
        "expected_rows_per_prediction_artifact": 19_999,
        "missing_predictable_rows": 0,
        "seed_continuous_rows": 18 * 19_999,
        "expected_full_panel_seed_rows": 18 * 20_000,
        "missing_full_panel_seed_rows": 18,
        "ensemble_continuous_rows": 6 * 19_999,
        "selected_action_rows": 6 * 19_999,
        "seed_action_value_cells": 18 * 19_999 * 5,
        "ensemble_action_value_cells": 6 * 19_999 * 5,
        "expected_predictable_family_rows": 6 * 19_999,
        "observed_predictable_family_rows": 6 * 19_999,
        "expected_panel_family_rows": 6 * 20_000,
        "missing_panel_family_rows": 6,
        "duplicate_artifact_keys": 0,
        "duplicate_seed_prediction_keys": 0,
        "duplicate_ensemble_prediction_keys": 0,
        "duplicate_selected_action_keys": 0,
        "unexpected_states": 0,
        "nonfinite_seed_values": 0,
        "nonfinite_ensemble_values": 0,
        "nonfinite_selected_values": 0,
    }


def _technical_missingness_count_contract() -> dict[str, int]:
    return {
        "state_level_missing_states": 1,
        "source_action_level_missing_events": 3,
        "masked_family_action_cells": 6 * 3,
        "combined_trajectory_incomplete_states": 4,
    }


def _manifest(
    authority: Mapping[str, Any], feature_inventory: Mapping[str, Any],
    checkpoint_lock: Mapping[str, Any], implementation: Mapping[str, Any],
    features: Mapping[str, Any], models: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    encoded = _stable(REPOSITORY / PREDICTION_INVENTORY)
    reader = csv.DictReader(StringIO(encoded.decode("utf-8"), newline=""))
    rows = list(reader)
    regenerated = _prediction_rows(
        feature_inventory, checkpoint_lock, implementation, features,
        authority["inventory"]["action_technical_missingness"]["rows"],
        models,
    )
    if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS or rows != regenerated:
        raise TrackAPreGTError("V9C_PREDICTION_INVENTORY_DRIFT")
    model_hashes = _model_hashes(checkpoint_lock)
    scaler_hashes = _scaler_hashes(models, checkpoint_lock)
    return {
        "schema_version": "rail3.tmlr-v9c-track-a-coco-prediction-manifest.v1",
        "status": "TMLR_V9C_TRACK_A_COCO_PREDICTIONS_FROZEN",
        "lock_id": "PREDICTION_BEFORE_GT_LOCK",
        "track": "A_ORIGINAL_ZERO_SHOT_EXTERNAL_REPLICATION",
        "track_b_no_go": authority["track_b_no_go"],
        "panel": {"lock_id": PANEL_LOCK_ID, "sha256": PANEL_SHA256, "states": 20_000},
        "trajectory": {
            "lock": authority["lock_identity"],
            "inventory": authority["inventory_identity"],
            "sha256": authority["inventory_identity"]["sha256"],
        },
        "feature_inventory": _identity(FEATURE_INVENTORY),
        "feature_schema": authority["feature_schema"],
        "checkpoint_lock": authority["checkpoint_identity"],
        "model_sha256": model_hashes,
        "scaler_identity": scaler_hashes,
        "model_families": list(FAMILIES),
        "seed_order": list(SEEDS),
        "ensemble_rule": "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION",
        "majority_vote": False,
        "selection_lambda": 0.0,
        "selection_action_order": list(ACTION_ORDER),
        "coco_threshold_tuning": False,
        "calibration": False,
        "prediction_counts": _prediction_count_contract(),
        "technical_missingness_counts": _technical_missingness_count_contract(),
        "technical_missing_prediction_state_ids": feature_inventory[
            "technical_missing_prediction_state_ids"
        ],
        "prediction_key_definitions": {
            "seed_continuous": ["family", "seed", "state_id", "action"],
            "ensemble_continuous": ["family", "state_id", "action"],
            "selected_action": ["family", "state_id"],
        },
        "action_technical_missingness": authority["report"]["action_level"],
        "prediction_inventory": {
            "path": PREDICTION_INVENTORY.as_posix(),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "rows": len(rows),
        },
        "prediction_artifacts": rows,
        "implementation": dict(implementation),
        "counters_at_lock": {
            "COCO_GT_READS": 0,
            "MODEL_TRAINING_JOBS": 0,
            "MODEL_PREDICTION_JOBS": 18,
            "PERFORMANCE_RESULTS": 0,
        },
        "checks": {
            "all_predictable_states_predicted": True,
            "panel_missingness_preserved_without_imputation": True,
            "future_action_outcomes_not_used_as_features": True,
            "voc_fit_checkpoint_scalers_only": True,
            "all_scaler_payload_hashes_recorded": True,
            "no_coco_scaler_fit": True,
            "all_checkpoint_hashes_match": True,
            "continuous_float64_mean_before_selection": True,
            "majority_vote_not_used": True,
            "coco_threshold_tuning_not_used": True,
            "calibration_not_used": True,
            "zero_missing_expected_prediction": True,
            "zero_duplicate_prediction_key": True,
            "zero_nonfinite_prediction": True,
            "exact_six_family_prediction_tree": True,
            "technical_missing_actions_masked_only_at_selection": True,
            "manifest_rebuild_stable": True,
            "coco_gt_unread": True,
            "performance_results_zero": True,
        },
    }


def _prediction_lock_payload(
    *, authority: Mapping[str, Any], feature_inventory: Mapping[str, Any],
    manifest: Mapping[str, Any], implementation: Mapping[str, Any],
) -> dict[str, Any]:
    build1 = _identity(PREDICTION_MANIFEST_BUILD1)
    build2 = _identity(PREDICTION_MANIFEST_BUILD2)
    build1_bytes = _stable(REPOSITORY / PREDICTION_MANIFEST_BUILD1)
    build2_bytes = _stable(REPOSITORY / PREDICTION_MANIFEST_BUILD2)
    if build1["sha256"] != build2["sha256"] or build1_bytes != build2_bytes:
        raise TrackAPreGTError("V9C_PREDICTION_MANIFEST_DOUBLE_BUILD_DRIFT")
    return {
        "schema_version": "rail3.tmlr-v9c-track-a-coco-prediction-lock.v1",
        "status": "TMLR_V9C_TRACK_A_PREDICTION_LOCK_COMPLETE",
        "lock_id": "PREDICTION_BEFORE_GT_LOCK",
        "track_b_no_go": authority["track_b_no_go"],
        "prediction_manifests": {
            "build_count": 2,
            "build1": build1,
            "build2": build2,
            "canonical_bytes_equal": True,
            "manifest_sha256": build1["sha256"],
        },
        "prediction_inventory": _identity(PREDICTION_INVENTORY),
        "feature_inventory": _identity(FEATURE_INVENTORY),
        "implementation": dict(implementation),
        "panel": manifest["panel"],
        "trajectory_lock": authority["lock_identity"],
        "trajectory_inventory": authority["inventory_identity"],
        "checkpoint_lock": authority["checkpoint_identity"],
        "checkpoint_identities": manifest["model_sha256"]["checkpoint_identities"],
        "model_sha256": manifest["model_sha256"],
        "scaler_identity": manifest["scaler_identity"],
        "feature_schema": authority["feature_schema"],
        "model_families": list(FAMILIES),
        "seed_order": list(SEEDS),
        "ensemble_rule": "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION",
        "majority_vote": False,
        "coco_threshold_tuning": False,
        "calibration": False,
        "prediction_counts": manifest["prediction_counts"],
        "technical_missingness_counts": manifest["technical_missingness_counts"],
        "counters_at_lock": {
            "COCO_GT_READS": 0,
            "MODEL_TRAINING_JOBS": 0,
            "MODEL_PREDICTION_JOBS": 18,
            "PERFORMANCE_RESULTS": 0,
        },
        "checks": {
            "trajectory_sha_verified": True,
            "panel_sha_verified": True,
            "checkpoint_sha_verified": True,
            "feature_schema_verified": True,
            "voc_fit_scaler_identities_verified": True,
            "prediction_completeness_verified": True,
            "zero_duplicate_prediction_key": True,
            "zero_nonfinite_prediction": True,
            "exact_six_family_prediction_tree": True,
            "manifest_builds_byte_identical": True,
            "coco_gt_unread": True,
            "performance_results_zero": True,
        },
        "gt_read_authorized_only_after_this_lock_is_committed": True,
    }


def _predict(
    authority: Mapping[str, Any], feature_inventory: Mapping[str, Any],
    features: Mapping[str, Any], implementation: Mapping[str, Any],
) -> dict[str, Any]:
    from rail3.models.tmlr_v9b1.inference import (
        ensemble_continuous_predictions, predict_continuous_residuals,
        select_lambda_zero_actions,
    )
    from scripts.predict_tmlr_v9b1_label_free import _load_checkpoint_authority

    _assert_prediction_tree(complete=False)
    checkpoint_lock, _, models = _load_checkpoint_authority()
    expected_pairs = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if (
        _identity(CHECKPOINT_LOCK) != authority["checkpoint_identity"]
        or set(models) != expected_pairs
    ):
        raise TrackAPreGTError("V9C_CHECKPOINT_AUTHORITY_DRIFT")
    _checkpoint_identities(checkpoint_lock)
    scaler_registry = _scaler_hashes(models, checkpoint_lock)
    feature_by_role = {row["role"]: row for row in feature_inventory["roles"]}
    checkpoint_by_pair = {
        (row["family"], int(row["seed"])): row["checkpoint"]["sha256"]
        for row in checkpoint_lock["runs"]
    }
    missing_rows = authority["inventory"]["action_technical_missingness"]["rows"]
    for family in FAMILIES:
        role = FAMILY_ROLE[family]
        seed_records: dict[int, Any] = {}
        for seed in SEEDS:
            path = _seed_path(family, seed)
            scaler_sha256 = scaler_registry["by_family"][family][
                "scaler_sha256_by_seed"
            ][str(seed)]
            if not (REPOSITORY / path).exists():
                record = predict_continuous_residuals(
                    checkpoint=models[(family, seed)],
                    features=features[role],
                    feature_payload_sha256=feature_by_role[role]["payload_sha256"],
                    device=torch.device("cpu"),
                )
                payload = {
                    "family": np.asarray(family),
                    "bundle_role": np.asarray(role),
                    "seed": np.asarray(seed, dtype=np.int64),
                    "checkpoint_sha256": np.asarray(record.checkpoint_sha256),
                    "scaler_sha256": np.asarray(scaler_sha256),
                    "feature_payload_sha256": np.asarray(
                        record.feature_payload_sha256
                    ),
                    "producer_commit": np.asarray(implementation["commit"]),
                    "producer_sha256": np.asarray(
                        implementation["sha256"][
                            "scripts/run_tmlr_v9c_track_a_pre_gt.py"
                        ]
                    ),
                    "state_ids": np.asarray(record.state_ids),
                    "prediction": np.asarray(record.values, dtype=np.float32),
                }
                _create_once(path, _npz_bytes(payload))
            seed_records[seed] = _load_seed_record(
                family=family,
                seed=seed,
                feature_row=feature_by_role[role],
                features=features[role],
                checkpoint_sha256=checkpoint_by_pair[(family, seed)],
                scaler_sha256=scaler_sha256,
                implementation=implementation,
            )
        ensemble = ensemble_continuous_predictions(seed_records)
        feasibility = _mask_action_technical_missingness(
            features[role].state_ids,
            features[role].action_x16[:, :, 0],
            missing_rows,
        )
        actions, selected = select_lambda_zero_actions(
            ensemble, feasibility, state_ids=features[role].state_ids,
        )
        payload = {
            "family": np.asarray(family),
            "bundle_role": np.asarray(role),
            "ensemble_id": np.asarray(
                "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"
            ),
            "checkpoint_sha256_by_seed": np.asarray([
                seed_records[seed].checkpoint_sha256 for seed in SEEDS
            ]),
            "scaler_sha256_by_seed": np.asarray([
                scaler_registry["by_family"][family]["scaler_sha256_by_seed"][
                    str(seed)
                ]
                for seed in SEEDS
            ]),
            "feature_payload_sha256": np.asarray(
                feature_by_role[role]["payload_sha256"]
            ),
            "producer_commit": np.asarray(implementation["commit"]),
            "producer_sha256": np.asarray(
                implementation["sha256"]["scripts/run_tmlr_v9c_track_a_pre_gt.py"]
            ),
            "state_ids": np.asarray(ensemble.state_ids),
            "prediction": np.asarray(ensemble.values, dtype=np.float64),
            "feasibility": feasibility,
            "selected_action": actions,
            "selected_prediction": selected.astype(np.float64),
        }
        ensemble_path = _ensemble_path(family)
        if not (REPOSITORY / ensemble_path).exists():
            _create_once(ensemble_path, _npz_bytes(payload))
    _assert_prediction_tree(complete=True)
    rows = _prediction_rows(
        feature_inventory, checkpoint_lock, implementation, features,
        authority["inventory"]["action_technical_missingness"]["rows"],
        models,
    )
    _create_once(PREDICTION_INVENTORY, _inventory_bytes(rows))
    first = _manifest(
        authority, feature_inventory, checkpoint_lock, implementation, features,
        models,
    )
    second = _manifest(
        authority, feature_inventory, checkpoint_lock, implementation, features,
        models,
    )
    first_encoded = _canonical(first) + b"\n"
    second_encoded = _canonical(second) + b"\n"
    if first_encoded != second_encoded:
        raise TrackAPreGTError("V9C_PREDICTION_MANIFEST_DOUBLE_BUILD_DRIFT")
    _create_once(PREDICTION_MANIFEST_BUILD1, first_encoded)
    _create_once(PREDICTION_MANIFEST_BUILD2, second_encoded)
    lock = _prediction_lock_payload(
        authority=authority,
        feature_inventory=feature_inventory,
        manifest=first,
        implementation=implementation,
    )
    _create_once(PREDICTION_LOCK, _canonical(lock) + b"\n")
    return first


def build() -> dict[str, Any]:
    if any((REPOSITORY / path).exists() for path in LEGACY_OUTPUTS):
        raise TrackAPreGTError("V9C_LEGACY_PRE_GT_OUTPUT_PRESENT")
    if (REPOSITORY / PREDICTION_LOCK).exists():
        return check()
    implementation = _implementation()
    authority = _validate_authorities()
    if (REPOSITORY / FEATURE_INVENTORY).exists():
        feature_inventory, features = _load_features(authority)
        if feature_inventory.get("implementation") != implementation:
            raise TrackAPreGTError("V9C_FEATURE_IMPLEMENTATION_IDENTITY_DRIFT")
    else:
        feature_inventory, features = _build_features(authority, implementation)
    if _implementation(implementation["commit"]) != implementation:
        raise TrackAPreGTError("V9C_IMPLEMENTATION_CHANGED_DURING_FEATURE_BUILD")
    result = _predict(authority, feature_inventory, features, implementation)
    if _implementation(implementation["commit"]) != implementation:
        raise TrackAPreGTError("V9C_IMPLEMENTATION_CHANGED_DURING_PREDICTION")
    verified = check()
    if verified != result:
        raise TrackAPreGTError("V9C_POST_BUILD_VERIFICATION_DRIFT")
    return verified


def check() -> dict[str, Any]:
    if any((REPOSITORY / path).exists() for path in LEGACY_OUTPUTS):
        raise TrackAPreGTError("V9C_LEGACY_PRE_GT_OUTPUT_PRESENT")
    authority = _validate_authorities()
    published = _load_json(PREDICTION_MANIFEST_BUILD1)
    published_second = _load_json(PREDICTION_MANIFEST_BUILD2)
    first_bytes = _stable(REPOSITORY / PREDICTION_MANIFEST_BUILD1)
    second_bytes = _stable(REPOSITORY / PREDICTION_MANIFEST_BUILD2)
    if published != published_second or first_bytes != second_bytes:
        raise TrackAPreGTError("V9C_PREDICTION_MANIFEST_DOUBLE_BUILD_DRIFT")
    implementation = _implementation(
        str(published.get("implementation", {}).get("commit", ""))
    )
    feature_inventory, features = _load_features(authority)
    if feature_inventory.get("implementation") != implementation:
        raise TrackAPreGTError("V9C_FEATURE_IMPLEMENTATION_IDENTITY_DRIFT")
    from scripts.predict_tmlr_v9b1_label_free import _load_checkpoint_authority

    checkpoint_lock, _, models = _load_checkpoint_authority()
    expected_pairs = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if (
        set(models) != expected_pairs
        or _identity(CHECKPOINT_LOCK) != authority["checkpoint_identity"]
    ):
        raise TrackAPreGTError("V9C_CHECKPOINT_COUNT_DRIFT")
    _checkpoint_identities(checkpoint_lock)
    _assert_prediction_tree(complete=True)
    first = _manifest(
        authority, feature_inventory, checkpoint_lock, implementation, features,
        models,
    )
    second = _manifest(
        authority, feature_inventory, checkpoint_lock, implementation, features,
        models,
    )
    expected_bytes = _canonical(first) + b"\n"
    if (
        published != first
        or first != second
        or first_bytes != expected_bytes
        or second_bytes != expected_bytes
    ):
        raise TrackAPreGTError("V9C_PREDICTION_MANIFEST_REBUILD_DRIFT")
    expected_lock = _prediction_lock_payload(
        authority=authority,
        feature_inventory=feature_inventory,
        manifest=first,
        implementation=implementation,
    )
    lock = _load_json(PREDICTION_LOCK)
    if lock != expected_lock:
        raise TrackAPreGTError("V9C_PREDICTION_LOCK_DRIFT")
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build", metavar="TOKEN")
    modes.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            result = check()
        else:
            if arguments.build != TOKEN:
                raise TrackAPreGTError("V9C_PRE_GT_TOKEN_MISMATCH")
            result = build()
    except Exception as error:
        print(f"STOP-BLOCKED_TMLR_V9C_TRACK_A_PRE_GT: {error}", file=sys.stderr)
        return 2
    counts = result["prediction_counts"]
    print(json.dumps({
        "status": "TMLR_V9C_TRACK_A_PREDICTION_LOCK_COMPLETE",
        "prediction_rows": {
            "seed_continuous": counts["seed_continuous_rows"],
            "ensemble_continuous": counts["ensemble_continuous_rows"],
            "selected_actions": counts["selected_action_rows"],
        },
        "missing_rows": {
            "predictable": counts["missing_predictable_rows"],
            "panel_states": counts["missing_states"],
            "full_panel_seed": counts["missing_full_panel_seed_rows"],
            "full_panel_family": counts["missing_panel_family_rows"],
        },
        "technical_missingness_count": result["technical_missingness_counts"][
            "combined_trajectory_incomplete_states"
        ],
        "technical_missingness_breakdown": result[
            "technical_missingness_counts"
        ],
        "prediction_manifest_sha256": _identity(
            PREDICTION_MANIFEST_BUILD1
        )["sha256"],
        "prediction_lock_sha256": _identity(PREDICTION_LOCK)["sha256"],
        "checkpoint_identities": result["model_sha256"][
            "checkpoint_identities"
        ],
        "COCO_GT_READS": 0,
        "PERFORMANCE_RESULTS": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
