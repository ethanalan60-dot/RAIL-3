#!/usr/bin/env python3
"""Account for frozen technical missingness and close V9C trajectories.

This controller never imports an annotation reader and never invokes SAM.  It
accepts only the one frozen E4 state-level missingness record and the three
consumed R4A A4 failures, then merges existing immutable cache objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from rail3.contracts import canonical_json_bytes
from rail3.sam.coco_v9b1_cache import (
    SecureCandidateCache,
    SecureTrajectoryCache,
    secure_create_once,
    secure_read_bytes,
)


TOKEN = "TMLR_V9C_TRAJECTORY_MISSINGNESS_FINALIZE"
BASELINE_HEAD = "4840f9ca807a5837f907c0e4f247462ca07abc87"
ADDENDUM = Path("configs/experiments/tmlr_v9c_utility_aligned_addendum.json")
V9B1_CONFIG = Path("configs/experiments/tmlr_v9b1_coco_label_free_zeroshot.json")
V9B1_CONFIG_SHA256 = "5fcc7f09f28103391e861bf4d615ba6e43866519c19bd705303b6564fa5dd177"
R2D_DISPOSITION = Path("configs/experiments/tmlr_v9b1_e4_missingness_disposition_v1.json")
R2D_DISPOSITION_SHA256 = "e3118fc03b03b81487705516450156eb75f5921c9f77f91850f596fa9df35db1"
R4A_AGGREGATE = Path("artifacts/paper/source_data/tmlr_v9b1/r4a_three_a4_recovery.json")
R4A_AGGREGATE_SHA256 = "36ed7d654b100891ff864e5053468b0a2213d8fc37522b3749396bb7084e8de6"
PANEL = Path("data/manifests/tmlr_v9a_r_coco_external_panel_v1.json")
PANEL_SHA256 = "a0feb5f180d473d791cd9163ce062717b602ce3b1a049c3c3cc25132e06b2529"
SOURCE_ROOT = Path("artifacts/paper/source_data/tmlr_v9b1")
V9C_ROOT = Path("artifacts/paper/source_data/tmlr_v9c")
SHARD1_INVENTORY = SOURCE_ROOT / "coco_external_trajectory_inventory_shard1.json"
FINAL_INVENTORY = SOURCE_ROOT / "coco_external_trajectory_inventory.json"
TECHNICAL_REPORT = V9C_ROOT / "technical_missingness_report.json"
FINAL_LOCK = V9C_ROOT / "final_trajectory_lock.json"
STATE_MANIFEST = SOURCE_ROOT / "coco_external_state_manifest_label_free.csv"
SHARD_MANIFEST = SOURCE_ROOT / "coco_external_shard_manifest.json"
SHARD0_INVENTORY = SOURCE_ROOT / "coco_external_trajectory_inventory_shard0.json"
SHARD0_INVENTORY_SHA256 = "e532a7debeee00413aacb2b5dc4a0b6f5a248a7bcf7a29a0373b6568e2670f42"
SHARD0_TELEMETRY = SOURCE_ROOT / "coco_external_trajectory_run_telemetry_shard0.json"
SHARD1_TELEMETRY = SOURCE_ROOT / "coco_external_trajectory_run_telemetry_shard1.json"
LAUNCH_FAILURE = SOURCE_ROOT / "coco_external_trajectory_failure_launch.json"
SHARD1_FAILURE = SOURCE_ROOT / "coco_external_trajectory_failure_shard1.json"
LAUNCH_FAILURE_SHA256 = "825158ded305d7296af104f5745d8c1622520f32bc8e43aea23e042f08d67cbc"
SHARD1_FAILURE_SHA256 = "f03c53431580157bc3a8fe3302850c281b5a8ff93334f2dd22ae540e28dd5124"
SHARD_CACHE_ROOT = Path("artifacts/candidates/tmlr-v9b1/shards")
FINAL_ROOT = Path("artifacts/candidates/tmlr-v9b1/final")
FINAL_A0 = FINAL_ROOT / "a0"
FINAL_ACTIONS = FINAL_ROOT / "actions"
FINAL_PLANS = FINAL_ROOT / "plans"
GPU_UUIDS = (
    "GPU-0c4fc6e5-d153-8780-ed6c-4c612f8e3de6",
    "GPU-18a76220-b743-4a44-dcc1-14d69b505593",
)
HEX64 = re.compile(r"[0-9a-f]{64}")


class V9CTrajectoryError(RuntimeError):
    pass


def _encoded(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(payload)) + b"\n"


def _read(relative: Path) -> bytes:
    return secure_read_bytes(REPOSITORY / relative)


def _load(relative: Path, *, canonical: bool = True) -> dict[str, Any]:
    encoded = _read(relative)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V9CTrajectoryError(f"V9C_JSON_INVALID:{relative}") from error
    if not isinstance(value, dict) or (canonical and encoded != _encoded(value)):
        raise V9CTrajectoryError(f"V9C_JSON_NOT_CANONICAL:{relative}")
    return value


def _identity(relative: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    encoded = _read(relative)
    digest = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise V9CTrajectoryError(f"V9C_IDENTITY_DRIFT:{relative}")
    return {"path": relative.as_posix(), "bytes": len(encoded), "sha256": digest}


def _create_or_verify(relative: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    encoded = _encoded(payload)
    try:
        secure_create_once(REPOSITORY / relative, encoded)
    except FileExistsError:
        if _read(relative) != encoded:
            raise V9CTrajectoryError(f"V9C_CREATE_ONCE_CONFLICT:{relative}")
    return {
        "path": relative.as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise V9CTrajectoryError(f"V9C_GIT_FAILED:{' '.join(arguments)}")
    return completed.stdout.strip()


def _require_clean_repository() -> dict[str, Any]:
    if Path.cwd().resolve() != REPOSITORY:
        raise V9CTrajectoryError("V9C_CWD_DRIFT")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", BASELINE_HEAD, head],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise V9CTrajectoryError("V9C_HEAD_ANCESTRY_DRIFT")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise V9CTrajectoryError("V9C_WORKTREE_NOT_CLEAN")
    implementation = Path("scripts/finalize_tmlr_v9c_trajectories.py")
    if _git("ls-files", "--error-unmatch", implementation.as_posix()) != implementation.as_posix():
        raise V9CTrajectoryError("V9C_IMPLEMENTATION_NOT_TRACKED")
    return {"commit": head, "implementation": _identity(implementation)}


def _validate_addendum() -> dict[str, Any]:
    payload = _load(ADDENDUM, canonical=False)
    action = payload.get("action_technical_missingness", {})
    zero = payload.get("zero_counters_at_freeze", {})
    if not (
        payload.get("schema_version") == "rail3.tmlr-v9c-utility-aligned-addendum.v1"
        and payload.get("status") == "FROZEN_BEFORE_COCO_GT_AND_PERFORMANCE"
        and payload.get("result_blind") is True
        and action.get("disposition") == "ACTION_EXECUTION_TECHNICAL_MISSINGNESS"
        and action.get("affected_a4_actions") == 3
        and action.get("affected_states") == 3
        and action.get("affected_images") == 3
        and action.get("retry_maximum") == 1
        and action.get("tolerance_states") == 100
        and action.get("tolerance_fraction") == 0.005
        and all(action.get(key) is False for key in (
            "fabricate_action", "impute_residual_or_utility", "infer_class_absence",
            "relabel_as_infeasible",
        ))
        and zero == {
            "COCO_GT_READS": 0,
            "MODEL_PREDICTIONS": 0,
            "PERFORMANCE_RESULTS": 0,
        }
    ):
        raise V9CTrajectoryError("V9C_ADDENDUM_DRIFT")
    return payload


def _validate_zero_boundary() -> None:
    aggregate = _load(R4A_AGGREGATE)
    if not (
        aggregate.get("COCO_GT_READS") == 0
        and aggregate.get("MODEL_PREDICTIONS") == 0
        and aggregate.get("PERFORMANCE_RESULTS") == 0
    ):
        raise V9CTrajectoryError("V9C_NONZERO_RESULT_BOUNDARY")
    forbidden_outputs = (
        Path("artifacts/predictions/tmlr-v9b1"),
        Path("artifacts/predictions/tmlr-v9c"),
        SOURCE_ROOT / "coco_external_prediction_manifest.json",
        V9C_ROOT / "all_coco_prediction_manifest.json",
        V9C_ROOT / "track_a_external_results.csv",
        V9C_ROOT / "track_b_external_results.csv",
    )
    if any((REPOSITORY / path).exists() for path in forbidden_outputs):
        raise V9CTrajectoryError("V9C_PRE_GT_OUTPUT_ALREADY_EXISTS")


def _model_identity() -> dict[str, Any]:
    from rail3.sam.sam31_backend import official_sam31_model_spec, real_backend_config_hash

    model = official_sam31_model_spec()
    return {
        "model_spec_id": model.model_spec_id,
        "model_source_commit": model.source_commit,
        "checkpoint_sha256": model.checkpoint_sha256,
        "sam_config_hash": real_backend_config_hash(),
    }


def _load_authorities() -> dict[str, Any]:
    for path, digest in (
        (V9B1_CONFIG, V9B1_CONFIG_SHA256),
        (R2D_DISPOSITION, R2D_DISPOSITION_SHA256),
        (R4A_AGGREGATE, R4A_AGGREGATE_SHA256),
        (PANEL, PANEL_SHA256),
    ):
        _identity(path, digest)
    from rail3.sam.coco_v9b1_trajectory import (
        build_semantic_states,
        load_config,
        load_frozen_panel_images,
        load_taxonomy,
        validate_e4_execution_registry,
        validate_frozen_execution_manifests,
        validate_small_authorities,
    )

    config = load_config(REPOSITORY)
    validate_small_authorities(REPOSITORY, config)
    states = build_semantic_states(load_frozen_panel_images(REPOSITORY), load_taxonomy(REPOSITORY))
    manifest = validate_frozen_execution_manifests(
        REPOSITORY, config, states, require_global_clean=True,
    )
    e4 = validate_e4_execution_registry(REPOSITORY, config)
    if len(e4) != 1:
        raise V9CTrajectoryError("V9C_E4_MISSINGNESS_COUNT_DRIFT")
    return {
        "config": config,
        "states": states,
        "manifest": manifest,
        "e4": e4,
        "model_identity": _model_identity(),
    }


def _validate_r4a() -> tuple[dict[str, Any], ...]:
    aggregate = _load(R4A_AGGREGATE)
    rows = aggregate.get("results")
    if not (
        aggregate.get("status") == "STOP-BLOCKED_TMLR_V9B1_PERSISTENT_A4_FAILURE"
        and aggregate.get("TOTAL_RETRIES") == 3
        and aggregate.get("TOTAL_RECOVERED") == 0
        and aggregate.get("TOTAL_PERSISTENT_FAILURES") == 3
        and aggregate.get("COCO_GT_READS") == 0
        and aggregate.get("MODEL_PREDICTIONS") == 0
        and aggregate.get("PERFORMANCE_RESULTS") == 0
        and isinstance(rows, list)
        and len(rows) == 3
    ):
        raise V9CTrajectoryError("V9C_R4A_AGGREGATE_DRIFT")
    seen_states: set[str] = set()
    seen_images: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        state = row.get("state_identity", {})
        ledger = row.get("result_ledger", {})
        path = Path(str(ledger.get("path", "")))
        if not (
            row.get("structural_feasibility") == "STRUCTURALLY_FEASIBLE"
            and row.get("retry_performed") is True
            and row.get("new_sam_calls") == 1
            and row.get("retry_result") == "failure"
            and row.get("retry_reason_code") == "SAM31_PROMPT_CALL_FAILED"
            and row.get("canonical_recovery_status")
            == "PERSISTENT_ACTION_EXECUTION_FAILURE"
            and isinstance(state.get("semantic_state_id"), str)
            and isinstance(state.get("image_id"), str)
            and _identity(path) == dict(ledger)
        ):
            raise V9CTrajectoryError("V9C_R4A_ROW_DRIFT")
        result = _load(path)
        if not (
            result.get("status") == "PERSISTENT_A4_FAILURE"
            and result.get("retry_count") == 1
            and result.get("maximum_retry") == 1
            and result.get("additional_retry_authorized") is False
            and result.get("new_sam_calls") == 1
            and result.get("COCO_GT_CONTENT_READ_COUNT") == 0
            and result.get("PERFORMANCE_RESULT_COUNT") == 0
            and result.get("target", {}).get("semantic_state_id")
            == state["semantic_state_id"]
            and result.get("target", {}).get("action_id") == state.get("action_id")
        ):
            raise V9CTrajectoryError("V9C_R4A_RESULT_LEDGER_DRIFT")
        if state["semantic_state_id"] in seen_states or state["image_id"] in seen_images:
            raise V9CTrajectoryError("V9C_R4A_TARGET_COLLISION")
        seen_states.add(state["semantic_state_id"])
        seen_images.add(state["image_id"])
        validated.append(dict(row))
    return tuple(validated)


def _derive_shard(authorities: Mapping[str, Any], shard: int) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    from rail3.sam.coco_v9b1_trajectory import derive_bundles_from_a0_cache

    states = tuple(state for state in authorities["states"] if state.shard == shard)
    state_ids = {state.semantic_state_id for state in states}
    technical = tuple(
        entry for entry in authorities["e4"]
        if entry.record.semantic_state_id in state_ids
    )
    root = REPOSITORY / SHARD_CACHE_ROOT / f"shard-{shard}"
    a0 = SecureCandidateCache(root / "a0")
    actions = SecureTrajectoryCache(root / "actions")
    plans = root / "plans"
    bundles = derive_bundles_from_a0_cache(
        states,
        model_spec_id=authorities["model_identity"]["model_spec_id"],
        a0_cache=a0,
        technical_missingness=technical,
    )
    return a0, actions, plans, bundles


def _augment(
    inventory: dict[str, Any], authorities: Mapping[str, Any], shard: int,
) -> dict[str, Any]:
    from scripts.run_tmlr_v9b1_sam_trajectory_worker import _augment_inventory

    telemetry = SHARD0_TELEMETRY if shard == 0 else SHARD1_TELEMETRY
    telemetry_identity = _identity(telemetry)
    return _augment_inventory(
        inventory,
        shard=shard,
        physical_gpu=shard,
        gpu_uuid=GPU_UUIDS[shard],
        state_bytes=_read(STATE_MANIFEST),
        shard_bytes=_read(SHARD_MANIFEST),
        implementation_commit=authorities["manifest"]["implementation_commit"],
        telemetry_path=REPOSITORY / telemetry,
        telemetry_sha256=telemetry_identity["sha256"],
        model_identity=dict(authorities["model_identity"]),
    )


def _validate_raw_inventories(
    authorities: Mapping[str, Any], r4a: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    from rail3.sam.coco_v9b1_trajectory import validate_inventory

    shard_bundles: list[tuple[Any, ...]] = []
    raw: list[dict[str, Any]] = []
    for shard in (0, 1):
        a0, actions, plans, bundles = _derive_shard(authorities, shard)
        technical = authorities["e4"] if shard == 0 else ()
        base = validate_inventory(
            bundles,
            a0_cache=a0,
            action_cache=actions,
            plan_lock_root=plans,
            expected_model_identity=dict(authorities["model_identity"]),
            technical_missingness=technical,
        )
        raw.append(_augment(base, authorities, shard))
        shard_bundles.append(tuple(bundles))
    shard0 = _load(SHARD0_INVENTORY)
    _identity(SHARD0_INVENTORY, SHARD0_INVENTORY_SHA256)
    if shard0 != raw[0] or not shard0.get("ready"):
        raise V9CTrajectoryError("V9C_SHARD0_INVENTORY_DRIFT")
    a4 = raw[1].get("counts_by_action_and_outcome", {}).get("A4", {})
    if not (
        raw[1].get("state_count") == 10_380
        and raw[1].get("trajectory_state_count") == 10_380
        and raw[1].get("semantic_record_count") == 72_660
        and raw[1].get("trajectory_record_count") == 72_660
        and raw[1].get("failure_count") == 3
        and raw[1].get("ready") is False
        and a4.get("failure") == 3
        and sum(
            counts.get("failure", -1)
            for action, counts in raw[1]["counts_by_action_and_outcome"].items()
            if action != "A4"
        ) == 0
        and {row["state_identity"]["semantic_state_id"] for row in r4a}
        <= {bundle.state.semantic_state_id for bundle in shard_bundles[1]}
    ):
        raise V9CTrajectoryError("V9C_SHARD1_RAW_INVENTORY_DRIFT")
    return raw[0], raw[1], shard_bundles[0], shard_bundles[1]


def _missingness_overlay(
    raw: Mapping[str, Any], r4a: Sequence[Mapping[str, Any]], *, scope: str,
) -> dict[str, Any]:
    action_rows = [
        {
            "action_code": "A4",
            "action_id": row["state_identity"]["action_id"],
            "image_id": row["state_identity"]["image_id"],
            "reason_code": "SAM31_PROMPT_CALL_FAILED",
            "result_ledger": row["result_ledger"],
            "semantic_state_id": row["state_identity"]["semantic_state_id"],
        }
        for row in r4a
    ]
    result = dict(raw)
    result.update({
        "schema_version": f"rail3.tmlr-v9c-{scope}-trajectory-inventory.v1",
        "status": f"TMLR_V9C_{scope.upper()}_TRAJECTORIES_ACCOUNTED",
        "ready": True,
        "raw_execution_failure_count": 3,
        "unresolved_execution_failure_count": 0,
        "action_technical_missingness": {
            "count": 3,
            "affected_state_count": 3,
            "affected_image_count": 3,
            "fraction_of_20000_state_panel": 3 / 20_000,
            "disposition": "ACTION_EXECUTION_TECHNICAL_MISSINGNESS",
            "scientific_outcome_imputed": False,
            "rows": action_rows,
        },
        "r4a_aggregate": _identity(R4A_AGGREGATE, R4A_AGGREGATE_SHA256),
        "failure_ledger_evidence": [
            _identity(LAUNCH_FAILURE, LAUNCH_FAILURE_SHA256),
            _identity(SHARD1_FAILURE, SHARD1_FAILURE_SHA256),
        ],
        "complete_oracle_denominator_excluded_state_count": 3,
        "per_action_excluded_row_count": 3,
        "additional_retry_authorized": False,
        "coco_gt_reads": 0,
        "model_predictions": 0,
        "performance_results": 0,
    })
    return result


def _copy_candidate(identity: Any, source: SecureCandidateCache, destination: SecureCandidateCache) -> None:
    value = source.load(identity)
    destination.store(identity, value)
    if destination.load(identity).to_dict() != value.to_dict():
        raise V9CTrajectoryError(f"V9C_A0_COPY_DRIFT:{identity.cache_key}")


def _copy_action(identity: Any, source: SecureTrajectoryCache, destination: SecureTrajectoryCache) -> None:
    value = source.load(identity)
    destination.store(identity, value)
    if destination.load(identity).to_dict() != value.to_dict():
        raise V9CTrajectoryError(f"V9C_ACTION_COPY_DRIFT:{identity.cache_key}")


def _copy_e4_evidence(entry: Any) -> None:
    source = Path(entry.cache_path)
    digest = entry.cache_sha256
    encoded = _read(source)
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise V9CTrajectoryError("V9C_E4_EVIDENCE_DRIFT")
    key = entry.cache_key
    destination = FINAL_A0 / key.rsplit("_", 1)[-1][:2] / f"{key}.json"
    try:
        secure_create_once(REPOSITORY / destination, encoded)
    except FileExistsError:
        if _read(destination) != encoded:
            raise V9CTrajectoryError("V9C_E4_EVIDENCE_COPY_CONFLICT")


def _copy_plans(source_roots: Sequence[Path]) -> None:
    from rail3.sam.coco_v9b1_trajectory import _bounded_plan_paths

    sources = [_bounded_plan_paths(REPOSITORY / root) for root in source_roots]
    names = [{path.name for path in paths} for paths in sources]
    if names[0] & names[1] or len(names[0] | names[1]) != 1_000:
        raise V9CTrajectoryError("V9C_PLAN_SOURCE_GRID_DRIFT")
    for path in sorted(sources[0] | sources[1], key=lambda value: value.name):
        relative = path.relative_to(REPOSITORY)
        encoded = _read(relative)
        destination = FINAL_PLANS / path.parent.name / path.name
        try:
            secure_create_once(REPOSITORY / destination, encoded)
        except FileExistsError:
            if _read(destination) != encoded:
                raise V9CTrajectoryError(f"V9C_PLAN_COPY_CONFLICT:{path.name}")


def _merge_and_validate(
    authorities: Mapping[str, Any], shard_bundles: Sequence[Sequence[Any]],
    r4a: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from rail3.sam.coco_v9b1_trajectory import validate_inventory

    source_a0 = [
        SecureCandidateCache(REPOSITORY / SHARD_CACHE_ROOT / f"shard-{shard}" / "a0")
        for shard in (0, 1)
    ]
    source_actions = [
        SecureTrajectoryCache(REPOSITORY / SHARD_CACHE_ROOT / f"shard-{shard}" / "actions")
        for shard in (0, 1)
    ]
    final_a0 = SecureCandidateCache(REPOSITORY / FINAL_A0)
    final_actions = SecureTrajectoryCache(REPOSITORY / FINAL_ACTIONS)
    by_state = {
        bundle.state.semantic_state_id: bundle
        for bundle in (*shard_bundles[0], *shard_bundles[1])
    }
    all_bundles = tuple(
        by_state[state.semantic_state_id]
        for state in authorities["states"]
        if state.semantic_state_id in by_state
    )
    if len(all_bundles) != 19_999 or len({b.state.semantic_state_id for b in all_bundles}) != 19_999:
        raise V9CTrajectoryError("V9C_EXECUTABLE_STATE_GRID_DRIFT")
    for bundle in all_bundles:
        shard = bundle.state.shard
        _copy_candidate(bundle.a0_identity, source_a0[shard], final_a0)
        for identity in bundle.action_identities:
            _copy_action(identity, source_actions[shard], final_actions)
    _copy_e4_evidence(authorities["e4"][0])
    _copy_plans(tuple(SHARD_CACHE_ROOT / f"shard-{s}" / "plans" for s in (0, 1)))
    raw = validate_inventory(
        all_bundles,
        a0_cache=final_a0,
        action_cache=final_actions,
        plan_lock_root=REPOSITORY / FINAL_PLANS,
        require_full_panel=True,
        expected_model_identity=dict(authorities["model_identity"]),
        technical_missingness=authorities["e4"],
    )
    if not (
        raw.get("state_count") == 20_000
        and raw.get("trajectory_state_count") == 19_999
        and raw.get("semantic_record_count") == 140_000
        and raw.get("trajectory_record_count") == 139_993
        and raw.get("technical_missingness", {}).get("count") == 1
        and raw.get("failure_count") == 3
        and raw.get("ready") is False
    ):
        raise V9CTrajectoryError("V9C_FINAL_RAW_INVENTORY_DRIFT")
    result = _missingness_overlay(raw, r4a, scope="final")
    result.update({
        "state_dispositions": {
            "COMPLETE": 19_996,
            "LEGALLY_INFEASIBLE": 0,
            "TECHNICAL_MISSINGNESS": 4,
        },
        "trajectory_incomplete_state_count": 4,
        "trajectory_incomplete_image_count": 4,
        "trajectory_incomplete_fraction": 4 / 20_000,
        "trajectory_incomplete_tolerance_count": 100,
        "trajectory_incomplete_tolerance_fraction": 0.005,
        "completeness_gate": "PASS",
        "state_manifest_sha256": _identity(STATE_MANIFEST)["sha256"],
        "shard_manifest_sha256": _identity(SHARD_MANIFEST)["sha256"],
        "shard_inventory_sha256": {
            "0": _identity(SHARD0_INVENTORY)["sha256"],
            "1": _identity(SHARD1_INVENTORY)["sha256"],
        },
        "sam_identity": dict(authorities["model_identity"]),
        "final_cache_roots": {
            "a0": FINAL_A0.as_posix(),
            "actions": FINAL_ACTIONS.as_posix(),
            "plans": FINAL_PLANS.as_posix(),
        },
        "shard_count": 2,
    })
    return result


def _technical_report(r4a: Sequence[Mapping[str, Any]], e4: Sequence[Any]) -> dict[str, Any]:
    e4_record = e4[0].record
    return {
        "schema_version": "rail3.tmlr-v9c-technical-missingness-report.v1",
        "status": "INFERENCE_COMPLETENESS_PASS",
        "action_level": {
            "disposition": "ACTION_EXECUTION_TECHNICAL_MISSINGNESS",
            "missing_action_count": 3,
            "affected_state_count": 3,
            "affected_image_count": 3,
            "fraction_of_20000_state_panel": 3 / 20_000,
            "semantic_state_ids": sorted(
                row["state_identity"]["semantic_state_id"] for row in r4a
            ),
        },
        "state_level": {
            "prior_e4_missing_state_count": 1,
            "semantic_state_ids": [e4_record.semantic_state_id],
        },
        "combined": {
            "trajectory_incomplete_state_count": 4,
            "affected_image_count": 4,
            "fraction_of_20000_state_panel": 4 / 20_000,
            "maximum_allowed_states": 100,
            "maximum_allowed_fraction": 0.005,
            "gate": "PASS",
        },
        "denominator_policy": {
            "complete_oracle_metrics": "EXCLUDE_EXACT_AFFECTED_STATE",
            "per_action_diagnostics": "EXCLUDE_ONLY_MISSING_ACTION_WHERE_DEFINED",
            "preserve_panel_membership": True,
            "scientific_outcome_imputed": False,
        },
        "COCO_GT_READS": 0,
        "MODEL_PREDICTIONS": 0,
        "PERFORMANCE_RESULTS": 0,
    }


def finalize() -> dict[str, Any]:
    implementation = _require_clean_repository()
    _validate_addendum()
    _validate_zero_boundary()
    authorities = _load_authorities()
    r4a = _validate_r4a()
    _, shard1_raw, shard0_bundles, shard1_bundles = _validate_raw_inventories(
        authorities, r4a,
    )
    report_identity = _create_or_verify(
        TECHNICAL_REPORT, _technical_report(r4a, authorities["e4"]),
    )
    shard1 = _missingness_overlay(shard1_raw, r4a, scope="shard1")
    shard1.update({
        "state_dispositions": {
            "COMPLETE": 10_377,
            "LEGALLY_INFEASIBLE": 0,
            "TECHNICAL_MISSINGNESS": 3,
        },
        "trajectory_incomplete_state_count": 3,
        "trajectory_incomplete_fraction": 3 / 20_000,
        "completeness_gate": "PASS",
    })
    shard1_identity = _create_or_verify(SHARD1_INVENTORY, shard1)
    final = _merge_and_validate(
        authorities, (shard0_bundles, shard1_bundles), r4a,
    )
    final_identity = _create_or_verify(FINAL_INVENTORY, final)
    lock = {
        "schema_version": "rail3.tmlr-v9c-final-trajectory-lock.v1",
        "status": "TMLR_V9C_FINAL_TRAJECTORIES_READY",
        "implementation": implementation,
        "addendum": _identity(ADDENDUM),
        "panel": _identity(PANEL, PANEL_SHA256),
        "technical_missingness_report": report_identity,
        "shard_inventories": {
            "0": _identity(SHARD0_INVENTORY, SHARD0_INVENTORY_SHA256),
            "1": shard1_identity,
        },
        "final_inventory": final_identity,
        "states": 20_000,
        "trajectory_incomplete_states": 4,
        "action_level_technical_missingness": 3,
        "completeness_gate": "PASS",
        "COCO_GT_READS": 0,
        "MODEL_PREDICTIONS": 0,
        "PERFORMANCE_RESULTS": 0,
    }
    lock_identity = _create_or_verify(FINAL_LOCK, lock)
    return {
        "status": lock["status"],
        "final_inventory": final_identity,
        "final_lock": lock_identity,
        "technical_missingness": 4,
        "action_level_technical_missingness": 3,
        "COCO_GT_READS": 0,
        "MODEL_PREDICTIONS": 0,
        "PERFORMANCE_RESULTS": 0,
    }


def audit() -> dict[str, Any]:
    implementation = _require_clean_repository()
    _validate_addendum()
    _validate_zero_boundary()
    authorities = _load_authorities()
    r4a = _validate_r4a()
    shard0, shard1, shard0_bundles, shard1_bundles = _validate_raw_inventories(
        authorities, r4a,
    )
    return {
        "status": "TMLR_V9C_TRAJECTORY_PREFLIGHT_PASS",
        "implementation": implementation,
        "shard0_ready": shard0["ready"],
        "shard0_trajectory_states": len(shard0_bundles),
        "shard1_raw_ready": shard1["ready"],
        "shard1_trajectory_states": len(shard1_bundles),
        "persistent_a4_failures": len(r4a),
        "prior_e4_state_missingness": len(authorities["e4"]),
        "combined_trajectory_incomplete_states": len(r4a) + len(authorities["e4"]),
        "completeness_gate": "PASS",
        "COCO_GT_READS": 0,
        "MODEL_PREDICTIONS": 0,
        "PERFORMANCE_RESULTS": 0,
    }


def check() -> dict[str, Any]:
    expected = finalize()
    lock = _load(FINAL_LOCK)
    if lock.get("final_inventory") != _identity(FINAL_INVENTORY):
        raise V9CTrajectoryError("V9C_FINAL_LOCK_INVENTORY_DRIFT")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--finalize")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.check:
            payload = check()
        elif arguments.audit:
            payload = audit()
        else:
            if arguments.finalize != TOKEN:
                raise V9CTrajectoryError("V9C_FINALIZE_TOKEN_MISMATCH")
            payload = finalize()
    except Exception as error:
        print(json.dumps({
            "status": "STOP-BLOCKED_TMLR_V9C_TRAJECTORIES",
            "reason": f"{type(error).__name__}:{error}",
            "COCO_GT_READS": 0,
            "MODEL_PREDICTIONS": 0,
            "PERFORMANCE_RESULTS": 0,
        }, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
