"""Result-blind COCO V9B1 state, shard, trajectory, and resume contracts.

This module contains no annotation or evaluation dependency.  It consumes only
the frozen label-blind panel identities and executes prompts through an injected
session backend.  The real SAM adapter is deliberately separate so unit and
synthetic smoke tests cannot construct or load it accidentally.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from rail3.cache import (
    CacheIdentity,
    CandidateCache,
    CandidateGenerationResult,
    CandidateNoResult,
)
from rail3.contracts import canonical_json_bytes, stable_id
from rail3.sam.coco_v9b1_cache import (
    SecureCandidateCache,
    SecureTrajectoryCache,
    secure_create_once,
    secure_read_bytes,
)
from rail3.sam.e4_missingness import (
    E4ExecutionEvidence,
    load_e4_execution_registry,
    partition_executable_states,
)
from rail3.sam.protocols import EncodedImageState, PromptRequest
from rail3.sam.sam31_backend import PromptMetadata
from rail3.sam.voc_actions import (
    ACTION_CODES,
    PROTOCOL_SHA256,
    ActionCacheIdentity,
    ActionOutcome,
    ActionPlan,
    TrajectoryCache,
    TrajectoryCacheError,
    action_metadata,
    build_action_plans,
    outcome_from_generation,
    select_source_candidate,
    shard_index,
)
from rail3.sam.voc_protocol import canonical_voc_prompts, voc_cache_identity


CONFIG_SCHEMA = "rail3.tmlr-v9b1-coco-label-free-zeroshot.v1"
STATE_MANIFEST_SCHEMA = "rail3.tmlr-v9b1-label-free-state-manifest.v1"
PLAN_LOCK_SCHEMA = "rail3.tmlr-v9b1-label-free-image-plan-lock.v1"
R2D_PLAN_LOCK_SCHEMA = "rail3.tmlr-v9b1-label-free-image-plan-lock-r2d.v1"
INVENTORY_SCHEMA = "rail3.tmlr-v9b1-label-free-inventory.v1"
ADDENDUM_SCHEMA = "rail3.tmlr-v9b1-label-free-sequencing-addendum.v1"
INFERENCE_OUTCOME_POLICY_SCHEMA = "rail3.tmlr-v9b1-inference-outcome-policy.v1"
E4_MISSINGNESS_DISPOSITION_SCHEMA = "rail3.tmlr-v9b1-e4-missingness-disposition.v1"
SHARD_MANIFEST_SCHEMA = "rail3.tmlr-v9b1-label-free-shard-manifest.v1"

CONFIG_PATH = Path("configs/experiments/tmlr_v9b1_coco_label_free_zeroshot.json")
SEQUENCING_ADDENDUM_PATH = Path(
    "configs/experiments/tmlr_v9b1_label_free_sequencing_addendum.json"
)
SEQUENCING_ADDENDUM_SHA256 = (
    "0de7a5972f62ff788ae14c607902d5ce8ccc58352dcd6082039c234736d84c61"
)
INFERENCE_OUTCOME_POLICY_PATH = Path(
    "configs/experiments/tmlr_v9b1_inference_outcome_policy_v1.json"
)
INFERENCE_OUTCOME_POLICY_SHA256 = (
    "828689d050ea96ccebf57ce42103e2022b3cae4b796402b0401b51ac60fd0169"
)
E4_MISSINGNESS_DISPOSITION_PATH = Path(
    "configs/experiments/tmlr_v9b1_e4_missingness_disposition_v1.json"
)
E4_MISSINGNESS_DISPOSITION_SHA256 = (
    "e3118fc03b03b81487705516450156eb75f5921c9f77f91850f596fa9df35db1"
)
E4_EXECUTION_REGISTRY_PATH = Path(
    "configs/experiments/tmlr_v9b1_e4_execution_registry_v1.json"
)
E4_EXECUTION_REGISTRY_SHA256 = (
    "acb8befbeaefa1e76479e34133c5f331d1bc175634a6c4e04475da82e18f8396"
)
PANEL_MANIFEST_PATH = Path("data/manifests/tmlr_v9a_r_coco_external_panel_v1.json")
PANEL_MANIFEST_SHA256 = (
    "a0feb5f180d473d791cd9163ce062717b602ce3b1a049c3c3cc25132e06b2529"
)
PANEL_IDENTITY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9a_r/coco_external_panel_identity.csv"
)
PANEL_IDENTITY_SHA256 = (
    "0ccecda97caf90bba18146635a03d5050d2f0ddd7ca4d032f18432cf5e3b5364"
)
TAXONOMY_PATH = Path(
    "artifacts/paper/source_data/tmlr_v9b0_r2/coco_voc20_taxonomy_verification.csv"
)
TAXONOMY_SHA256 = (
    "a2a177f926d34bbac2e131c21c517d768c004d1acc615a41db68e73d3ba7071c"
)
IMAGE_ROOT = Path("data/public/coco2017/raw/val2017")
SHARD_CACHE_ROOT = Path("artifacts/candidates/tmlr-v9b1/shards")
FINAL_CACHE_ROOT = Path("artifacts/candidates/tmlr-v9b1/final")
FINAL_A0_CACHE_ROOT = FINAL_CACHE_ROOT / "a0"
FINAL_ACTION_CACHE_ROOT = FINAL_CACHE_ROOT / "actions"
FINAL_PLAN_LOCK_ROOT = FINAL_CACHE_ROOT / "plans"
FORMAL_IMPLEMENTATION_PATHS = (
    Path("src/rail3/cache/atomic_io.py"),
    Path("src/rail3/cache/candidates.py"),
    Path("src/rail3/contracts/__init__.py"),
    Path("src/rail3/sam/checkpoint_compat.py"),
    Path("src/rail3/sam/coco_v9b1_asset_gate.py"),
    Path("src/rail3/sam/coco_v9b1_cache.py"),
    Path("src/rail3/sam/coco_v9b1_r2d_control.py"),
    Path("src/rail3/sam/e4_missingness.py"),
    Path("src/rail3/sam/coco_v9b1_real.py"),
    Path("src/rail3/sam/coco_v9b1_synthetic.py"),
    Path("src/rail3/sam/coco_v9b1_trajectory.py"),
    Path("src/rail3/sam/coco_v9b1_worker_gate.py"),
    Path("src/rail3/sam/protocols.py"),
    Path("src/rail3/sam/sam31_backend.py"),
    Path("src/rail3/sam/strict_builder.py"),
    Path("src/rail3/sam/voc_actions.py"),
    Path("src/rail3/sam/voc_protocol.py"),
    Path("scripts/build_tmlr_v9b1_label_free_state_manifest.py"),
    Path("scripts/finalize_tmlr_v9b1_sam_trajectories.py"),
    Path("scripts/launch_tmlr_v9b1_sam_trajectories.py"),
    Path("scripts/control_tmlr_v9b1_r2d_resume.py"),
    Path("scripts/postcheck_tmlr_v9b1_r2d_execution.py"),
    Path("scripts/run_tmlr_v9b1_sam_trajectory_worker.py"),
)

PANEL_LOCK_ID = (
    "tmlr_v9a_r_coco_panel_"
    "9e80da655198b353e1f0e224282710acf214b96ea6f685f2d43c613f8d418886"
)
EXPECTED_IMAGES = 1_000
EXPECTED_CLASSES = 20
EXPECTED_STATES = 20_000
EXPECTED_SEMANTIC_RECORDS = 140_000
EXPECTED_SHARD_IMAGES = (481, 519)
EXPECTED_SHARD_SEMANTIC_RECORDS = (67_340, 72_660)
EXPECTED_GPU_UUIDS = (
    "GPU-0c4fc6e5-d153-8780-ed6c-4c612f8e3de6",
    "GPU-18a76220-b743-4a44-dcc1-14d69b505593",
)
ACTION_ORDER = ("A0", *ACTION_CODES)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CANONICAL_IMAGE_ID_RE = re.compile(r"[0-9]{12}")
_CACHE_KEY_RE = re.compile(r"(?:candidate|trajectory)_cache_[0-9a-f]{64}")

EXPECTED_SMALL_AUTHORITIES = {
    "action_protocol": (
        Path("configs/experiments/voc_m04_m05_action_protocol_v1.json"),
        "c558595b6be151307f04ae189ff0b8c74f1de2adbf7e0c046db57048fd12f985",
    ),
    "epoch_lock": (
        Path("artifacts/paper/source_data/tmlr_v9b0_r2/v9_fullfit_epoch_lock.json"),
        "4bc275c44cb0ea842ddd66ad7bfad70bd271e4fd0afcabc097d85fdaee93ab37",
    ),
    "panel_identity": (PANEL_IDENTITY_PATH, PANEL_IDENTITY_SHA256),
    "panel_manifest": (PANEL_MANIFEST_PATH, PANEL_MANIFEST_SHA256),
    "protocol_v2": (
        Path("docs/experiments/TMLR_V9_COCO_EXTERNAL_PROTOCOL_V2.md"),
        "4c66825371a4a00438e5fa18d5cc9928a4e4289550adc064ae6bb5fd8b7ef6a0",
    ),
    "r2_recovery_addendum": (
        Path("configs/experiments/tmlr_v9b0_r2_recovery_addendum.json"),
        "8703a83523034112077eeff88e6c3034e3b44eb01706b8bca2a24541c4efda6c",
    ),
    "r2_safe_summary": (
        Path("artifacts/safe_handoffs/tmlr_v9b0_r2_recovery_safe_summary.json"),
        "74f5b37ae185185426c38a2dcf826bec641c619b08bbe0f7e5a7bc05615d78a4",
    ),
    "sam_asset_lock": (
        Path("artifacts/paper/source_data/tmlr_v9b0_r2/sam_frozen_asset_lock.json"),
        "f3ecf7d6e044bff58711e51c8f1bf79523cbe45f49f9fbc2cd22a676587af257",
    ),
    "sequencing_addendum": (SEQUENCING_ADDENDUM_PATH, SEQUENCING_ADDENDUM_SHA256),
    "inference_outcome_policy": (
        INFERENCE_OUTCOME_POLICY_PATH, INFERENCE_OUTCOME_POLICY_SHA256,
    ),
    "e4_missingness_disposition": (
        E4_MISSINGNESS_DISPOSITION_PATH, E4_MISSINGNESS_DISPOSITION_SHA256,
    ),
    "e4_execution_registry": (
        E4_EXECUTION_REGISTRY_PATH, E4_EXECUTION_REGISTRY_SHA256,
    ),
    "taxonomy_verification": (TAXONOMY_PATH, TAXONOMY_SHA256),
    "voc_taxonomy": (
        Path("src/rail3/data/voc_taxonomy.py"),
        "d82b2ddf9004c955c680ead57e9136e465f9c551f791d00dff7a8f3bd364e75a",
    ),
}


class V9B1ContractError(RuntimeError):
    """Fail-closed result-blind execution contract error."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def read_stable_regular_bytes(path: Path) -> bytes:
    return secure_read_bytes(path)


def _repo_path(repo_root: Path, relative: Path) -> Path:
    root = repo_root.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise V9B1ContractError("V9B1_PATH_NOT_EXACT", "authority path must be repository-relative")
    candidate = root.joinpath(relative)
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise V9B1ContractError("V9B1_PATH_ESCAPES_REPOSITORY", str(relative)) from exc
    return candidate


def _verify_regular_file(repo_root: Path, relative: Path, expected_sha256: str) -> Path:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise V9B1ContractError("V9B1_AUTHORITY_SHA_INVALID", str(relative))
    path = _repo_path(repo_root, relative)
    try:
        encoded = secure_read_bytes(path)
    except FileNotFoundError as exc:
        raise V9B1ContractError("V9B1_AUTHORITY_MISSING", str(relative)) from exc
    except (OSError, RuntimeError) as exc:
        raise V9B1ContractError("V9B1_AUTHORITY_NOT_REGULAR", str(relative)) from exc
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise V9B1ContractError("V9B1_AUTHORITY_HASH_DRIFT", str(relative))
    return path


def load_config(repo_root: Path, path: Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = _repo_path(repo_root, path)
    try:
        payload = json.loads(secure_read_bytes(config_path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError("V9B1_CONFIG_INVALID", str(path)) from exc
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise V9B1ContractError("V9B1_CONFIG_SCHEMA_DRIFT", str(path))
    return payload


def validate_sequencing_addendum(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    identity = config.get("authorities", {}).get("sequencing_addendum", {})
    if (
        identity.get("path") != str(SEQUENCING_ADDENDUM_PATH)
        or identity.get("sha256") != SEQUENCING_ADDENDUM_SHA256
    ):
        raise V9B1ContractError(
            "V9B1_SEQUENCING_ADDENDUM_IDENTITY_DRIFT",
            "trajectory config does not bind the exact reviewed sequencing addendum",
        )
    path = _verify_regular_file(
        repo_root, SEQUENCING_ADDENDUM_PATH, SEQUENCING_ADDENDUM_SHA256
    )
    try:
        payload = json.loads(secure_read_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError("V9B1_SEQUENCING_ADDENDUM_INVALID", str(path)) from exc
    required_true = {
        "result_blind": True,
        "execution_order_change": True,
        "scientific_change": False,
        "performance_evidence_used": False,
        "coco_per_image_gt_used": False,
    }
    if payload.get("schema_version") != ADDENDUM_SCHEMA or any(
        payload.get(key) is not value for key, value in required_true.items()
    ):
        raise V9B1ContractError(
            "V9B1_SEQUENCING_ADDENDUM_SEMANTICS_DRIFT",
            "sequencing addendum no longer expresses the reviewed result-blind resolution",
        )
    resolved = payload.get("resolved_conflict", {})
    boundary = payload.get("stage_boundary", {})
    zeros = payload.get("v9b1_hard_zero_counters", {})
    if not (
        resolved.get("category_registry_verified_before_sam") is True
        and resolved.get("per_image_validation_deferred_to_v9b2") is True
        and resolved.get("per_image_validation_before_target_construction") is True
        and resolved.get("per_image_validation_before_performance_evaluation") is True
        and boundary.get("real_coco_annotation_open_allowed") is False
        and boundary.get("target_construction_allowed") is False
        and boundary.get("performance_evaluation_allowed") is False
        and all(int(zeros.get(key, -1)) == 0 for key in (
            "coco_annotation_rows_materialized",
            "coco_image_label_rows_materialized",
            "coco_per_image_gt_materialized",
            "coco_public_schema_additional_reads",
            "performance_results",
        ))
    ):
        raise V9B1ContractError(
            "V9B1_SEQUENCING_ADDENDUM_BOUNDARY_DRIFT",
            "sequencing addendum no longer preserves the V9B1 label-free boundary",
        )
    return payload


def validate_inference_outcome_policy(
    repo_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    identity = config.get("authorities", {}).get("inference_outcome_policy", {})
    if (
        identity.get("path") != str(INFERENCE_OUTCOME_POLICY_PATH)
        or identity.get("sha256") != INFERENCE_OUTCOME_POLICY_SHA256
    ):
        raise V9B1ContractError(
            "V9B1_INFERENCE_OUTCOME_POLICY_IDENTITY_DRIFT",
            "trajectory config does not bind the exact result-blind outcome policy",
        )
    path = _verify_regular_file(
        repo_root, INFERENCE_OUTCOME_POLICY_PATH, INFERENCE_OUTCOME_POLICY_SHA256
    )
    try:
        payload = json.loads(secure_read_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError(
            "V9B1_INFERENCE_OUTCOME_POLICY_INVALID", str(path)
        ) from exc
    retained = payload.get("candidate_retention_contract", {})
    outcomes = payload.get("outcome_families", {})
    valid_empty = outcomes.get("VALID_EMPTY_A0", {})
    persistent = outcomes.get("PERSISTENT_EXECUTION_FAILURE", {})
    classification = payload.get("classification_contract", {})
    retry = payload.get("retry_policy", {})
    missingness = payload.get("technical_missingness_gate", {})
    state_gate = missingness.get("state_gate", {})
    image_gate = missingness.get("image_group_gate", {})
    invariants = payload.get("result_blind_invariants", {})
    if not (
        payload.get("schema_version") == INFERENCE_OUTCOME_POLICY_SCHEMA
        and payload.get("policy_version") == "V9B1_INFERENCE_OUTCOME_POLICY_V1"
        and payload.get("status") == "FROZEN_RESULT_BLIND"
        and invariants.get("result_blind") is True
        and invariants.get("performance_evidence_used") is False
        and invariants.get("coco_per_image_gt_used") is False
        and int(invariants.get("coco_gt_content_reads", -1)) == 0
        and int(invariants.get("performance_results", -1)) == 0
        and retained.get("raw_empty_mask_handling") == "DISCARD_AS_NOT_RETAINED"
        and retained.get("raw_nonempty_mask_handling") == "RETAIN_IN_RAW_ORDER"
        and retained.get("zero_retained_masks_outcome") == "COMPLETED_NO_RESULT"
        and retained.get("empty_candidate_record_fabrication_allowed") is False
        and valid_empty.get("candidate_set") == "EMPTY_SET"
        and valid_empty.get("source_candidate") == "ABSENT"
        and valid_empty.get("stop", {}).get("valid") is True
        and valid_empty.get("eligibility", {}).get("terminal_classification") == "E0_OR_E1"
        and valid_empty.get("eligibility", {}).get(
            "complete_raw_composition_evidence_available"
        ) is True
        and persistent.get("state_status") == "INFERENCE_UNAVAILABLE"
        and persistent.get("additional_retry") is False
        and persistent.get("eligibility", {}).get("terminal_classification") == "E3"
        and persistent.get("eligibility", {}).get("execution_failure_proven") is True
        and persistent.get("eligibility", {}).get("classification_not_E4") is True
        and classification.get("kind") == "MUTUALLY_EXCLUSIVE_TERMINAL_CLASSIFICATION"
        and classification.get("precedence") == ["E0", "E1", "E2", "E3", "E4"]
        and classification.get("evidence_predicates_are_not_terminal_classifications") is True
        and int(retry.get("maximum_retries_per_state", -1)) == 1
        and retry.get("additional_retry_authorized") is False
        and state_gate.get("denominator") == EXPECTED_STATES
        and state_gate.get("maximum_count") == 100
        and state_gate.get("maximum_fraction") == 0.005
        and image_gate.get("denominator") == EXPECTED_IMAGES
        and image_gate.get("maximum_count") == 10
        and image_gate.get("maximum_fraction") == 0.01
    ):
        raise V9B1ContractError(
            "V9B1_INFERENCE_OUTCOME_POLICY_SEMANTICS_DRIFT",
            "result-blind inference outcome policy semantics drifted",
        )
    return payload


def validate_e4_missingness_disposition(
    repo_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    identity = config.get("authorities", {}).get("e4_missingness_disposition", {})
    if (
        identity.get("path") != str(E4_MISSINGNESS_DISPOSITION_PATH)
        or identity.get("sha256") != E4_MISSINGNESS_DISPOSITION_SHA256
    ):
        raise V9B1ContractError(
            "V9B1_E4_MISSINGNESS_DISPOSITION_IDENTITY_DRIFT",
            "trajectory config does not bind the exact E4 disposition",
        )
    path = _verify_regular_file(
        repo_root, E4_MISSINGNESS_DISPOSITION_PATH, E4_MISSINGNESS_DISPOSITION_SHA256
    )
    try:
        payload = json.loads(secure_read_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError(
            "V9B1_E4_MISSINGNESS_DISPOSITION_INVALID", str(path)
        ) from exc
    scope = payload.get("scope", {})
    semantic = payload.get("semantic_layer", {})
    execution = payload.get("execution_layer", {})
    retry = payload.get("retry_policy", {})
    gate = payload.get("technical_missingness_gate", {})
    denominator = payload.get("evaluation_denominator", {})
    parser = payload.get("future_parser_semantics", {})
    resume = payload.get("resume_gate", {})
    accounting = payload.get("result_blind_accounting", {})
    if not (
        payload.get("schema_version") == E4_MISSINGNESS_DISPOSITION_SCHEMA
        and payload.get("disposition_version") == "V9B1_E4_MISSINGNESS_DISPOSITION_V1"
        and payload.get("status") == "FROZEN_RESULT_BLIND"
        and scope.get("global") is True
        and scope.get("image_or_class_special_cases") is False
        and semantic.get("semantic_classification") == "E4_INSUFFICIENT_EVIDENCE"
        and semantic.get("semantic_resolution") == "UNRESOLVED_E4"
        and semantic.get("semantic_outcome_defined") is False
        and semantic.get("trajectory_defined") is False
        and semantic.get("scientific_outcome_imputed") is False
        and execution.get("execution_disposition")
        == "QUARANTINED_TECHNICAL_MISSINGNESS"
        and execution.get("execution_disposition_resolved") is True
        and execution.get("preserve_panel_membership") is True
        and execution.get("count_each_semantic_state_once") is True
        and execution.get("fabricate_action_outcome") is False
        and execution.get("impute_residual") is False
        and execution.get("impute_utility") is False
        and retry == {"retry_count": 1, "maximum_retry": 1, "additional_retry": False}
        and gate.get("thresholds_changed") is False
        and gate.get("state_denominator") == EXPECTED_STATES
        and gate.get("maximum_technical_unavailable_states") == 100
        and gate.get("maximum_state_fraction") == 0.005
        and gate.get("image_group_denominator") == EXPECTED_IMAGES
        and gate.get("maximum_affected_image_groups") == 10
        and gate.get("maximum_image_group_fraction") == 0.01
        and gate.get("failure_terminal") == "INFERENCE_COMPLETENESS_FAILED"
        and denominator.get("preserve_state_in_panel_completeness") is True
        and denominator.get(
            "exclude_only_from_numerical_denominators_requiring_defined_trajectory"
        ) is True
        and denominator.get("retain_other_class_states_from_same_image") is True
        and denominator.get("retain_image_group_for_bootstrap") is True
        and denominator.get("assign_error_zero") is False
        and denominator.get("impute_residual_or_utility") is False
        and parser.get("changed_by_R2D") is False
        and parser.get("historical_E4_special_case_in_backend") is False
        and resume.get("current_state_semantic_resolution") == "UNRESOLVED_E4"
        and resume.get("current_state_execution_disposition")
        == "QUARANTINED_TECHNICAL_MISSINGNESS"
        and resume.get("current_state_execution_disposition_resolved") is True
        and resume.get("semantic_resolution_claimed_resolved") is False
        and resume.get("resume_allowed_under_frozen_missingness_policy") is True
        and resume.get("automatic_shard0_resume") is False
        and accounting == {
            "COCO_GT_CONTENT_READ_COUNT": 0,
            "PERFORMANCE_RESULT_COUNT": 0,
            "REAL_SAM_CALL_COUNT": 0,
        }
    ):
        raise V9B1ContractError(
            "V9B1_E4_MISSINGNESS_DISPOSITION_SEMANTICS_DRIFT",
            "E4 technical-missingness disposition semantics drifted",
        )
    return payload


def validate_e4_execution_registry(
    repo_root: Path, config: dict[str, Any]
) -> tuple[E4ExecutionEvidence, ...]:
    identity = config.get("authorities", {}).get("e4_execution_registry", {})
    if identity != {
        "path": str(E4_EXECUTION_REGISTRY_PATH),
        "sha256": E4_EXECUTION_REGISTRY_SHA256,
    }:
        raise V9B1ContractError(
            "V9B1_E4_EXECUTION_REGISTRY_IDENTITY_DRIFT",
            "trajectory config does not bind the exact E4 execution registry",
        )
    try:
        return load_e4_execution_registry(repo_root, identity)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise V9B1ContractError(
            "V9B1_E4_EXECUTION_REGISTRY_INVALID", str(exc)
        ) from exc


def validate_small_authorities(repo_root: Path, config: dict[str, Any]) -> dict[str, str]:
    authorities = config.get("authorities", {})
    validated: dict[str, str] = {}
    for name, (relative, digest) in EXPECTED_SMALL_AUTHORITIES.items():
        identity = authorities.get(name, {})
        if identity.get("path") != str(relative) or identity.get("sha256") != digest:
            raise V9B1ContractError(
                "V9B1_AUTHORITY_REGISTRY_DRIFT", f"authority identity drift: {name}"
            )
        _verify_regular_file(repo_root, relative, digest)
        validated[str(relative)] = digest
    image_root = config.get("read_boundary", {}).get("coco_image_root")
    if image_root != str(IMAGE_ROOT):
        raise V9B1ContractError("V9B1_IMAGE_ROOT_DRIFT", str(image_root))
    root_path = _repo_path(repo_root, IMAGE_ROOT)
    try:
        metadata = root_path.lstat()
    except FileNotFoundError as exc:
        raise V9B1ContractError("V9B1_IMAGE_ROOT_MISSING", str(IMAGE_ROOT)) from exc
    if not stat.S_ISDIR(metadata.st_mode) or root_path.is_symlink():
        raise V9B1ContractError("V9B1_IMAGE_ROOT_NOT_DIRECTORY", str(IMAGE_ROOT))
    state_grid = config.get("state_grid", {})
    trajectory = config.get("trajectory", {})
    if not (
        state_grid.get("panel_images") == EXPECTED_IMAGES
        and state_grid.get("classes_per_image") == EXPECTED_CLASSES
        and state_grid.get("states") == EXPECTED_STATES
        and state_grid.get("canonical_image_id") == "12-digit zero-padded ASCII filename stem"
        and trajectory.get("a0_records") == EXPECTED_STATES
        and trajectory.get("a1_a6_records") == EXPECTED_STATES * len(ACTION_CODES)
        and trajectory.get("semantic_records") == EXPECTED_SEMANTIC_RECORDS
        and trajectory.get("physical_prompt_calls_max") == 160_000
        and trajectory.get("a6_fresh_session_canonical_replay_required") is True
        and trajectory.get("stop_references_a0_without_sam_call") is True
    ):
        raise V9B1ContractError("V9B1_GRID_CONFIG_DRIFT", "state/trajectory grid drift")
    shards = trajectory.get("shards")
    expected_shards = [
        {
            "shard": shard,
            "physical_gpu": shard,
            "gpu_uuid": EXPECTED_GPU_UUIDS[shard],
            "images": EXPECTED_SHARD_IMAGES[shard],
            "states": EXPECTED_SHARD_IMAGES[shard] * EXPECTED_CLASSES,
            "semantic_records": EXPECTED_SHARD_SEMANTIC_RECORDS[shard],
        }
        for shard in (0, 1)
    ]
    if shards != expected_shards:
        raise V9B1ContractError("V9B1_GPU_SHARD_CONFIG_DRIFT", str(shards))
    if trajectory.get("prospective_cost_contract") != {
        "A0": "CandidateGenerationResult.runtime_seconds",
        "A1_A2": (
            "V6_FINALIZE_OUTCOME_V1_PROMPT_PLUS_OUTCOME_PLUS_"
            "CANONICAL_SERIALIZATION_SHARED_ENCODE_NOT_AMORTIZED"
        ),
        "A3_A6_feature_eligible": False,
    }:
        raise V9B1ContractError(
            "V9B1_PROSPECTIVE_COST_CONTRACT_DRIFT", "trajectory cost contract drift"
        )
    validate_sequencing_addendum(repo_root, config)
    validate_inference_outcome_policy(repo_root, config)
    validate_e4_missingness_disposition(repo_root, config)
    validate_e4_execution_registry(repo_root, config)
    safe_summary = json.loads(
        secure_read_bytes(
            _repo_path(repo_root, EXPECTED_SMALL_AUTHORITIES["r2_safe_summary"][0])
        )
    )
    if not (
        safe_summary.get("terminal_state") == "TMLR_V9B0_R2_RECOVERY_LOCK_READY"
        and safe_summary.get("panel_sha256") == PANEL_MANIFEST_SHA256
        and safe_summary.get("coco_per_image_gt_materialized") == 0
        and safe_summary.get("performance_results") == 0
        and safe_summary.get("sam_asset_status") == "SAM_ASSET_LOAD_PASS"
    ):
        raise V9B1ContractError("V9B1_RECOVERY_SAFE_SUMMARY_DRIFT", "R2 recovery is not ready")
    return validated


@dataclass(frozen=True)
class PanelImageIdentity:
    image_index: int
    image_id: int
    canonical_image_id: str
    filename: str
    width: int
    height: int
    file_sha256: str
    pixel_sha256: str
    selection_digest: str
    duplicate_component: str
    asset_id: str

    def __post_init__(self) -> None:
        if self.image_index < 0 or self.image_id < 0:
            raise ValueError("panel image indices must be non-negative")
        if self.canonical_image_id != f"{self.image_id:012d}" or _CANONICAL_IMAGE_ID_RE.fullmatch(
            self.canonical_image_id
        ) is None:
            raise ValueError("canonical COCO image ID must be the 12-digit decimal stem")
        if self.filename != f"{self.canonical_image_id}.jpg":
            raise ValueError("panel filename differs from canonical COCO image identity")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("panel image dimensions must be positive")
        for digest in (self.file_sha256, self.pixel_sha256, self.selection_digest):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("panel image digest is invalid")
        expected_asset_id = stable_id("coco_val2017_image", {
            "canonical_image_id": self.canonical_image_id,
            "filename": self.filename,
            "file_sha256": self.file_sha256,
            "pixel_sha256": self.pixel_sha256,
            "width": self.width,
            "height": self.height,
        })
        if self.asset_id != expected_asset_id:
            raise ValueError("panel asset ID differs from the frozen label-free identity")

    def item_payload(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "image_sha256": self.file_sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class SemanticState:
    semantic_state_id: str
    panel_lock_id: str
    image: PanelImageIdentity
    class_id: int
    source_name: str
    canonical_prompt: str
    coco_category_id: int
    coco_category_name: str
    shard: int

    def __post_init__(self) -> None:
        if self.panel_lock_id != PANEL_LOCK_ID or not 1 <= self.class_id <= EXPECTED_CLASSES:
            raise ValueError("semantic state panel/class identity is invalid")
        if self.shard != shard_index(self.image.canonical_image_id, 2):
            raise ValueError("semantic state shard differs from canonical image-ID shard")
        expected = stable_id("v9b1_label_free_state", {
            "panel_lock_id": self.panel_lock_id,
            "asset_id": self.image.asset_id,
            "image_sha256": self.image.file_sha256,
            "class_id": self.class_id,
        })
        if self.semantic_state_id != expected:
            raise ValueError("semantic state ID differs from its frozen identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_state_id": self.semantic_state_id,
            "panel_lock_id": self.panel_lock_id,
            "image_index": self.image.image_index,
            "image_id": self.image.image_id,
            "canonical_image_id": self.image.canonical_image_id,
            "filename": self.image.filename,
            "asset_id": self.image.asset_id,
            "width": self.image.width,
            "height": self.image.height,
            "file_sha256": self.image.file_sha256,
            "pixel_sha256": self.image.pixel_sha256,
            "class_id": self.class_id,
            "source_name": self.source_name,
            "canonical_prompt": self.canonical_prompt,
            "coco_category_id": self.coco_category_id,
            "coco_category_name": self.coco_category_name,
            "shard": self.shard,
        }


def _asset_id(row: dict[str, str], canonical_image_id: str) -> str:
    return stable_id("coco_val2017_image", {
        "canonical_image_id": canonical_image_id,
        "filename": row["filename"],
        "file_sha256": row["file_sha256"],
        "pixel_sha256": row["pixel_sha256"],
        "width": int(row["width"]),
        "height": int(row["height"]),
    })


def load_frozen_panel_images(repo_root: Path) -> tuple[PanelImageIdentity, ...]:
    panel_path = _verify_regular_file(repo_root, PANEL_MANIFEST_PATH, PANEL_MANIFEST_SHA256)
    identity_path = _verify_regular_file(repo_root, PANEL_IDENTITY_PATH, PANEL_IDENTITY_SHA256)
    panel = json.loads(secure_read_bytes(panel_path))
    selected = tuple(int(value) for value in panel.get("selected_image_ids", ()))
    if (
        panel.get("panel_lock_id") != PANEL_LOCK_ID
        or panel.get("canonical_image_id") != "12-digit zero-padded ASCII filename stem"
        or panel.get("panel_n") != EXPECTED_IMAGES
        or len(selected) != EXPECTED_IMAGES
        or len(set(selected)) != EXPECTED_IMAGES
    ):
        raise V9B1ContractError("V9B1_PANEL_MANIFEST_DRIFT", "panel lock/grid is not exact")
    with io.StringIO(secure_read_bytes(identity_path).decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        required = (
            "image_id", "filename", "width", "height", "file_sha256", "pixel_sha256",
            "selection_digest", "duplicate_component", "eligibility_status",
        )
        if tuple(reader.fieldnames or ()) != required:
            raise V9B1ContractError("V9B1_PANEL_IDENTITY_SCHEMA_DRIFT", "CSV header drift")
        rows = list(reader)
    if len(rows) != EXPECTED_IMAGES or tuple(int(row["image_id"]) for row in rows) != selected:
        raise V9B1ContractError("V9B1_PANEL_IDENTITY_ROWS_DRIFT", "panel CSV/order drift")
    images: list[PanelImageIdentity] = []
    for index, row in enumerate(rows):
        if row["eligibility_status"] != "SELECTED_LABEL_BLIND":
            raise V9B1ContractError("V9B1_PANEL_IDENTITY_ELIGIBILITY_DRIFT", row["image_id"])
        image_id = int(row["image_id"])
        canonical = f"{image_id:012d}"
        images.append(PanelImageIdentity(
            image_index=index,
            image_id=image_id,
            canonical_image_id=canonical,
            filename=row["filename"],
            width=int(row["width"]),
            height=int(row["height"]),
            file_sha256=row["file_sha256"],
            pixel_sha256=row["pixel_sha256"],
            selection_digest=row["selection_digest"],
            duplicate_component=row["duplicate_component"],
            asset_id=_asset_id(row, canonical),
        ))
    shard_counts = tuple(sum(shard_index(image.canonical_image_id, 2) == value for image in images) for value in (0, 1))
    if shard_counts != EXPECTED_SHARD_IMAGES:
        raise V9B1ContractError("V9B1_SHARD_COUNT_DRIFT", str(shard_counts))
    return tuple(images)


def load_taxonomy(repo_root: Path) -> dict[int, dict[str, Any]]:
    path = _verify_regular_file(repo_root, TAXONOMY_PATH, TAXONOMY_SHA256)
    with io.StringIO(secure_read_bytes(path).decode("utf-8"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CLASSES:
        raise V9B1ContractError("V9B1_TAXONOMY_COUNT_DRIFT", str(len(rows)))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        class_id = int(row["voc_index"])
        if row.get("status") != "PASS" or class_id in result:
            raise V9B1ContractError("V9B1_TAXONOMY_MAPPING_DRIFT", row.get("voc_name", ""))
        result[class_id] = {
            "source_name": row["voc_name"],
            "coco_category_id": int(row["coco_category_id"]),
            "coco_category_name": row["expected_coco_name"],
        }
    if tuple(sorted(result)) != tuple(range(1, EXPECTED_CLASSES + 1)) or len({
        row["coco_category_id"] for row in result.values()
    }) != EXPECTED_CLASSES:
        raise V9B1ContractError("V9B1_TAXONOMY_MAPPING_DRIFT", "mapping is not 20/20 unique")
    return result


def build_semantic_states(
    images: Sequence[PanelImageIdentity],
    taxonomy: dict[int, dict[str, Any]],
    *,
    require_full_panel: bool = True,
) -> tuple[SemanticState, ...]:
    if require_full_panel and len(images) != EXPECTED_IMAGES:
        raise V9B1ContractError("V9B1_PANEL_SIZE_DRIFT", str(len(images)))
    prompts = canonical_voc_prompts()
    if len(prompts) != EXPECTED_CLASSES or tuple(sorted(taxonomy)) != tuple(range(1, 21)):
        raise V9B1ContractError("V9B1_CLASS_GRID_DRIFT", "class authority is not exact")
    states: list[SemanticState] = []
    for image in images:
        for class_id, (prompt, metadata) in enumerate(prompts, start=1):
            mapping = taxonomy[class_id]
            if metadata.class_id != class_id or prompt.text != metadata.canonical_text:
                raise V9B1ContractError("V9B1_PROMPT_ORDER_DRIFT", str(class_id))
            state_id = stable_id("v9b1_label_free_state", {
                "panel_lock_id": PANEL_LOCK_ID,
                "asset_id": image.asset_id,
                "image_sha256": image.file_sha256,
                "class_id": class_id,
            })
            states.append(SemanticState(
                semantic_state_id=state_id,
                panel_lock_id=PANEL_LOCK_ID,
                image=image,
                class_id=class_id,
                source_name=mapping["source_name"],
                canonical_prompt=metadata.canonical_text,
                coco_category_id=mapping["coco_category_id"],
                coco_category_name=mapping["coco_category_name"],
                shard=shard_index(image.canonical_image_id, 2),
            ))
    expected = len(images) * EXPECTED_CLASSES
    if len(states) != expected or len({state.semantic_state_id for state in states}) != expected:
        raise V9B1ContractError("V9B1_STATE_GRID_NOT_CARTESIAN", str(len(states)))
    return tuple(states)


def state_manifest_payload(states: Sequence[SemanticState]) -> dict[str, Any]:
    image_count = len({state.image.asset_id for state in states})
    shard_images = [
        len({state.image.asset_id for state in states if state.shard == shard}) for shard in (0, 1)
    ]
    payload = {
        "schema_version": STATE_MANIFEST_SCHEMA,
        "panel_lock_id": PANEL_LOCK_ID,
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "panel_identity_sha256": PANEL_IDENTITY_SHA256,
        "taxonomy_sha256": TAXONOMY_SHA256,
        "canonical_image_id": "12-digit zero-padded ASCII filename stem",
        "shard_rule": "sha256(canonical_image_id UTF-8)[0:8] big-endian modulo 2",
        "counts": {
            "images": image_count,
            "classes": EXPECTED_CLASSES,
            "states": len(states),
            "semantic_records_expected": len(states) * len(ACTION_ORDER),
            "shard_images": shard_images,
            "shard_semantic_records": [value * EXPECTED_CLASSES * len(ACTION_ORDER) for value in shard_images],
        },
        "states": [state.to_dict() for state in states],
    }
    return payload


def freeze_state_manifest(path: Path, payload: dict[str, Any]) -> str:
    encoded = canonical_json_bytes(payload) + b"\n"
    try:
        secure_create_once(path, encoded)
    except FileExistsError:
        try:
            existing = read_stable_regular_bytes(path)
        except OSError as exc:
            raise V9B1ContractError("V9B1_STATE_MANIFEST_UNREADABLE", str(path)) from exc
        if existing != encoded:
            raise V9B1ContractError("V9B1_STATE_MANIFEST_CONFLICT", str(path))
    return hashlib.sha256(encoded).hexdigest()


STATE_MANIFEST_FIELDS = (
    "semantic_state_id",
    "panel_lock_id",
    "image_index",
    "image_id",
    "canonical_image_id",
    "filename",
    "asset_id",
    "width",
    "height",
    "file_sha256",
    "pixel_sha256",
    "class_id",
    "source_name",
    "canonical_prompt",
    "coco_category_id",
    "coco_category_name",
    "shard",
)


def state_manifest_csv_bytes(states: Sequence[SemanticState]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=STATE_MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for state in states:
        payload = state.to_dict()
        writer.writerow({key: payload[key] for key in STATE_MANIFEST_FIELDS})
    return stream.getvalue().encode("utf-8")


def freeze_state_manifest_csv(path: Path, states: Sequence[SemanticState]) -> str:
    encoded = state_manifest_csv_bytes(states)
    try:
        secure_create_once(path, encoded)
    except FileExistsError:
        try:
            existing = read_stable_regular_bytes(path)
        except OSError as exc:
            raise V9B1ContractError("V9B1_STATE_MANIFEST_UNREADABLE", str(path)) from exc
        if existing != encoded:
            raise V9B1ContractError("V9B1_STATE_MANIFEST_CONFLICT", str(path))
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, *arguments: str, expected_codes: tuple[int, ...] = (0,)) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode not in expected_codes:
        raise V9B1ContractError(
            "V9B1_IMPLEMENTATION_GIT_COMMAND_FAILED", " ".join(arguments)
        )
    return completed.stdout.strip()


def committed_implementation_identity(
    repo_root: Path, *, require_global_clean: bool
) -> dict[str, Any]:
    if require_global_clean and _git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise V9B1ContractError(
            "V9B1_FORMAL_WORKTREE_NOT_CLEAN", "formal implementation must be committed-clean"
        )
    head = _git(repo_root, "rev-parse", "HEAD")
    hashes: dict[str, str] = {}
    for relative in FORMAL_IMPLEMENTATION_PATHS:
        _git(repo_root, "ls-files", "--error-unmatch", str(relative))
        if _git(repo_root, "diff", "--name-only", "HEAD", "--", str(relative)):
            raise V9B1ContractError("V9B1_IMPLEMENTATION_WORKTREE_DRIFT", str(relative))
        hashes[str(relative)] = hashlib.sha256(
            secure_read_bytes(repo_root / relative)
        ).hexdigest()
    config_sha256 = hashlib.sha256(
        secure_read_bytes(repo_root / CONFIG_PATH)
    ).hexdigest()
    return {
        "commit": head,
        "implementation_sha256": hashes,
        "config_sha256": config_sha256,
    }


def shard_manifest_payload(
    *,
    state_manifest_path: Path,
    state_manifest_sha256: str,
    state_payload: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "panel_lock_id": PANEL_LOCK_ID,
        "state_manifest_path": str(state_manifest_path),
        "state_manifest_sha256": state_manifest_sha256,
        "implementation_commit": implementation["commit"],
        "implementation_sha256": implementation["implementation_sha256"],
        "config_path": str(CONFIG_PATH),
        "config_sha256": implementation["config_sha256"],
        "shard_rule": state_payload["shard_rule"],
        "counts": state_payload["counts"],
        "shards": [
            {
                "shard": shard,
                "physical_gpu": shard,
                "gpu_uuid": EXPECTED_GPU_UUIDS[shard],
                "images": EXPECTED_SHARD_IMAGES[shard],
                "states": EXPECTED_SHARD_IMAGES[shard] * EXPECTED_CLASSES,
                "semantic_records": EXPECTED_SHARD_SEMANTIC_RECORDS[shard],
                "cache_roots": {
                    "a0": str(SHARD_CACHE_ROOT / f"shard-{shard}" / "a0"),
                    "actions": str(SHARD_CACHE_ROOT / f"shard-{shard}" / "actions"),
                    "plans": str(SHARD_CACHE_ROOT / f"shard-{shard}" / "plans"),
                },
            }
            for shard in (0, 1)
        ],
        "final_cache_roots": {
            "a0": str(FINAL_A0_CACHE_ROOT),
            "actions": str(FINAL_ACTION_CACHE_ROOT),
            "plans": str(FINAL_PLAN_LOCK_ROOT),
        },
    }


def validate_frozen_execution_manifests(
    repo_root: Path,
    config: dict[str, Any],
    states: Sequence[SemanticState],
    *,
    require_global_clean: bool,
) -> dict[str, Any]:
    output = config.get("outputs", {})
    source_root = Path(str(output.get("source_data_root", "")))
    state_relative = source_root / str(output.get("state_manifest", ""))
    shard_relative = source_root / str(output.get("shard_manifest", ""))
    expected_state = state_manifest_csv_bytes(states)
    state_path = _repo_path(repo_root, state_relative)
    shard_path = _repo_path(repo_root, shard_relative)
    try:
        actual_state = read_stable_regular_bytes(state_path)
        shard_encoded = read_stable_regular_bytes(shard_path)
        manifest = json.loads(shard_encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V9B1ContractError("V9B1_FROZEN_EXECUTION_MANIFEST_INVALID", str(exc)) from exc
    state_sha = hashlib.sha256(actual_state).hexdigest()
    if actual_state != expected_state or manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise V9B1ContractError("V9B1_STATE_OR_SHARD_MANIFEST_DRIFT", str(state_relative))
    if shard_encoded != canonical_json_bytes(manifest) + b"\n":
        raise V9B1ContractError("V9B1_SHARD_MANIFEST_NOT_CANONICAL", str(shard_relative))
    implementation = committed_implementation_identity(
        repo_root, require_global_clean=require_global_clean
    )
    frozen_commit = manifest.get("implementation_commit")
    if not isinstance(frozen_commit, str) or re.fullmatch(r"[0-9a-f]{40}", frozen_commit) is None:
        raise V9B1ContractError("V9B1_IMPLEMENTATION_COMMIT_INVALID", str(frozen_commit))
    ancestry = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", str(frozen_commit), implementation["commit"]],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    expected_payload = state_manifest_payload(states)
    expected_core = shard_manifest_payload(
        state_manifest_path=state_relative,
        state_manifest_sha256=state_sha,
        state_payload=expected_payload,
        implementation={
            "commit": frozen_commit,
            "implementation_sha256": implementation["implementation_sha256"],
            "config_sha256": implementation["config_sha256"],
        },
    )
    if ancestry != 0 or manifest != expected_core:
        raise V9B1ContractError("V9B1_FROZEN_EXECUTION_IDENTITY_DRIFT", str(shard_relative))
    return manifest


def action_sequence_index(action_code: str, class_id: int) -> int:
    if action_code not in ACTION_ORDER or not 1 <= class_id <= EXPECTED_CLASSES:
        raise ValueError("action sequence identity is invalid")
    return ACTION_ORDER.index(action_code) * EXPECTED_CLASSES + class_id - 1


def trajectory_id_for(state: EncodedImageState) -> str:
    return stable_id("trajectory", {
        "model_spec_id": state.model_spec_id,
        "asset_id": state.asset_id,
        "state_id": state.state_id,
        "protocol": "independent-canonical-text-v1",
    })


@dataclass(frozen=True)
class SessionHandle:
    state: EncodedImageState
    purpose: str


class LabelFreeSessionBackend(Protocol):
    model_spec_id: str
    physical_gpu_index: int

    def prepare_image(self, image: PanelImageIdentity) -> None: ...

    def release_image(self, image: PanelImageIdentity) -> None: ...

    def open_original(self, image: PanelImageIdentity, *, purpose: str) -> SessionHandle: ...

    def open_flip(self, image: PanelImageIdentity) -> SessionHandle: ...

    def open_crop(
        self, image: PanelImageIdentity, crop_box_xyxy: tuple[int, int, int, int], *, action_id: str
    ) -> SessionHandle: ...

    def run_prompt(
        self,
        session: SessionHandle,
        *,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        trajectory_id: str,
        upstream_object_id: int | None = None,
    ) -> CandidateGenerationResult: ...

    def close_session(self, session: SessionHandle) -> None: ...

    def counters(self) -> dict[str, int]: ...

    def evidence(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StatePlanBundle:
    state: SemanticState
    a0_identity: CacheIdentity
    a0_result: CandidateGenerationResult
    plans: tuple[ActionPlan, ...]
    action_identities: tuple[ActionCacheIdentity, ...]

    def __post_init__(self) -> None:
        if tuple(plan.action_code for plan in self.plans) != ACTION_CODES:
            raise ValueError("state plan bundle action order is invalid")
        if len(self.action_identities) != len(self.plans):
            raise ValueError("state plan bundle cache identities are incomplete")


def _result_status(result: CandidateGenerationResult) -> str:
    if result.candidate is not None:
        return "result"
    if result.no_result is not None:
        return "no_result"
    return "failure"


def _candidate_replay_projection(result: CandidateGenerationResult) -> dict[str, Any]:
    return {
        "status": _result_status(result),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "trajectory_id": candidate.trajectory_id,
                "prompt_id": candidate.prompt_id,
                "step_index": candidate.step_index,
                "candidate_index": candidate.candidate_index,
                "width": candidate.width,
                "height": candidate.height,
                "mask_rle": list(candidate.mask_rle),
                "mask_sha256": candidate.mask_sha256,
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "model_score": candidate.model_score,
                "upstream_object_id": candidate.upstream_object_id,
                "upstream_bbox_xywh": (
                    None if candidate.upstream_bbox_xywh is None
                    else list(candidate.upstream_bbox_xywh)
                ),
            }
            for candidate in result.all_candidates
        ],
    }


def assert_a6_replay_closure(
    frozen_a0: CandidateGenerationResult, replay: CandidateGenerationResult
) -> str:
    frozen_projection = _candidate_replay_projection(frozen_a0)
    replay_projection = _candidate_replay_projection(replay)
    if frozen_projection != replay_projection:
        raise V9B1ContractError(
            "V9B1_A6_CANONICAL_REPLAY_MISMATCH",
            "fresh-session canonical replay differs from the frozen A0 source",
        )
    source = select_source_candidate(frozen_a0)
    replay_source = select_source_candidate(replay)
    if (
        source is None
        or replay_source is None
        or source.candidate_id != replay_source.candidate_id
        or source.upstream_object_id is None
        or replay_source.upstream_object_id != source.upstream_object_id
    ):
        raise V9B1ContractError(
            "V9B1_A6_SOURCE_OBJECT_CLOSURE_FAILED",
            "A6 replay did not preserve the selected upstream object identity",
        )
    return hashlib.sha256(canonical_json_bytes(frozen_projection)).hexdigest()


def filter_a6_source_object(
    result: CandidateGenerationResult, upstream_object_id: int
) -> CandidateGenerationResult:
    if result.candidate is None:
        return result
    selected = tuple(
        candidate for candidate in result.all_candidates
        if candidate.upstream_object_id == upstream_object_id
    )
    if not selected:
        return CandidateGenerationResult(no_result=CandidateNoResult(
            "REFINED_OBJECT_NOT_RETURNED",
            "point refinement completed but did not return the source object",
            None,
        ))
    return CandidateGenerationResult(
        candidate=selected[0], additional_candidates=selected[1:]
    )


def _load_a0(cache: CandidateCache, identity: CacheIdentity) -> CandidateGenerationResult | None:
    try:
        return cache.load(identity)
    except FileNotFoundError:
        return None


def _load_action(cache: TrajectoryCache, identity: ActionCacheIdentity) -> ActionOutcome | None:
    try:
        return cache.load(identity)
    except TrajectoryCacheError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise


def _plan_lock_payload(
    image: PanelImageIdentity,
    bundles: Sequence[StatePlanBundle],
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> dict[str, Any]:
    payload = {
        "schema_version": (
            R2D_PLAN_LOCK_SCHEMA if technical_missingness else PLAN_LOCK_SCHEMA
        ),
        "panel_lock_id": PANEL_LOCK_ID,
        "asset_id": image.asset_id,
        "canonical_image_id": image.canonical_image_id,
        "a0": [
            {
                "semantic_state_id": bundle.state.semantic_state_id,
                "class_id": bundle.state.class_id,
                "cache_key": bundle.a0_identity.cache_key,
                "status": _result_status(bundle.a0_result),
                "candidate_ids": sorted(candidate.candidate_id for candidate in bundle.a0_result.all_candidates),
            }
            for bundle in bundles
        ],
        "actions": [
            {
                "semantic_state_id": bundle.state.semantic_state_id,
                "class_id": bundle.state.class_id,
                "action_code": plan.action_code,
                "action_id": plan.action_id,
                "cache_key": identity.cache_key,
                "feasible": plan.feasible,
                "reason_code": plan.reason_code,
                "identity": plan.identity_payload(),
            }
            for bundle in bundles
            for plan, identity in zip(bundle.plans, bundle.action_identities)
        ],
    }
    if technical_missingness:
        payload["technical_missingness"] = [
            entry.disposition_dict()
            for entry in sorted(
                technical_missingness, key=lambda item: item.record.semantic_state_id
            )
        ]
    return payload


def _freeze_plan_lock(
    root: Path,
    image: PanelImageIdentity,
    bundles: Sequence[StatePlanBundle],
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> Path:
    payload = _plan_lock_payload(image, bundles, technical_missingness)
    encoded = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(image.asset_id.encode("utf-8")).hexdigest()
    path = root / digest[:2] / f"image_plan_{digest}.json"
    try:
        secure_create_once(path, encoded)
    except FileExistsError:
        try:
            existing = read_stable_regular_bytes(path)
        except OSError as exc:
            raise V9B1ContractError("V9B1_PLAN_LOCK_UNREADABLE", image.canonical_image_id) from exc
        if existing != encoded:
            raise V9B1ContractError("V9B1_PLAN_LOCK_CONFLICT", image.canonical_image_id)
    return path


def _plan_lock_path(root: Path, image: PanelImageIdentity) -> Path:
    digest = hashlib.sha256(image.asset_id.encode("utf-8")).hexdigest()
    return root / digest[:2] / f"image_plan_{digest}.json"


def derive_bundles_from_a0_cache(
    states: Sequence[SemanticState],
    *,
    model_spec_id: str,
    a0_cache: CandidateCache,
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> tuple[StatePlanBundle, ...]:
    prompts = {metadata.class_id: prompt for prompt, metadata in canonical_voc_prompts()}
    bundles: list[StatePlanBundle] = []
    executable, _ = partition_executable_states(states, technical_missingness)
    for state in executable:
        prompt = prompts[state.class_id]
        identity = voc_cache_identity(
            state.image.item_payload(), prompt, model_spec_id=model_spec_id
        )
        result = _load_a0(a0_cache, identity)
        if result is None:
            raise FileNotFoundError(a0_cache.path_for(identity))
        plans = build_action_plans(
            state.image.item_payload(),
            class_id=state.class_id,
            canonical_result=result,
            canonical_cache_key=identity.cache_key,
        )
        action_identities = tuple(ActionCacheIdentity.create(
            model_spec_id=model_spec_id,
            item=state.image.item_payload(),
            plan=plan,
        ) for plan in plans)
        bundles.append(StatePlanBundle(state, identity, result, plans, action_identities))
    return tuple(bundles)


def _bounded_plan_paths(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    paths: set[Path] = set()
    with os.scandir(root) as first_level:
        for directory in first_level:
            if directory.is_symlink() or not directory.is_dir(follow_symlinks=False) or re.fullmatch(r"[0-9a-f]{2}", directory.name) is None:
                raise V9B1ContractError("V9B1_PLAN_LOCK_LAYOUT_INVALID", directory.path)
            with os.scandir(directory.path) as entries:
                for entry in entries:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise V9B1ContractError("V9B1_PLAN_LOCK_NOT_REGULAR", entry.path)
                    if re.fullmatch(r"image_plan_[0-9a-f]{64}\.json", entry.name) is None:
                        raise V9B1ContractError("V9B1_PLAN_LOCK_NAME_INVALID", entry.name)
                    paths.add(Path(entry.path).resolve())
    return paths


def validate_plan_locks(
    bundles: Sequence[StatePlanBundle],
    plan_lock_root: Path,
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> dict[str, Any]:
    grouped: dict[str, list[StatePlanBundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.state.image.asset_id, []).append(bundle)
    expected: dict[Path, bytes] = {}
    entries: list[dict[str, str]] = []
    missing_by_image: dict[str, list[E4ExecutionEvidence]] = {}
    for entry in technical_missingness:
        missing_by_image.setdefault(entry.record.image_group_id, []).append(entry)
    if not set(missing_by_image) <= {
        bundle.state.image.canonical_image_id for bundle in bundles
    }:
        raise V9B1ContractError(
            "V9B1_PLAN_LOCK_MISSINGNESS_IMAGE_ABSENT",
            "technical-missingness image has no preserved executable states",
        )
    for image_bundles in grouped.values():
        ordered = tuple(sorted(image_bundles, key=lambda bundle: bundle.state.class_id))
        image = ordered[0].state.image
        missing = tuple(sorted(
            missing_by_image.get(image.canonical_image_id, ()),
            key=lambda entry: entry.record.class_id,
        ))
        observed_classes = tuple(bundle.state.class_id for bundle in ordered)
        missing_classes = tuple(entry.record.class_id for entry in missing)
        if (
            len(set((*observed_classes, *missing_classes)))
            != len(observed_classes) + len(missing_classes)
            or tuple(sorted((*observed_classes, *missing_classes))) != tuple(range(1, 21))
        ):
            raise V9B1ContractError("V9B1_PLAN_LOCK_CLASS_GRID_DRIFT", image.canonical_image_id)
        path = _plan_lock_path(plan_lock_root, image).resolve()
        encoded = canonical_json_bytes(_plan_lock_payload(image, ordered, missing)) + b"\n"
        expected[path] = encoded
        entries.append({
            "canonical_image_id": image.canonical_image_id,
            "plan_lock_sha256": hashlib.sha256(encoded).hexdigest(),
        })
    actual = _bounded_plan_paths(plan_lock_root)
    if actual != set(expected):
        raise V9B1ContractError("V9B1_PLAN_LOCK_SET_MISMATCH", "missing or extra image plan lock")
    for path, encoded in expected.items():
        if read_stable_regular_bytes(path) != encoded:
            raise V9B1ContractError("V9B1_PLAN_LOCK_CONTENT_MISMATCH", str(path))
    aggregate = hashlib.sha256(canonical_json_bytes(sorted(
        entries, key=lambda entry: entry["canonical_image_id"]
    ))).hexdigest()
    return {
        "plan_lock_count": len(expected),
        "plan_record_count": len(bundles) * len(ACTION_CODES),
        "technical_missingness_count": len(technical_missingness),
        "plan_lock_bytes": sum(len(encoded) for encoded in expected.values()),
        "plan_lock_aggregate_sha256": aggregate,
    }


def merge_expected_plan_locks(
    source_roots: Sequence[Path],
    destination_root: Path,
    bundles: Sequence[StatePlanBundle],
) -> dict[str, Any]:
    grouped: dict[str, list[StatePlanBundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.state.image.asset_id, []).append(bundle)
    source_sets = [_bounded_plan_paths(root) for root in source_roots]
    seen_names: set[str] = set()
    for paths in source_sets:
        names = {path.name for path in paths}
        if seen_names & names:
            raise V9B1ContractError("V9B1_PLAN_MERGE_DUPLICATE_SOURCE", "plan shards overlap")
        seen_names |= names
    expected_names: set[str] = set()
    for image_bundles in grouped.values():
        image = image_bundles[0].state.image
        expected_names.add(_plan_lock_path(Path("."), image).name)
    if seen_names != expected_names:
        raise V9B1ContractError("V9B1_PLAN_MERGE_SOURCE_SET_MISMATCH", "plan shards incomplete")
    for image_bundles in grouped.values():
        ordered = tuple(sorted(image_bundles, key=lambda bundle: bundle.state.class_id))
        image = ordered[0].state.image
        name = _plan_lock_path(Path("."), image).name
        matches = [path for paths in source_sets for path in paths if path.name == name]
        if len(matches) != 1:
            raise V9B1ContractError("V9B1_PLAN_MERGE_SOURCE_OWNERSHIP_INVALID", name)
        expected_bytes = canonical_json_bytes(_plan_lock_payload(image, ordered)) + b"\n"
        if read_stable_regular_bytes(matches[0]) != expected_bytes:
            raise V9B1ContractError("V9B1_PLAN_LOCK_CONTENT_MISMATCH", name)
        destination = _plan_lock_path(destination_root, image)
        try:
            secure_create_once(destination, expected_bytes)
        except FileExistsError:
            if read_stable_regular_bytes(destination) != expected_bytes:
                raise V9B1ContractError("V9B1_PLAN_LOCK_CONFLICT", name)
    return validate_plan_locks(bundles, destination_root)


class V9B1TrajectoryDriver:
    """Execute A0 then immutable A1--A6 plans through an injected backend."""

    def __init__(
        self,
        *,
        backend: LabelFreeSessionBackend,
        a0_cache: CandidateCache,
        action_cache: TrajectoryCache,
        plan_lock_root: Path,
        technical_missingness: Sequence[E4ExecutionEvidence] = (),
    ) -> None:
        if not isinstance(a0_cache, SecureCandidateCache) or not isinstance(
            action_cache, SecureTrajectoryCache
        ):
            raise V9B1ContractError(
                "V9B1_INSECURE_CACHE_IMPLEMENTATION",
                "formal/synthetic trajectory driver requires no-follow immutable caches",
            )
        self.backend = backend
        self.a0_cache = a0_cache
        self.action_cache = action_cache
        self.plan_lock_root = plan_lock_root
        self.technical_missingness = tuple(technical_missingness)
        self._missingness_by_state = {
            entry.record.semantic_state_id: entry
            for entry in self.technical_missingness
        }
        if len(self._missingness_by_state) != len(self.technical_missingness):
            raise V9B1ContractError(
                "V9B1_E4_EXECUTION_REGISTRY_DUPLICATE",
                "technical-missingness records duplicate a semantic state",
            )
        self._prompts = {metadata.class_id: (prompt, metadata) for prompt, metadata in canonical_voc_prompts()}

    def _run_prompt(
        self,
        session: SessionHandle,
        *,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        upstream_object_id: int | None = None,
    ) -> CandidateGenerationResult:
        return self.backend.run_prompt(
            session,
            prompt=prompt,
            metadata=metadata,
            trajectory_id=trajectory_id_for(session.state),
            upstream_object_id=upstream_object_id,
        )

    def _store_action(
        self,
        *,
        bundle: StatePlanBundle,
        plan: ActionPlan,
        identity: ActionCacheIdentity,
        result: CandidateGenerationResult | None,
        telemetry: dict[str, Any],
        action_started: float | None = None,
    ) -> ActionOutcome:
        provisional = outcome_from_generation(
            plan=plan,
            item=bundle.state.image.item_payload(),
            result=result,
            telemetry=telemetry,
        )
        outcome = provisional
        if plan.action_code in {"A1", "A2"}:
            if action_started is None:
                raise V9B1ContractError(
                    "V9B1_BASE_ACTION_TIMER_MISSING", plan.action_id
                )
            serialization_started = time.perf_counter()
            canonical_json_bytes(provisional.to_dict())
            serialization_seconds = time.perf_counter() - serialization_started
            action_cost_seconds = time.perf_counter() - action_started
            if action_cost_seconds < serialization_seconds or action_cost_seconds <= 0.0:
                raise V9B1ContractError(
                    "V9B1_BASE_ACTION_COST_INVALID", plan.action_id
                )
            outcome = replace(provisional, telemetry={
                **telemetry,
                "wall_seconds": action_cost_seconds,
                "action_cost_seconds": action_cost_seconds,
                "serialization_seconds": serialization_seconds,
            })
        self.action_cache.store(identity, outcome)
        return outcome

    def _action_telemetry(
        self,
        *,
        bundle: StatePlanBundle,
        plan: ActionPlan,
        prompt_calls: int,
        session: SessionHandle | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence = self.backend.evidence()
        required = {
            "model_spec_id",
            "model_source_commit",
            "checkpoint_sha256",
            "sam_config_hash",
        }
        if set(evidence) != required:
            raise V9B1ContractError(
                "V9B1_BACKEND_EVIDENCE_SCHEMA_DRIFT", str(sorted(evidence))
            )
        physical = None if session is None else {
            "physical_input_asset_id": session.state.asset_id,
            "physical_input_image_sha256": session.state.image_sha256,
            "physical_input_state_id": session.state.state_id,
            "physical_input_width": session.state.width,
            "physical_input_height": session.state.height,
        }
        return {
            **evidence,
            "panel_lock_id": PANEL_LOCK_ID,
            "action_protocol_sha256": PROTOCOL_SHA256,
            "semantic_state_id": bundle.state.semantic_state_id,
            "original_asset_id": bundle.state.image.asset_id,
            "original_image_sha256": bundle.state.image.file_sha256,
            "action_id": plan.action_id,
            "prompt_call_count": prompt_calls,
            "physical_gpu_index": self.backend.physical_gpu_index,
            "logical_device": "cuda:0" if evidence["checkpoint_sha256"] is not None else "synthetic",
            "physical_input": physical,
            **(extra or {}),
        }

    def _derive_bundles(
        self, image_states: Sequence[SemanticState], a0_results: dict[str, CandidateGenerationResult]
    ) -> tuple[StatePlanBundle, ...]:
        bundles: list[StatePlanBundle] = []
        for state in image_states:
            prompt, _ = self._prompts[state.class_id]
            a0_identity = voc_cache_identity(
                state.image.item_payload(), prompt, model_spec_id=self.backend.model_spec_id
            )
            result = a0_results[state.semantic_state_id]
            plans = build_action_plans(
                state.image.item_payload(),
                class_id=state.class_id,
                canonical_result=result,
                canonical_cache_key=a0_identity.cache_key,
            )
            identities = tuple(ActionCacheIdentity.create(
                model_spec_id=self.backend.model_spec_id,
                item=state.image.item_payload(),
                plan=plan,
            ) for plan in plans)
            bundles.append(StatePlanBundle(state, a0_identity, result, plans, identities))
        return tuple(bundles)

    def run_image(
        self,
        image_states: Sequence[SemanticState],
        *,
        actions_allowed: bool = True,
    ) -> tuple[StatePlanBundle, ...]:
        if (
            len(image_states) != EXPECTED_CLASSES
            or tuple(state.class_id for state in image_states) != tuple(range(1, 21))
            or len({state.image.asset_id for state in image_states}) != 1
        ):
            raise V9B1ContractError("V9B1_IMAGE_STATE_GRID_DRIFT", "image does not have all 20 states")
        image = image_states[0].image
        image_missingness = tuple(
            entry for entry in self.technical_missingness
            if entry.record.image_group_id == image.canonical_image_id
        )
        try:
            executable_states, _ = partition_executable_states(
                image_states, image_missingness
            )
        except ValueError as exc:
            raise V9B1ContractError(
                "V9B1_E4_EXECUTION_PARTITION_INVALID", str(exc)
            ) from exc
        self.backend.prepare_image(image)
        base_session: SessionHandle | None = None
        try:
            a0_results: dict[str, CandidateGenerationResult] = {}
            missing_a0: list[tuple[SemanticState, CacheIdentity]] = []
            for state in executable_states:
                prompt, _ = self._prompts[state.class_id]
                identity = voc_cache_identity(
                    image.item_payload(), prompt, model_spec_id=self.backend.model_spec_id
                )
                result = _load_a0(self.a0_cache, identity)
                if result is None:
                    missing_a0.append((state, identity))
                else:
                    a0_results[state.semantic_state_id] = result
            if missing_a0:
                base_session = self.backend.open_original(image, purpose="A0_A1_A2_BASE")
                for state, identity in missing_a0:
                    prompt, metadata = self._prompts[state.class_id]
                    result = self._run_prompt(base_session, prompt=prompt, metadata=metadata)
                    self.a0_cache.store(identity, result)
                    a0_results[state.semantic_state_id] = self.a0_cache.load(identity)
            if len(a0_results) != len(executable_states):
                raise V9B1ContractError("V9B1_A0_INVENTORY_INCOMPLETE", image.canonical_image_id)
            bundles = self._derive_bundles(executable_states, a0_results)
            _freeze_plan_lock(
                self.plan_lock_root, image, bundles, image_missingness
            )
            if not actions_allowed:
                return bundles

            base_missing = [
                (bundle, plan, identity)
                for bundle in bundles
                for plan, identity in zip(bundle.plans, bundle.action_identities)
                if plan.action_code in {"A1", "A2"} and _load_action(self.action_cache, identity) is None
            ]
            if base_missing and base_session is None:
                base_session = self.backend.open_original(image, purpose="A1_A2_RESUME")
            for bundle, plan, identity in base_missing:
                action_started = time.perf_counter()
                result = self._run_prompt(
                    base_session,
                    prompt=plan.prompt,
                    metadata=action_metadata(plan, action_sequence_index(plan.action_code, plan.class_id)),
                )
                self._store_action(
                    bundle=bundle,
                    plan=plan,
                    identity=identity,
                    result=result,
                    telemetry=self._action_telemetry(
                        bundle=bundle, plan=plan, prompt_calls=1, session=base_session
                    ),
                    action_started=action_started,
                )
            if base_session is not None:
                self.backend.close_session(base_session)
                base_session = None

            a3_missing = [
                (bundle, plan, identity)
                for bundle in bundles
                for plan, identity in zip(bundle.plans, bundle.action_identities)
                if plan.action_code == "A3" and _load_action(self.action_cache, identity) is None
            ]
            if a3_missing:
                session = self.backend.open_flip(image)
                try:
                    for bundle, plan, identity in a3_missing:
                        result = self._run_prompt(
                            session,
                            prompt=plan.prompt,
                            metadata=action_metadata(plan, action_sequence_index("A3", plan.class_id)),
                        )
                        self._store_action(
                            bundle=bundle, plan=plan, identity=identity, result=result,
                            telemetry=self._action_telemetry(
                                bundle=bundle, plan=plan, prompt_calls=1, session=session
                            ),
                        )
                finally:
                    self.backend.close_session(session)

            for bundle in bundles:
                by_code = {
                    plan.action_code: (plan, identity)
                    for plan, identity in zip(bundle.plans, bundle.action_identities)
                }
                for code in ("A4", "A5", "A6"):
                    plan, identity = by_code[code]
                    if _load_action(self.action_cache, identity) is not None:
                        continue
                    if not plan.feasible:
                        self._store_action(
                            bundle=bundle, plan=plan, identity=identity, result=None,
                            telemetry=self._action_telemetry(
                                bundle=bundle, plan=plan, prompt_calls=0, session=None
                            ),
                        )
                        continue
                    if code == "A4":
                        session = self.backend.open_crop(
                            image,
                            tuple(int(value) for value in plan.transform["crop_box_xyxy"]),
                            action_id=plan.action_id,
                        )
                        try:
                            result = self._run_prompt(
                                session,
                                prompt=plan.prompt,
                                metadata=action_metadata(plan, action_sequence_index(code, plan.class_id)),
                            )
                        finally:
                            self.backend.close_session(session)
                        calls = 1
                        extra: dict[str, Any] = {}
                    elif code == "A5":
                        session = self.backend.open_original(image, purpose=f"A5:{plan.action_id}")
                        try:
                            result = self._run_prompt(
                                session,
                                prompt=plan.prompt,
                                metadata=action_metadata(plan, action_sequence_index(code, plan.class_id)),
                            )
                        finally:
                            self.backend.close_session(session)
                        calls = 1
                        extra = {}
                    else:
                        session = self.backend.open_original(image, purpose=f"A6:{plan.action_id}")
                        try:
                            a0_prompt, a0_metadata = self._prompts[plan.class_id]
                            replay = self._run_prompt(
                                session, prompt=a0_prompt, metadata=a0_metadata
                            )
                            replay_sha = assert_a6_replay_closure(bundle.a0_result, replay)
                            result = self._run_prompt(
                                session,
                                prompt=plan.prompt,
                                metadata=action_metadata(plan, action_sequence_index(code, plan.class_id)),
                                upstream_object_id=plan.source_upstream_object_id,
                            )
                            result = filter_a6_source_object(
                                result, int(plan.source_upstream_object_id)
                            )
                        finally:
                            self.backend.close_session(session)
                        calls = 2
                        extra = {"canonical_replay_verified": True, "canonical_replay_sha256": replay_sha}
                    self._store_action(
                        bundle=bundle,
                        plan=plan,
                        identity=identity,
                        result=result,
                        telemetry=self._action_telemetry(
                            bundle=bundle,
                            plan=plan,
                            prompt_calls=calls,
                            session=session,
                            extra=extra,
                        ),
                    )
            return bundles
        finally:
            if base_session is not None:
                self.backend.close_session(base_session)
            self.backend.release_image(image)

    def run(self, states: Sequence[SemanticState]) -> tuple[StatePlanBundle, ...]:
        grouped: dict[str, list[SemanticState]] = {}
        order: list[str] = []
        for state in states:
            key = state.image.asset_id
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(state)
        a0_bundles: list[StatePlanBundle] = []
        for key in order:
            a0_bundles.extend(self.run_image(grouped[key], actions_allowed=False))
        # Global stage barrier: all A0 cache objects and all image plan locks
        # exist before the first A1--A6 prompt can execute.
        validate_a0_stage(
            tuple(a0_bundles), self.a0_cache, self.technical_missingness
        )
        validate_plan_locks(
            tuple(a0_bundles), self.plan_lock_root, self.technical_missingness
        )
        validate_action_resume_prefix(tuple(a0_bundles), self.action_cache)
        completed: list[StatePlanBundle] = []
        for key in order:
            completed.extend(self.run_image(grouped[key], actions_allowed=True))
        return tuple(completed)


def _cache_root(cache: CandidateCache | TrajectoryCache) -> Path:
    return Path.cwd().resolve() / cache.root


def _workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError as exc:
        raise V9B1ContractError("V9B1_OUTPUT_ROOT_ESCAPES_WORKSPACE", str(path)) from exc


def _bounded_cache_keys(root: Path, prefix: str) -> set[str]:
    if not root.exists():
        return set()
    keys: set[str] = set()
    with os.scandir(root) as first_level:
        for directory in first_level:
            if directory.is_symlink() or not directory.is_dir(follow_symlinks=False) or re.fullmatch(r"[0-9a-f]{2}", directory.name) is None:
                raise V9B1ContractError("V9B1_CACHE_LAYOUT_INVALID", directory.path)
            with os.scandir(directory.path) as objects:
                for entry in objects:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise V9B1ContractError("V9B1_CACHE_OBJECT_NOT_REGULAR", entry.path)
                    expected_prefix = f"{prefix}_cache_"
                    if not entry.name.startswith(expected_prefix) or not entry.name.endswith(".json"):
                        raise V9B1ContractError("V9B1_CACHE_OBJECT_NAME_INVALID", entry.name)
                    key = entry.name[:-5]
                    if _CACHE_KEY_RE.fullmatch(key) is None or key in keys:
                        raise V9B1ContractError("V9B1_CACHE_KEY_INVALID", key)
                    keys.add(key)
    return keys


def validate_a0_stage(
    bundles: Sequence[StatePlanBundle],
    a0_cache: CandidateCache,
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> dict[str, Any]:
    expected = {bundle.a0_identity.cache_key: bundle.a0_identity for bundle in bundles}
    if len(expected) != len(bundles):
        raise V9B1ContractError("V9B1_A0_STAGE_KEY_COLLISION", "A0 cache key collision")
    actual = _bounded_cache_keys(_cache_root(a0_cache), "candidate")
    evidence_keys = {entry.cache_key for entry in technical_missingness}
    if len(evidence_keys) != len(technical_missingness):
        raise V9B1ContractError(
            "V9B1_E4_EVIDENCE_KEY_COLLISION", "E4 evidence cache key collision"
        )
    if actual != set(expected) | evidence_keys:
        raise V9B1ContractError(
            "V9B1_A0_STAGE_SET_MISMATCH", "A0 stage is missing records or has extras"
        )
    for identity in expected.values():
        if _load_a0(a0_cache, identity) is None:
            raise V9B1ContractError("V9B1_A0_STAGE_LOAD_MISSING", identity.cache_key)
    return {
        "a0_record_count": len(expected),
        "technical_missingness_count": len(technical_missingness),
        "a0_cache_aggregate": _cache_aggregate(_cache_root(a0_cache), expected),
    }


def validate_action_resume_prefix(
    bundles: Sequence[StatePlanBundle], action_cache: TrajectoryCache
) -> dict[str, int]:
    expected = {
        identity.cache_key: identity
        for bundle in bundles
        for identity in bundle.action_identities
    }
    if len(expected) != len(bundles) * len(ACTION_CODES):
        raise V9B1ContractError(
            "V9B1_ACTION_RESUME_KEY_COLLISION", "action cache key collision"
        )
    actual = _bounded_cache_keys(_cache_root(action_cache), "trajectory")
    if not actual <= set(expected):
        raise V9B1ContractError(
            "V9B1_ACTION_RESUME_EXTRA_KEY", "action resume contains an unexpected key"
        )
    for cache_key in actual:
        action_cache.load(expected[cache_key])
    return {"expected": len(expected), "present": len(actual), "missing": len(expected) - len(actual)}


def _cache_object_path(root: Path, cache_key: str) -> Path:
    digest = cache_key.rsplit("_", 1)[-1]
    return root / digest[:2] / f"{cache_key}.json"


def _cache_aggregate(root: Path, cache_keys: Iterable[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for cache_key in sorted(cache_keys):
        path = _cache_object_path(root, cache_key)
        try:
            metadata = path.lstat()
            encoded = read_stable_regular_bytes(path)
        except OSError as exc:
            raise V9B1ContractError("V9B1_CACHE_OBJECT_UNREADABLE", cache_key) from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise V9B1ContractError("V9B1_CACHE_OBJECT_NOT_SINGLE_LINK_REGULAR", cache_key)
        entries.append({
            "cache_key": cache_key,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        })
    return {
        "objects": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "aggregate_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def observable_cache_snapshot(
    bundles: Sequence[StatePlanBundle],
    a0_cache: CandidateCache,
    action_cache: TrajectoryCache,
) -> dict[str, Any]:
    """Freeze the exact A0/A1/A2 payloads available to the feature graph."""

    entries: list[dict[str, str]] = []
    for bundle in bundles:
        a0 = a0_cache.load(bundle.a0_identity)
        if a0 is None or canonical_json_bytes(a0.to_dict()) != canonical_json_bytes(
            bundle.a0_result.to_dict()
        ):
            raise V9B1ContractError(
                "V9B1_OBSERVABLE_A0_DRIFT", bundle.a0_identity.cache_key,
            )
        entries.append({
            "action": "A0",
            "cache_key": bundle.a0_identity.cache_key,
            "payload_sha256": hashlib.sha256(
                canonical_json_bytes(a0.to_dict())
            ).hexdigest(),
        })
        for plan, identity in zip(bundle.plans, bundle.action_identities):
            if plan.action_code not in {"A1", "A2"}:
                continue
            outcome = action_cache.load(identity)
            if outcome is None:
                raise V9B1ContractError(
                    "V9B1_OBSERVABLE_ACTION_MISSING", identity.cache_key,
                )
            entries.append({
                "action": plan.action_code,
                "cache_key": identity.cache_key,
                "payload_sha256": hashlib.sha256(
                    canonical_json_bytes(outcome.to_dict())
                ).hexdigest(),
            })
    expected_records = len(bundles) * 3
    if len(entries) != expected_records or len(
        {entry["cache_key"] for entry in entries}
    ) != expected_records:
        raise V9B1ContractError(
            "V9B1_OBSERVABLE_CACHE_GRID_DRIFT", str(len(entries)),
        )
    return {
        "records": len(entries),
        "actions": ["A0", "A1", "A2"],
        "aggregate_sha256": hashlib.sha256(
            canonical_json_bytes(entries)
        ).hexdigest(),
    }


def _validate_action_cost_contract(
    expected_actions: dict[str, tuple[ActionCacheIdentity, str]],
    action_cache: TrajectoryCache,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    counts = {
        code: {status: 0 for status in ("result", "no_result", "failure", "infeasible")}
        for code in ACTION_ORDER
    }
    base_costs: list[float] = []
    for identity, code in expected_actions.values():
        outcome = action_cache.load(identity)
        counts[code][outcome.outcome] += 1
        timing_keys = {"wall_seconds", "action_cost_seconds", "serialization_seconds"}
        present = timing_keys & set(outcome.telemetry)
        if code in {"A1", "A2"}:
            if present != timing_keys:
                raise V9B1ContractError("V9B1_BASE_ACTION_COST_MISSING", identity.cache_key)
            try:
                wall = float(outcome.telemetry["wall_seconds"])
                cost = float(outcome.telemetry["action_cost_seconds"])
                serialization = float(outcome.telemetry["serialization_seconds"])
            except (TypeError, ValueError) as exc:
                raise V9B1ContractError(
                    "V9B1_BASE_ACTION_COST_INVALID", identity.cache_key
                ) from exc
            if not (
                math.isfinite(cost)
                and math.isfinite(serialization)
                and math.isfinite(wall)
                and wall == cost
                and cost > 0.0
                and serialization >= 0.0
                and cost >= serialization
            ):
                raise V9B1ContractError("V9B1_BASE_ACTION_COST_INVALID", identity.cache_key)
            base_costs.append(cost)
        elif present:
            raise V9B1ContractError(
                "V9B1_DECISION_ACTION_COST_FORBIDDEN", identity.cache_key
            )
    return counts, {
        "a0_cost_source": "CandidateGenerationResult.runtime_seconds",
        "a1_a2_cost_source": (
            "V6_FINALIZE_OUTCOME_V1_PROMPT_PLUS_OUTCOME_PLUS_"
            "CANONICAL_SERIALIZATION_SHARED_ENCODE_NOT_AMORTIZED"
        ),
        "a1_a2_cost_record_count": len(base_costs),
        "a1_a2_cost_sum_seconds": sum(base_costs),
        "a3_a6_runtime_feature_eligible": False,
    }


def _validate_action_evidence(
    bundles: Sequence[StatePlanBundle],
    action_cache: TrajectoryCache,
    expected_model_identity: dict[str, Any] | None,
) -> None:
    required = {
        "model_spec_id", "model_source_commit", "checkpoint_sha256", "sam_config_hash",
        "panel_lock_id", "action_protocol_sha256", "semantic_state_id",
        "original_asset_id", "original_image_sha256", "action_id",
        "prompt_call_count", "physical_gpu_index", "logical_device", "physical_input",
    }
    observed_model_identity: dict[str, Any] | None = None
    for bundle in bundles:
        for plan, identity in zip(bundle.plans, bundle.action_identities):
            outcome = action_cache.load(identity)
            telemetry = outcome.telemetry
            model_identity = {
                key: telemetry.get(key)
                for key in (
                    "model_spec_id", "model_source_commit", "checkpoint_sha256",
                    "sam_config_hash",
                )
            }
            if observed_model_identity is None:
                observed_model_identity = model_identity
            elif model_identity != observed_model_identity:
                raise V9B1ContractError(
                    "V9B1_ACTION_MODEL_IDENTITY_INCONSISTENT", identity.cache_key
                )
            expected_keys = set(required)
            if plan.action_code in {"A1", "A2"}:
                expected_keys |= {
                    "wall_seconds", "action_cost_seconds", "serialization_seconds"
                }
            if plan.action_code == "A6" and plan.feasible:
                expected_keys |= {
                    "canonical_replay_verified", "canonical_replay_sha256"
                }
            if set(telemetry) != expected_keys:
                raise V9B1ContractError("V9B1_ACTION_EVIDENCE_MISSING", identity.cache_key)
            physical = telemetry["physical_input"]
            physical_valid = physical is None
            if plan.feasible and isinstance(physical, dict) and set(physical) == {
                "physical_input_asset_id", "physical_input_image_sha256",
                "physical_input_state_id", "physical_input_width", "physical_input_height",
            }:
                physical_asset_id = physical["physical_input_asset_id"]
                physical_sha256 = physical["physical_input_image_sha256"]
                physical_width = physical["physical_input_width"]
                physical_height = physical["physical_input_height"]
                physical_valid = (
                    isinstance(physical_asset_id, str)
                    and bool(physical_asset_id)
                    and _SHA256_RE.fullmatch(str(physical_sha256)) is not None
                    and type(physical_width) is int
                    and type(physical_height) is int
                    and physical_width > 0
                    and physical_height > 0
                    and physical["physical_input_state_id"]
                    == stable_id("encoded_state", {
                        "model_spec_id": telemetry["model_spec_id"],
                        "asset_id": physical_asset_id,
                        "image_sha256": physical_sha256,
                        "width": physical_width,
                        "height": physical_height,
                    })
                )
                if plan.action_code in {"A1", "A2", "A5", "A6"}:
                    physical_valid = physical_valid and (
                        physical_asset_id == bundle.state.image.asset_id
                        and physical_sha256 == bundle.state.image.file_sha256
                        and physical_width == bundle.state.image.width
                        and physical_height == bundle.state.image.height
                    )
                elif plan.action_code == "A3":
                    physical_valid = physical_valid and (
                        physical_width == bundle.state.image.width
                        and physical_height == bundle.state.image.height
                    )
                    if telemetry["checkpoint_sha256"] is not None:
                        physical_valid = physical_valid and physical_asset_id == stable_id(
                            "v9b1_physical_input", {
                                "asset_id": bundle.state.image.asset_id,
                                "transform": "horizontal_flip",
                            }
                        )
                elif plan.action_code == "A4":
                    x0, y0, x1, y1 = (
                        int(value) for value in plan.transform["crop_box_xyxy"]
                    )
                    physical_valid = physical_valid and (
                        physical_width == x1 - x0 and physical_height == y1 - y0
                    )
                    if telemetry["checkpoint_sha256"] is not None:
                        physical_valid = physical_valid and physical_asset_id == stable_id(
                            "v9b1_physical_input", {
                                "asset_id": bundle.state.image.asset_id,
                                "transform": "crop",
                                "crop_box_xyxy": (x0, y0, x1, y1),
                                "action_id": plan.action_id,
                            }
                        )
            if not (
                telemetry["model_spec_id"] == identity.model_spec_id
                and telemetry["panel_lock_id"] == PANEL_LOCK_ID
                and telemetry["action_protocol_sha256"] == PROTOCOL_SHA256
                and telemetry["semantic_state_id"] == bundle.state.semantic_state_id
                and telemetry["original_asset_id"] == bundle.state.image.asset_id
                and telemetry["original_image_sha256"] == bundle.state.image.file_sha256
                and telemetry["action_id"] == plan.action_id
                and type(telemetry["prompt_call_count"]) is int
                and telemetry["prompt_call_count"] == (0 if not plan.feasible else (2 if plan.action_code == "A6" else 1))
                and type(telemetry["physical_gpu_index"]) is int
                and re.fullmatch(r"[0-9a-f]{40}", str(telemetry["model_source_commit"]))
                and re.fullmatch(r"[0-9a-f]{64}", str(telemetry["sam_config_hash"]))
                and (
                    telemetry["checkpoint_sha256"] is None
                    or re.fullmatch(r"[0-9a-f]{64}", str(telemetry["checkpoint_sha256"]))
                )
                and physical_valid
                and telemetry["logical_device"]
                == ("cuda:0" if telemetry["checkpoint_sha256"] is not None else "synthetic")
                and (
                    telemetry["checkpoint_sha256"] is None
                    or telemetry["physical_gpu_index"] == bundle.state.shard
                )
                and (
                    plan.action_code != "A6"
                    or not plan.feasible
                    or (
                        telemetry["canonical_replay_verified"] is True
                        and _SHA256_RE.fullmatch(
                            str(telemetry["canonical_replay_sha256"])
                        )
                        is not None
                    )
                )
            ):
                raise V9B1ContractError("V9B1_ACTION_EVIDENCE_DRIFT", identity.cache_key)
    if expected_model_identity is not None and (
        set(expected_model_identity) != {
            "model_spec_id", "model_source_commit", "checkpoint_sha256", "sam_config_hash"
        }
        or observed_model_identity != expected_model_identity
    ):
        raise V9B1ContractError(
            "V9B1_ACTION_MODEL_IDENTITY_DRIFT", str(observed_model_identity)
        )


def _validate_a0_evidence(
    bundles: Sequence[StatePlanBundle], action_cache: TrajectoryCache
) -> None:
    for bundle in bundles:
        a1_identity = next(
            identity
            for plan, identity in zip(bundle.plans, bundle.action_identities)
            if plan.action_code == "A1"
        )
        expected = action_cache.load(a1_identity).telemetry
        for candidate in bundle.a0_result.all_candidates:
            if not (
                candidate.model_spec_id == bundle.a0_identity.model_spec_id
                and candidate.asset_id == bundle.state.image.asset_id
                and candidate.image_sha256 == bundle.state.image.file_sha256
                and candidate.model_source_commit == expected["model_source_commit"]
                and candidate.checkpoint_sha256 == expected["checkpoint_sha256"]
                and candidate.config_hash == expected["sam_config_hash"]
                and candidate.prompt_class_id == bundle.state.class_id
                and candidate.canonical_prompt_text == bundle.state.canonical_prompt
                and candidate.runtime_seconds is not None
                and math.isfinite(candidate.runtime_seconds)
                and candidate.runtime_seconds >= 0.0
                and candidate.physical_gpu_index == expected["physical_gpu_index"]
                and candidate.logical_device == expected["logical_device"]
            ):
                raise V9B1ContractError(
                    "V9B1_A0_EVIDENCE_DRIFT", bundle.a0_identity.cache_key
                )
        result_telemetry = (
            bundle.a0_result.no_result.telemetry
            if bundle.a0_result.no_result is not None
            else bundle.a0_result.failure.telemetry
            if bundle.a0_result.failure is not None
            else None
        )
        if result_telemetry is not None:
            if not (
                result_telemetry.asset_id == bundle.state.image.asset_id
                and result_telemetry.image_sha256 == bundle.state.image.file_sha256
                and result_telemetry.model_source_commit == expected["model_source_commit"]
                and result_telemetry.checkpoint_sha256 == expected["checkpoint_sha256"]
                and result_telemetry.config_hash == expected["sam_config_hash"]
                and result_telemetry.prompt_class_id == bundle.state.class_id
                and result_telemetry.canonical_prompt_text == bundle.state.canonical_prompt
                and math.isfinite(result_telemetry.runtime_seconds)
                and result_telemetry.runtime_seconds >= 0.0
                and result_telemetry.physical_gpu_index == expected["physical_gpu_index"]
                and result_telemetry.logical_device == expected["logical_device"]
            ):
                raise V9B1ContractError(
                    "V9B1_A0_EVIDENCE_DRIFT", bundle.a0_identity.cache_key
                )
        elif expected["checkpoint_sha256"] is not None and not bundle.a0_result.all_candidates:
            raise V9B1ContractError(
                "V9B1_A0_RESULT_TELEMETRY_MISSING", bundle.a0_identity.cache_key
            )
def validate_inventory(
    bundles: Sequence[StatePlanBundle],
    *,
    a0_cache: CandidateCache,
    action_cache: TrajectoryCache,
    plan_lock_root: Path,
    require_full_panel: bool = False,
    expected_model_identity: dict[str, Any] | None = None,
    technical_missingness: Sequence[E4ExecutionEvidence] = (),
) -> dict[str, Any]:
    expected_a0 = {bundle.a0_identity.cache_key: bundle.a0_identity for bundle in bundles}
    expected_actions = {
        identity.cache_key: (identity, plan.action_code)
        for bundle in bundles
        for plan, identity in zip(bundle.plans, bundle.action_identities)
    }
    if len(expected_a0) != len(bundles) or len(expected_actions) != len(bundles) * len(ACTION_CODES):
        raise V9B1ContractError("V9B1_EXPECTED_KEY_COLLISION", "semantic cache key collision")
    actual_a0 = _bounded_cache_keys(_cache_root(a0_cache), "candidate")
    actual_actions = _bounded_cache_keys(_cache_root(action_cache), "trajectory")
    evidence_keys = {entry.cache_key for entry in technical_missingness}
    if len(evidence_keys) != len(technical_missingness):
        raise V9B1ContractError(
            "V9B1_E4_EVIDENCE_KEY_COLLISION", "E4 evidence cache key collision"
        )
    if actual_a0 != set(expected_a0) | evidence_keys:
        raise V9B1ContractError("V9B1_A0_INVENTORY_SET_MISMATCH", "missing or extra A0 cache keys")
    if actual_actions != set(expected_actions):
        raise V9B1ContractError("V9B1_ACTION_INVENTORY_SET_MISMATCH", "missing or extra action cache keys")
    counts, cost_contract = _validate_action_cost_contract(expected_actions, action_cache)
    _validate_action_evidence(bundles, action_cache, expected_model_identity)
    _validate_a0_evidence(bundles, action_cache)
    for identity in expected_a0.values():
        counts["A0"][_result_status(a0_cache.load(identity))] += 1
    plan_inventory = validate_plan_locks(
        bundles, plan_lock_root, technical_missingness
    )
    trajectory_records = len(expected_a0) + len(expected_actions)
    panel_state_count = len(expected_a0) + len(technical_missingness)
    semantic_records = panel_state_count * len(ACTION_ORDER)
    failure_count = sum(values["failure"] for values in counts.values())
    if require_full_panel and semantic_records != EXPECTED_SEMANTIC_RECORDS:
        raise V9B1ContractError("V9B1_SEMANTIC_INVENTORY_COUNT_DRIFT", str(semantic_records))
    a0_inventory = _cache_aggregate(_cache_root(a0_cache), expected_a0)
    action_inventory = _cache_aggregate(_cache_root(action_cache), expected_actions)
    observable_inventory = observable_cache_snapshot(
        bundles, a0_cache, action_cache,
    )
    return {
        "schema_version": INVENTORY_SCHEMA,
        "panel_lock_id": PANEL_LOCK_ID,
        "state_count": panel_state_count,
        "trajectory_state_count": len(bundles),
        "semantic_record_count": semantic_records,
        "trajectory_record_count": trajectory_records,
        "expected_semantic_record_count": (
            EXPECTED_SEMANTIC_RECORDS
            if require_full_panel
            else panel_state_count * len(ACTION_ORDER)
        ),
        "counts_by_action_and_outcome": counts,
        "failure_count": failure_count,
        "technical_missingness": {
            "count": len(technical_missingness),
            "semantic_state_ids": sorted(
                entry.record.semantic_state_id for entry in technical_missingness
            ),
            "scientific_outcome_imputed": False,
        },
        "ready": failure_count == 0,
        "cache_roots": {
            "a0": _workspace_relative(_cache_root(a0_cache)),
            "actions": _workspace_relative(_cache_root(action_cache)),
            "plans": _workspace_relative(plan_lock_root),
        },
        "a0_cache": a0_inventory,
        "action_cache": action_inventory,
        "observable_cache_snapshot": observable_inventory,
        "plan_locks": plan_inventory,
        "cache_bytes_total": (
            a0_inventory["bytes"]
            + action_inventory["bytes"]
            + plan_inventory["plan_lock_bytes"]
        ),
        "prospective_cost_contract": cost_contract,
        "coco_per_image_gt_materialized": 0,
        "coco_annotation_rows_materialized": 0,
        "coco_image_label_rows_materialized": 0,
        "coco_public_schema_additional_reads": 0,
        "model_predictions": 0,
        "training_jobs": 0,
        "performance_results": 0,
    }


def validate_run_telemetry_receipt(
    payload: dict[str, Any],
    *,
    shard: int,
    gpu_uuid: str,
    model_identity: dict[str, Any],
) -> None:
    counters = payload.get("backend_counters", {})
    base_keys = {
        "schema_version", "shard", "physical_gpu", "gpu_uuid",
        "backend_counters", "sam_identity", "real_backend", "wall_seconds",
        "coco_annotation_rows_materialized", "coco_image_label_rows_materialized",
        "model_predictions", "performance_results",
    }
    missingness_keys = {
        "technical_missingness_count", "technical_missingness_semantic_state_ids"
    }
    payload_keys = set(payload)
    has_missingness = payload_keys == base_keys | missingness_keys
    if not (
        payload_keys in (base_keys, base_keys | missingness_keys)
        and payload.get("schema_version")
        == "rail3.tmlr-v9b1-label-free-run-telemetry.v1"
        and payload.get("shard") == shard
        and payload.get("physical_gpu") == shard
        and payload.get("gpu_uuid") == gpu_uuid
        and payload.get("sam_identity") == model_identity
        and payload.get("real_backend") is True
        and type(payload.get("wall_seconds")) in {int, float}
        and math.isfinite(payload["wall_seconds"])
        and payload["wall_seconds"] > 0.0
        and set(counters) == {
            "model_builds", "checkpoint_loads", "session_initializations", "prompt_calls"
        }
        and counters.get("model_builds") == 1
        and counters.get("checkpoint_loads") == 1
        and isinstance(counters.get("session_initializations"), int)
        and counters["session_initializations"] > 0
        and isinstance(counters.get("prompt_calls"), int)
        and counters["prompt_calls"] > 0
        and all(payload.get(key) == 0 for key in (
            "coco_annotation_rows_materialized", "coco_image_label_rows_materialized",
            "model_predictions", "performance_results",
        ))
        and (
            not has_missingness
            or (
                isinstance(payload.get("technical_missingness_count"), int)
                and payload["technical_missingness_count"] >= 0
                and isinstance(payload.get("technical_missingness_semantic_state_ids"), list)
                and len(set(payload["technical_missingness_semantic_state_ids"]))
                == payload["technical_missingness_count"]
            )
        )
    ):
        raise V9B1ContractError("V9B1_RUN_TELEMETRY_RECEIPT_DRIFT", str(shard))


def merge_expected_action_caches(
    source_caches: Sequence[TrajectoryCache],
    destination: TrajectoryCache,
    expected: Sequence[ActionCacheIdentity],
) -> dict[str, int]:
    expected_by_key = {identity.cache_key: identity for identity in expected}
    if len(expected_by_key) != len(expected):
        raise V9B1ContractError("V9B1_MERGE_EXPECTED_KEY_COLLISION", "action cache keys collide")
    source_sets = [_bounded_cache_keys(_cache_root(cache), "trajectory") for cache in source_caches]
    union: set[str] = set()
    for keys in source_sets:
        if union & keys:
            raise V9B1ContractError("V9B1_MERGE_DUPLICATE_SOURCE_KEY", "shard caches overlap")
        union |= keys
    if union != set(expected_by_key):
        raise V9B1ContractError("V9B1_MERGE_SOURCE_SET_MISMATCH", "shard action sets are not exhaustive")
    for key in sorted(expected_by_key):
        source_indexes = [index for index, keys in enumerate(source_sets) if key in keys]
        if len(source_indexes) != 1:
            raise V9B1ContractError("V9B1_MERGE_SOURCE_OWNERSHIP_INVALID", key)
        identity = expected_by_key[key]
        outcome = source_caches[source_indexes[0]].load(identity)
        destination.store(identity, outcome)
        destination.load(identity)
    if _bounded_cache_keys(_cache_root(destination), "trajectory") != set(expected_by_key):
        raise V9B1ContractError("V9B1_MERGE_DESTINATION_SET_MISMATCH", "merged action set drift")
    return {"expected": len(expected_by_key), "merged": len(expected_by_key), "duplicates": 0}


def merge_expected_a0_caches(
    source_caches: Sequence[CandidateCache],
    destination: CandidateCache,
    expected: Sequence[CacheIdentity],
) -> dict[str, int]:
    expected_by_key = {identity.cache_key: identity for identity in expected}
    if len(expected_by_key) != len(expected):
        raise V9B1ContractError("V9B1_MERGE_EXPECTED_KEY_COLLISION", "A0 cache keys collide")
    source_sets = [_bounded_cache_keys(_cache_root(cache), "candidate") for cache in source_caches]
    union: set[str] = set()
    for keys in source_sets:
        if union & keys:
            raise V9B1ContractError("V9B1_MERGE_DUPLICATE_SOURCE_KEY", "shard A0 caches overlap")
        union |= keys
    if union != set(expected_by_key):
        raise V9B1ContractError("V9B1_MERGE_SOURCE_SET_MISMATCH", "shard A0 sets are not exhaustive")
    for key in sorted(expected_by_key):
        source_indexes = [index for index, keys in enumerate(source_sets) if key in keys]
        identity = expected_by_key[key]
        result = source_caches[source_indexes[0]].load(identity)
        destination.store(identity, result)
        destination.load(identity)
    if _bounded_cache_keys(_cache_root(destination), "candidate") != set(expected_by_key):
        raise V9B1ContractError("V9B1_MERGE_DESTINATION_SET_MISMATCH", "merged A0 set drift")
    return {"expected": len(expected_by_key), "merged": len(expected_by_key), "duplicates": 0}


def bundles_action_identities(bundles: Iterable[StatePlanBundle]) -> tuple[ActionCacheIdentity, ...]:
    return tuple(identity for bundle in bundles for identity in bundle.action_identities)


def bundles_a0_identities(bundles: Iterable[StatePlanBundle]) -> tuple[CacheIdentity, ...]:
    return tuple(bundle.a0_identity for bundle in bundles)
