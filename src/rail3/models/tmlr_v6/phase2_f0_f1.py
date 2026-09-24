"""Cross-fitting and immutable audit for the 120 registered F0/F1 jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.m06e.models import SharedAtomicResidual, parameter_count
from rail3.models.tmlr_v6.models import ProspectiveSharedAtomicResidual
from rail3.models.tmlr_v6.phase2_cost_profiles import profiled_cost_vector
from rail3.models.tmlr_v6.phase2_io import (
    A3_PHYSICAL_INDEX,
    FILTRATIONS,
    FOLDS,
    PHASE2_F0_F1_OUTPUT_ROOT,
    PHASE2_VIEW_REGISTRY_PATH,
    PHYSICAL_ACTION_ORDER,
    REMAINING_ACTIONS,
    REMAINING_PHYSICAL_INDICES,
    SCALES,
    SEEDS,
    ZERO_EXTERNAL_ACCESS,
    Phase2View,
    _assert_single_link_regular,
    load_phase2_addendum,
    load_f0_f1_view_registry,
    load_registered_f0_f1_view,
    split_groups,
    view_registry_row,
)
from rail3.models.tmlr_v6.phase2_training import (
    FROZEN_TRAINING_RECIPE,
    _Phase2GpuLease,
    _assert_active_phase2_gpu_lease,
    _registered_visible_device,
    exclusive_phase2_gpu_lock,
)
from rail3.models.tmlr_v6.phase2_scaling import (
    remaining_scaler_payload,
    scale_remaining_predictor_features,
)
from rail3.models.tmlr_v6.schema import ACTION_DIM, ATOM_DIM, STATE_DIM
from rail3.models.tmlr_v6.training import (
    MEMORY_BOUNDED_ATOM_CHUNK,
    MEMORY_BOUNDED_ATOM_THRESHOLD,
    PROTOCOL_PATH,
    RUN_SECONDS_LIMIT,
    VRAM_LIMIT,
    _assert_no_stale_run_staging,
    _atomic_checkpoint,
    _atomic_json,
    _atomic_npz,
    _device,
    _initialize_cuda_telemetry,
    _logical_path,
    _prepare_run_staging,
    _publish_run_staging,
    _repository_path,
    _run_staging_path,
    _seed,
    _snapshot,
    _state_huber,
    _tensor,
    assert_cuda_determinism_environment,
    assert_zero_external_access,
    cuda_determinism_environment_contract,
    inference_weights_from_features,
    load_protocol,
    memory_bounded_shared_atomic_forward,
    sha256_file,
)


RUN_SCHEMA = "rail3.tmlr-v6.phase2-f0-f1-run.v1"
RUN_STATUS = "PASS"
RUN_PHASE = "fit_phase2_f0_f1_outer_crossfit"
INVENTORY_SCHEMA = "rail3.tmlr-v6.phase2-f0-f1-run-inventory.v1"
INVENTORY_STATUS = "TMLR_V6_PHASE2_F0_F1_RUN_INVENTORY_PASS"
F0_F1_INVENTORY_PATH = Path(
    "artifacts/audits/tmlr_v6/phase2_f0_f1_run_inventory.json"
)
PREDICTION_ARRAY_NAMES = frozenset({
    "state_index", "state_id", "state_prediction",
})


def _single_link_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
PARAMETER_COUNT = 103_521
F0_F1_STATIC_RUN_FIELDS = frozenset({
    "schema_version",
    "status",
    "phase",
    "run_id",
    "model_id",
    "filtration_id",
    "filtration_if_applicable",
    "lambda_if_model_specific",
    "registered_view_id",
    "scale",
    "outer_fold",
    "seed",
    "parent_protocol_sha256",
    "semantic_addendum_sha256",
    "launch_lock_sha256",
    "implementation_commit",
    "implementation_combined_sha256",
    "implementation_source_sha256",
    "feature_manifest_sha256",
    "feature_array_sha256",
    "target_manifest_sha256",
    "target_array_sha256",
    "fold_manifest_sha256",
    "cost_profile_registry_sha256",
    "view_registry",
    "cuda_determinism_environment",
    "feature_schema_id",
    "input_dimensions",
    "architecture",
    "parameter_count",
    "physical_action_order",
    "remaining_physical_indices",
    "remaining_actions",
    "a3_mask_both_sides",
    "a3_scaled_row_all_zero",
    "action_scaler_fit_rows",
    "action_scaler_transform_rows",
    "state_scaler_population",
    "atom_scaler_population",
    "primary_training_loss",
    "inner_stopping_loss",
    "atom_target_auxiliary_loss",
    "predictor_cost_input",
    "future_A4_A6_outcome_or_runtime_input",
    "heldout_state_specific_runtime_input",
    "post_action_diagnostics_opened",
    "inference_weight_source",
    "target_weight_source",
    "target_weight_in_prediction_graph",
    "train_group_ids",
    "inner_stop_group_ids",
    "outer_heldout_fit_group_ids",
    "counts",
    "excluded_state_ids_sha256",
    "heldout_access",
    "fit_only",
    "sam_inference",
    "prompt_or_trajectory_generation",
    "protected_reader_invoked",
    "ddp",
})
F0_F1_DYNAMIC_RUN_FIELDS = frozenset({
    "architecture_class",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "patience",
    "minimum_inner_stop_improvement",
    "best_epoch",
    "epochs_completed",
    "inner_stop_loss",
    "execution_memory_mode",
    "started_at_utc",
    "ended_at_utc",
    "wall_seconds",
    "gpu_wall_seconds",
    "peak_vram_bytes",
    "visible_device",
    "prediction",
    "checkpoint",
})
F0_F1_RUN_FIELDS = F0_F1_STATIC_RUN_FIELDS | F0_F1_DYNAMIC_RUN_FIELDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_logical_path_not_symlinked(
    repository: Path, logical_path: Path, *, description: str,
) -> None:
    """Reject a logical path and any in-repository symlink ancestor."""

    root = repository.resolve()
    requested = logical_path if logical_path.is_absolute() else root / logical_path
    if requested.is_symlink() or any(
        parent.is_symlink() for parent in requested.parents
        if parent == root or root in parent.parents
    ):
        raise RuntimeError(f"TMLR V6 F0/F1 {description} is symlinked")


def _frozen_recipe(addendum: Mapping[str, Any]) -> dict[str, Any]:
    recipe = addendum.get("training_recipe", {})
    observed = {
        key: recipe.get(key) if isinstance(recipe, Mapping) else None
        for key in FROZEN_TRAINING_RECIPE
    }
    if observed != FROZEN_TRAINING_RECIPE:
        raise RuntimeError("TMLR V6 F0/F1 frozen training recipe drift")
    return observed


def _utc_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"TMLR V6 F0/F1 {field} is not UTC")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"TMLR V6 F0/F1 {field} is invalid") from error
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise RuntimeError(f"TMLR V6 F0/F1 {field} is not UTC")
    return result


def expected_f0_f1_job_keys() -> set[tuple[str, str, str, int]]:
    """The exact scale/filtration/fold/seed product (120 jobs)."""

    result = {
        (scale, filtration_id, outer_fold, seed)
        for scale in SCALES
        for filtration_id in FILTRATIONS
        for outer_fold in FOLDS
        for seed in SEEDS
    }
    if len(result) != 120:
        raise RuntimeError("TMLR V6 F0/F1 job matrix is not exactly 120")
    return result


def _validate_exact_f0_f1_tree(
    root: Path, expected_dirs: set[Path],
) -> None:
    """Accept only the exact run directories and single-link regular files."""

    expected_files = {
        directory / name
        for directory in expected_dirs
        for name in ("run-manifest.json", "predictions.npz", "checkpoint.pt")
    }
    observed_files: set[Path] = set()
    observed_dirs: set[Path] = {root}
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            observed_dirs.add(path)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            observed_files.add(path)
        else:
            raise RuntimeError("TMLR V6 F0/F1 run tree has a non-regular entry")
    expected_all_dirs = {root}
    for directory in expected_dirs:
        current = directory
        while current != root:
            expected_all_dirs.add(current)
            current = current.parent
    if observed_files != expected_files or observed_dirs != expected_all_dirs:
        raise RuntimeError("TMLR V6 F0/F1 run tree contains an extra/missing file")


def build_f0_f1_model() -> nn.Module:
    model = ProspectiveSharedAtomicResidual(ATOM_DIM, STATE_DIM, ACTION_DIM)
    if parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("TMLR V6 F0/F1 R1_P parameter-count drift")
    return model


def _load_launch_contract(repository: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use the shared phase-2 validator; never duplicate lock semantics here."""

    from rail3.models.tmlr_v6.phase2_contract import (  # local: avoids a cycle
        PHASE2_LAUNCH_LOCK_PATH,
        load_scale_cost_profile_registry,
        validate_phase2_launch_lock,
    )

    _assert_logical_path_not_symlinked(
        repository,
        PHASE2_LAUNCH_LOCK_PATH,
        description="launch-lock logical path",
    )
    validated = validate_phase2_launch_lock(repository)
    costs = load_scale_cost_profile_registry(repository, validated)
    lock = dict(validated.payload)
    lock_path = _repository_path(repository, PHASE2_LAUNCH_LOCK_PATH)
    if validated.path != lock_path or validated.sha256 != sha256_file(lock_path):
        raise RuntimeError("TMLR V6 phase-2 validated launch-lock identity drift")
    return lock, costs, validated.sha256


def _view_registry_identity(repository: Path) -> dict[str, Any]:
    _assert_logical_path_not_symlinked(
        repository,
        PHASE2_VIEW_REGISTRY_PATH,
        description="view-registry logical path",
    )
    path = _repository_path(repository, PHASE2_VIEW_REGISTRY_PATH)
    _assert_single_link_regular(path, description="view registry")
    return {
        "path": PHASE2_VIEW_REGISTRY_PATH.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _implementation_fields(lock: Mapping[str, Any]) -> tuple[str, str, dict[str, str]]:
    value = lock.get("phase2_implementation", {})
    commit = str(value.get("commit", ""))
    combined = str(value.get("combined_source_sha256", ""))
    sources = value.get("source_sha256", {})
    if len(commit) != 40 or len(combined) != 64 or not isinstance(sources, Mapping):
        raise RuntimeError("TMLR V6 phase-2 implementation identity drift")
    return commit, combined, {str(key): str(item) for key, item in sources.items()}


def _artifact_identity(repository: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _logical_path(repository, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run_dir(
    repository: Path, *, scale: str, filtration_id: str, outer_fold: str,
    seed: int, output_root: Path = PHASE2_F0_F1_OUTPUT_ROOT,
) -> Path:
    if (scale, filtration_id, outer_fold, seed) not in expected_f0_f1_job_keys():
        raise ValueError("unregistered TMLR V6 F0/F1 job")
    _assert_logical_path_not_symlinked(
        repository,
        PHASE2_F0_F1_OUTPUT_ROOT,
        description="registered output root",
    )
    _assert_logical_path_not_symlinked(
        repository,
        output_root,
        description="requested output root",
    )
    if _repository_path(repository, output_root) != _repository_path(
        repository, PHASE2_F0_F1_OUTPUT_ROOT,
    ):
        raise RuntimeError("TMLR V6 F0/F1 output root is not registered")
    return _repository_path(
        repository,
        output_root / scale / filtration_id / outer_fold / f"seed_{seed}",
    )


def _load_existing_f0_f1_report(run_dir: Path) -> dict[str, Any] | None:
    """Preflight an existing run tree before opening its manifest."""

    if not run_dir.exists() and not run_dir.is_symlink():
        return None
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("incomplete TMLR V6 F0/F1 run exists")
    entries = list(run_dir.iterdir())
    if (
        {item.name for item in entries}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(not _single_link_regular(item) for item in entries)
    ):
        raise RuntimeError("incomplete TMLR V6 F0/F1 run exists")
    report = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise RuntimeError("TMLR V6 F0/F1 completed manifest is not an object")
    return dict(report)


def _expected_manifest(
    *, repository: Path, view: Phase2View, lock: Mapping[str, Any],
    launch_lock_sha256: str, view_registry_identity: Mapping[str, Any],
    scale: str, filtration_id: str, outer_fold: str, seed: int,
    train_groups: set[str], stop_groups: set[str], heldout_groups: set[str],
    train_mask: np.ndarray, stop_mask: np.ndarray, heldout_mask: np.ndarray,
) -> dict[str, Any]:
    implementation_commit, implementation_sha, source_sha = _implementation_fields(lock)
    parent = lock["parent_protocol"]
    addendum = lock["phase2_semantic_addendum"]
    fold = lock["fold_manifests"][scale]
    cost_json = lock["phase2_scale_cost_profiles"]["json"]
    view_row = view_registry_row(view)
    run_id = stable_id("tmlr_v6_phase2_f0_f1_run", {
        "parent_protocol_sha256": parent["sha256"],
        "semantic_addendum_sha256": addendum["sha256"],
        "launch_lock_sha256": launch_lock_sha256,
        "implementation_combined_sha256": implementation_sha,
        "feature_manifest_sha256": view.bundle.feature_manifest_sha256,
        "feature_array_sha256": view.bundle.feature_array_sha256,
        "target_manifest_sha256": view.bundle.target_manifest_sha256,
        "target_array_sha256": view.bundle.target_array_sha256,
        "fold_manifest_sha256": fold["sha256"],
        "cost_profile_registry_sha256": cost_json["sha256"],
        "view_registry_sha256": view_registry_identity["sha256"],
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "registered_view_id": view.view_id,
        "model_id": filtration_id,
        "scale": scale,
        "filtration_if_applicable": filtration_id,
        "lambda_if_model_specific": None,
        "outer_fold": outer_fold,
        "seed": seed,
    })
    state_target = np.asarray(view.targets["state_target"])
    feasible = np.asarray(view.targets["feasible"])
    train_defined = np.isfinite(state_target) & feasible & train_mask[:, None]
    incomplete = ~view.complete_case
    state_ids = np.asarray(view.features["state_ids"]).astype(str)
    return {
        "schema_version": RUN_SCHEMA,
        "status": RUN_STATUS,
        "phase": RUN_PHASE,
        "run_id": run_id,
        "model_id": filtration_id,
        "filtration_id": filtration_id,
        "filtration_if_applicable": filtration_id,
        "lambda_if_model_specific": None,
        "registered_view_id": view.view_id,
        "scale": scale,
        "outer_fold": outer_fold,
        "seed": seed,
        "parent_protocol_sha256": parent["sha256"],
        "semantic_addendum_sha256": addendum["sha256"],
        "launch_lock_sha256": launch_lock_sha256,
        "implementation_commit": implementation_commit,
        "implementation_combined_sha256": implementation_sha,
        "implementation_source_sha256": source_sha,
        "feature_manifest_sha256": view.bundle.feature_manifest_sha256,
        "feature_array_sha256": view.bundle.feature_array_sha256,
        "target_manifest_sha256": view.bundle.target_manifest_sha256,
        "target_array_sha256": view.bundle.target_array_sha256,
        "fold_manifest_sha256": fold["sha256"],
        "cost_profile_registry_sha256": cost_json["sha256"],
        "view_registry": dict(view_registry_identity),
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "feature_schema_id": "PROSPECTIVE_V1",
        "input_dimensions": {"atom": 27, "state": 41, "action_projected": 16},
        "architecture": "R1_P",
        "parameter_count": PARAMETER_COUNT,
        "physical_action_order": list(PHYSICAL_ACTION_ORDER),
        "remaining_physical_indices": list(REMAINING_PHYSICAL_INDICES),
        "remaining_actions": list(REMAINING_ACTIONS),
        "a3_mask_both_sides": True,
        "a3_scaled_row_all_zero": True,
        "action_scaler_fit_rows": list(REMAINING_PHYSICAL_INDICES),
        "action_scaler_transform_rows": list(REMAINING_PHYSICAL_INDICES),
        "state_scaler_population": "scale_outer_training_states",
        "atom_scaler_population": "atoms_of_scale_outer_training_states",
        "primary_training_loss": (
            "state-level Huber(delta=0.05) over remaining defined actions only"
        ),
        "inner_stopping_loss": (
            "state-level Huber(delta=0.05) over remaining defined actions only"
        ),
        "atom_target_auxiliary_loss": False,
        "predictor_cost_input": False,
        "future_A4_A6_outcome_or_runtime_input": False,
        "heldout_state_specific_runtime_input": False,
        "post_action_diagnostics_opened": False,
        "inference_weight_source": "feature_area_fraction_full_raster",
        "target_weight_source": "valid_pixel_fraction_target_only",
        "target_weight_in_prediction_graph": False,
        "train_group_ids": sorted(train_groups),
        "inner_stop_group_ids": sorted(stop_groups),
        "outer_heldout_fit_group_ids": sorted(heldout_groups),
        "counts": {
            "inner_train_states": int(train_mask.sum()),
            "inner_stop_states": int(stop_mask.sum()),
            "outer_heldout_fit_states": int(heldout_mask.sum()),
            "view_states": len(state_ids),
            "view_atoms": len(view.features["atom_ids"]),
            "objective_supervised_states": int(np.any(train_defined, axis=1).sum()),
            "objective_supervised_state_actions": int(train_defined.sum()),
            "formal_complete_case_states": int(view.complete_case.sum()),
            "plan_feasible_target_undefined_excluded_states": int(incomplete.sum()),
        },
        "excluded_state_ids_sha256": _canonical_sha256(
            sorted(state_ids[incomplete].tolist())
        ),
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "fit_only": True,
        "sam_inference": False,
        "prompt_or_trajectory_generation": False,
        "protected_reader_invoked": False,
        "ddp": False,
    }


def validate_f0_f1_run_manifest(
    repository: Path, report: Mapping[str, Any], *, run_dir: Path,
    view: Phase2View | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one immutable run, exact predictions, and heldout coverage."""

    _assert_logical_path_not_symlinked(
        repository, run_dir, description="run directory logical path",
    )
    repository = repository.resolve()
    if set(report) != F0_F1_RUN_FIELDS:
        raise RuntimeError("TMLR V6 F0/F1 completed manifest schema drift")
    if expected_identity is not None:
        identity_drift = {
            key: (report.get(key), value)
            for key, value in expected_identity.items()
            if report.get(key) != value
        }
        if identity_drift:
            raise RuntimeError(
                f"TMLR V6 F0/F1 completed run frozen identity drift: {identity_drift}"
            )
    required = {
        "schema_version": RUN_SCHEMA,
        "status": RUN_STATUS,
        "phase": RUN_PHASE,
        "physical_action_order": list(PHYSICAL_ACTION_ORDER),
        "remaining_physical_indices": list(REMAINING_PHYSICAL_INDICES),
        "remaining_actions": list(REMAINING_ACTIONS),
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "feature_schema_id": "PROSPECTIVE_V1",
        "input_dimensions": {"atom": 27, "state": 41, "action_projected": 16},
        "architecture": "R1_P",
        "parameter_count": PARAMETER_COUNT,
        "a3_mask_both_sides": True,
        "a3_scaled_row_all_zero": True,
        "action_scaler_fit_rows": list(REMAINING_PHYSICAL_INDICES),
        "action_scaler_transform_rows": list(REMAINING_PHYSICAL_INDICES),
        "state_scaler_population": "scale_outer_training_states",
        "atom_scaler_population": "atoms_of_scale_outer_training_states",
        "primary_training_loss": (
            "state-level Huber(delta=0.05) over remaining defined actions only"
        ),
        "inner_stopping_loss": (
            "state-level Huber(delta=0.05) over remaining defined actions only"
        ),
        "atom_target_auxiliary_loss": False,
        "predictor_cost_input": False,
        "future_A4_A6_outcome_or_runtime_input": False,
        "heldout_state_specific_runtime_input": False,
        "post_action_diagnostics_opened": False,
        "inference_weight_source": "feature_area_fraction_full_raster",
        "target_weight_source": "valid_pixel_fraction_target_only",
        "target_weight_in_prediction_graph": False,
        "fit_only": True,
        "sam_inference": False,
        "prompt_or_trajectory_generation": False,
        "protected_reader_invoked": False,
        "ddp": False,
    }
    drift = {key: (report.get(key), value) for key, value in required.items() if report.get(key) != value}
    if drift:
        raise RuntimeError(f"TMLR V6 F0/F1 completed run contract drift: {drift}")
    try:
        scale = str(report.get("scale", ""))
        filtration_id = str(report.get("filtration_id", ""))
        outer_fold = str(report.get("outer_fold", ""))
        seed = int(report.get("seed", -1))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("TMLR V6 F0/F1 completed run job identity drift") from error
    if (
        report.get("model_id") != filtration_id
        or report.get("filtration_if_applicable") != filtration_id
        or report.get("lambda_if_model_specific") is not None
        or (scale, filtration_id, outer_fold, seed) not in expected_f0_f1_job_keys()
        or report.get("registered_view_id")
        != f"{filtration_id}_{scale}_REMAINING_A3_MASKED"
    ):
        raise RuntimeError("TMLR V6 F0/F1 completed run job identity drift")
    protocol = load_protocol(repository)
    expected_visible_device = _registered_visible_device(protocol, outer_fold)
    assert_zero_external_access(report.get("heldout_access", {}))
    if (
        run_dir.is_symlink() or not run_dir.is_dir()
        or {item.name for item in run_dir.iterdir()}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(not _single_link_regular(item) for item in run_dir.iterdir())
    ):
        raise RuntimeError("TMLR V6 F0/F1 run file set drift")
    prediction_path = run_dir / "predictions.npz"
    checkpoint_path = run_dir / "checkpoint.pt"
    for key, path in (("prediction", prediction_path), ("checkpoint", checkpoint_path)):
        identity = report.get(key, {})
        if (
            _repository_path(repository, Path(str(identity.get("path", "")))) != path
            or path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(path) != str(identity.get("sha256", ""))
        ):
            raise RuntimeError(f"TMLR V6 F0/F1 {key} identity drift")
    with np.load(prediction_path, allow_pickle=False) as payload:
        if set(payload.files) != PREDICTION_ARRAY_NAMES:
            raise RuntimeError("TMLR V6 F0/F1 prediction schema drift")
        indices = np.asarray(payload["state_index"])
        state_ids = np.asarray(payload["state_id"])
        prediction = np.asarray(payload["state_prediction"])
    if (
        indices.ndim != 1 or indices.dtype != np.dtype(np.int32)
        or state_ids.ndim != 1 or state_ids.dtype.kind not in {"U", "S"}
        or prediction.shape != (len(indices), 5)
        or prediction.dtype != np.dtype(np.float32)
        or len(state_ids) != len(indices)
        or len(set(indices.tolist())) != len(indices)
        or np.any(np.isfinite(prediction[:, A3_PHYSICAL_INDEX]))
        or not np.isfinite(prediction[:, REMAINING_PHYSICAL_INDICES]).all()
    ):
        raise RuntimeError("TMLR V6 F0/F1 prediction value contract drift")
    if view is not None:
        if view.scale != scale or view.filtration_id != filtration_id:
            raise RuntimeError("TMLR V6 F0/F1 audit view/job drift")
        groups = np.asarray(view.features["image_group_ids"]).astype(str)
        heldout = set(str(value) for value in report["outer_heldout_fit_group_ids"])
        local = np.flatnonzero(np.isin(groups, tuple(heldout)))
        expected_indices = view.global_state_indices[local].astype(np.int32)
        expected_ids = np.asarray(view.features["state_ids"])[local].astype(str)
        if not np.array_equal(indices, expected_indices) or not np.array_equal(
            state_ids.astype(str), expected_ids,
        ):
            raise RuntimeError("TMLR V6 F0/F1 prediction heldout coverage drift")
    if {
        key: report.get(key) for key in FROZEN_TRAINING_RECIPE
    } != FROZEN_TRAINING_RECIPE:
        raise RuntimeError("TMLR V6 F0/F1 frozen optimizer/recipe drift")
    if report.get("architecture_class") != (
        "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual"
    ) or report.get("execution_memory_mode") not in {
        "FULL_BATCH",
        f"ACTIVATION_CHECKPOINTED_ATOM_CHUNKS_{MEMORY_BOUNDED_ATOM_CHUNK}",
    }:
        raise RuntimeError("TMLR V6 F0/F1 frozen architecture/execution drift")
    try:
        best_epoch = int(report["best_epoch"])
        epochs_completed = int(report["epochs_completed"])
        inner_stop_loss = float(report["inner_stop_loss"])
        wall_seconds = float(report["wall_seconds"])
        gpu_wall_seconds = float(report["gpu_wall_seconds"])
        peak_vram_bytes = int(report["peak_vram_bytes"])
        started_at = _utc_timestamp(report["started_at_utc"], field="started_at_utc")
        ended_at = _utc_timestamp(report["ended_at_utc"], field="ended_at_utc")
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("TMLR V6 F0/F1 execution observation is invalid") from error
    if (
        isinstance(report["best_epoch"], bool)
        or isinstance(report["epochs_completed"], bool)
        or not 1 <= best_epoch <= epochs_completed
        <= FROZEN_TRAINING_RECIPE["max_epochs"]
        or not math.isfinite(inner_stop_loss)
        or not math.isfinite(wall_seconds)
        or not 0.0 <= wall_seconds <= RUN_SECONDS_LIMIT
        or not math.isfinite(gpu_wall_seconds)
        or gpu_wall_seconds != wall_seconds
        or not 0 <= peak_vram_bytes < VRAM_LIMIT
        or ended_at < started_at
        or report["visible_device"] != expected_visible_device
    ):
        raise RuntimeError("TMLR V6 F0/F1 run resource contract drift")
    return dict(report)


def run_f0_f1_crossfit(
    *, scale: str, filtration_id: str, outer_fold: str, seed: int,
    repository: Path, output_root: Path = PHASE2_F0_F1_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate one job, then train/resume it under its mandatory GPU lease."""

    assert_cuda_determinism_environment()
    repository = repository.resolve()
    if (scale, filtration_id, outer_fold, seed) not in expected_f0_f1_job_keys():
        raise ValueError("unregistered TMLR V6 F0/F1 job")
    run_dir = _run_dir(
        repository,
        scale=scale,
        filtration_id=filtration_id,
        outer_fold=outer_fold,
        seed=seed,
        output_root=output_root,
    )
    if not _single_link_regular(_repository_path(repository, PROTOCOL_PATH)):
        raise RuntimeError(
            "TMLR V6 F0/F1 protocol must be one regular single-link file"
        )
    protocol = load_protocol(repository)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_device != _registered_visible_device(protocol, outer_fold):
        raise RuntimeError(
            "TMLR V6 F0/F1 worker violates physical-GPU assignment"
        )
    lock, costs, lock_sha = _load_launch_contract(repository)
    recipe = _frozen_recipe(load_phase2_addendum(repository))
    with exclusive_phase2_gpu_lock(
        repository, str(visible_device),
    ) as gpu_lease:
        return _run_f0_f1_with_gpu_lease(
            scale=scale,
            filtration_id=filtration_id,
            outer_fold=outer_fold,
            seed=seed,
            repository=repository,
            run_dir=run_dir,
            lock=lock,
            costs=costs,
            lock_sha=lock_sha,
            recipe=recipe,
            protocol=protocol,
            visible_device=str(visible_device),
            gpu_lease=gpu_lease,
        )


def _run_f0_f1_with_gpu_lease(
    *, scale: str, filtration_id: str, outer_fold: str, seed: int,
    repository: Path, run_dir: Path, lock: Mapping[str, Any],
    costs: Mapping[str, Any], lock_sha: str, recipe: Mapping[str, Any],
    protocol: Mapping[str, Any], visible_device: str,
    gpu_lease: _Phase2GpuLease,
) -> dict[str, Any]:
    """Run after the public API has authenticated and leased the GPU."""

    _assert_active_phase2_gpu_lease(gpu_lease, visible_device)
    if visible_device != _registered_visible_device(protocol, outer_fold):
        raise RuntimeError("TMLR V6 F0/F1 GPU lease assignment drift")
    # Load/validate the fold profile but do not expose it to the predictor.
    profiled_cost_vector(costs, scale=scale, outer_fold=outer_fold)
    view = load_registered_f0_f1_view(
        repository, filtration_id=filtration_id, scale=scale,
    )
    view_registry = load_f0_f1_view_registry(
        repository, lock, verified_views=[view],
    )
    view_registry_identity = _view_registry_identity(repository)
    row = next(
        (item for item in view_registry["views"] if item["view_id"] == view.view_id),
        None,
    )
    if row != view_registry_row(view):
        raise RuntimeError("TMLR V6 F0/F1 registered view identity drift")
    train_groups, stop_groups, heldout_groups = split_groups(
        view.fold_manifest, outer_fold=outer_fold, seed=seed,
    )
    group_ids = np.asarray(view.features["image_group_ids"]).astype(str)
    train_np = np.isin(group_ids, tuple(train_groups))
    stop_np = np.isin(group_ids, tuple(stop_groups))
    heldout_np = np.isin(group_ids, tuple(heldout_groups))
    outer_train_np = train_np | stop_np
    if (
        not train_np.any() or not stop_np.any() or not heldout_np.any()
        or np.any(train_np & stop_np) or np.any(train_np & heldout_np)
        or np.any(stop_np & heldout_np) or not np.all(train_np | stop_np | heldout_np)
    ):
        raise RuntimeError("TMLR V6 F0/F1 fit partition is not isolated")
    expected = _expected_manifest(
        repository=repository,
        view=view,
        lock=lock,
        launch_lock_sha256=lock_sha,
        view_registry_identity=view_registry_identity,
        scale=scale,
        filtration_id=filtration_id,
        outer_fold=outer_fold,
        seed=seed,
        train_groups=train_groups,
        stop_groups=stop_groups,
        heldout_groups=heldout_groups,
        train_mask=train_np,
        stop_mask=stop_np,
        heldout_mask=heldout_np,
    )
    staging = _run_staging_path(run_dir)
    _assert_no_stale_run_staging(staging)
    report = _load_existing_f0_f1_report(run_dir)
    if report is not None:
        drift = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
        if drift:
            raise RuntimeError(f"completed TMLR V6 F0/F1 run identity drift: {drift}")
        return validate_f0_f1_run_manifest(
            repository, report, run_dir=run_dir, view=view,
        )

    scaled = scale_remaining_predictor_features(
        state_x=view.features["state_x"],
        action_x16=view.features["action_x16"],
        atom_x=view.features["atom_x"],
        state_lengths=view.features["state_lengths"],
        outer_train_mask=outer_train_np,
    )
    import time
    started = time.perf_counter()
    started_at = _utc_now()
    _seed(seed)
    device = _device()
    telemetry_index = _initialize_cuda_telemetry(device)
    staging = _prepare_run_staging(run_dir)
    state_x = _tensor(scaled.state, device, torch.float32)
    action_x = _tensor(scaled.action, device, torch.float32)
    atom_x = _tensor(scaled.atom, device, torch.float32)
    target = _tensor(view.targets["state_target"], device, torch.float32)
    lengths = _tensor(view.features["state_lengths"], device, torch.long)
    inference_weights = _tensor(
        inference_weights_from_features(
            view.features["atom_x"], view.features["state_lengths"],
        ), device, torch.float32,
    )
    train_mask = _tensor(train_np, device, torch.bool)
    stop_mask = _tensor(stop_np, device, torch.bool)
    model = build_f0_f1_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    memory_bounded = atom_x.shape[0] >= MEMORY_BOUNDED_ATOM_THRESHOLD

    def objective(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(model, SharedAtomicResidual)
        if memory_bounded:
            _, prediction = memory_bounded_shared_atomic_forward(
                model, atom_x, state_x, action_x, lengths, inference_weights,
                checkpoint_activations=model.training,
            )
        else:
            _, prediction = model(
                atom_x, state_x, action_x, lengths, inference_weights,
            )
        return _state_huber(prediction, target, mask), prediction

    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    completed = 0
    for epoch in range(1, int(recipe["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(train_mask)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite TMLR V6 F0/F1 training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            inner = float(objective(stop_mask)[0].item())
        completed = epoch
        if inner < best_value - float(recipe["minimum_inner_stop_improvement"]):
            best_value = inner
            best_epoch = epoch
            stale = 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= int(recipe["patience"]):
                break
        if time.perf_counter() - started > RUN_SECONDS_LIMIT:
            raise RuntimeError("TMLR V6 F0/F1 single-run wall limit exceeded")
    if best_state is None:
        raise RuntimeError("TMLR V6 F0/F1 early stopping produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        _, state_prediction = objective(stop_mask)
    state_prediction = state_prediction.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(state_prediction).all():
        raise RuntimeError("non-finite TMLR V6 F0/F1 model prediction")
    heldout_local = np.flatnonzero(heldout_np)
    prediction = state_prediction[heldout_local].copy()
    prediction[:, A3_PHYSICAL_INDEX] = np.nan
    global_indices = view.global_state_indices[heldout_local].astype(np.int32)
    state_ids = np.asarray(view.features["state_ids"])[heldout_local]
    prediction_path = staging / "predictions.npz"
    checkpoint_path = staging / "checkpoint.pt"
    _atomic_npz(
        prediction_path,
        state_index=global_indices,
        state_id=state_ids,
        state_prediction=prediction,
    )
    _atomic_checkpoint(checkpoint_path, {
        "model": best_state,
        "model_id": filtration_id,
        "filtration_id": filtration_id,
        "registered_view_id": view.view_id,
        "scale": scale,
        "feature_schema_id": "PROSPECTIVE_V1",
        "input_dimensions": expected["input_dimensions"],
        "remaining_physical_indices": np.asarray(
            REMAINING_PHYSICAL_INDICES, dtype=np.int8,
        ),
        "scalers": remaining_scaler_payload(scaled.scalers),
    })
    wall_seconds = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_reserved(telemetry_index))
    if wall_seconds > RUN_SECONDS_LIMIT or peak_vram >= VRAM_LIMIT:
        raise RuntimeError("TMLR V6 F0/F1 per-run resource limit exceeded")
    final_prediction = run_dir / "predictions.npz"
    final_checkpoint = run_dir / "checkpoint.pt"
    report = {
        **expected,
        "architecture_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        **recipe,
        "best_epoch": best_epoch,
        "epochs_completed": completed,
        "inner_stop_loss": best_value,
        "execution_memory_mode": (
            f"ACTIVATION_CHECKPOINTED_ATOM_CHUNKS_{MEMORY_BOUNDED_ATOM_CHUNK}"
            if memory_bounded else "FULL_BATCH"
        ),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "wall_seconds": wall_seconds,
        "gpu_wall_seconds": wall_seconds,
        "peak_vram_bytes": peak_vram,
        "visible_device": visible_device,
        "prediction": {
            "path": _logical_path(repository, final_prediction),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        },
        "checkpoint": {
            "path": _logical_path(repository, final_checkpoint),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
    }
    _atomic_json(staging / "run-manifest.json", report)
    _publish_run_staging(staging, run_dir)
    return report


def audit_f0_f1_runs(
    repository: Path, launch_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact 120-run tree and return consumer-ready rows."""

    repository = repository.resolve()
    from rail3.models.tmlr_v6.phase2_contract import validate_phase2_launch_lock
    validated_lock = validate_phase2_launch_lock(repository)
    if launch_lock is not None:
        supplied = (
            launch_lock.payload if hasattr(launch_lock, "payload") else launch_lock
        )
        if not isinstance(supplied, Mapping) or dict(supplied) != dict(validated_lock.payload):
            raise RuntimeError("TMLR V6 F0/F1 supplied launch-lock identity drift")
    launch_lock = dict(validated_lock.payload)
    if not isinstance(launch_lock, Mapping):
        raise RuntimeError("TMLR V6 F0/F1 audit launch-lock type drift")
    # One full all-eight deterministic reconstruction authenticates the
    # derived view registry for the complete 120-run audit.
    load_f0_f1_view_registry(repository, launch_lock)
    view_registry_identity = _view_registry_identity(repository)
    _assert_logical_path_not_symlinked(
        repository,
        PHASE2_F0_F1_OUTPUT_ROOT,
        description="audit output root",
    )
    root = _repository_path(repository, PHASE2_F0_F1_OUTPUT_ROOT)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("TMLR V6 F0/F1 run root is missing or symlinked")
    expected = expected_f0_f1_job_keys()
    expected_dirs = {
        _run_dir(
            repository, scale=scale, filtration_id=filtration,
            outer_fold=fold, seed=seed,
        )
        for scale, filtration, fold, seed in expected
    }
    _validate_exact_f0_f1_tree(root, expected_dirs)
    views: dict[tuple[str, str], Phase2View] = {}
    runs: list[dict[str, Any]] = []
    for scale, filtration, fold, seed in sorted(expected):
        key = (scale, filtration)
        if key not in views:
            views[key] = load_registered_f0_f1_view(
                repository, filtration_id=filtration, scale=scale,
            )
        run_dir = _run_dir(
            repository, scale=scale, filtration_id=filtration,
            outer_fold=fold, seed=seed,
        )
        report = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
        view = views[key]
        train_groups, stop_groups, heldout_groups = split_groups(
            view.fold_manifest, outer_fold=fold, seed=seed,
        )
        group_ids = np.asarray(view.features["image_group_ids"]).astype(str)
        train_mask = np.isin(group_ids, tuple(train_groups))
        stop_mask = np.isin(group_ids, tuple(stop_groups))
        heldout_mask = np.isin(group_ids, tuple(heldout_groups))
        frozen_identity = _expected_manifest(
            repository=repository,
            view=view,
            lock=launch_lock,
            launch_lock_sha256=validated_lock.sha256,
            view_registry_identity=view_registry_identity,
            scale=scale,
            filtration_id=filtration,
            outer_fold=fold,
            seed=seed,
            train_groups=train_groups,
            stop_groups=stop_groups,
            heldout_groups=heldout_groups,
            train_mask=train_mask,
            stop_mask=stop_mask,
            heldout_mask=heldout_mask,
        )
        validated = validate_f0_f1_run_manifest(
            repository,
            report,
            run_dir=run_dir,
            view=view,
            expected_identity=frozen_identity,
        )
        runs.append({
            "job": {
                "scale": scale,
                "filtration_id": filtration,
                "model_id": filtration,
                "outer_fold": fold,
                "seed": seed,
            },
            "manifest_path": _logical_path(repository, run_dir / "run-manifest.json"),
            "manifest": _artifact_identity(
                repository, run_dir / "run-manifest.json",
            ),
            "report": validated,
        })
    if len(runs) != 120 or len({item["report"]["run_id"] for item in runs}) != 120:
        raise RuntimeError("TMLR V6 F0/F1 audit did not validate 120 unique runs")
    return {
        "schema_version": INVENTORY_SCHEMA,
        "status": INVENTORY_STATUS,
        "phase2_launch_lock": {
            "path": _logical_path(repository, validated_lock.path),
            "bytes": validated_lock.bytes,
            "sha256": validated_lock.sha256,
        },
        "view_registry": view_registry_identity,
        "expected_jobs": 120,
        "validated_runs": 120,
        "runs": runs,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "checks": {
            "exact_job_product_120": True,
            "exact_three_files_per_run": True,
            "a3_prediction_nan": True,
            "remaining_predictions_finite": True,
            "outer_heldout_coverage_exact": True,
            "scales_complete": True,
            "filtrations_complete": True,
            "protected_access_zero": True,
        },
    }


def f0_f1_inventory_artifact_payload(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Strip in-memory full reports to a compact immutable inventory."""

    if audit.get("status") != INVENTORY_STATUS or int(audit.get("validated_runs", -1)) != 120:
        raise RuntimeError("TMLR V6 F0/F1 audit is incomplete")
    rows = []
    for item in audit["runs"]:
        report = item["report"]
        rows.append({
            **item["job"],
            "run_id": report["run_id"],
            "manifest_path": item["manifest_path"],
            "manifest": item["manifest"],
            "prediction": report["prediction"],
            "checkpoint": report["checkpoint"],
            "inner_stop_loss": report["inner_stop_loss"],
            "best_epoch": report["best_epoch"],
            "epochs_completed": report["epochs_completed"],
            "wall_seconds": report["wall_seconds"],
            "peak_vram_bytes": report["peak_vram_bytes"],
        })
    return {
        "schema_version": INVENTORY_SCHEMA,
        "status": INVENTORY_STATUS,
        "phase2_launch_lock": dict(audit["phase2_launch_lock"]),
        "view_registry": dict(audit["view_registry"]),
        "expected_jobs": 120,
        "validated_runs": 120,
        "rows": rows,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "checks": dict(audit["checks"]),
    }


__all__ = [
    "F0_F1_INVENTORY_PATH",
    "INVENTORY_SCHEMA",
    "INVENTORY_STATUS",
    "PARAMETER_COUNT",
    "PREDICTION_ARRAY_NAMES",
    "RUN_PHASE",
    "RUN_SCHEMA",
    "RUN_STATUS",
    "audit_f0_f1_runs",
    "build_f0_f1_model",
    "expected_f0_f1_job_keys",
    "f0_f1_inventory_artifact_payload",
    "run_f0_f1_crossfit",
    "validate_f0_f1_run_manifest",
]
