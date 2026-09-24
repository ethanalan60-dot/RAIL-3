"""Immutable 225-job TMLR V6 phase-two second-batch training.

Every entry point authenticates the identity-only phase-two launch lock before
opening a prospective bundle.  Predictor tensors contain only PROSPECTIVE_V1
features; fold-specific costs are passed solely to registered objectives.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping

import numpy as np
import torch

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.m06e.models import SharedAtomicResidual
from rail3.models.tmlr_v6.phase2_contract import (
    Phase2Job,
    Phase2LaunchLock,
    expected_second_batch_jobs,
    load_phase2_addendum,
    load_s1364_fold_cost_profile,
    second_batch_run_path,
    validate_phase2_launch_lock,
)

from rail3.models.tmlr_v6.phase2_models import (
    ATOMIC_MODEL_IDS,
    DIRECT_INPUT_DIM,
    DIRECT_MODEL_IDS,
    MODEL_REGISTRY,
    PREDICTION_SEMANTICS,
    build_phase2_model,
)
from rail3.models.tmlr_v6.phase2_objectives import (
    AtomicObjectiveInputs,
    direct_objective,
    q2_objective,
    r2_objective,
    r3_objective,
)
from rail3.models.tmlr_v6.schema import (
    ACTION_DIM,
    ATOM_DIM,
    GLOBAL_DIM,
    PROSPECTIVE_SCHEMA_ID,
    STATE_DIM,
    scale_predictor_features,
    scaler_payload,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    CUBLAS_WORKSPACE_CONFIG,
    FOLD_MANIFEST_PATH,
    MEMORY_BOUNDED_ATOM_CHUNK,
    MEMORY_BOUNDED_ATOM_THRESHOLD,
    PROTOCOL_PATH,
    RUN_SECONDS_LIMIT,
    VRAM_LIMIT,
    ZERO_EXTERNAL_ACCESS,
    _atomic_checkpoint,
    _atomic_json,
    _atomic_npz,
    _device,
    _GpuLease,
    _assert_active_gpu_lease,
    _exclusive_gpu_worker_lock,
    _gpu_kernel_lease_address,
    _initialize_cuda_telemetry,
    _inner_split,
    _logical_path,
    _prepare_run_staging,
    _publish_run_staging,
    _repository_path,
    _run_staging_path,
    _seed,
    _snapshot,
    _strings,
    _tensor,
    _validate_artifact_identity,
    _validate_bundle_population,
    assert_cuda_determinism_environment,
    assert_zero_external_access,
    cuda_determinism_environment_contract,
    inference_weights_from_features,
    load_protocol,
    load_registered_bundle,
    memory_bounded_shared_atomic_forward,
    sha256_file,
    validate_fold_manifest,
)


_Phase2GpuLease = _GpuLease
_assert_active_phase2_gpu_lease = _assert_active_gpu_lease
_phase2_gpu_kernel_lease_address = _gpu_kernel_lease_address
exclusive_phase2_gpu_lock = _exclusive_gpu_worker_lock


def _single_link_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


DEFAULT_SECOND_BATCH_OUTPUT_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/phase2-second-batch-crossfit"
)
SECOND_BATCH_RUN_SCHEMA = "rail3.tmlr-v6.phase2-second-batch-run.v1"
SECOND_BATCH_RUN_STATUS = "PASS"
SECOND_BATCH_PHASE = "fit_s1364_phase2_second_batch_outer_crossfit"

PREDICTION_ARRAY_NAMES = {
    "R2_P": frozenset({
        "state_index", "state_id", "state_prediction",
        "atom_index", "atom_id", "atom_prediction",
    }),
    "R3_P": frozenset({
        "state_index", "state_id", "state_prediction",
        "atom_index", "atom_id", "atom_prediction",
    }),
    "Q2_P": frozenset({
        "state_index", "state_id", "raw_gain_prediction",
    }),
    "L2D_P": frozenset({"state_index", "state_id", "action_logits"}),
    "SPO_PLUS_P": frozenset({
        "state_index", "state_id", "action_value_prediction",
    }),
}
OBJECTIVE_DESCRIPTIONS = {
    "R2_P": "L_atom + L_abs + 0.5*L_PIDR",
    "R3_P": "L_atom + L_abs + L_rel + 0.5*L_sign_weighted + 0.5*L_rank",
    "Q2_P": "state-balanced Huber(delta=0.05) on direct STOP-relative gain",
    "L2D_P": "unweighted feasible-masked multiclass cross entropy",
    "SPO_PLUS_P": "finite-action SPO+ with deterministic first-minimum tie break",
}
ARCHITECTURE_CLASSES = {
    "R2_P": (
        "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual"
    ),
    "R3_P": (
        "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual"
    ),
    "Q2_P": "rail3.models.m06e.models.DirectGainBaseline",
    "L2D_P": "rail3.diagnostic.baselines.MultiActionNetwork",
    "SPO_PLUS_P": "rail3.diagnostic.baselines.MultiActionNetwork",
}
FROZEN_TRAINING_RECIPE = {
    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "max_epochs": 120,
    "patience": 15,
    "minimum_inner_stop_improvement": 1e-8,
}
SECOND_BATCH_DYNAMIC_RUN_FIELDS = frozenset({
    "best_epoch",
    "epochs_completed",
    "inner_stop_loss",
    "started_at_utc",
    "ended_at_utc",
    "wall_seconds",
    "gpu_wall_seconds",
    "peak_vram_bytes",
    "visible_device",
    "prediction",
    "checkpoint",
})


def phase2_direct_input(
    state_features: torch.Tensor, action_features: torch.Tensor,
) -> torch.Tensor:
    """Build the exact 41 + 5*16 predictor input; no cost argument exists."""

    if (
        state_features.ndim != 2
        or state_features.shape[1] != STATE_DIM
        or action_features.ndim != 3
        or action_features.shape[0] != state_features.shape[0]
        or action_features.shape[1:] != (len(ACTIONS), ACTION_DIM)
    ):
        raise ValueError("TMLR V6 phase-two direct feature geometry drift")
    result = torch.cat((
        state_features,
        action_features.reshape(len(state_features), len(ACTIONS) * ACTION_DIM),
    ), dim=1)
    if result.shape[1] != DIRECT_INPUT_DIM:
        raise RuntimeError("TMLR V6 phase-two direct input dimension drift")
    return result


def prediction_array_names(model_id: str) -> frozenset[str]:
    try:
        return PREDICTION_ARRAY_NAMES[model_id]
    except KeyError as error:
        raise ValueError("unregistered TMLR V6 phase-two model") from error


def _artifact_identity(
    repository: Path, path: Path, *, logical_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "path": _logical_path(
            repository, path if logical_path is None else logical_path,
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _complete_case_audit(
    *, state_ids: np.ndarray, state_target: np.ndarray, feasible: np.ndarray,
    train: np.ndarray, stop: np.ndarray, heldout: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(state_target)
    plan = np.asarray(feasible, dtype=bool)
    missing = plan & ~np.isfinite(target)
    complete = ~missing.any(axis=1)
    formal = complete & plan[:, 0] & np.isfinite(target[:, 0]) & (
        plan[:, 1:] & np.isfinite(target[:, 1:])
    ).any(axis=1)
    excluded = heldout & ~complete
    return {
        "objective_supervised_states_and_actions": {
            "inner_train_states": int((train & np.isfinite(target).any(1)).sum()),
            "inner_train_actions": int((train[:, None] & np.isfinite(target)).sum()),
            "inner_stop_states": int((stop & np.isfinite(target).any(1)).sum()),
            "inner_stop_actions": int((stop[:, None] & np.isfinite(target)).sum()),
        },
        "formal_complete_case_states": int((heldout & formal).sum()),
        "plan_feasible_target_undefined_excluded_states": int(excluded.sum()),
        "excluded_state_ids_sha256": hashlib.sha256(canonical_json_bytes(
            sorted(np.asarray(state_ids).astype(str)[excluded].tolist())
        )).hexdigest(),
    }


def _registered_visible_device(
    protocol: Mapping[str, Any], outer_fold: str,
) -> str:
    training = protocol.get("training", {})
    assignments = training.get("gpu_assignment", {}) if isinstance(
        training, Mapping,
    ) else {}
    if not isinstance(assignments, Mapping):
        raise RuntimeError("TMLR V6 phase-two GPU assignment is absent")
    devices = {
        str(device) for device, folds in assignments.items()
        if isinstance(folds, list) and outer_fold in folds
    }
    if len(devices) != 1 or not next(iter(devices)).isdigit():
        raise RuntimeError("TMLR V6 phase-two fold/GPU assignment drift")
    return next(iter(devices))


def _validate_phase2_execution_fields(
    report: Mapping[str, Any], *, job: Phase2Job,
    protocol: Mapping[str, Any],
) -> None:
    """Validate dynamic training/resource observations for one completed run."""

    try:
        started = time.strptime(
            str(report.get("started_at_utc", "")), "%Y-%m-%dT%H:%M:%SZ",
        )
        ended = time.strptime(
            str(report.get("ended_at_utc", "")), "%Y-%m-%dT%H:%M:%SZ",
        )
        best_epoch = int(report.get("best_epoch", 0))
        completed = int(report.get("epochs_completed", 0))
        inner_stop = float(report.get("inner_stop_loss", math.nan))
        wall = float(report.get("wall_seconds", math.nan))
        gpu_wall = float(report.get("gpu_wall_seconds", math.nan))
        peak_vram = int(report.get("peak_vram_bytes", -1))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "TMLR V6 phase-two execution observation is invalid"
        ) from error
    if (
        not 1 <= best_epoch <= completed <= FROZEN_TRAINING_RECIPE["max_epochs"]
        or not math.isfinite(inner_stop)
        or not math.isfinite(wall)
        or not 0.0 <= wall <= RUN_SECONDS_LIMIT
        or not math.isfinite(gpu_wall)
        or gpu_wall != wall
        or not 0 <= peak_vram < VRAM_LIMIT
        or ended < started
        or report.get("visible_device")
        != _registered_visible_device(protocol, job.outer_fold)
    ):
        raise RuntimeError("TMLR V6 phase-two run resource/training drift")


def _run_static_identity(
    *, repository: Path, job: Phase2Job, lock: Phase2LaunchLock,
    bundle: Any, fold_sha: str, train_groups: set[str], stop_groups: set[str],
    heldout_groups: set[str], group_ids: np.ndarray,
) -> dict[str, Any]:
    implementation = lock.payload["phase2_implementation"]
    parent = lock.payload["parent_protocol"]
    addendum = lock.payload["phase2_semantic_addendum"]
    cost_identity = lock.payload["s1364_cost_profiles"]["json"]
    run_id_fields = {
        "parent_protocol_sha256": parent["sha256"],
        "semantic_addendum_sha256": addendum["sha256"],
        "launch_lock_sha256": lock.sha256,
        "implementation_combined_sha256": implementation[
            "combined_source_sha256"
        ],
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "cost_profile_registry_sha256": cost_identity["sha256"],
        "model_id": job.model_id,
        "scale": job.scale,
        "lambda_if_model_specific": job.cost_lambda,
        "filtration_if_applicable": "F0_P",
        "outer_fold": job.outer_fold,
        "seed": job.seed,
    }
    return {
        "schema_version": SECOND_BATCH_RUN_SCHEMA,
        "status": SECOND_BATCH_RUN_STATUS,
        "phase": SECOND_BATCH_PHASE,
        "run_id": stable_id("tmlr_v6_phase2_second_batch_run", run_id_fields),
        "model_id": job.model_id,
        "outer_fold": job.outer_fold,
        "seed": job.seed,
        "scale": job.scale,
        "cost_lambda": job.cost_lambda,
        "lambda_if_model_specific": job.cost_lambda,
        "bundle_role_id": "F0_P",
        "filtration": "F0_P",
        "filtration_if_applicable": "F0_P",
        "prediction_semantics": PREDICTION_SEMANTICS[job.model_id],
        "architecture_class": ARCHITECTURE_CLASSES[job.model_id],
        "parent_protocol_sha256": parent["sha256"],
        "semantic_addendum_sha256": addendum["sha256"],
        "launch_lock_sha256": lock.sha256,
        "implementation_commit": implementation["commit"],
        "implementation_sha256": implementation["combined_source_sha256"],
        "implementation_combined_sha256": implementation[
            "combined_source_sha256"
        ],
        "implementation_source_sha256": dict(implementation["source_sha256"]),
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "cost_profile_registry_sha256": cost_identity["sha256"],
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "feature_schema_id": PROSPECTIVE_SCHEMA_ID,
        "input_dimensions": {
            "atom": ATOM_DIM,
            "state": STATE_DIM,
            "action_projected": ACTION_DIM,
            "global_per_action": GLOBAL_DIM,
            "direct_all_actions": DIRECT_INPUT_DIM,
        },
        "physical_input_partitions": ["prospective_features", "targets"],
        "post_action_diagnostics_opened": False,
        "predictor_cost_input": False,
        "cost_objective_input": job.model_id in {
            "R2_P", "L2D_P", "SPO_PLUS_P",
        },
        "inference_weight_source": "feature_area_fraction_full_raster",
        "target_weight_source": "valid_pixel_fraction_target_only",
        "target_weight_in_prediction_graph": False,
        "objective": OBJECTIVE_DESCRIPTIONS[job.model_id],
        "parameter_count": MODEL_REGISTRY[job.model_id].parameter_count,
        **FROZEN_TRAINING_RECIPE,
        "execution_memory_mode": (
            f"ACTIVATION_CHECKPOINTED_ATOM_CHUNKS_{MEMORY_BOUNDED_ATOM_CHUNK}"
            if (
                job.model_id in ATOMIC_MODEL_IDS
                and len(bundle.features["atom_ids"])
                >= MEMORY_BOUNDED_ATOM_THRESHOLD
            )
            else "FULL_BATCH"
        ),
        "train_group_ids": sorted(train_groups),
        "inner_stop_group_ids": sorted(stop_groups),
        "outer_heldout_fit_group_ids": sorted(heldout_groups),
        "counts": {
            "inner_train_states": int(np.isin(group_ids, tuple(train_groups)).sum()),
            "inner_stop_states": int(np.isin(group_ids, tuple(stop_groups)).sum()),
            "outer_heldout_fit_states": int(
                np.isin(group_ids, tuple(heldout_groups)).sum()
            ),
            "bundle_states": len(group_ids),
            "bundle_atoms": len(bundle.features["atom_ids"]),
        },
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "fit_only": True,
        "sam_inference": False,
        "prompt_or_trajectory_generation": False,
        "protected_reader_invoked": False,
        "ddp": False,
    }


def _verify_existing_phase2_run(
    repository: Path, run_dir: Path, expected: Mapping[str, Any], *,
    job: Phase2Job, protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    registered_run_dir = job.run_dir(
        _repository_path(repository, DEFAULT_SECOND_BATCH_OUTPUT_ROOT)
    )
    if _repository_path(repository, run_dir) != registered_run_dir:
        raise RuntimeError("TMLR V6 phase-two resume path is outside exact run tree")
    if not run_dir.exists() and not run_dir.is_symlink():
        return None
    manifest_path = run_dir / "run-manifest.json"
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError(f"incomplete TMLR V6 phase-two run: {run_dir}")
    entries = list(run_dir.iterdir())
    if (
        {item.name for item in entries}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(not _single_link_regular(item) for item in entries)
    ):
        raise RuntimeError(f"incomplete TMLR V6 phase-two run: {run_dir}")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, Mapping)
        or set(report) != set(expected) | SECOND_BATCH_DYNAMIC_RUN_FIELDS
    ):
        raise RuntimeError("completed TMLR V6 phase-two manifest schema drift")
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items() if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"completed TMLR V6 phase-two run drift: {drift}")
    assert_zero_external_access(report.get("heldout_access", {}))
    _validate_phase2_execution_fields(report, job=job, protocol=protocol)
    expected_prediction = _repository_path(repository, run_dir / "predictions.npz")
    expected_checkpoint = _repository_path(repository, run_dir / "checkpoint.pt")
    if (
        _repository_path(
            repository, Path(str(report["prediction"].get("path", ""))),
        ) != expected_prediction
        or _repository_path(
            repository, Path(str(report["checkpoint"].get("path", ""))),
        ) != expected_checkpoint
    ):
        raise RuntimeError(
            "completed TMLR V6 phase-two artifact logical path drift"
        )
    _validate_artifact_identity(repository, report["prediction"], kind="prediction")
    _validate_artifact_identity(repository, report["checkpoint"], kind="checkpoint")
    return report


def run_phase2_second_batch_crossfit(
    *, job: Phase2Job, repository: Path,
    output_root: Path = DEFAULT_SECOND_BATCH_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate one job, then train/resume it under its mandatory GPU lease."""

    assert_cuda_determinism_environment()
    repository = repository.resolve()
    if job not in set(expected_second_batch_jobs()):
        raise ValueError("unregistered TMLR V6 phase-two second-batch job")
    # Validate every launch argument and immutable launch identity before the
    # lock directory or durable lock file can be created.
    lock = validate_phase2_launch_lock(repository)
    addendum = load_phase2_addendum(repository)
    if not _single_link_regular(_repository_path(repository, PROTOCOL_PATH)):
        raise RuntimeError(
            "TMLR V6 phase-two protocol must be one regular single-link file"
        )
    protocol = load_protocol(repository)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_device != _registered_visible_device(protocol, job.outer_fold):
        raise RuntimeError(
            "TMLR V6 phase-two worker violates physical-GPU assignment"
        )
    logical_output_root = repository / DEFAULT_SECOND_BATCH_OUTPUT_ROOT
    if logical_output_root.is_symlink() or any(
        parent.is_symlink() for parent in logical_output_root.parents
        if parent == repository or repository in parent.parents
    ):
        raise RuntimeError("TMLR V6 phase-two output root is symlinked")
    if _repository_path(repository, output_root) != _repository_path(
        repository, DEFAULT_SECOND_BATCH_OUTPUT_ROOT,
    ):
        raise RuntimeError("TMLR V6 phase-two output root is not registered")
    with exclusive_phase2_gpu_lock(
        repository, str(visible_device),
    ) as gpu_lease:
        return _run_phase2_second_batch_with_gpu_lease(
            job=job,
            repository=repository,
            output_root=output_root,
            lock=lock,
            addendum=addendum,
            protocol=protocol,
            visible_device=str(visible_device),
            gpu_lease=gpu_lease,
        )


def _run_phase2_second_batch_with_gpu_lease(
    *, job: Phase2Job, repository: Path, output_root: Path,
    lock: Phase2LaunchLock, addendum: Mapping[str, Any],
    protocol: Mapping[str, Any], visible_device: str,
    gpu_lease: _Phase2GpuLease,
) -> dict[str, Any]:
    """Run after the public API has authenticated and leased the GPU."""

    _assert_active_phase2_gpu_lease(gpu_lease, visible_device)
    if visible_device != _registered_visible_device(protocol, job.outer_fold):
        raise RuntimeError("TMLR V6 phase-two GPU lease assignment drift")
    run_dir = second_batch_run_path(
        _repository_path(repository, output_root), job,
    )
    if any(parent.is_symlink() for parent in run_dir.parents if parent != repository):
        raise RuntimeError("TMLR V6 phase-two run parent is symlinked")
    staging = _run_staging_path(run_dir)
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(
            f"stale incomplete TMLR V6 phase-two run needs resolution: {staging}"
        )

    fold_path = _repository_path(repository, FOLD_MANIFEST_PATH)
    if not _single_link_regular(fold_path):
        raise RuntimeError(
            "TMLR V6 phase-two fold manifest must be one regular single-link file"
        )
    fold_manifest = validate_fold_manifest(repository, protocol)
    fold_sha = sha256_file(fold_path)
    if fold_sha != lock.payload["fold_manifests"]["S1364"]["sha256"]:
        raise RuntimeError("TMLR V6 phase-two S1364 fold identity drift")
    bundle = load_registered_bundle(repository, "F0_P")
    _validate_bundle_population(bundle, fold_manifest)
    cost_np = load_s1364_fold_cost_profile(repository, lock, job.outer_fold)

    train_groups, stop_groups, heldout_groups = _inner_split(
        fold_manifest, job.outer_fold, job.seed,
    )
    group_ids = _strings(bundle.features["image_group_ids"], name="image_group_ids")
    train_np = np.isin(group_ids, tuple(train_groups))
    stop_np = np.isin(group_ids, tuple(stop_groups))
    heldout_np = np.isin(group_ids, tuple(heldout_groups))
    outer_train_np = train_np | stop_np
    if (
        not train_np.any()
        or not stop_np.any()
        or not heldout_np.any()
        or np.any(train_np & stop_np)
        or np.any(train_np & heldout_np)
        or np.any(stop_np & heldout_np)
        or not np.all(train_np | stop_np | heldout_np)
    ):
        raise RuntimeError("TMLR V6 phase-two FIT partition isolation failed")

    expected = _run_static_identity(
        repository=repository,
        job=job,
        lock=lock,
        bundle=bundle,
        fold_sha=fold_sha,
        train_groups=train_groups,
        stop_groups=stop_groups,
        heldout_groups=heldout_groups,
        group_ids=group_ids,
    )
    expected["missingness_audit"] = _complete_case_audit(
        state_ids=_strings(bundle.features["state_ids"], name="state_ids"),
        state_target=np.asarray(bundle.targets["state_target"]),
        feasible=np.asarray(bundle.targets["feasible"]),
        train=train_np,
        stop=stop_np,
        heldout=heldout_np,
    )
    existing = _verify_existing_phase2_run(
        repository, run_dir, expected, job=job, protocol=protocol,
    )
    if existing is not None:
        return existing

    is_atomic = job.model_id in ATOMIC_MODEL_IDS
    scaled = scale_predictor_features(
        atom_x=bundle.features["atom_x"] if is_atomic else None,
        state_lengths=bundle.features["state_lengths"] if is_atomic else None,
        state_x=bundle.features["state_x"],
        action_x16=bundle.features["action_x16"],
        outer_train_mask=outer_train_np,
    )
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed(job.seed)
    device = _device()
    telemetry_index = _initialize_cuda_telemetry(device)
    staging = _prepare_run_staging(run_dir)

    state_x = _tensor(scaled.state, device, torch.float32)
    action_x = _tensor(scaled.action, device, torch.float32)
    state_target = _tensor(bundle.targets["state_target"], device, torch.float32)
    feasible = _tensor(bundle.targets["feasible"], device, torch.bool)
    train_mask = _tensor(train_np, device, torch.bool)
    stop_mask = _tensor(stop_np, device, torch.bool)
    cost = _tensor(cost_np, device, torch.float32)
    global_x = torch.cat((
        state_x[:, None, :].expand(-1, len(ACTIONS), -1), action_x,
    ), dim=2)
    direct_x = phase2_direct_input(state_x, action_x)
    model = build_phase2_model(job.model_id).to(device)
    architecture_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
    if architecture_class != expected["architecture_class"]:
        raise RuntimeError("TMLR V6 phase-two architecture identity drift")
    recipe = addendum["training_recipe"]
    observed_recipe = {
        key: recipe.get(key) for key in FROZEN_TRAINING_RECIPE if key != "optimizer"
    }
    if (
        recipe.get("optimizer") != FROZEN_TRAINING_RECIPE["optimizer"]
        or observed_recipe != {
            key: value for key, value in FROZEN_TRAINING_RECIPE.items()
            if key != "optimizer"
        }
    ):
        raise RuntimeError("TMLR V6 phase-two frozen training recipe drift")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(addendum["training_recipe"]["learning_rate"]),
        weight_decay=float(addendum["training_recipe"]["weight_decay"]),
    )

    atom_x: torch.Tensor | None = None
    atom_target: torch.Tensor | None = None
    atom_weights: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    inference_weights: torch.Tensor | None = None
    memory_bounded = False
    if is_atomic:
        if scaled.atom is None:
            raise RuntimeError("TMLR V6 phase-two atomic scaler is absent")
        atom_x = _tensor(scaled.atom, device, torch.float32)
        atom_target = _tensor(bundle.targets["atom_target"], device, torch.float32)
        atom_weights = _tensor(bundle.targets["atom_weights"], device, torch.float32)
        lengths = _tensor(bundle.features["state_lengths"], device, torch.long)
        inference_weights = _tensor(
            inference_weights_from_features(
                bundle.features["atom_x"], bundle.features["state_lengths"],
            ),
            device,
            torch.float32,
        )
        memory_bounded = len(atom_x) >= MEMORY_BOUNDED_ATOM_THRESHOLD

    def objective(mask: torch.Tensor) -> tuple[
        Mapping[str, torch.Tensor], torch.Tensor, torch.Tensor | None,
    ]:
        if is_atomic:
            assert (
                isinstance(model, SharedAtomicResidual)
                and atom_x is not None
                and atom_target is not None
                and atom_weights is not None
                and lengths is not None
                and inference_weights is not None
            )
            if memory_bounded:
                atom_prediction, state_prediction = (
                    memory_bounded_shared_atomic_forward(
                        model,
                        atom_x,
                        state_x,
                        action_x,
                        lengths,
                        inference_weights,
                        checkpoint_activations=model.training,
                    )
                )
            else:
                atom_prediction, state_prediction = model(
                    atom_x, state_x, action_x, lengths, inference_weights,
                )
            inputs = AtomicObjectiveInputs(
                atom_prediction=atom_prediction,
                state_prediction=state_prediction,
                atom_target=atom_target,
                state_target=state_target,
                atom_weights=atom_weights,
                state_lengths=lengths,
                state_mask=mask,
                normalized_cost=cost,
            )
            terms = r2_objective(inputs) if job.model_id == "R2_P" else r3_objective(inputs)
            return terms, state_prediction, atom_prediction
        if job.model_id == "Q2_P":
            raw_gain = model(global_x.reshape(-1, GLOBAL_DIM)).reshape(-1, 5)
            return q2_objective(raw_gain, state_target, mask), raw_gain, None
        output = model(direct_x)
        if job.cost_lambda is None:
            raise RuntimeError("TMLR V6 direct model has no registered lambda")
        return direct_objective(
            model_id=job.model_id,
            output=output,
            state_target=state_target,
            feasible=feasible,
            state_mask=mask,
            normalized_cost=cost,
            cost_lambda=job.cost_lambda,
        ), output, None

    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    completed = 0
    max_epochs = int(recipe["max_epochs"])
    patience = int(recipe["patience"])
    minimum_improvement = float(recipe["minimum_inner_stop_improvement"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        terms, _, _ = objective(train_mask)
        loss = terms["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite TMLR V6 phase-two training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            inner_value = float(objective(stop_mask)[0]["total"].item())
        completed = epoch
        if inner_value < best_value - minimum_improvement:
            best_value = inner_value
            best_epoch = epoch
            stale = 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= patience:
                break
        if time.perf_counter() - started > RUN_SECONDS_LIMIT:
            raise RuntimeError("TMLR V6 phase-two single-run wall limit exceeded")
    if best_state is None:
        raise RuntimeError("TMLR V6 phase-two early stopping has no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        _, state_output, atom_output = objective(stop_mask)
    if not torch.isfinite(state_output).all():
        raise RuntimeError("non-finite TMLR V6 phase-two state output")

    heldout_index = np.flatnonzero(heldout_np).astype(np.int32)
    state_ids = _strings(bundle.features["state_ids"], name="state_ids")
    output: dict[str, np.ndarray] = {
        "state_index": heldout_index,
        "state_id": state_ids[heldout_index],
    }
    state_array = (
        state_output[heldout_index].detach().cpu().numpy().astype(np.float32)
    )
    if job.model_id in ATOMIC_MODEL_IDS:
        output["state_prediction"] = state_array
        if atom_output is None or not torch.isfinite(atom_output).all():
            raise RuntimeError("non-finite TMLR V6 phase-two atom output")
        lengths_np = np.asarray(bundle.features["state_lengths"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths_np)]
        atom_index = np.concatenate([
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index in heldout_index
        ]).astype(np.int32)
        atom_ids = _strings(bundle.features["atom_ids"], name="atom_ids")
        output.update({
            "atom_index": atom_index,
            "atom_id": atom_ids[atom_index],
            "atom_prediction": (
                atom_output[atom_index].detach().cpu().numpy().astype(np.float32)
            ),
        })
    elif job.model_id == "Q2_P":
        output["raw_gain_prediction"] = state_array
    elif job.model_id == "L2D_P":
        output["action_logits"] = state_array
    else:
        output["action_value_prediction"] = state_array
    if set(output) != prediction_array_names(job.model_id):
        raise RuntimeError("TMLR V6 phase-two prediction schema drift")

    prediction_path = staging / "predictions.npz"
    checkpoint_path = staging / "checkpoint.pt"
    _atomic_npz(prediction_path, **output)
    _atomic_checkpoint(checkpoint_path, {
        "model": best_state,
        "model_id": job.model_id,
        "bundle_role_id": "F0_P",
        "feature_schema_id": PROSPECTIVE_SCHEMA_ID,
        "prediction_semantics": PREDICTION_SEMANTICS[job.model_id],
        "cost_lambda": job.cost_lambda,
        "normalized_fold_cost_objective_only": cost_np,
        "predictor_cost_input": False,
        "input_dimensions": expected["input_dimensions"],
        "scalers": scaler_payload(scaled.scalers),
    })
    wall_seconds = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_reserved(telemetry_index))
    if wall_seconds > RUN_SECONDS_LIMIT or peak_vram >= VRAM_LIMIT:
        raise RuntimeError("TMLR V6 phase-two run exceeds resource limits")
    report = {
        **expected,
        "best_epoch": best_epoch,
        "epochs_completed": completed,
        "inner_stop_loss": best_value,
        "started_at_utc": started_at,
        "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": wall_seconds,
        "gpu_wall_seconds": wall_seconds,
        "peak_vram_bytes": peak_vram,
        "visible_device": visible_device,
        "prediction": _artifact_identity(
            repository, prediction_path,
            logical_path=run_dir / prediction_path.name,
        ),
        "checkpoint": _artifact_identity(
            repository, checkpoint_path,
            logical_path=run_dir / checkpoint_path.name,
        ),
    }
    _atomic_json(staging / "run-manifest.json", report)
    staging_entries = list(staging.iterdir())
    if (
        {item.name for item in staging_entries}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(item.is_symlink() or not item.is_file() for item in staging_entries)
    ):
        raise RuntimeError("TMLR V6 phase-two staging tree is not exactly three files")
    _publish_run_staging(staging, run_dir)
    return report


__all__ = [
    "DEFAULT_SECOND_BATCH_OUTPUT_ROOT",
    "OBJECTIVE_DESCRIPTIONS",
    "PREDICTION_ARRAY_NAMES",
    "SECOND_BATCH_PHASE",
    "SECOND_BATCH_RUN_SCHEMA",
    "SECOND_BATCH_RUN_STATUS",
    "phase2_direct_input",
    "prediction_array_names",
    "run_phase2_second_batch_crossfit",
]
