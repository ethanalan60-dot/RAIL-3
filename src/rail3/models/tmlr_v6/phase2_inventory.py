"""Exact-tree inventory for the 225 TMLR V6 phase-two second-batch runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from rail3.models.tmlr_v6.phase2_contract import (
    PHASE2_LAUNCH_LOCK_PATH,
    Phase2Job,
    Phase2LaunchLock,
    expected_second_batch_jobs,
    file_identity,
    validate_phase2_launch_lock,
)
from rail3.models.tmlr_v6.phase2_models import (
    LOGICAL_ALIAS_REGISTRY,
    SECOND_BATCH_MODEL_IDS,
)
from rail3.models.tmlr_v6.phase2_training import (
    DEFAULT_SECOND_BATCH_OUTPUT_ROOT,
    SECOND_BATCH_DYNAMIC_RUN_FIELDS,
    SECOND_BATCH_RUN_SCHEMA,
    SECOND_BATCH_RUN_STATUS,
    _complete_case_audit,
    _run_static_identity,
    _validate_phase2_execution_fields,
    prediction_array_names,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    FOLD_MANIFEST_PATH,
    _inner_split,
    _repository_path,
    _strings,
    _validate_artifact_identity,
    _validate_bundle_population,
    assert_zero_external_access,
    inference_weights_from_features,
    load_protocol,
    load_registered_bundle,
    sha256_file,
    validate_fold_manifest,
)


DEFAULT_SECOND_BATCH_INVENTORY_PATH = Path(
    "artifacts/audits/tmlr_v6/phase2_second_batch_run_inventory.json"
)
SECOND_BATCH_INVENTORY_SCHEMA = (
    "rail3.tmlr-v6.phase2-second-batch-run-inventory.v1"
)
SECOND_BATCH_INVENTORY_STATUS = (
    "TMLR_V6_PHASE2_SECOND_BATCH_RUN_INVENTORY_PASS"
)
@dataclass(frozen=True)
class SecondBatchValidationContext:
    lock: Phase2LaunchLock
    protocol: Mapping[str, Any]
    fold_manifest: Mapping[str, Any]
    fold_sha256: str
    bundle: Any


def _single_link_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def build_second_batch_validation_context(
    repository: Path,
) -> SecondBatchValidationContext:
    repository = repository.resolve()
    lock = validate_phase2_launch_lock(repository)
    protocol = load_protocol(repository)
    fold_manifest = validate_fold_manifest(repository, protocol)
    fold_sha = sha256_file(_repository_path(repository, FOLD_MANIFEST_PATH))
    if fold_sha != lock.payload["fold_manifests"]["S1364"]["sha256"]:
        raise RuntimeError("TMLR V6 phase-two inventory fold identity drift")
    bundle = load_registered_bundle(repository, "F0_P")
    _validate_bundle_population(bundle, fold_manifest)
    return SecondBatchValidationContext(
        lock=lock,
        protocol=protocol,
        fold_manifest=fold_manifest,
        fold_sha256=fold_sha,
        bundle=bundle,
    )


def _job_from_report(report: Mapping[str, Any]) -> Phase2Job:
    value = report.get("cost_lambda")
    return Phase2Job(
        model_id=str(report.get("model_id", "")),
        scale=str(report.get("scale", "")),
        outer_fold=str(report.get("outer_fold", "")),
        seed=int(report.get("seed", -1)),
        cost_lambda=None if value is None else float(value),
    )


def _expected_atom_index(lengths: np.ndarray, states: np.ndarray) -> np.ndarray:
    offsets = np.r_[0, np.cumsum(lengths)]
    return np.concatenate([
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index in states
    ]).astype(np.int32)


def validate_second_batch_run_manifest(
    repository: Path,
    manifest_path: Path,
    *,
    expected_job: Phase2Job | None = None,
    launch_lock: Phase2LaunchLock | None = None,
    verify_artifacts: bool = True,
    context: SecondBatchValidationContext | None = None,
) -> dict[str, Any]:
    """Validate one run and return its authenticated manifest."""

    repository = repository.resolve()
    ctx = context or build_second_batch_validation_context(repository)
    if launch_lock is not None and launch_lock.sha256 != ctx.lock.sha256:
        raise RuntimeError("TMLR V6 phase-two launch-lock context drift")
    requested_path = (
        manifest_path if manifest_path.is_absolute()
        else repository / manifest_path
    )
    if requested_path.is_symlink() or any(
        parent.is_symlink() for parent in requested_path.parents
        if parent == repository or repository in parent.parents
    ):
        raise RuntimeError("TMLR V6 phase-two run path is symlinked")
    path = _repository_path(repository, manifest_path)
    registered_root = _repository_path(repository, DEFAULT_SECOND_BATCH_OUTPUT_ROOT)
    if expected_job is None:
        registered_paths = {
            job.run_dir(registered_root) / "run-manifest.json": job
            for job in expected_second_batch_jobs()
        }
        job = registered_paths.get(path)
        if job is None:
            raise RuntimeError(
                "TMLR V6 phase-two manifest is outside exact run tree"
            )
    else:
        job = expected_job
        if path != job.run_dir(registered_root) / "run-manifest.json":
            raise RuntimeError(
                "TMLR V6 phase-two manifest is outside exact run tree"
            )
    run_dir = path.parent
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("TMLR V6 phase-two run directory is absent or unsafe")
    entries = list(run_dir.iterdir())
    if (
        {item.name for item in entries}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(not _single_link_regular(item) for item in entries)
    ):
        raise RuntimeError("TMLR V6 phase-two run manifest is absent or unsafe")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise RuntimeError("TMLR V6 phase-two run manifest is not an object")
    if expected_job is None and _job_from_report(report) != job:
        raise RuntimeError("TMLR V6 phase-two manifest job/path identity drift")
    train_groups, stop_groups, heldout_groups = _inner_split(
        ctx.fold_manifest, job.outer_fold, job.seed,
    )
    group_ids = _strings(
        ctx.bundle.features["image_group_ids"], name="image_group_ids",
    )
    train = np.isin(group_ids, tuple(train_groups))
    stop = np.isin(group_ids, tuple(stop_groups))
    heldout = np.isin(group_ids, tuple(heldout_groups))
    expected = _run_static_identity(
        repository=repository,
        job=job,
        lock=ctx.lock,
        bundle=ctx.bundle,
        fold_sha=ctx.fold_sha256,
        train_groups=train_groups,
        stop_groups=stop_groups,
        heldout_groups=heldout_groups,
        group_ids=group_ids,
    )
    expected["missingness_audit"] = _complete_case_audit(
        state_ids=_strings(ctx.bundle.features["state_ids"], name="state_ids"),
        state_target=np.asarray(ctx.bundle.targets["state_target"]),
        feasible=np.asarray(ctx.bundle.targets["feasible"]),
        train=train,
        stop=stop,
        heldout=heldout,
    )
    if set(report) != set(expected) | SECOND_BATCH_DYNAMIC_RUN_FIELDS:
        raise RuntimeError("TMLR V6 phase-two run manifest field-set drift")
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items() if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"TMLR V6 phase-two run identity drift: {drift}")
    assert_zero_external_access(report.get("heldout_access", {}))
    if (
        report.get("schema_version") != SECOND_BATCH_RUN_SCHEMA
        or report.get("status") != SECOND_BATCH_RUN_STATUS
    ):
        raise RuntimeError("TMLR V6 phase-two run status drift")
    _validate_phase2_execution_fields(report, job=job, protocol=ctx.protocol)

    run_dir = path.parent
    entries = list(run_dir.iterdir())
    if (
        {item.name for item in entries}
        != {"run-manifest.json", "predictions.npz", "checkpoint.pt"}
        or any(item.is_symlink() or not item.is_file() for item in entries)
    ):
        raise RuntimeError("TMLR V6 phase-two immutable three-file tree drift")
    if verify_artifacts:
        _validate_artifact_identity(
            repository, report["prediction"], kind="phase-two prediction",
        )
        _validate_artifact_identity(
            repository, report["checkpoint"], kind="phase-two checkpoint",
        )
    prediction_path = _repository_path(
        repository, Path(str(report["prediction"]["path"])),
    )
    checkpoint_path = _repository_path(
        repository, Path(str(report["checkpoint"]["path"])),
    )
    if (
        prediction_path != run_dir / "predictions.npz"
        or checkpoint_path != run_dir / "checkpoint.pt"
    ):
        raise RuntimeError("TMLR V6 phase-two artifact escapes its run")
    if not verify_artifacts:
        return dict(report)

    expected_state_index = np.flatnonzero(heldout).astype(np.int32)
    state_ids = _strings(ctx.bundle.features["state_ids"], name="state_ids")
    with np.load(prediction_path, allow_pickle=False) as payload:
        if set(payload.files) != prediction_array_names(job.model_id):
            raise RuntimeError("TMLR V6 phase-two prediction field-set drift")
        state_index = np.asarray(payload["state_index"])
        output_state_ids = _strings(
            payload["state_id"], name="prediction state_id",
        )
        if (
            state_index.dtype != np.dtype(np.int32)
            or not np.array_equal(state_index, expected_state_index)
            or not np.array_equal(output_state_ids, state_ids[state_index])
        ):
            raise RuntimeError("TMLR V6 phase-two heldout state coverage drift")
        surface_name = {
            "R2_P": "state_prediction",
            "R3_P": "state_prediction",
            "Q2_P": "raw_gain_prediction",
            "L2D_P": "action_logits",
            "SPO_PLUS_P": "action_value_prediction",
        }[job.model_id]
        surface = np.asarray(payload[surface_name])
        if (
            surface.dtype != np.dtype(np.float32)
            or surface.shape != (len(state_index), len(ACTIONS))
            or not np.isfinite(surface).all()
        ):
            raise RuntimeError("TMLR V6 phase-two state surface drift")
        if job.model_id in {"R2_P", "R3_P"}:
            lengths = np.asarray(
                ctx.bundle.features["state_lengths"], dtype=np.int64,
            )
            expected_atom = _expected_atom_index(lengths, state_index)
            atom_index = np.asarray(payload["atom_index"])
            atom_ids = _strings(payload["atom_id"], name="prediction atom_id")
            full_atom_ids = _strings(
                ctx.bundle.features["atom_ids"], name="atom_ids",
            )
            atom_prediction = np.asarray(payload["atom_prediction"])
            if (
                atom_index.dtype != np.dtype(np.int32)
                or not np.array_equal(atom_index, expected_atom)
                or not np.array_equal(atom_ids, full_atom_ids[atom_index])
                or atom_prediction.dtype != np.dtype(np.float32)
                or atom_prediction.shape != (len(atom_index), len(ACTIONS))
                or not np.isfinite(atom_prediction).all()
            ):
                raise RuntimeError("TMLR V6 phase-two heldout atom coverage drift")
            weights = inference_weights_from_features(
                np.asarray(ctx.bundle.features["atom_x"])[atom_index],
                lengths[state_index],
            )
            reconstructed = np.add.reduceat(
                atom_prediction.astype(np.float64) * weights[:, None],
                np.r_[0, np.cumsum(lengths[state_index][:-1])],
            )
            if not np.allclose(
                reconstructed, surface.astype(np.float64), rtol=0.0, atol=2e-6,
            ):
                raise RuntimeError("TMLR V6 phase-two additive prediction drift")
    return dict(report)


def expected_second_batch_run_dirs(
    root: Path = DEFAULT_SECOND_BATCH_OUTPUT_ROOT,
) -> set[Path]:
    return {job.run_dir(root) for job in expected_second_batch_jobs()}


def _validate_exact_tree(root: Path) -> None:
    run_dirs = expected_second_batch_run_dirs(root)
    expected_dirs: set[Path] = set()
    for run_dir in run_dirs:
        current = run_dir
        while current != root:
            expected_dirs.add(current)
            current = current.parent
    expected_files = {
        run_dir / name for run_dir in run_dirs
        for name in ("run-manifest.json", "predictions.npz", "checkpoint.pt")
    }
    observed_dirs: set[Path] = set()
    observed_files: set[Path] = set()
    for path in root.rglob("*") if root.exists() else ():
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            observed_dirs.add(path)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            observed_files.add(path)
        else:
            raise RuntimeError(f"non-regular TMLR V6 phase-two artifact: {path}")
    if observed_dirs != expected_dirs or observed_files != expected_files:
        raise RuntimeError(
            "TMLR V6 phase-two exact artifact tree drift: "
            f"extra_dirs={sorted(observed_dirs - expected_dirs)[:1]}, "
            f"missing_dirs={sorted(expected_dirs - observed_dirs)[:1]}, "
            f"extra_files={sorted(observed_files - expected_files)[:1]}, "
            f"missing_files={sorted(expected_files - observed_files)[:1]}"
        )


def audit_phase2_second_batch_runs(
    *, repository: Path,
    root: Path = DEFAULT_SECOND_BATCH_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Audit the exact 225-run cross product and OOF coverage."""

    repository = repository.resolve()
    context = build_second_batch_validation_context(repository)
    logical_root = repository / DEFAULT_SECOND_BATCH_OUTPUT_ROOT
    if logical_root.is_symlink() or any(
        parent.is_symlink() for parent in logical_root.parents
        if parent == repository or repository in parent.parents
    ):
        raise RuntimeError("TMLR V6 phase-two inventory root is symlinked")
    root = _repository_path(repository, root)
    if root != _repository_path(repository, DEFAULT_SECOND_BATCH_OUTPUT_ROOT):
        raise RuntimeError("TMLR V6 phase-two inventory root is not registered")
    _validate_exact_tree(root)
    expected_jobs = set(expected_second_batch_jobs())
    observed: dict[Phase2Job, dict[str, Any]] = {}
    run_ids: set[str] = set()
    implementation_commits: set[str] = set()
    implementation_hashes: set[str] = set()
    coverage: dict[tuple[str, float | None, int], list[np.ndarray]] = {}
    atom_coverage: dict[tuple[str, int], list[np.ndarray]] = {}
    total_prediction_bytes = 0
    total_checkpoint_bytes = 0
    gpu_wall_seconds = 0.0
    peak_vram_bytes = 0
    for job in expected_second_batch_jobs():
        manifest_path = job.run_dir(root) / "run-manifest.json"
        report = validate_second_batch_run_manifest(
            repository,
            manifest_path,
            expected_job=job,
            launch_lock=context.lock,
            context=context,
        )
        if job in observed:
            raise RuntimeError("duplicate TMLR V6 phase-two job")
        prediction_path = _repository_path(
            repository, Path(str(report["prediction"]["path"])),
        )
        with np.load(prediction_path, allow_pickle=False) as payload:
            state_id = _strings(payload["state_id"], name="prediction state_id")
            coverage.setdefault(
                (job.model_id, job.cost_lambda, job.seed), [],
            ).append(state_id.copy())
            if job.model_id in {"R2_P", "R3_P"}:
                atom_coverage.setdefault((job.model_id, job.seed), []).append(
                    _strings(payload["atom_id"], name="prediction atom_id").copy()
                )
        run_id = str(report["run_id"])
        if run_id in run_ids:
            raise RuntimeError("duplicate TMLR V6 phase-two run ID")
        run_ids.add(run_id)
        implementation_commits.add(str(report["implementation_commit"]))
        implementation_hashes.add(str(report["implementation_sha256"]))
        total_prediction_bytes += int(report["prediction"]["bytes"])
        total_checkpoint_bytes += int(report["checkpoint"]["bytes"])
        gpu_wall_seconds += float(report["gpu_wall_seconds"])
        peak_vram_bytes = max(
            peak_vram_bytes, int(report["peak_vram_bytes"]),
        )
        observed[job] = report
    if set(observed) != expected_jobs:
        raise RuntimeError("TMLR V6 phase-two 225-job coverage drift")

    expected_state_ids = set(_strings(
        context.bundle.features["state_ids"], name="state_ids",
    ).tolist())
    for key, parts in coverage.items():
        ids = np.concatenate(parts).tolist()
        if len(ids) != len(expected_state_ids) or set(ids) != expected_state_ids:
            raise RuntimeError(f"TMLR V6 phase-two OOF state coverage drift: {key}")
    expected_atom_ids = set(_strings(
        context.bundle.features["atom_ids"], name="atom_ids",
    ).tolist())
    for key, parts in atom_coverage.items():
        ids = np.concatenate(parts).tolist()
        if len(ids) != len(expected_atom_ids) or set(ids) != expected_atom_ids:
            raise RuntimeError(f"TMLR V6 phase-two OOF atom coverage drift: {key}")
    if (
        implementation_commits != {context.lock.implementation_commit}
        or implementation_hashes != {context.lock.implementation_sha256}
    ):
        raise RuntimeError("TMLR V6 phase-two run implementation mixture")
    by_model = {
        model_id: sum(job.model_id == model_id for job in observed)
        for model_id in SECOND_BATCH_MODEL_IDS
    }
    if by_model != {
        "R2_P": 15,
        "R3_P": 15,
        "Q2_P": 15,
        "L2D_P": 90,
        "SPO_PLUS_P": 90,
    }:
        raise RuntimeError("TMLR V6 phase-two per-model job counts drift")
    return {
        "schema_version": SECOND_BATCH_INVENTORY_SCHEMA,
        "status": SECOND_BATCH_INVENTORY_STATUS,
        "launch_lock": file_identity(repository, PHASE2_LAUNCH_LOCK_PATH),
        "expected_runs": 225,
        "completed_valid_runs": len(observed),
        "missing_runs": 0,
        "duplicate_runs": 0,
        "unexpected_runs": 0,
        "temporary_or_incomplete_runs": 0,
        "by_model": by_model,
        "logical_alias_registry": {
            "LL4TTA_P": {
                "alias_of": LOGICAL_ALIAS_REGISTRY["LL4TTA_P"],
                "new_training_jobs": 0,
                "artifact_strategy": (
                    "DIRECT_REFERENCE_TO_FIRST_BATCH_R0_SMALL_P_RUNS_NO_COPY"
                ),
                "first_batch_run_inventory_sha256": context.lock.payload[
                    "first_batch_run_inventory"
                ]["sha256"],
            }
        },
        "implementation_commits": sorted(implementation_commits),
        "implementation_sha256": sorted(implementation_hashes),
        "total_prediction_bytes": total_prediction_bytes,
        "total_checkpoint_bytes": total_checkpoint_bytes,
        "gpu_wall_seconds": gpu_wall_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "heldout_access": dict(context.lock.payload["heldout_access"]),
        "checks": {
            "launch_lock_no_bypass": True,
            "exact_cross_product_225": len(observed) == 225,
            "ll4tta_is_zero_job_logical_alias": True,
            "immutable_three_file_run_tree": True,
            "all_artifact_hashes_valid": True,
            "all_run_ids_unique": len(run_ids) == 225,
            "single_committed_clean_implementation": True,
            "predictor_cost_input_zero": True,
            "exact_state_oof_coverage": True,
            "exact_atom_oof_coverage_for_r2_r3": True,
            "q2_raw_stop_saved_but_not_scientifically_consumed_before_ensemble": True,
            "all_training_states_finite": True,
            "resource_limits_pass": True,
            "protected_access_zero": True,
        },
    }


__all__ = [
    "DEFAULT_SECOND_BATCH_INVENTORY_PATH",
    "SECOND_BATCH_INVENTORY_SCHEMA",
    "SECOND_BATCH_INVENTORY_STATUS",
    "SecondBatchValidationContext",
    "audit_phase2_second_batch_runs",
    "build_second_batch_validation_context",
    "expected_second_batch_run_dirs",
    "validate_second_batch_run_manifest",
]
