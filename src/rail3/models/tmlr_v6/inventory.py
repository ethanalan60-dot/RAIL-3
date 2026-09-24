"""Exhaustive 75-run inventory and first-batch completion gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    BUNDLE_REGISTRY,
    CUBLAS_WORKSPACE_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    FIRST_BATCH_MODEL_IDS,
    FOLD_MANIFEST_PATH,
    FOLDS,
    PROTOCOL_PATH,
    RUNTIME_FEATURE_AUDIT_PATH,
    RUNTIME_FEATURE_AUDIT_STATUS,
    SEEDS,
    ACTION_DIM,
    ATOM_DIM,
    GLOBAL_DIM,
    MEMORY_BOUNDED_ATOM_CHUNK,
    MEMORY_BOUNDED_ATOM_THRESHOLD,
    RUN_SECONDS_LIMIT,
    VRAM_LIMIT,
    IMPLEMENTATION_PATHS,
    STATE_DIM,
    ZERO_EXTERNAL_ACCESS,
    _inner_split,
    _logical_path,
    _model_registry,
    _registered_gpu_for_fold,
    _repository_path,
    _strings,
    _validate_artifact_identity,
    assert_zero_external_access,
    cuda_determinism_environment_contract,
    inference_weights_from_features,
    load_protocol,
    load_registered_bundle,
    sha256_file,
    validate_fold_manifest,
    verify_runtime_feature_audit,
)


DEFAULT_INVENTORY_PATH = Path(
    "artifacts/audits/tmlr_v6/first_batch_run_inventory.json"
)
DEFAULT_GATE_PATH = Path(
    "artifacts/audits/tmlr_v6/first_batch_completion_gate.json"
)
RUNTIME_AUDIT_PATH = RUNTIME_FEATURE_AUDIT_PATH
FIRST_BATCH_EVALUATION_PATH = Path(
    "artifacts/paper/source_data/tmlr_v6/first_batch/"
    "first_batch_evaluation_report.json"
)
RUNTIME_AUDIT_STATUS = RUNTIME_FEATURE_AUDIT_STATUS
FIRST_BATCH_EVALUATION_STATUS = "TMLR_V6_FIRST_BATCH_EVALUATION_PASS"
FIRST_BATCH_EVALUATION_OUTPUT_NAMES = frozenset({
    "first_batch_prediction_metrics.csv",
    "historical_c1_c2_prediction_metrics.csv",
    "first_batch_decision_metrics.csv",
    "first_batch_sparse_budget_metrics.csv",
    "first_batch_primary_group_bootstrap.csv",
    "historical_c1_c2_group_bootstrap.csv",
    "primary_metric_per_group.csv",
    "first_batch_run_inventory.csv",
    "historical_c1_c2_run_inventory.csv",
    "first_batch_metrics.json",
})
FIRST_BATCH_EVALUATION_IMPLEMENTATION_PATHS = (
    "scripts/evaluate_tmlr_v6_first_batch_fit.py",
    "scripts/materialize_tmlr_v6_historical_fit_oof.py",
    "src/rail3/__init__.py",
    "src/rail3/analysis/__init__.py",
    "src/rail3/analysis/tmlr_v6_first_batch.py",
    "src/rail3/analysis/tmlr_reviewer_revision.py",
    "src/rail3/analysis/m06f_decision_audit.py",
    "src/rail3/models/tmlr_v6/inventory.py",
    "src/rail3/models/tmlr_v6/training.py",
    "src/rail3/models/tmlr_v6/cost_profiles.py",
    "src/rail3/models/tmlr_v6/schema.py",
    "src/rail3/models/tmlr_v6/models.py",
    "src/rail3/models/tmlr_v6/__init__.py",
    "src/rail3/models/__init__.py",
    "src/rail3/models/m06b/__init__.py",
    "src/rail3/models/m06b/protocol.py",
    "src/rail3/models/m06e/__init__.py",
    "src/rail3/models/m06e/data.py",
    "src/rail3/models/m06e/training.py",
    "src/rail3/models/m06e/features.py",
    "src/rail3/models/m06e/losses.py",
    "src/rail3/models/m06e/models.py",
    "src/rail3/contracts/__init__.py",
    "src/rail3/contracts/records.py",
    "src/rail3/contracts/serialization.py",
    "src/rail3/evaluation/__init__.py",
    "src/rail3/evaluation/voc_v2a.py",
    "src/rail3/models/m06b/metrics.py",
    "src/rail3/regions/__init__.py",
    "src/rail3/regions/atomic.py",
    "src/rail3/regions/m06e_causal.py",
    "src/rail3/regions/voc_state_atomic.py",
    "configs/experiments/tmlr_v6_p0_prospective_capacity.json",
)


def expected_job_keys() -> set[tuple[str, str, int]]:
    return {
        (model_id, outer_fold, seed)
        for model_id in FIRST_BATCH_MODEL_IDS
        for outer_fold in FOLDS
        for seed in SEEDS
    }


def _validate_exact_run_tree(
    root: Path, expected_keys: set[tuple[str, str, int]],
) -> None:
    expected_run_dirs = {
        root / "S1364" / model_id / outer_fold / f"seed_{seed}"
        for model_id, outer_fold, seed in expected_keys
    }
    expected_dirs = {root / "S1364"}
    expected_dirs.update(path.parent for path in expected_run_dirs)
    expected_dirs.update(path.parent.parent for path in expected_run_dirs)
    expected_dirs.update(expected_run_dirs)
    expected_files = {
        run_dir / name
        for run_dir in expected_run_dirs
        for name in ("run-manifest.json", "predictions.npz", "checkpoint.pt")
    }
    observed_dirs: set[Path] = set()
    observed_files: set[Path] = set()
    for path in root.rglob("*") if root.exists() else ():
        if path.is_symlink():
            raise RuntimeError(f"linked TMLR V6 first-batch artifact: {path}")
        if path.is_dir():
            observed_dirs.add(path)
        elif path.is_file():
            observed_files.add(path)
        else:
            raise RuntimeError(f"non-regular TMLR V6 first-batch artifact: {path}")
    extra_dirs = sorted(observed_dirs - expected_dirs)
    missing_dirs = sorted(expected_dirs - observed_dirs)
    extra_files = sorted(observed_files - expected_files)
    missing_files = sorted(expected_files - observed_files)
    if extra_dirs or missing_dirs or extra_files or missing_files:
        raise RuntimeError(
            "TMLR V6 first-batch artifact tree drift: "
            f"extra_dirs={extra_dirs[:1]}, missing_dirs={missing_dirs[:1]}, "
            f"extra_files={extra_files[:1]}, missing_files={missing_files[:1]}"
        )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_artifact_scope_and_resources(
    *, repository: Path, manifest_path: Path, report: Mapping[str, Any],
) -> None:
    expected_prediction = manifest_path.parent / "predictions.npz"
    expected_checkpoint = manifest_path.parent / "checkpoint.pt"
    if (
        _repository_path(
            repository, Path(str(report.get("prediction", {}).get("path", ""))),
        ) != expected_prediction
        or _repository_path(
            repository, Path(str(report.get("checkpoint", {}).get("path", ""))),
        ) != expected_checkpoint
    ):
        raise RuntimeError("TMLR V6 run artifact path escapes its own run directory")
    if (
        not np.isfinite(float(report.get("gpu_wall_seconds", np.nan)))
        or float(report.get("gpu_wall_seconds", np.inf)) > RUN_SECONDS_LIMIT
        or int(report.get("peak_vram_bytes", VRAM_LIMIT)) >= VRAM_LIMIT
    ):
        raise RuntimeError("TMLR V6 run exceeds frozen resource limits")


def _assert_commit_ancestor(repository: Path, commit: str) -> None:
    if len(commit) != 40:
        raise RuntimeError("TMLR V6 implementation commit is malformed")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("TMLR V6 run implementation is outside HEAD ancestry")


def _verify_committed_sources(
    repository: Path, commit: str, declared: Mapping[str, Any],
) -> str:
    hashes: dict[str, str] = {}
    for relative, expected in sorted(declared.items()):
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        actual = hashlib.sha256(result.stdout).hexdigest()
        if actual != str(expected):
            raise RuntimeError("TMLR V6 committed implementation hash drift")
        hashes[str(relative)] = actual
    return hashlib.sha256(canonical_json_bytes(hashes)).hexdigest()


def reconstruct_state_prediction(
    atom_prediction: np.ndarray,
    inference_weights: np.ndarray,
    state_lengths: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(atom_prediction, dtype=np.float64)
    weights = np.asarray(inference_weights, dtype=np.float64)
    lengths = np.asarray(state_lengths, dtype=np.int64)
    if (
        prediction.ndim != 2
        or prediction.shape[1] != len(ACTIONS)
        or weights.shape != (len(prediction),)
        or lengths.ndim != 1
        or np.any(lengths <= 0)
        or int(lengths.sum()) != len(prediction)
    ):
        raise ValueError("TMLR V6 atomic reconstruction inputs are invalid")
    offsets = np.r_[0, np.cumsum(lengths)]
    return np.stack([
        np.sum(
            prediction[offsets[index]:offsets[index + 1]]
            * weights[offsets[index]:offsets[index + 1], None],
            axis=0,
        )
        for index in range(len(lengths))
    ])


def _same_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in {"f", "c"} or right.dtype.kind in {"f", "c"}:
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _validate_cross_role_contract(bundles: Mapping[str, Any]) -> None:
    baseline = bundles["F0_P"]
    for role_id in ("RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P"):
        candidate = bundles[role_id]
        # state_x intentionally contains representation-local region summary
        # statistics, so it may differ.  IDs, action descriptors, state-level
        # supervision, and feasibility must remain identical.
        for name in ("state_ids", "image_group_ids", "action_x16"):
            if not _same_with_nan(
                np.asarray(baseline.features[name]),
                np.asarray(candidate.features[name]),
            ):
                raise RuntimeError(
                    f"TMLR V6 cross-role prospective feature drift: {role_id}:{name}"
                )
        for name in ("state_ids", "state_target", "feasible"):
            if not _same_with_nan(
                np.asarray(baseline.targets[name]),
                np.asarray(candidate.targets[name]),
            ):
                raise RuntimeError(
                    f"TMLR V6 cross-role target drift: {role_id}:{name}"
                )


def _run_expected(
    *, report: Mapping[str, Any], model_id: str, outer_fold: str, seed: int,
    repository: Path, protocol: Mapping[str, Any], bundle: Any, fold_sha: str,
    runtime_audit_identity: Mapping[str, Any],
    train_groups: set[str], stop_groups: set[str], heldout_groups: set[str],
) -> dict[str, Any]:
    source_hashes = report.get("implementation_source_sha256", {})
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise RuntimeError("TMLR V6 run lacks implementation source identities")
    expected_paths = {path.as_posix() for path in IMPLEMENTATION_PATHS}
    if set(source_hashes) != expected_paths:
        raise RuntimeError("TMLR V6 run implementation path set drift")
    implementation_sha = hashlib.sha256(
        canonical_json_bytes(dict(source_hashes))
    ).hexdigest()
    if implementation_sha != report.get("implementation_sha256"):
        raise RuntimeError("TMLR V6 implementation aggregate identity drift")
    protocol_sha = sha256_file(_repository_path(repository, PROTOCOL_PATH))
    run_id = stable_id("tmlr_v6_first_batch_run", {
        "protocol_sha256": protocol_sha,
        "implementation_sha256": implementation_sha,
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "runtime_feature_audit_sha256": runtime_audit_identity["sha256"],
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "model_id": model_id,
        "outer_fold": outer_fold,
        "seed": seed,
    })
    group_ids = _strings(bundle.features["image_group_ids"], name="image_group_ids")
    return {
        "schema_version": "rail3.tmlr-v6.first-batch-run.v1",
        "status": "PASS",
        "phase": "fit_s1364_first_batch_outer_crossfit",
        "run_id": run_id,
        "model_id": model_id,
        "bundle_role_id": str(_model_registry(protocol)[model_id]["bundle"]),
        "geometry": bundle.registration.geometry,
        "outer_fold": outer_fold,
        "seed": seed,
        "scale": "S1364",
        "protocol_sha256": protocol_sha,
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "runtime_feature_audit": dict(runtime_audit_identity),
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "visible_device": _registered_gpu_for_fold(outer_fold),
        "feature_schema_id": "PROSPECTIVE_V1",
        "input_dimensions": {
            "atom": ATOM_DIM,
            "state": STATE_DIM,
            "action_projected": ACTION_DIM,
            "global": GLOBAL_DIM,
        },
        "physical_input_partitions": ["prospective_features", "targets"],
        "post_action_diagnostics_opened": False,
        "predictor_cost_input": False,
        "cost_objective_input": False,
        "inference_weight_source": "feature_area_fraction_full_raster",
        "target_weight_source": "valid_pixel_fraction_target_only",
        "target_weight_in_prediction_graph": False,
        "primary_training_loss": protocol["training"]["primary_training_loss"],
        "inner_stopping_loss": protocol["training"]["inner_stopping_loss"],
        "atom_target_auxiliary_loss": False,
        "parameter_count": int(_model_registry(protocol)[model_id]["parameters"]),
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


def _validate_cuda_determinism_contract(report: Mapping[str, Any]) -> None:
    if report.get(
        "cuda_determinism_environment"
    ) != cuda_determinism_environment_contract():
        raise RuntimeError("TMLR V6 run CUDA determinism environment drift")


def audit_first_batch_runs(
    *, repository: Path, root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Validate the complete immutable cross product, without external data."""

    repository = repository.resolve()
    protocol = load_protocol(repository)
    fold_manifest = validate_fold_manifest(repository, protocol)
    fold_sha = sha256_file(_repository_path(repository, FOLD_MANIFEST_PATH))
    runtime_audit_identity = verify_runtime_feature_audit(repository, protocol)
    root = _repository_path(repository, root)
    if root != _repository_path(repository, DEFAULT_OUTPUT_ROOT):
        raise RuntimeError("TMLR V6 inventory root is not registered")
    bundles = {
        role_id: load_registered_bundle(repository, role_id)
        for role_id in BUNDLE_REGISTRY
    }
    _validate_cross_role_contract(bundles)
    expected_keys = expected_job_keys()
    temporaries = sorted(
        path.relative_to(repository).as_posix()
        for path in root.rglob("*")
        if (
            path.name.endswith((".tmp", ".partial", ".incomplete"))
            or path.name.startswith(".run-manifest")
        )
    ) if root.exists() else []
    if temporaries:
        raise RuntimeError(f"incomplete TMLR V6 first-batch artifacts: {temporaries[:3]}")
    _validate_exact_run_tree(root, expected_keys)
    observed: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    run_ids: set[str] = set()
    implementation_commits: set[str] = set()
    source_contracts: set[str] = set()
    state_coverage: dict[tuple[str, int], list[np.ndarray]] = {}
    atom_coverage: dict[tuple[str, int], list[np.ndarray]] = {}
    total_prediction_bytes = 0
    total_checkpoint_bytes = 0
    gpu_wall_seconds = 0.0
    peak_vram_bytes = 0
    for manifest_path in sorted(root.rglob("run-manifest.json")) if root.exists() else []:
        report = _read(manifest_path)
        _validate_cuda_determinism_contract(report)
        key = (
            str(report.get("model_id")),
            str(report.get("outer_fold")),
            int(report.get("seed", -1)),
        )
        if key in observed:
            raise RuntimeError(f"duplicate TMLR V6 first-batch job: {key}")
        if key not in expected_keys:
            raise RuntimeError(f"unexpected TMLR V6 first-batch job: {key}")
        model_id, outer_fold, seed = key
        expected_path = (
            root / "S1364" / model_id / outer_fold / f"seed_{seed}"
            / "run-manifest.json"
        )
        if manifest_path != expected_path:
            raise RuntimeError("TMLR V6 run manifest is outside the exact inventory tree")
        role_id = str(_model_registry(protocol)[model_id]["bundle"])
        bundle = bundles[role_id]
        train_groups, stop_groups, heldout_groups = _inner_split(
            fold_manifest, outer_fold, seed,
        )
        expected = _run_expected(
            report=report,
            model_id=model_id,
            outer_fold=outer_fold,
            seed=seed,
            repository=repository,
            protocol=protocol,
            bundle=bundle,
            fold_sha=fold_sha,
            runtime_audit_identity=runtime_audit_identity,
            train_groups=train_groups,
            stop_groups=stop_groups,
            heldout_groups=heldout_groups,
        )
        drift = {
            name: (report.get(name), value)
            for name, value in expected.items()
            if report.get(name) != value
        }
        if drift:
            raise RuntimeError(
                f"TMLR V6 run identity drift at {manifest_path}: {drift}"
            )
        assert_zero_external_access(report.get("heldout_access", {}))
        if (
            not np.isfinite(float(report.get("inner_stop_loss", np.nan)))
            or int(report.get("best_epoch", 0)) <= 0
            or int(report.get("epochs_completed", 0)) <= 0
        ):
            raise RuntimeError("TMLR V6 run did not finish with finite training state")
        expected_memory_mode = (
            f"ACTIVATION_CHECKPOINTED_ATOM_CHUNKS_{MEMORY_BOUNDED_ATOM_CHUNK}"
            if model_id in {"R1_P", "RECT_P", "UNION_P"}
            and len(bundle.features["atom_ids"]) >= MEMORY_BOUNDED_ATOM_THRESHOLD
            else "FULL_BATCH"
        )
        if report.get("execution_memory_mode") != expected_memory_mode:
            raise RuntimeError("TMLR V6 activation-checkpoint execution drift")
        commit = str(report.get("implementation_commit"))
        _assert_commit_ancestor(repository, commit)
        source_sha = _verify_committed_sources(
            repository, commit, report["implementation_source_sha256"],
        )
        if source_sha != report["implementation_sha256"]:
            raise RuntimeError("TMLR V6 committed source aggregate drift")
        _validate_run_artifact_scope_and_resources(
            repository=repository, manifest_path=manifest_path, report=report,
        )
        _validate_artifact_identity(
            repository, report["prediction"], kind="prediction",
        )
        _validate_artifact_identity(
            repository, report["checkpoint"], kind="checkpoint",
        )
        prediction_path = _repository_path(
            repository, Path(str(report["prediction"]["path"])),
        )
        is_atomic = model_id in {"R1_P", "RECT_P", "UNION_P"}
        required = {
            "state_index", "state_id", "state_prediction",
        }
        if is_atomic:
            required |= {"atom_index", "atom_id", "atom_prediction"}
        with np.load(prediction_path, allow_pickle=False) as payload:
            if set(payload.files) != required:
                raise RuntimeError("TMLR V6 prediction payload schema drift")
            group_ids = _strings(
                bundle.features["image_group_ids"], name="image_group_ids",
            )
            expected_state_index = np.flatnonzero(
                np.isin(group_ids, tuple(heldout_groups))
            ).astype(np.int32)
            state_index = np.asarray(payload["state_index"])
            state_id = _strings(payload["state_id"], name="prediction state_id")
            state_prediction = np.asarray(payload["state_prediction"])
            if (
                not np.array_equal(state_index, expected_state_index)
                or not np.array_equal(
                    state_id,
                    _strings(bundle.features["state_ids"], name="state_ids")[state_index],
                )
                or state_prediction.shape != (len(state_index), len(ACTIONS))
                or not np.isfinite(state_prediction).all()
            ):
                raise RuntimeError("TMLR V6 held-out state prediction coverage drift")
            state_coverage.setdefault((model_id, seed), []).append(state_id.copy())
            if is_atomic:
                lengths = np.asarray(bundle.features["state_lengths"], dtype=np.int64)
                offsets = np.r_[0, np.cumsum(lengths)]
                expected_atom_index = np.concatenate([
                    np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
                    for index in state_index
                ]).astype(np.int32)
                atom_index = np.asarray(payload["atom_index"])
                atom_id = _strings(payload["atom_id"], name="prediction atom_id")
                atom_prediction = np.asarray(payload["atom_prediction"])
                if (
                    not np.array_equal(atom_index, expected_atom_index)
                    or not np.array_equal(
                        atom_id,
                        _strings(bundle.features["atom_ids"], name="atom_ids")[
                            atom_index
                        ],
                    )
                    or atom_prediction.shape != (len(atom_index), len(ACTIONS))
                    or not np.isfinite(atom_prediction).all()
                ):
                    raise RuntimeError("TMLR V6 held-out atom prediction coverage drift")
                reconstructed = reconstruct_state_prediction(
                    atom_prediction,
                    inference_weights_from_features(
                        np.asarray(bundle.features["atom_x"])[atom_index],
                        lengths[state_index],
                    ),
                    lengths[state_index],
                )
                if not np.allclose(
                    reconstructed,
                    state_prediction.astype(np.float64),
                    rtol=0.0,
                    atol=2e-6,
                ):
                    raise RuntimeError("TMLR V6 saved additive prediction drift")
                atom_coverage.setdefault((model_id, seed), []).append(atom_id.copy())
        run_id = str(report["run_id"])
        if run_id in run_ids:
            raise RuntimeError("duplicate TMLR V6 first-batch run ID")
        run_ids.add(run_id)
        implementation_commits.add(commit)
        source_contracts.add(source_sha)
        total_prediction_bytes += int(report["prediction"]["bytes"])
        total_checkpoint_bytes += int(report["checkpoint"]["bytes"])
        gpu_wall_seconds += float(report["gpu_wall_seconds"])
        peak_vram_bytes = max(peak_vram_bytes, int(report["peak_vram_bytes"]))
        observed[key] = manifest_path, report
    missing = sorted(expected_keys - set(observed))
    if missing:
        raise RuntimeError(
            f"missing TMLR V6 first-batch jobs: {missing[:3]} ({len(missing)} total)"
        )
    for model_id in FIRST_BATCH_MODEL_IDS:
        role_id = str(_model_registry(protocol)[model_id]["bundle"])
        bundle = bundles[role_id]
        expected_states = set(
            _strings(bundle.features["state_ids"], name="state_ids").tolist()
        )
        for seed in SEEDS:
            state_ids = np.concatenate(state_coverage[(model_id, seed)]).tolist()
            if len(state_ids) != len(expected_states) or set(state_ids) != expected_states:
                raise RuntimeError("TMLR V6 state OOF coverage is not exact")
            if model_id in {"R1_P", "RECT_P", "UNION_P"}:
                expected_atoms = set(
                    _strings(bundle.features["atom_ids"], name="atom_ids").tolist()
                )
                atom_ids = np.concatenate(atom_coverage[(model_id, seed)]).tolist()
                if len(atom_ids) != len(expected_atoms) or set(atom_ids) != expected_atoms:
                    raise RuntimeError("TMLR V6 atom OOF coverage is not exact")
    by_model = {
        model_id: sum(key[0] == model_id for key in observed)
        for model_id in FIRST_BATCH_MODEL_IDS
    }
    if len(implementation_commits) != 1 or len(source_contracts) != 1:
        raise RuntimeError(
            "TMLR V6 first batch mixes training implementations"
        )
    return {
        "schema_version": "rail3.tmlr-v6.first-batch-run-inventory.v1",
        "status": "TMLR_V6_FIRST_BATCH_RUN_INVENTORY_PASS",
        "protocol_sha256": sha256_file(_repository_path(repository, PROTOCOL_PATH)),
        "fold_manifest_sha256": fold_sha,
        "runtime_feature_audit": runtime_audit_identity,
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "expected_runs": 75,
        "completed_valid_runs": len(observed),
        "missing_runs": 0,
        "duplicate_runs": 0,
        "unexpected_runs": 0,
        "temporary_or_incomplete_runs": 0,
        "by_model": by_model,
        "bundle_identities": {
            role_id: {
                "feature_manifest_sha256": bundle.feature_manifest_sha256,
                "feature_array_sha256": bundle.feature_array_sha256,
                "target_manifest_sha256": bundle.target_manifest_sha256,
                "target_array_sha256": bundle.target_array_sha256,
            }
            for role_id, bundle in bundles.items()
        },
        "implementation_commits": sorted(implementation_commits),
        "implementation_sha256": sorted(source_contracts),
        "total_prediction_bytes": total_prediction_bytes,
        "total_checkpoint_bytes": total_checkpoint_bytes,
        "gpu_wall_seconds": gpu_wall_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "checks": {
            "exact_cross_product_75": len(observed) == 75,
            "physical_feature_target_separation": True,
            "prospective_schema_only": True,
            "runtime_feature_count_zero": True,
            "post_action_diagnostic_access_zero": True,
            "ground_truth_target_weights_absent_from_prediction_graph": True,
            "same_state_target_across_roles": True,
            "same_action_feasibility_across_roles": True,
            "exact_state_oof_coverage": True,
            "exact_atom_oof_coverage_for_atomic_models": True,
            "saved_additive_reconstruction_exact": True,
            "all_declared_files_hash_valid": True,
            "all_run_ids_deterministic_and_unique": len(run_ids) == 75,
            "all_training_states_finite": True,
            "single_training_implementation_commit": len(implementation_commits) == 1,
            "single_training_source_contract": len(source_contracts) == 1,
            "exact_cuda_determinism_environment": True,
            "protected_access_zero": True,
        },
    }


def validate_supporting_report(
    *, repository: Path, path: Path, expected_status: str,
    expected_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    actual = _repository_path(repository, path)
    report = _read(actual)
    if report.get("status") != expected_status:
        raise RuntimeError(f"TMLR V6 supporting report is not PASS: {path}")
    access = report.get("heldout_access", report.get("protected_access", {}))
    assert_zero_external_access(access)
    if expected_inventory_sha256 is not None and (
        report.get("first_batch_run_inventory_sha256")
        != expected_inventory_sha256
        or int(report.get("completed_valid_runs", -1)) != 75
        or not bool(report.get("metric_pipeline_pass"))
    ):
        raise RuntimeError("TMLR V6 first-batch evaluation is not inventory-bound")
    if expected_inventory_sha256 is not None:
        checks = report.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or not all(bool(value) for value in checks.values())
        ):
            raise RuntimeError("TMLR V6 evaluation checks are incomplete")
        if report.get("schema_version") != "rail3.tmlr-v6.first-batch-fit-evaluation.v1":
            raise RuntimeError("TMLR V6 evaluation report schema drift")
        protocol_identity = report.get("protocol")
        if (
            not isinstance(protocol_identity, Mapping)
            or set(protocol_identity) != {"path", "sha256"}
            or protocol_identity.get("path") != PROTOCOL_PATH.as_posix()
            or protocol_identity.get("sha256")
            != sha256_file(_repository_path(repository, PROTOCOL_PATH))
        ):
            raise RuntimeError("TMLR V6 evaluation protocol identity drift")
        inventory_identity = report.get("inventory")
        if (
            not isinstance(inventory_identity, Mapping)
            or set(inventory_identity) != {"path", "bytes", "sha256"}
            or inventory_identity.get("path") != DEFAULT_INVENTORY_PATH.as_posix()
            or inventory_identity.get("sha256") != expected_inventory_sha256
        ):
            raise RuntimeError("TMLR V6 evaluation inventory identity drift")
        _validate_artifact_identity(
            repository, inventory_identity, kind="evaluation-bound run inventory",
        )
        implementation_commit = str(report.get("implementation_commit", ""))
        implementation_sources = report.get("implementation_source_sha256")
        if (
            not isinstance(implementation_sources, Mapping)
            or set(map(str, implementation_sources))
            != set(FIRST_BATCH_EVALUATION_IMPLEMENTATION_PATHS)
        ):
            raise RuntimeError("TMLR V6 evaluation implementation identity is absent")
        _assert_commit_ancestor(repository, implementation_commit)
        implementation_sha = _verify_committed_sources(
            repository, implementation_commit, implementation_sources,
        )
        if implementation_sha != report.get("implementation_combined_sha256"):
            raise RuntimeError("TMLR V6 evaluation implementation aggregate drift")
        outputs = report.get("outputs")
        if (
            not isinstance(outputs, Mapping)
            or set(map(str, outputs)) != FIRST_BATCH_EVALUATION_OUTPUT_NAMES
        ):
            raise RuntimeError("TMLR V6 evaluation output registry drift")
        observed_paths: set[Path] = set()
        for name in sorted(FIRST_BATCH_EVALUATION_OUTPUT_NAMES):
            identity = outputs[name]
            if not isinstance(identity, Mapping) or set(identity) != {
                "path", "bytes", "sha256", "rows",
            }:
                raise RuntimeError(f"TMLR V6 evaluation output schema drift: {name}")
            output_path = _repository_path(
                repository, Path(str(identity.get("path", ""))),
            )
            if (
                output_path.parent != actual.parent
                or output_path.name != name
                or output_path in observed_paths
                or int(identity.get("rows", -1)) < 0
            ):
                raise RuntimeError(f"TMLR V6 evaluation output path drift: {name}")
            _validate_artifact_identity(
                repository, identity, kind=f"evaluation output {name}",
            )
            observed_paths.add(output_path)
    return report


def build_first_batch_gate(
    *, repository: Path, inventory_path: Path = DEFAULT_INVENTORY_PATH,
    runtime_audit_path: Path = RUNTIME_AUDIT_PATH,
    evaluation_path: Path = FIRST_BATCH_EVALUATION_PATH,
) -> dict[str, Any]:
    """Require schema audit, 75 valid runs, and the metric-pipeline report."""

    repository = repository.resolve()
    protocol = load_protocol(repository)
    runtime_audit_identity = verify_runtime_feature_audit(repository, protocol)
    if _repository_path(repository, runtime_audit_path) != _repository_path(
        repository, RUNTIME_AUDIT_PATH,
    ):
        raise RuntimeError("TMLR V6 runtime-audit path is not registered")
    if _repository_path(repository, evaluation_path) != _repository_path(
        repository, FIRST_BATCH_EVALUATION_PATH,
    ):
        raise RuntimeError("TMLR V6 evaluation path is not registered")
    inventory_file = _repository_path(repository, inventory_path)
    inventory = _read(inventory_file)
    recomputed_inventory = audit_first_batch_runs(repository=repository)
    if inventory_file.read_bytes() != canonical_json_bytes(recomputed_inventory) + b"\n":
        raise RuntimeError("TMLR V6 first-batch inventory is not reproducible")
    inventory = recomputed_inventory
    if (
        inventory.get("status") != "TMLR_V6_FIRST_BATCH_RUN_INVENTORY_PASS"
        or int(inventory.get("completed_valid_runs", -1)) != 75
        or not all(bool(value) for value in inventory.get("checks", {}).values())
    ):
        raise RuntimeError("TMLR V6 first-batch inventory is not complete")
    assert_zero_external_access(inventory.get("heldout_access", {}))
    if inventory.get("runtime_feature_audit") != runtime_audit_identity:
        raise RuntimeError("TMLR V6 run inventory runtime-audit identity drift")
    inventory_sha = sha256_file(inventory_file)
    runtime = validate_supporting_report(
        repository=repository,
        path=runtime_audit_path,
        expected_status=RUNTIME_AUDIT_STATUS,
    )
    runtime_schema = runtime.get("prospective_feature_schema", {})
    runtime_checks = runtime.get("checks", {})
    if (
        runtime_schema.get("id") != "PROSPECTIVE_V1"
        or int(runtime_schema.get("action_dimension", -1)) != ACTION_DIM
        or runtime_schema.get("predictor_cost_input") is not False
        or runtime_schema.get("post_action_mask_input") is not False
        or runtime_schema.get("future_action_outcome_input") is not False
        or not bool(runtime_checks.get("prospective_action_dimension_16"))
        or not bool(runtime_checks.get("prospective_cost_api_physically_separate"))
        or not bool(runtime_checks.get("prospective_full_raster_guard_present"))
    ):
        raise RuntimeError("TMLR V6 prospective runtime-audit contract drift")
    evaluation = validate_supporting_report(
        repository=repository,
        path=evaluation_path,
        expected_status=FIRST_BATCH_EVALUATION_STATUS,
        expected_inventory_sha256=inventory_sha,
    )
    gate_status = str(protocol["stage_gates"]["first_batch"])
    return {
        "schema_version": "rail3.tmlr-v6.first-batch-completion-gate.v1",
        "status": gate_status,
        "protocol_sha256": sha256_file(_repository_path(repository, PROTOCOL_PATH)),
        "inventory": {
            "path": _logical_path(repository, inventory_file),
            "bytes": inventory_file.stat().st_size,
            "sha256": inventory_sha,
        },
        "runtime_feature_audit": {
            **runtime_audit_identity,
            "status": runtime["status"],
        },
        "first_batch_evaluation": {
            "path": _logical_path(repository, evaluation_path),
            "sha256": sha256_file(_repository_path(repository, evaluation_path)),
            "status": evaluation["status"],
        },
        "completed_valid_runs": 75,
        "second_batch_authorized_by_gate": True,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "gate_id": stable_id("tmlr_v6_first_batch_gate", {
            "protocol_sha256": sha256_file(_repository_path(repository, PROTOCOL_PATH)),
            "inventory_sha256": inventory_sha,
            "runtime_audit_sha256": sha256_file(
                _repository_path(repository, runtime_audit_path)
            ),
            "evaluation_sha256": sha256_file(
                _repository_path(repository, evaluation_path)
            ),
        }),
        "checks": {
            "prospective_runtime_audit_pass": True,
            "exact_cross_product_75_pass": True,
            "all_training_finite": True,
            "metric_pipeline_pass": True,
            "protected_access_zero": True,
        },
    }


__all__ = [
    "DEFAULT_GATE_PATH",
    "DEFAULT_INVENTORY_PATH",
    "audit_first_batch_runs",
    "build_first_batch_gate",
    "expected_job_keys",
    "reconstruct_state_prediction",
]
