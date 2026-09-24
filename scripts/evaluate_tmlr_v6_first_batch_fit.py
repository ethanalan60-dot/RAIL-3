#!/usr/bin/env python3
"""Evaluate the frozen TMLR V6 first batch on S1364 FIT OOF only.

This entry point never trains a model, invokes SAM, imports a protected reader,
or opens an external split.  Formal execution requires the complete audited
75-run inventory and the complete historical FIT OOF inputs used by C1/C2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rail3.analysis.m06f_decision_audit import validate_exact_oof_indices
from rail3.analysis.tmlr_reviewer_revision import committed_source_identity
from rail3.analysis.tmlr_v6_first_batch import (
    HISTORICAL_COMPARISON_METADATA,
    HISTORICAL_CONTRASTS,
    PRIMARY_CONTRASTS,
    CostSurface,
    ResidualSurface,
    annotate_historical_comparison,
    assert_aligned_surfaces,
    budget_metrics,
    complete_case_state_mask,
    decision_metrics,
    paired_primary_statistics,
    prediction_metrics,
    strict_seed_ensemble,
    validate_evaluation_protocol,
    validate_report_contract,
)
from rail3.contracts import canonical_json_bytes
from rail3.models.m06e.data import TensorBundle
from rail3.models.m06e.training import _scale_view
from rail3.models.tmlr_v6.cost_profiles import (
    assert_fit_only_path,
    csv_bytes as cost_csv_bytes,
    validate_cost_profile_implementation_identity,
    validate_cost_profile_source_identities,
)
from rail3.models.tmlr_v6.inventory import (
    DEFAULT_INVENTORY_PATH,
    FIRST_BATCH_EVALUATION_STATUS,
    audit_first_batch_runs,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    BUNDLE_REGISTRY,
    DEFAULT_OUTPUT_ROOT,
    FIRST_BATCH_MODEL_IDS,
    FOLD_MANIFEST_PATH,
    FOLDS,
    SEEDS,
    ZERO_EXTERNAL_ACCESS,
    _model_registry,
    _repository_path as _validated_repository_path,
    _strings,
    assert_legacy_zero_external_access,
    assert_zero_external_access,
    load_protocol,
    load_registered_bundle,
    sha256_file,
    validate_fold_manifest,
)


SCHEMA_VERSION = "rail3.tmlr-v6.first-batch-fit-evaluation.v1"
DEFAULT_OUTPUT = Path("artifacts/paper/source_data/tmlr_v6/first_batch")
EXPECTED_OUTPUT_NAMES = frozenset({
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
DEFAULT_COST_JSON = Path("artifacts/source_data/tmlr_v6/action_type_cost_profiles.json")
DEFAULT_COST_CSV = Path("artifacts/source_data/tmlr_v6/action_type_cost_profiles.csv")
EXPECTED_COST_PROFILE_CHECKS = {
    "feasible_runtime_strictly_positive": True,
    "implementation_committed_clean": True,
    "infeasible_runtime_exactly_zero": True,
    "label_free_trajectory_reports_only": True,
    "outer_heldout_excluded_per_fold": True,
    "predictor_input_used": False,
    "profile_seed_independent": True,
    "protected_read_count": 0,
    "protocol_registered_source_set_verified": True,
    "stop_exactly_zero": True,
}
LEGACY_BUNDLE = Path("artifacts/voc2012/m06e/atomic-bundle-v2")
LEGACY_MIRROR_ROOT = Path("data/v6_fit_only/historical_oof_mirror")
LEGACY_CANONICAL_RUN_ROOT = Path(
    "artifacts/voc2012/m06e/crossfit-v2-final/S_MAX"
)
LEGACY_RUN_ROOT = LEGACY_MIRROR_ROOT / LEGACY_CANONICAL_RUN_ROOT
LEGACY_INVENTORY = Path(
    "artifacts/audits/tmlr_v6/historical_fit_oof_inventory.json"
)
LEGACY_INVENTORY_SCHEMA = "rail3.tmlr-v6.historical-fit-oof-materialization.v1"
LEGACY_INVENTORY_STATUS = "TMLR_V6_HISTORICAL_FIT_OOF_COMPLETE"
LEGACY_MODELS = {
    "RETRO_R0_FIT_OOF": "R0_GLOBAL_RESIDUAL",
    "RETRO_R1_FIT_OOF": "R1_ATOMIC_RESIDUAL",
}
LEGACY_BUNDLE_IDENTITIES = {
    "manifest.json": {
        "bytes": 6_857_557,
        "sha256": "473c12e6a4b1eab73b4aa621ec7c0f7432ad0f99fa0dcf1d45d04e199c9e6e8f",
    },
    "arrays.npz": {
        "bytes": 3_699_414,
        "sha256": "ffca683aa27114cc4e4732397589e8145b7b1665e358331fde3086ef05e9e3c7",
    },
}
IMPLEMENTATION_PATHS = (
    Path("scripts/evaluate_tmlr_v6_first_batch_fit.py"),
    Path("scripts/materialize_tmlr_v6_historical_fit_oof.py"),
    Path("src/rail3/__init__.py"),
    Path("src/rail3/analysis/__init__.py"),
    Path("src/rail3/analysis/tmlr_v6_first_batch.py"),
    Path("src/rail3/analysis/tmlr_reviewer_revision.py"),
    Path("src/rail3/analysis/m06f_decision_audit.py"),
    Path("src/rail3/models/tmlr_v6/inventory.py"),
    Path("src/rail3/models/tmlr_v6/training.py"),
    Path("src/rail3/models/tmlr_v6/cost_profiles.py"),
    Path("src/rail3/models/tmlr_v6/schema.py"),
    Path("src/rail3/models/tmlr_v6/models.py"),
    Path("src/rail3/models/tmlr_v6/__init__.py"),
    Path("src/rail3/models/__init__.py"),
    Path("src/rail3/models/m06b/__init__.py"),
    Path("src/rail3/models/m06e/data.py"),
    Path("src/rail3/models/m06b/protocol.py"),
    Path("src/rail3/models/m06e/__init__.py"),
    Path("src/rail3/models/m06e/training.py"),
    Path("src/rail3/models/m06e/features.py"),
    Path("src/rail3/models/m06e/losses.py"),
    Path("src/rail3/models/m06e/models.py"),
    Path("src/rail3/contracts/__init__.py"),
    Path("src/rail3/contracts/records.py"),
    Path("src/rail3/contracts/serialization.py"),
    Path("src/rail3/evaluation/__init__.py"),
    Path("src/rail3/evaluation/voc_v2a.py"),
    Path("src/rail3/models/m06b/metrics.py"),
    Path("src/rail3/regions/__init__.py"),
    Path("src/rail3/regions/atomic.py"),
    Path("src/rail3/regions/m06e_causal.py"),
    Path("src/rail3/regions/voc_state_atomic.py"),
    Path("configs/experiments/tmlr_v6_p0_prospective_capacity.json"),
)


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 evaluation must run from the worktree root")
    return root


def _repository_path(repository: Path, path: Path) -> Path:
    """Use the training boundary's lexical, symlink, and protected checks."""

    return _validated_repository_path(repository, path)


def _logical(repository: Path, path: Path) -> str:
    return _repository_path(repository, path).relative_to(
        repository.resolve()
    ).as_posix()


def _read_json(repository: Path, path: Path) -> dict[str, Any]:
    actual = _repository_path(repository, path)
    if actual.is_symlink() or not actual.is_file():
        raise FileNotFoundError(actual)
    value = json.loads(actual.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"TMLR V6 JSON input is not an object: {path}")
    return value


def _verify_identity(path: Path, identity: Mapping[str, Any], *, name: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != str(identity.get("sha256"))
    ):
        raise RuntimeError(f"TMLR V6 {name} identity drift")


def _artifact_identity(repository: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _logical(repository, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _protocol_registered_artifact(
    *, repository: Path, requested_path: Path,
    registration: Mapping[str, Any] | None, name: str,
) -> tuple[Path, dict[str, Any]]:
    """Require one exact path/bytes/SHA identity frozen in the protocol."""

    if not isinstance(registration, Mapping) or set(registration) != {
        "path", "bytes", "sha256",
    }:
        raise RuntimeError(
            f"TMLR_V6_PROTOCOL_ARTIFACT_IDENTITY_REQUIRED: {name}"
        )
    path_text = registration.get("path")
    byte_count = registration.get("bytes")
    digest = registration.get("sha256")
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
        or ".." in Path(path_text).parts
        or Path(path_text).as_posix() != path_text
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"TMLR V6 {name} protocol identity is malformed")
    registered = _repository_path(repository, Path(path_text))
    requested = _repository_path(repository, requested_path)
    if requested != registered:
        raise RuntimeError(f"TMLR V6 {name} path is not protocol-registered")
    identity = {
        "path": path_text,
        "bytes": byte_count,
        "sha256": digest,
    }
    _verify_identity(registered, identity, name=f"protocol-registered {name}")
    return registered, identity


def _recorded_artifact(
    repository: Path, item: Mapping[str, Any], *, name: str,
    path_key: str = "path", relative_base: Path = Path(),
) -> Path:
    recorded = Path(str(item.get(path_key, "")))
    if recorded.is_absolute() or ".." in recorded.parts or not recorded.parts:
        raise RuntimeError(f"TMLR V6 {name} recorded path is unsafe")
    path = _repository_path(repository, relative_base / recorded)
    _verify_identity(path, item, name=name)
    return path


def _prospective_feature_plan_feasibility(
    action_x16: np.ndarray, stored_feasible: np.ndarray,
) -> np.ndarray:
    """Derive plan feasibility only from the two pre-action feature flags."""

    action = np.asarray(action_x16)
    stored = np.asarray(stored_feasible)
    if (
        action.ndim != 3
        or action.shape[1:] != (len(ACTIONS), 16)
        or stored.shape != action.shape[:2]
        or stored.dtype.kind != "b"
    ):
        raise RuntimeError("TMLR V6 prospective feasibility geometry drift")
    feasible_value = action[:, :, 0]
    precondition_value = action[:, :, 7]
    if (
        not np.isin(feasible_value, (0.0, 1.0)).all()
        or not np.isin(precondition_value, (0.0, 1.0)).all()
        or not np.array_equal(feasible_value, precondition_value)
    ):
        raise RuntimeError("TMLR V6 pre-action feasibility feature flags drift")
    derived = feasible_value.astype(bool)
    if not np.array_equal(derived, stored):
        raise RuntimeError("TMLR V6 feature/target plan-feasibility join drift")
    return derived


def _legacy_feature_plan_feasibility(
    action_x: np.ndarray, stored_feasible: np.ndarray,
) -> np.ndarray:
    """Recover immutable RETRO plan feasibility from its feature column 0."""

    action = np.asarray(action_x)
    stored = np.asarray(stored_feasible)
    if (
        action.ndim != 3
        or action.shape[:2] != stored.shape
        or action.shape[2] < 1
        or stored.dtype.kind != "b"
        or not np.isin(action[:, :, 0], (0.0, 1.0)).all()
    ):
        raise RuntimeError("TMLR V6 historical pre-action feasibility geometry drift")
    derived = action[:, :, 0].astype(bool)
    if not np.array_equal(derived, stored):
        raise RuntimeError("TMLR V6 historical feature/target feasibility drift")
    return derived


def load_first_batch_surfaces(
    *, repository: Path, protocol: Mapping[str, Any], fold_manifest: Mapping[str, Any],
    run_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, ResidualSurface], list[dict[str, Any]]]:
    """Load exactly five models x five folds x three seeds."""

    root = _repository_path(repository, run_root)
    if root != _repository_path(repository, DEFAULT_OUTPUT_ROOT):
        raise RuntimeError("TMLR V6 first-batch run root is not registered")
    bundles = {
        role_id: load_registered_bundle(repository, role_id)
        for role_id in BUNDLE_REGISTRY
    }
    group_to_fold = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in fold_manifest["groups"]
    }
    surfaces: dict[str, ResidualSurface] = {}
    inventory_rows: list[dict[str, Any]] = []
    for model_id in FIRST_BATCH_MODEL_IDS:
        role_id = str(_model_registry(protocol)[model_id]["bundle"])
        bundle = bundles[role_id]
        state_ids = _strings(bundle.features["state_ids"], name="state_ids")
        groups = _strings(bundle.features["image_group_ids"], name="image_group_ids")
        plan_feasible = _prospective_feature_plan_feasibility(
            bundle.features["action_x16"], bundle.targets["feasible"],
        )
        fold_ids = np.asarray([group_to_fold[str(group)] for group in groups])
        by_seed = {
            seed: np.full(
                np.asarray(bundle.targets["state_target"]).shape,
                np.nan, dtype=np.float64,
            )
            for seed in SEEDS
        }
        covered = {
            seed: np.zeros(len(state_ids), dtype=bool) for seed in SEEDS
        }
        for fold in FOLDS:
            expected_index = np.flatnonzero(fold_ids == fold).astype(np.int32)
            for seed in SEEDS:
                manifest_path = (
                    root / "S1364" / model_id / fold / f"seed_{seed}"
                    / "run-manifest.json"
                )
                report = _read_json(
                    repository, manifest_path.relative_to(repository),
                )
                if (
                    report.get("status") != "PASS"
                    or report.get("model_id") != model_id
                    or report.get("outer_fold") != fold
                    or int(report.get("seed", -1)) != seed
                    or report.get("bundle_role_id") != role_id
                    or report.get("fit_only") is not True
                    or report.get("protected_reader_invoked") is not False
                    or report.get("post_action_diagnostics_opened") is not False
                ):
                    raise RuntimeError("TMLR V6 first-batch run manifest drift")
                assert_zero_external_access(report.get("heldout_access", {}))
                prediction_path = _recorded_artifact(
                    repository, report["prediction"], name="first-batch prediction",
                )
                with np.load(prediction_path, allow_pickle=False) as payload:
                    observed_index = np.asarray(payload["state_index"], dtype=np.int32)
                    observed_ids = _strings(
                        payload["state_id"], name="prediction state_id",
                    )
                    prediction = np.asarray(
                        payload["state_prediction"], dtype=np.float64,
                    )
                    if (
                        not np.array_equal(observed_index, expected_index)
                        or not np.array_equal(observed_ids, state_ids[expected_index])
                        or prediction.shape != (len(expected_index), len(ACTIONS))
                        or not np.isfinite(prediction).all()
                        or covered[seed][observed_index].any()
                    ):
                        raise RuntimeError("TMLR V6 OOF state stitching drift")
                    by_seed[seed][observed_index] = prediction
                    covered[seed][observed_index] = True
                inventory_rows.append({
                    "model": model_id,
                    "outer_fold": fold,
                    "seed": seed,
                    "run_id": str(report["run_id"]),
                    "manifest_path": _logical(repository, manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "prediction_path": str(report["prediction"]["path"]),
                    "prediction_sha256": str(report["prediction"]["sha256"]),
                    "checkpoint_path": str(report["checkpoint"]["path"]),
                    "checkpoint_sha256": str(report["checkpoint"]["sha256"]),
                    "implementation_commit": str(report["implementation_commit"]),
                    "implementation_sha256": str(report["implementation_sha256"]),
                })
        if any(not value.all() for value in covered.values()):
            raise RuntimeError("TMLR V6 five-fold OOF coverage is incomplete")
        surfaces[model_id] = ResidualSurface(
            model_id=model_id,
            prediction=strict_seed_ensemble(by_seed),
            target=np.asarray(bundle.targets["state_target"], dtype=np.float64),
            feasible=plan_feasible,
            groups=groups,
            state_ids=tuple(state_ids.tolist()),
        ).validated()
    if len(inventory_rows) != 75:
        raise RuntimeError("TMLR V6 first-batch evaluator did not load 75 runs")
    assert_aligned_surfaces(surfaces)
    return surfaces, inventory_rows


def load_cost_surface(
    *, repository: Path, protocol: Mapping[str, Any], groups: Sequence[str],
    fold_manifest: Mapping[str, Any], json_path: Path = DEFAULT_COST_JSON,
    csv_path: Path = DEFAULT_COST_CSV,
) -> tuple[CostSurface, dict[str, Any]]:
    """Load fold-specific, seed-invariant cost profiles outside predictors."""

    cost_registration = protocol.get("cost_profile", {})
    if not isinstance(cost_registration, Mapping):
        cost_registration = {}
    artifact_registration = cost_registration.get("artifacts", {})
    if not isinstance(artifact_registration, Mapping):
        artifact_registration = {}
    actual_json, json_identity = _protocol_registered_artifact(
        repository=repository, requested_path=json_path,
        registration=artifact_registration.get("json"),
        name="action cost JSON",
    )
    actual_csv, csv_identity = _protocol_registered_artifact(
        repository=repository, requested_path=csv_path,
        registration=artifact_registration.get("csv"),
        name="action cost CSV",
    )
    payload = _read_json(repository, json_path)
    if (
        payload.get("schema_version")
        != "rail3.tmlr-v6.action-type-cost-profiles.v1"
        or payload.get("status") != "TMLR_V6_FIT_ONLY_COST_PROFILES_PASS"
        or payload.get("actions") != list(ACTIONS)
        or payload.get("profile_seed_dependency") is not False
        or int(payload.get("fit_group_count", -1)) != 1364
        or int(payload.get("class_states_per_group", -1)) != 20
        or payload.get("checks") != EXPECTED_COST_PROFILE_CHECKS
    ):
        raise RuntimeError("TMLR V6 action cost profile contract drift")
    source_identities = validate_cost_profile_source_identities(
        repository=repository,
        protocol=protocol,
        identities=payload.get("source_identities"),
    )
    implementation_identity = validate_cost_profile_implementation_identity(
        repository, payload.get("implementation"),
    )
    expected_fit_sha = str(
        protocol["frozen_fit_inputs"]["s1364_manifest"]["sha256"]
    )
    if any(
        str(row.get("source_manifest_sha256")) != expected_fit_sha
        for row in payload.get("rows", ())
    ):
        raise RuntimeError("TMLR V6 cost profile FIT source identity drift")
    protected_access = payload.get("protected_access", {})
    if (
        not isinstance(protected_access, Mapping)
        or set(protected_access) != set(ZERO_EXTERNAL_ACCESS)
        or any(
            type(protected_access[name]) is not int
            or protected_access[name] != 0
            for name in ZERO_EXTERNAL_ACCESS
        )
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")
    if actual_csv.read_bytes() != cost_csv_bytes(payload):
        raise RuntimeError("TMLR V6 action cost CSV/JSON drift")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", ()):
        key = (str(row.get("fold")), str(row.get("action")))
        if key in indexed:
            raise RuntimeError("TMLR V6 cost profile repeats a fold/action")
        indexed[key] = row
    expected = {
        (fold, action) for fold in (*FOLDS, "FULLFIT") for action in ACTIONS
    }
    if set(indexed) != expected:
        raise RuntimeError("TMLR V6 cost profile fold/action coverage drift")
    for fold in (*FOLDS, "FULLFIT"):
        profile_hashes = {str(indexed[(fold, action)]["profile_sha256"]) for action in ACTIONS}
        if (
            len(profile_hashes) != 1
            or indexed[(fold, "STOP")]["normalized_median"] != 0.0
            or indexed[(fold, "STOP")]["median_seconds"] != 0.0
            or any(str(indexed[(fold, action)]["seed"]) != "ALL" for action in ACTIONS)
        ):
            raise RuntimeError("TMLR V6 cost profile seed/STOP contract drift")
    group_to_fold = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in fold_manifest["groups"]
    }
    try:
        state_folds = [group_to_fold[str(group)] for group in groups]
    except KeyError as error:
        raise RuntimeError("TMLR V6 cost surface contains an out-of-fit group") from error
    normalized = np.asarray([
        [float(indexed[(fold, action)]["normalized_median"]) for action in ACTIONS]
        for fold in state_folds
    ], dtype=np.float64)
    seconds = np.asarray([
        [float(indexed[(fold, action)]["median_seconds"]) for action in ACTIONS]
        for fold in state_folds
    ], dtype=np.float64)
    cost = CostSurface(normalized=normalized, seconds=seconds).validated(
        (len(state_folds), len(ACTIONS)),
    )
    return cost, {
        "json": json_identity,
        "csv": csv_identity,
        "source_identities": source_identities,
        "implementation": implementation_identity,
        "protocol_registered": True,
        "fold_specific": True,
        "outer_heldout_excluded": True,
        "seed_invariant": True,
        "predictor_input_used": False,
    }


def _require_legacy_inputs(
    repository: Path, bundle_root: Path, run_root: Path, inventory_path: Path,
) -> None:
    """Fail closed; C1/C2 never fall back to aggregate historical tables."""

    bundle = _repository_path(repository, bundle_root)
    root = _repository_path(repository, run_root)
    missing = [
        bundle / "manifest.json", bundle / "arrays.npz",
        _repository_path(repository, inventory_path),
        *[
            root / source / fold / f"seed_{seed}" / "run-manifest.json"
            for source in LEGACY_MODELS.values()
            for fold in FOLDS for seed in SEEDS
        ],
    ]
    absent = [path for path in missing if not path.is_file() or path.is_symlink()]
    if absent:
        raise RuntimeError(
            "TMLR_V6_C1_C2_SAFE_FIT_OOF_INPUT_REQUIRED: "
            f"{_logical(repository, absent[0])} ({len(absent)} missing)"
        )


HISTORICAL_FILE_NAMES = {
    "run_manifest": "run-manifest.json",
    "prediction": "predictions.npz",
    "checkpoint": "checkpoint.pt",
}


def _historical_inventory_maps(
    payload: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, int, str], Mapping[str, Any]],
    dict[tuple[str, str, int], Mapping[str, Any]],
]:
    """Map the exact registered 2x5x3 run/file cross-product."""

    observed_files = {
        str(item.get("path", "")): item for item in payload.get("files", ())
    }
    if len(observed_files) != 90:
        raise RuntimeError("TMLR V6 historical FIT inventory file-map drift")
    expected_files: dict[
        str, tuple[tuple[str, str, int, str], str]
    ] = {}
    for public_model, source_model in LEGACY_MODELS.items():
        for fold in FOLDS:
            for seed in SEEDS:
                canonical_root = (
                    LEGACY_CANONICAL_RUN_ROOT / source_model / fold / f"seed_{seed}"
                )
                mirror_root = LEGACY_MIRROR_ROOT / canonical_root
                for category, filename in HISTORICAL_FILE_NAMES.items():
                    canonical = (canonical_root / filename).as_posix()
                    path = (mirror_root / filename).as_posix()
                    expected_files[path] = (
                        (public_model, fold, seed, category), canonical,
                    )
    if set(observed_files) != set(expected_files):
        raise RuntimeError("TMLR V6 historical FIT inventory exact path-set drift")
    files_by_run: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for path, (key, canonical) in expected_files.items():
        item = observed_files[path]
        if (
            str(item.get("category")) != key[3]
            or str(item.get("canonical_relative_path")) != canonical
        ):
            raise RuntimeError("TMLR V6 historical FIT inventory file binding drift")
        files_by_run[key] = item

    runs_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for item in payload.get("runs", ()):
        key = (
            str(item.get("public_model")),
            str(item.get("outer_fold")),
            int(item.get("seed", -1)),
        )
        if key in runs_by_key:
            raise RuntimeError("TMLR V6 historical FIT inventory repeats a run")
        runs_by_key[key] = item
    expected_run_keys = {
        (model, fold, seed)
        for model in LEGACY_MODELS for fold in FOLDS for seed in SEEDS
    }
    if set(runs_by_key) != expected_run_keys:
        raise RuntimeError("TMLR V6 historical FIT inventory exact run-set drift")
    for key, item in runs_by_key.items():
        public_model, fold, seed = key
        source_model = LEGACY_MODELS[public_model]
        manifest = files_by_run[(*key, "run_manifest")]
        prediction = files_by_run[(*key, "prediction")]
        checkpoint = files_by_run[(*key, "checkpoint")]
        git_commit = str(item.get("git_commit", ""))
        if (
            str(item.get("source_model")) != source_model
            or str(item.get("public_model")) != public_model
            or str(item.get("outer_fold")) != fold
            or int(item.get("seed", -1)) != seed
            or int(item.get("heldout_states", -1)) <= 0
            or len(str(item.get("run_id", ""))) == 0
            or len(git_commit) != 40
            or any(character not in "0123456789abcdef" for character in git_commit)
            or str(item.get("manifest_sha256")) != str(manifest.get("sha256"))
            or str(item.get("prediction_sha256")) != str(prediction.get("sha256"))
            or str(item.get("checkpoint_sha256")) != str(checkpoint.get("sha256"))
        ):
            raise RuntimeError("TMLR V6 historical FIT inventory run binding drift")
    return files_by_run, runs_by_key


def _validate_historical_run_binding(
    *, report: Mapping[str, Any], inventory_run: Mapping[str, Any],
    inventory_files: Mapping[str, Mapping[str, Any]], public_model: str,
    source_model: str, fold: str, seed: int, heldout_states: int,
) -> None:
    """Bind one loaded manifest and both artifacts to its inventory row."""

    manifest = inventory_files["run_manifest"]
    prediction = inventory_files["prediction"]
    checkpoint = inventory_files["checkpoint"]
    expected_prediction = {
        "path": str(prediction["canonical_relative_path"]),
        "bytes": int(prediction["bytes"]),
        "sha256": str(prediction["sha256"]),
    }
    expected_checkpoint = {
        "logical_path": str(checkpoint["canonical_relative_path"]),
        "bytes": int(checkpoint["bytes"]),
        "sha256": str(checkpoint["sha256"]),
    }
    if (
        report.get("status") != "PASS"
        or report.get("model_id") != source_model
        or report.get("outer_fold") != fold
        or int(report.get("seed", -1)) != seed
        or report.get("scale") != "S_MAX"
        or bool(report.get("robustness_only", False))
        or str(report.get("run_id")) != str(inventory_run.get("run_id"))
        or str(report.get("git_commit")) != str(inventory_run.get("git_commit"))
        or int(report.get("counts", {}).get("heldout_states", -1))
        != heldout_states
        or int(inventory_run.get("heldout_states", -1)) != heldout_states
        or str(inventory_run.get("public_model")) != public_model
        or str(inventory_run.get("source_model")) != source_model
        or str(inventory_run.get("manifest_sha256"))
        != str(manifest.get("sha256"))
        or report.get("prediction") != expected_prediction
        or report.get("checkpoint") != expected_checkpoint
    ):
        raise RuntimeError("TMLR V6 historical FIT loaded run/inventory binding drift")
    assert_legacy_zero_external_access(report.get("heldout_access", {}))


def _validate_historical_inventory(
    repository: Path, protocol: Mapping[str, Any], inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind C1/C2 to the exact 90-file fit-only mirror inventory."""

    historical_registration = protocol.get("historical_schema", {})
    if not isinstance(historical_registration, Mapping):
        historical_registration = {}
    _, registered_identity = _protocol_registered_artifact(
        repository=repository, requested_path=inventory_path,
        registration=historical_registration.get("fit_oof_inventory"),
        name="historical FIT OOF inventory",
    )
    payload = _read_json(repository, inventory_path)
    counts = payload.get("counts", {})
    materializer = payload.get("materializer", {})
    materializer_commit = str(
        materializer.get("commit", "") if isinstance(materializer, Mapping) else ""
    )
    materializer_path = (
        str(materializer.get("path", ""))
        if isinstance(materializer, Mapping) else ""
    )
    materializer_sha = (
        str(materializer.get("sha256", ""))
        if isinstance(materializer, Mapping) else ""
    )
    if (
        not isinstance(materializer, Mapping)
        or set(materializer) != {"path", "bytes", "sha256", "commit"}
        or materializer_path
        != "scripts/materialize_tmlr_v6_historical_fit_oof.py"
        or len(materializer_commit) != 40
        or any(character not in "0123456789abcdef" for character in materializer_commit)
        or len(materializer_sha) != 64
        or any(character not in "0123456789abcdef" for character in materializer_sha)
        or not isinstance(materializer.get("bytes"), int)
        or isinstance(materializer.get("bytes"), bool)
        or int(materializer["bytes"]) <= 0
    ):
        raise RuntimeError("TMLR V6 historical FIT materializer identity drift")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", materializer_commit, "HEAD"],
        cwd=repository, check=False, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("TMLR V6 historical FIT materializer is outside HEAD ancestry")
    committed_materializer = subprocess.run(
        ["git", "show", f"{materializer_commit}:{materializer_path}"],
        cwd=repository, check=True, capture_output=True,
    ).stdout
    if (
        len(committed_materializer) != int(materializer["bytes"])
        or hashlib.sha256(committed_materializer).hexdigest() != materializer_sha
    ):
        raise RuntimeError("TMLR V6 historical FIT materializer commit/hash drift")
    if (
        payload.get("schema_version") != LEGACY_INVENTORY_SCHEMA
        or payload.get("status") != LEGACY_INVENTORY_STATUS
        or payload.get("mirror_root") != LEGACY_MIRROR_ROOT.as_posix()
        or payload.get("canonical_run_root") != LEGACY_CANONICAL_RUN_ROOT.as_posix()
        or payload.get("models") != LEGACY_MODELS
        or payload.get("folds") != list(FOLDS)
        or payload.get("seeds") != list(SEEDS)
        or int(counts.get("runs", -1)) != 30
        or int(counts.get("files", -1)) != 90
        or int(counts.get("run_manifests", -1)) != 30
        or int(counts.get("predictions", -1)) != 30
        or int(counts.get("checkpoints", -1)) != 30
        or int(counts.get("state_oof_per_model_seed", -1)) != 27_280
        or int(counts.get("symlinks", -1)) != 0
        or int(counts.get("forbidden_path_matches", -1)) != 0
        or not payload.get("checks")
        or not all(bool(value) for value in payload["checks"].values())
    ):
        raise RuntimeError("TMLR V6 historical FIT OOF inventory contract drift")
    assert_zero_external_access(payload.get("heldout_access", {}))
    files = payload.get("files", ())
    observed_paths: set[str] = set()
    categories: dict[str, int] = {}
    total_bytes = 0
    for item in files:
        path_text = str(item.get("path", ""))
        canonical = str(item.get("canonical_relative_path", ""))
        canonical_path = Path(canonical)
        assert_fit_only_path(canonical_path)
        expected_path = (LEGACY_MIRROR_ROOT / canonical).as_posix()
        if (
            canonical_path.is_absolute()
            or ".." in canonical_path.parts
            or path_text != expected_path
            or not canonical.startswith(LEGACY_CANONICAL_RUN_ROOT.as_posix() + "/")
            or path_text in observed_paths
        ):
            raise RuntimeError("TMLR V6 historical FIT inventory path drift")
        observed_paths.add(path_text)
        path = _repository_path(repository, Path(path_text))
        _verify_identity(path, item, name="historical FIT mirrored file")
        category = str(item.get("category"))
        categories[category] = categories.get(category, 0) + 1
        total_bytes += int(item["bytes"])
    if (
        len(observed_paths) != 90
        or categories != {"checkpoint": 30, "prediction": 30, "run_manifest": 30}
        or total_bytes != int(counts.get("bytes", -1))
    ):
        raise RuntimeError("TMLR V6 historical FIT inventory file coverage drift")
    _, runs_by_key = _historical_inventory_maps(payload)
    run_ids = {str(item.get("run_id")) for item in runs_by_key.values()}
    if len(run_ids) != 30:
        raise RuntimeError("TMLR V6 historical FIT inventory run coverage drift")
    return payload, registered_identity


def load_historical_fit_surfaces(
    *, repository: Path, protocol: Mapping[str, Any], reference: ResidualSurface,
    fold_manifest_path: Path = FOLD_MANIFEST_PATH,
    bundle_root: Path = LEGACY_BUNDLE, run_root: Path = LEGACY_RUN_ROOT,
    inventory_path: Path = LEGACY_INVENTORY,
) -> tuple[dict[str, ResidualSurface], list[dict[str, Any]], dict[str, Any]]:
    """Load C1/C2 solely from complete historical S1364 FIT OOF evidence."""

    if (
        run_root.is_absolute()
        or run_root.as_posix() != LEGACY_RUN_ROOT.as_posix()
    ):
        raise RuntimeError("TMLR V6 historical FIT run root is not registered")
    root = _repository_path(repository, run_root)
    if root != _repository_path(repository, LEGACY_RUN_ROOT):
        raise RuntimeError("TMLR V6 historical FIT run root identity drift")
    historical_inventory, historical_inventory_identity = (
        _validate_historical_inventory(repository, protocol, inventory_path)
    )
    inventory_files, inventory_runs = _historical_inventory_maps(
        historical_inventory,
    )
    _require_legacy_inputs(repository, bundle_root, run_root, inventory_path)
    bundle_path = _repository_path(repository, bundle_root)
    for name, identity in LEGACY_BUNDLE_IDENTITIES.items():
        _verify_identity(bundle_path / name, identity, name=f"historical FIT {name}")
    folds_path = _repository_path(repository, fold_manifest_path)
    fold_manifest = json.loads(folds_path.read_text(encoding="utf-8"))
    bundle = TensorBundle(bundle_path)
    view = _scale_view(bundle, fold_manifest)
    state_global = np.asarray(view["state_global"], dtype=np.int64)
    global_to_local = {int(value): index for index, value in enumerate(state_global)}
    group_to_fold = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in fold_manifest["groups"]
    }
    fold_ids = np.asarray([
        group_to_fold[str(group)] for group in np.asarray(view["group_ids"])
    ])
    historical_ids = tuple(
        str(bundle.manifest["state_ids"][int(index)]) for index in state_global
    )
    historical_target = np.asarray(view["state_target"], dtype=np.float64)
    historical_feasible = _legacy_feature_plan_feasibility(
        view["action_x"], view["feasible"],
    )
    historical_groups = np.asarray(view["group_ids"], dtype=str)
    if (
        historical_ids != reference.state_ids
        or not np.array_equal(historical_groups, reference.groups)
        or not np.array_equal(historical_feasible, reference.feasible)
        or not np.array_equal(historical_target, reference.target, equal_nan=True)
    ):
        raise RuntimeError("TMLR V6 historical/current FIT target alignment drift")
    surfaces: dict[str, ResidualSurface] = {}
    rows: list[dict[str, Any]] = []
    for public_id, source_id in LEGACY_MODELS.items():
        by_seed = {
            seed: np.full(historical_target.shape, np.nan, dtype=np.float64)
            for seed in SEEDS
        }
        covered = {seed: np.zeros(len(state_global), dtype=bool) for seed in SEEDS}
        for fold in FOLDS:
            expected = state_global[fold_ids == fold]
            for seed in SEEDS:
                key = (public_id, fold, seed)
                file_items = {
                    category: inventory_files[(*key, category)]
                    for category in HISTORICAL_FILE_NAMES
                }
                inventory_run = inventory_runs[key]
                manifest_path = _repository_path(
                    repository, Path(str(file_items["run_manifest"]["path"])),
                )
                expected_manifest_path = (
                    root / source_id / fold / f"seed_{seed}" / "run-manifest.json"
                )
                if manifest_path != expected_manifest_path:
                    raise RuntimeError(
                        "TMLR V6 historical FIT manifest path/inventory drift"
                    )
                _verify_identity(
                    manifest_path, file_items["run_manifest"],
                    name="historical FIT registered run manifest",
                )
                report = _read_json(repository, manifest_path.relative_to(repository))
                _validate_historical_run_binding(
                    report=report, inventory_run=inventory_run,
                    inventory_files=file_items, public_model=public_id,
                    source_model=source_id, fold=fold, seed=seed,
                    heldout_states=len(expected),
                )
                prediction_path = _repository_path(
                    repository, Path(str(file_items["prediction"]["path"])),
                )
                checkpoint_path = _repository_path(
                    repository, Path(str(file_items["checkpoint"]["path"])),
                )
                _verify_identity(
                    prediction_path, file_items["prediction"],
                    name="historical FIT registered prediction",
                )
                _verify_identity(
                    checkpoint_path, file_items["checkpoint"],
                    name="historical FIT registered checkpoint",
                )
                with np.load(prediction_path, allow_pickle=False) as payload:
                    observed = np.asarray(payload["state_index"], dtype=np.int64)
                    validate_exact_oof_indices(
                        observed, expected, kind="TMLR V6 historical FIT state",
                    )
                    prediction = np.asarray(payload["state_prediction"], dtype=np.float64)
                    if prediction.shape != (len(observed), len(ACTIONS)) or not np.isfinite(prediction).all():
                        raise RuntimeError("TMLR V6 historical FIT prediction drift")
                    local = np.asarray([global_to_local[int(value)] for value in observed])
                    if covered[seed][local].any():
                        raise RuntimeError("TMLR V6 historical FIT OOF overlap")
                    by_seed[seed][local] = prediction
                    covered[seed][local] = True
                rows.append({
                    "model": public_id,
                    "source_model": source_id,
                    "outer_fold": fold,
                    "seed": seed,
                    "run_id": str(report["run_id"]),
                    "manifest_path": str(file_items["run_manifest"]["path"]),
                    "manifest_sha256": str(file_items["run_manifest"]["sha256"]),
                    "prediction_path": str(file_items["prediction"]["path"]),
                    "prediction_sha256": str(file_items["prediction"]["sha256"]),
                    "checkpoint_path": str(file_items["checkpoint"]["path"]),
                    "checkpoint_sha256": str(file_items["checkpoint"]["sha256"]),
                })
        if any(not value.all() for value in covered.values()):
            raise RuntimeError("TMLR V6 historical FIT OOF coverage is incomplete")
        surfaces[public_id] = ResidualSurface(
            model_id=public_id,
            prediction=strict_seed_ensemble(by_seed),
            target=historical_target,
            feasible=historical_feasible,
            groups=historical_groups,
            state_ids=historical_ids,
        ).validated()
    loaded_keys = {
        (str(row["model"]), str(row["outer_fold"]), int(row["seed"]))
        for row in rows
    }
    if (
        loaded_keys != set(inventory_runs)
        or {str(row["run_id"]) for row in rows}
        != {str(item["run_id"]) for item in inventory_runs.values()}
    ):
        raise RuntimeError("TMLR V6 historical FIT evaluator/inventory run-ID drift")
    return surfaces, rows, {
        "materialization_inventory": historical_inventory_identity,
        "bundle_manifest": _artifact_identity(repository, bundle_path / "manifest.json"),
        "bundle_arrays": _artifact_identity(repository, bundle_path / "arrays.npz"),
        "fold_manifest": _artifact_identity(repository, folds_path),
        "models": dict(LEGACY_MODELS),
        "comparison_framing": HISTORICAL_COMPARISON_METADATA,
        "complete_runs": len(rows),
        "aggregate_table_fallback_used": False,
        "fit_oof_only": True,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty TMLR V6 evaluation table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in fields} for row in rows)
    return handle.getvalue().encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _deduplicate_rows(
    rows: Iterable[Mapping[str, Any]], *, keys: Sequence[str],
) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in rows:
        key = tuple(item[name] for name in keys)
        value = dict(item)
        if key in result and result[key] != value:
            raise RuntimeError("TMLR V6 machine table has a conflicting duplicate row")
        result[key] = value
    return [result[key] for key in sorted(result)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cost-json", type=Path, default=DEFAULT_COST_JSON)
    parser.add_argument("--cost-csv", type=Path, default=DEFAULT_COST_CSV)
    parser.add_argument("--legacy-bundle", type=Path, default=LEGACY_BUNDLE)
    parser.add_argument("--legacy-run-root", type=Path, default=LEGACY_RUN_ROOT)
    parser.add_argument("--legacy-inventory", type=Path, default=LEGACY_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true",
        help="recompute and byte-verify all scientific output tables",
    )
    args = parser.parse_args(argv)
    repository = _repository()
    for value in vars(args).values():
        if isinstance(value, Path):
            _repository_path(repository, value)
    implementation = committed_source_identity(repository, IMPLEMENTATION_PATHS)
    protocol = load_protocol(repository)
    validate_evaluation_protocol(protocol)
    fold_manifest = validate_fold_manifest(repository, protocol)

    inventory_path = _repository_path(repository, args.inventory)
    inventory = _read_json(repository, args.inventory)
    current_inventory = audit_first_batch_runs(
        repository=repository, root=args.run_root,
    )
    if canonical_json_bytes(inventory) != canonical_json_bytes(current_inventory):
        raise RuntimeError("TMLR V6 first-batch run inventory is stale")
    if (
        inventory.get("status") != "TMLR_V6_FIRST_BATCH_RUN_INVENTORY_PASS"
        or int(inventory.get("completed_valid_runs", -1)) != 75
        or not inventory.get("checks")
        or not all(bool(value) for value in inventory["checks"].values())
    ):
        raise RuntimeError("TMLR V6 first-batch inventory is not complete")
    assert_zero_external_access(inventory.get("heldout_access", {}))
    inventory_sha = sha256_file(inventory_path)

    surfaces, run_rows = load_first_batch_surfaces(
        repository=repository, protocol=protocol,
        fold_manifest=fold_manifest, run_root=args.run_root,
    )
    reference = surfaces[FIRST_BATCH_MODEL_IDS[0]]
    _, complete_case = complete_case_state_mask(reference)
    _, states_per_group = np.unique(reference.groups, return_counts=True)
    if (
        len(states_per_group) != 1364
        or len(reference.state_ids) != 27_280
        or not np.all(states_per_group == 20)
    ):
        raise RuntimeError("TMLR V6 first-batch FIT population is not S1364 x 20")
    costs, cost_identity = load_cost_surface(
        repository=repository, protocol=protocol, groups=reference.groups,
        fold_manifest=fold_manifest, json_path=args.cost_json,
        csv_path=args.cost_csv,
    )
    historical, historical_run_rows, historical_identity = (
        load_historical_fit_surfaces(
            repository=repository, protocol=protocol, reference=reference,
            bundle_root=args.legacy_bundle, run_root=args.legacy_run_root,
            inventory_path=args.legacy_inventory,
        )
    )

    prediction_rows = [prediction_metrics(surfaces[model]) for model in FIRST_BATCH_MODEL_IDS]
    historical_prediction_rows = annotate_historical_comparison([
        {
            **prediction_metrics(historical[model]),
            "historical_comparison_id": (
                "C1" if model == "RETRO_R0_FIT_OOF" else "C2"
            ),
        }
        for model in sorted(historical)
    ], comparison_id_key="historical_comparison_id")
    lambdas = tuple(float(value) for value in protocol["evaluation"]["lambdas"])
    budgets = tuple(float(value) for value in protocol["evaluation"]["budgets"])
    decision_rows = [
        decision_metrics(surfaces[model], costs, cost_lambda=value)
        for model in FIRST_BATCH_MODEL_IDS for value in lambdas
    ]
    sparse_rows = [
        row
        for model in FIRST_BATCH_MODEL_IDS for value in lambdas
        for row in budget_metrics(
            surfaces[model], costs, cost_lambda=value, budgets=budgets,
        )
    ]
    statistics = protocol["statistics"]
    primary_rows, prospective_group_rows = paired_primary_statistics(
        surfaces,
        contrasts=PRIMARY_CONTRASTS,
        replicates=int(statistics["replicates"]),
        seed=int(statistics["seed"]),
        interval=tuple(float(value) for value in statistics["confidence_interval"]),
    )
    historical_panel = {
        "R0_SMALL_P": surfaces["R0_SMALL_P"],
        "R1_P": surfaces["R1_P"],
        **historical,
    }
    historical_rows, historical_group_rows = paired_primary_statistics(
        historical_panel,
        contrasts=HISTORICAL_CONTRASTS,
        replicates=int(statistics["replicates"]),
        seed=int(statistics["seed"]),
        interval=tuple(float(value) for value in statistics["confidence_interval"]),
    )
    historical_rows = annotate_historical_comparison(historical_rows)
    group_rows = _deduplicate_rows(
        [*prospective_group_rows, *historical_group_rows],
        keys=("model", "image_group_id", "metric"),
    )
    formal_metric_rows = [
        *prediction_rows, *historical_prediction_rows,
        *decision_rows, *sparse_rows, *primary_rows, *historical_rows,
        *group_rows,
    ]
    complete_case_fields_consistent = all(
        row.get(key) == value
        for row in formal_metric_rows
        for key, value in complete_case.items()
    )
    expected_decisions = {
        (model, value) for model in FIRST_BATCH_MODEL_IDS for value in lambdas
    }
    observed_decisions = {
        (str(row["model"]), float(row["lambda"])) for row in decision_rows
    }
    expected_sparse = {
        (model, value, budget, semantics)
        for model in FIRST_BATCH_MODEL_IDS for value in lambdas
        for budget in budgets for semantics in ("FORCED_K", "POSITIVE_GAIN_CAP")
    }
    observed_sparse = {
        (
            str(row["model"]), float(row["lambda"]), float(row["budget"]),
            str(row["budget_semantics"]),
        )
        for row in sparse_rows
    }
    metric_names_complete = all(
        all(metric in row for metric in protocol["evaluation"]["prediction_metrics"])
        for row in prediction_rows
    ) and all(
        all(metric in row for metric in protocol["evaluation"]["decision_metrics"])
        for row in decision_rows
    )
    if (
        len(prediction_rows) != 5
        or len(decision_rows) != 5 * len(lambdas)
        or len(sparse_rows) != 5 * len(lambdas) * len(budgets) * 2
        or len(primary_rows) != 4 * 2
        or len(historical_rows) != 2 * 2
        or len(historical_prediction_rows) != 2
        or len(run_rows) != 75
        or len(historical_run_rows) != 30
        or observed_decisions != expected_decisions
        or observed_sparse != expected_sparse
        or not metric_names_complete
        or not complete_case_fields_consistent
    ):
        raise RuntimeError("TMLR V6 evaluation row cardinality drift")

    output_root = _repository_path(repository, args.output_root)
    if args.check and not output_root.exists():
        raise FileNotFoundError(output_root)
    if not args.check and output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        tables: dict[str, Sequence[Mapping[str, Any]]] = {
            "first_batch_prediction_metrics.csv": prediction_rows,
            "historical_c1_c2_prediction_metrics.csv": historical_prediction_rows,
            "first_batch_decision_metrics.csv": decision_rows,
            "first_batch_sparse_budget_metrics.csv": sparse_rows,
            "first_batch_primary_group_bootstrap.csv": primary_rows,
            "historical_c1_c2_group_bootstrap.csv": historical_rows,
            "primary_metric_per_group.csv": group_rows,
            "first_batch_run_inventory.csv": run_rows,
            "historical_c1_c2_run_inventory.csv": historical_run_rows,
        }
        outputs: dict[str, dict[str, Any]] = {}
        for name, rows in tables.items():
            path = temporary / name
            _write_new(path, _csv_bytes(rows))
            outputs[name] = {
                "path": (args.output_root / name).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        metrics_payload = {
            "schema_version": "rail3.tmlr-v6.first-batch-machine-metrics.v1",
            "split": "FIT_OOF",
            "models": list(FIRST_BATCH_MODEL_IDS),
            "complete_case": complete_case,
            "prediction_metrics": prediction_rows,
            "historical_c1_c2_prediction_metrics": historical_prediction_rows,
            "decision_metrics": decision_rows,
            "sparse_budget_metrics": sparse_rows,
            "primary_group_bootstrap": primary_rows,
            "historical_c1_c2_group_bootstrap": historical_rows,
        }
        metrics_path = temporary / "first_batch_metrics.json"
        _write_new(metrics_path, canonical_json_bytes(metrics_payload) + b"\n")
        outputs["first_batch_metrics.json"] = {
            "path": (args.output_root / "first_batch_metrics.json").as_posix(),
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
            "rows": (
                len(prediction_rows) + len(historical_prediction_rows)
                + len(decision_rows) + len(sparse_rows)
                + len(primary_rows) + len(historical_rows)
            ),
        }
        output_artifact_identities_exhaustive = (
            set(outputs) == EXPECTED_OUTPUT_NAMES
            and all(
                set(identity) == {"path", "bytes", "sha256", "rows"}
                and isinstance(identity["path"], str)
                and int(identity["bytes"]) > 0
                and int(identity["rows"]) > 0
                and isinstance(identity["sha256"], str)
                and len(identity["sha256"]) == 64
                for identity in outputs.values()
            )
        )
        checks = {
            "exact_five_model_panel": set(surfaces) == set(FIRST_BATCH_MODEL_IDS),
            "exact_cross_product_75": len(run_rows) == 75,
            "five_folds_stitched_per_seed": True,
            "continuous_seed_mean_before_selection": True,
            "cost_profiles_fold_specific_seed_invariant": True,
            "cost_profiles_outside_predictor": True,
            "all_registered_prediction_metrics": (
                len(prediction_rows) == 5 and metric_names_complete
            ),
            "all_registered_lambda_decision_metrics": (
                len(decision_rows) == 30
                and observed_decisions == expected_decisions
            ),
            "all_lambda_budget_semantics": (
                len(sparse_rows) == 360 and observed_sparse == expected_sparse
            ),
            "paired_image_group_bootstrap_2000_seed13": (
                int(statistics["replicates"]) == 2000
                and int(statistics["seed"]) == 13
            ),
            "image_group_is_independent_unit": len(states_per_group) == 1364,
            "class_states_not_independent_replicates": True,
            "folds_not_independent_replicates": True,
            "group_win_fractions_reported": all(
                row.get("group_win_fraction") is not None
                for row in [*primary_rows, *historical_rows]
            ),
            "historical_c1_c2_complete_safe_fit_oof": len(historical_run_rows) == 30,
            "historical_c1_c2_not_label_isolated_causal_effect": all(
                row["label_isolated_causal_effect"] is False
                for row in historical_rows
            ),
            "historical_aggregate_fallback_not_used": True,
            "output_artifact_identities_exhaustive": (
                output_artifact_identities_exhaustive
            ),
            "whole_state_complete_case_applied_to_all_formal_metrics": (
                complete_case_fields_consistent
            ),
            "protected_access_zero": True,
            "training_not_invoked": True,
            "sam_inference_not_invoked": True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"TMLR V6 first-batch metric gate failed: {checks}")
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": FIRST_BATCH_EVALUATION_STATUS,
            "first_batch_run_inventory_sha256": inventory_sha,
            "completed_valid_runs": 75,
            "metric_pipeline_pass": True,
            "implementation_commit": implementation["commit"],
            "implementation_source_sha256": implementation["source_sha256"],
            "implementation_combined_sha256": implementation["combined_sha256"],
            "protocol": {
                "path": "configs/experiments/tmlr_v6_p0_prospective_capacity.json",
                "sha256": sha256_file(
                    repository / "configs/experiments/tmlr_v6_p0_prospective_capacity.json"
                ),
            },
            "inventory": _artifact_identity(repository, inventory_path),
            "run_contract": {
                "models": list(FIRST_BATCH_MODEL_IDS),
                "folds": list(FOLDS),
                "seeds": list(SEEDS),
                "expected_runs": 75,
                "completed_valid_runs": 75,
                "seed_aggregation": protocol["evaluation"]["seed_aggregation"],
            },
            "population": {
                "split": "FIT_OOF",
                "image_groups": 1364,
                "class_states_per_group": 20,
                "states": 27_280,
                **complete_case,
                "independent_statistical_unit": "image_group_id",
            },
            "cost_profiles": cost_identity,
            "historical_c1_c2": historical_identity,
            "statistics": {
                "paired": True,
                "unit": "image_group_id",
                "replicates": 2000,
                "seed": 13,
                "confidence_interval": [0.025, 0.975],
                "primary_metrics": ["absolute_residual_mae", "drre"],
                "effect": "left_model_minus_right_model",
                "group_win": (
                    "strictly_lower_group_metric; absolute differences at or below "
                    "1e-15 counted as ties"
                ),
                "multiplicity_adjustment": "none; fixed pre-registered contrasts; no p-values",
            },
            "metric_definitions": {
                "decision_value": "residual_prediction + lambda * fold_profiled_normalized_cost",
                "false_stop_rate_denominator": "all_evaluable_states",
                "false_intervention_rate_denominator": "all_evaluable_states",
                "negative_intervention_rate_denominator": "selected_nonstop_states",
                "beneficial_detection_rate_denominator": "oracle_beneficial_states",
                "beneficial_detection_event": (
                    "selected_action_realized_net_gain_strictly_greater_than_zero"
                ),
                "budget_oracle_denominator": "top_K_nonnegative_oracle_net_gains",
                "positive_gain_cap": "top_K_ranked_states_filtered_by_predicted_net_gain_strictly_greater_than_zero",
                "tie_order": list(ACTIONS),
                "state_ranking_tie_break": "state_id_ascending",
                "complete_case_rule": (
                    "exclude_entire_state_if_any_pre_action_plan_feasible_action_"
                    "has_undefined_post_execution_target"
                ),
                "selection_feasibility_source": (
                    "pre_action_action_x16_columns_0_and_7_not_target_definedness"
                ),
            },
            "outputs": outputs,
            "checks": checks,
            "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        }
        validate_report_contract(report, inventory_sha256=inventory_sha)
        report_path = temporary / "first_batch_evaluation_report.json"
        _write_new(report_path, canonical_json_bytes(report) + b"\n")
        if args.check:
            existing = {path.name: path for path in output_root.iterdir()}
            generated = {path.name: path for path in temporary.iterdir()}
            expected_names = set(EXPECTED_OUTPUT_NAMES) | {
                "first_batch_evaluation_report.json",
            }
            if set(existing) != expected_names or set(generated) != expected_names:
                raise RuntimeError("TMLR V6 first-batch output tree drift")
            for path in existing.values():
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise RuntimeError("TMLR V6 first-batch output is not regular")
            for name in EXPECTED_OUTPUT_NAMES:
                if existing[name].read_bytes() != generated[name].read_bytes():
                    raise RuntimeError(
                        f"TMLR V6 first-batch scientific rebuild drift: {name}"
                    )
            existing_report = json.loads(
                existing["first_batch_evaluation_report.json"].read_text(
                    encoding="utf-8",
                )
            )
            validate_report_contract(
                existing_report, inventory_sha256=inventory_sha,
            )
        else:
            os.replace(temporary, output_root)
    print(json.dumps({
        "status": FIRST_BATCH_EVALUATION_STATUS,
        "output_root": _logical(repository, output_root),
        "completed_valid_runs": 75,
        "models": len(surfaces),
        "bootstrap_replicates": 2000,
        "check": args.check,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
