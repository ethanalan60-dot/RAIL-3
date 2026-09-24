"""No-bypass contract and immutable identities for TMLR V6 phase two.

The semantic addendum was frozen before first-batch metric inspection.  This
module turns it into executable, result-blind gates: an exact 225-job second
batch, a 120-job F0/F1 batch, and one identity-only launch lock.  It never
opens a predictor target or a prediction surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    EXTERNAL_SPLITS,
    FOLDS,
    PROTOCOL_PATH,
    SEEDS,
    ZERO_EXTERNAL_ACCESS,
    assert_zero_external_access,
    sha256_file,
)
from rail3.models.tmlr_v6.phase2_models import (
    DIRECT_MODEL_IDS,
    SECOND_BATCH_MODEL_IDS,
)
from rail3.models.tmlr_v6.phase2_objectives import PHASE2_LAMBDAS


PHASE2_ADDENDUM_PATH = Path(
    "configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json"
)
PHASE2_LAUNCH_LOCK_PATH = Path(
    "artifacts/audits/tmlr_v6/phase2_launch_lock.json"
)
FIRST_BATCH_GATE_PATH = Path(
    "artifacts/audits/tmlr_v6/first_batch_completion_gate.json"
)
FIRST_BATCH_INVENTORY_PATH = Path(
    "artifacts/audits/tmlr_v6/first_batch_run_inventory.json"
)
FIRST_BATCH_EVALUATION_PATH = Path(
    "artifacts/paper/source_data/tmlr_v6/first_batch/"
    "first_batch_evaluation_report.json"
)
S1364_COST_CSV_PATH = Path(
    "artifacts/source_data/tmlr_v6/action_type_cost_profiles.csv"
)
S1364_COST_JSON_PATH = Path(
    "artifacts/source_data/tmlr_v6/action_type_cost_profiles.json"
)
PHASE2_SCALE_COST_CSV_PATH = Path(
    "artifacts/source_data/tmlr_v6/phase2_scale_action_type_cost_profiles.csv"
)
PHASE2_SCALE_COST_JSON_PATH = Path(
    "artifacts/source_data/tmlr_v6/phase2_scale_action_type_cost_profiles.json"
)
_FORBIDDEN_PATH_FRAGMENTS = (
    "artifacts/protected", "validation40", "locked40", "calibration30",
    "pilot-test30", "pilot_test30", "official-voc", "official_voc",
    "segppd", "uav-iap", "uav_iap",
)
PHASE2_SCALES = ("S250", "S500", "S1000", "S1364")
PHASE2_FOLD_PATHS = {
    "S250": Path("data/manifests/voc2012-m06e-s250-folds.json"),
    "S500": Path("data/manifests/voc2012-m06e-s500-folds.json"),
    "S1000": Path("data/manifests/voc2012-m06e-s1000-folds.json"),
    "S1364": Path("data/manifests/voc2012-m06e-s_max-folds.json"),
}

SECOND_BATCH_RUNS = 225
F0_F1_RUNS = 120
PHASE2_TOTAL_RUNS = 345

LAUNCH_LOCK_SCHEMA = "rail3.tmlr-v6.phase2-launch-lock.v1"
LAUNCH_LOCK_STATUS = "TMLR_V6_PHASE2_LAUNCH_LOCK_PASS"
PHASE2_ADDENDUM_SCHEMA = (
    "rail3.tmlr-v6-p0-phase2-semantic-addendum.v1"
)
PHASE2_ADDENDUM_STATUS = (
    "TMLR_V6_PHASE2_SEMANTICS_FROZEN_BEFORE_FIRST_BATCH_METRIC_INSPECTION"
)
FROZEN_ADDENDUM_BYTES = 43_072
FROZEN_ADDENDUM_SHA256 = (
    "c5356e4141ef21dda8956109a855a9b7f89ae51095b4233ee709ed4110c24131"
)
SCALE_COST_SCHEMA = "rail3.tmlr-v6.phase2-scale-cost-profiles.v1"
SCALE_COST_STATUS = "TMLR_V6_PHASE2_SCALE_COST_PROFILES_PASS"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FILE_IDENTITY_KEYS = frozenset({"path", "bytes", "sha256"})
_LAUNCH_LOCK_KEYS = frozenset({
    "schema_version",
    "status",
    "created_at_utc",
    "lock_id",
    "parent_protocol",
    "phase2_semantic_addendum",
    "first_batch_completion_gate",
    "first_batch_run_inventory",
    "first_batch_evaluation",
    "prospective_bundle_registry_canonical_sha256",
    "historical_fit_oof_inventory",
    "s1364_cost_profiles",
    "phase2_scale_cost_profiles",
    "fold_manifests",
    "phase2_implementation",
    "job_counts",
    "heldout_access",
    "checks",
})
_LAUNCH_CHECKS = frozenset({
    "parent_protocol_identity_exact",
    "addendum_committed_clean_and_ancestor",
    "first_batch_gate_identity_exact",
    "first_batch_gate_is_completeness_and_safety_only",
    "first_batch_75_runs_complete",
    "phase2_job_matrix_exact_345",
    "four_scale_cost_profiles_complete_and_seed_invariant",
    "protected_access_zero",
    "no_semantic_field_added_to_launch_lock",
})
_FIRST_GATE_CHECKS = {
    "prospective_runtime_audit_pass": True,
    "exact_cross_product_75_pass": True,
    "all_training_finite": True,
    "metric_pipeline_pass": True,
    "protected_access_zero": True,
}
_S1364_COST_CHECKS = {
    "label_free_trajectory_reports_only": True,
    "feasible_runtime_strictly_positive": True,
    "infeasible_runtime_exactly_zero": True,
    "outer_heldout_excluded_per_fold": True,
    "stop_exactly_zero": True,
    "profile_seed_independent": True,
    "predictor_input_used": False,
    "protected_read_count": 0,
    "protocol_registered_source_set_verified": True,
    "implementation_committed_clean": True,
}
_SCALE_COST_CHECKS = {
    "exact_100_rows": True,
    "all_four_scales": True,
    "five_outer_folds_per_scale": True,
    "five_physical_actions_per_fold": True,
    "outer_training_union_only": True,
    "outer_heldout_excluded": True,
    "seed_invariant": True,
    "stop_exactly_zero": True,
    "a3_profile_not_predictor_input": True,
    "heldout_state_specific_runtime_used": False,
    "predictor_input_used": False,
    "protected_read_count": 0,
}
_FIRST_GATE_KEYS = frozenset({
    "schema_version",
    "status",
    "protocol_sha256",
    "inventory",
    "runtime_feature_audit",
    "first_batch_evaluation",
    "completed_valid_runs",
    "second_batch_authorized_by_gate",
    "heldout_access",
    "gate_id",
    "checks",
})
# The terminal classifier executes this exact registered test set.  Keeping
# the paths explicit avoids discovery-pattern gaps such as
# ``test_audit_tmlr_v6_*.py`` and makes every executed test part of the launch
# identity.
PHASE2_INTEGRITY_TEST_PATHS = (
    Path("tests/unit/analysis/test_tmlr_v6_first_batch_evaluation.py"),
    Path("tests/unit/analysis/test_tmlr_v6_phase2.py"),
    Path("tests/unit/analysis/test_tmlr_v6_qualitative.py"),
    Path("tests/unit/analysis/test_tmlr_v6_sensitivity.py"),
    Path("tests/unit/models/test_tmlr_v6_cost_profiles.py"),
    Path("tests/unit/models/test_tmlr_v6_first_batch.py"),
    Path("tests/unit/models/test_tmlr_v6_inventory.py"),
    Path("tests/unit/models/test_tmlr_v6_phase2_contract.py"),
    Path("tests/unit/models/test_tmlr_v6_phase2_cost_profiles.py"),
    Path("tests/unit/models/test_tmlr_v6_phase2_f0_f1.py"),
    Path("tests/unit/models/test_tmlr_v6_phase2_inventory.py"),
    Path("tests/unit/models/test_tmlr_v6_phase2_objectives.py"),
    Path("tests/unit/models/test_tmlr_v6_path_security.py"),
    Path("tests/unit/models/test_tmlr_v6_provenance.py"),
    Path("tests/unit/models/test_tmlr_v6_runtime_audit.py"),
    Path("tests/unit/models/test_tmlr_v6_runtime_schema.py"),
    Path("tests/unit/scripts/test_audit_tmlr_v6_first_batch.py"),
    Path("tests/unit/scripts/test_build_tmlr_v6_prospective_bundles.py"),
    Path("tests/unit/scripts/test_generate_tmlr_v6_pretraining_registry.py"),
    Path("tests/unit/scripts/test_materialize_tmlr_v6_fit_only_inputs.py"),
    Path("tests/unit/scripts/test_materialize_tmlr_v6_historical_fit_oof.py"),
    Path("tests/unit/scripts/test_build_tmlr_v6_final_handoff.py"),
    Path("tests/unit/scripts/test_build_tmlr_v6_paper_figures_abc.py"),
    Path("tests/unit/scripts/test_build_tmlr_v6_paper_summary_tables.py"),
    Path("tests/unit/scripts/test_tmlr_v6_phase2_analysis_consumers.py"),
    Path("tests/unit/scripts/test_tmlr_v6_phase2_f0_f1.py"),
    Path("tests/unit/scripts/test_tmlr_v6_phase2_second_batch.py"),
)

# The launch lock binds the complete 345-job implementation, every reused
# objective implementation, and the exact named integrity tests executed by
# the terminal classifier.  This prevents a post-lock test deletion or
# weakening from being mistaken for scientific evidence.
PHASE2_IMPLEMENTATION_PATHS = (
    Path("configs/experiments/tmlr_v6_p0_prospective_capacity.json"),
    Path("configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json"),
    Path("src/rail3/__init__.py"),
    Path("src/rail3/analysis/__init__.py"),
    Path("src/rail3/cache/__init__.py"),
    Path("src/rail3/cache/atomic_io.py"),
    Path("src/rail3/cache/candidates.py"),
    Path("src/rail3/contracts/__init__.py"),
    Path("src/rail3/contracts/records.py"),
    Path("src/rail3/contracts/serialization.py"),
    Path("src/rail3/data/__init__.py"),
    Path("src/rail3/data/manifest.py"),
    Path("src/rail3/data/m06e_sources.py"),
    Path("src/rail3/data/pest.py"),
    Path("src/rail3/data/splits.py"),
    Path("src/rail3/data/voc_taxonomy.py"),
    Path("src/rail3/diagnostic/__init__.py"),
    Path("src/rail3/diagnostic/baselines.py"),
    Path("src/rail3/evaluation/__init__.py"),
    Path("src/rail3/evaluation/voc_v2a.py"),
    Path("src/rail3/models/__init__.py"),
    Path("src/rail3/models/m06b/__init__.py"),
    Path("src/rail3/models/m06b/metrics.py"),
    Path("src/rail3/models/m06b/protocol.py"),
    Path("src/rail3/models/m06e/__init__.py"),
    Path("src/rail3/models/m06e/data.py"),
    Path("src/rail3/models/m06e/features.py"),
    Path("src/rail3/models/m06e/losses.py"),
    Path("src/rail3/models/m06e/models.py"),
    Path("src/rail3/models/m06e/training.py"),
    Path("src/rail3/models/m06g/__init__.py"),
    Path("src/rail3/models/m06g/losses.py"),
    Path("src/rail3/models/tmlr_v6/__init__.py"),
    Path("src/rail3/models/tmlr_v6/cost_profiles.py"),
    Path("src/rail3/models/tmlr_v6/inventory.py"),
    Path("src/rail3/models/tmlr_v6/models.py"),
    Path("src/rail3/models/tmlr_v6/schema.py"),
    Path("src/rail3/models/tmlr_v6/training.py"),
    Path("src/rail3/models/tmlr_v6/phase2_contract.py"),
    Path("src/rail3/models/tmlr_v6/phase2_models.py"),
    Path("src/rail3/models/tmlr_v6/phase2_objectives.py"),
    Path("src/rail3/models/tmlr_v6/phase2_training.py"),
    Path("src/rail3/models/tmlr_v6/phase2_inventory.py"),
    Path("src/rail3/models/tmlr_v6/phase2_io.py"),
    Path("src/rail3/models/tmlr_v6/phase2_scaling.py"),
    Path("src/rail3/models/tmlr_v6/phase2_cost_profiles.py"),
    Path("src/rail3/models/tmlr_v6/phase2_f0_f1.py"),
    Path("src/rail3/analysis/tmlr_v6_phase2.py"),
    Path("src/rail3/analysis/tmlr_v6_sensitivity.py"),
    Path("src/rail3/analysis/tmlr_v6_qualitative.py"),
    Path("src/rail3/analysis/m06f_decision_audit.py"),
    Path("src/rail3/analysis/tmlr_reviewer_revision.py"),
    Path("src/rail3/analysis/tmlr_v6_first_batch.py"),
    Path("src/rail3/regions/__init__.py"),
    Path("src/rail3/regions/atomic.py"),
    Path("src/rail3/regions/m06e_causal.py"),
    Path("src/rail3/regions/voc_state_atomic.py"),
    Path("src/rail3/registry/__init__.py"),
    Path("src/rail3/registry/datasets.py"),
    Path("src/rail3/sam/__init__.py"),
    Path("src/rail3/sam/checkpoint.py"),
    Path("src/rail3/sam/checkpoint_compat.py"),
    Path("src/rail3/sam/compat.py"),
    Path("src/rail3/sam/protocols.py"),
    Path("src/rail3/sam/sam31_backend.py"),
    Path("src/rail3/sam/strict_builder.py"),
    Path("src/rail3/sam/voc_actions.py"),
    Path("src/rail3/sam/voc_protocol.py"),
    Path("scripts/evaluate_tmlr_v6_first_batch_fit.py"),
    Path("scripts/lock_tmlr_v6_phase2_launch.py"),
    Path("scripts/run_tmlr_v6_phase2_second_batch_crossfit.py"),
    Path("scripts/audit_tmlr_v6_phase2_second_batch.py"),
    Path("scripts/build_tmlr_v6_phase2_scale_cost_profiles.py"),
    Path("scripts/register_tmlr_v6_phase2_views.py"),
    Path("scripts/run_tmlr_v6_phase2_f0_f1_crossfit.py"),
    Path("scripts/audit_tmlr_v6_phase2_f0_f1.py"),
    Path("scripts/evaluate_tmlr_v6_phase2_second_batch_fit.py"),
    Path("scripts/evaluate_tmlr_v6_phase2_f0_f1_fit.py"),
    Path("scripts/evaluate_tmlr_v6_phase2_sensitivity_fit.py"),
    Path("scripts/build_tmlr_v6_phase2_qualitative_atlas.py"),
    Path("scripts/classify_tmlr_v6_phase2.py"),
    Path("scripts/build_tmlr_v6_final_handoff.py"),
    Path("scripts/build_tmlr_v6_paper_figures_abc.py"),
    Path("scripts/build_tmlr_v6_paper_summary_tables.py"),
)
PHASE2_IMPLEMENTATION_PATHS += PHASE2_INTEGRITY_TEST_PATHS


@dataclass(frozen=True)
class Phase2Job:
    """One exact phase-two second-batch training job."""

    model_id: str
    scale: str
    outer_fold: str
    seed: int
    cost_lambda: float | None

    def __post_init__(self) -> None:
        if (
            self.model_id not in SECOND_BATCH_MODEL_IDS
            or self.scale != "S1364"
            or self.outer_fold not in FOLDS
            or self.seed not in SEEDS
        ):
            raise ValueError("unregistered TMLR V6 phase-two job")
        if self.model_id in DIRECT_MODEL_IDS:
            if self.cost_lambda not in PHASE2_LAMBDAS:
                raise ValueError("direct TMLR V6 job requires a registered lambda")
        elif self.cost_lambda is not None:
            raise ValueError("non-direct TMLR V6 job must not have a lambda model")

    @property
    def lambda_token(self) -> str | None:
        if self.cost_lambda is None:
            return None
        return f"lambda_{self.cost_lambda:.3f}".replace(".", "p")

    def run_dir(self, root: Path) -> Path:
        base = Path(root) / self.scale / self.model_id
        if self.lambda_token is not None:
            base /= self.lambda_token
        return base / self.outer_fold / f"seed_{self.seed}"


@dataclass(frozen=True)
class Phase2LaunchLock:
    payload: Mapping[str, Any]
    path: Path
    bytes: int
    sha256: str

    @property
    def implementation_commit(self) -> str:
        return str(self.payload["phase2_implementation"]["commit"])

    @property
    def implementation_sha256(self) -> str:
        return str(
            self.payload["phase2_implementation"]["combined_source_sha256"]
        )


def expected_second_batch_jobs() -> tuple[Phase2Job, ...]:
    jobs: list[Phase2Job] = []
    for model_id in SECOND_BATCH_MODEL_IDS:
        lambdas: Sequence[float | None] = (
            PHASE2_LAMBDAS if model_id in DIRECT_MODEL_IDS else (None,)
        )
        for value in lambdas:
            for outer_fold in FOLDS:
                for seed in SEEDS:
                    jobs.append(Phase2Job(
                        model_id=model_id,
                        scale="S1364",
                        outer_fold=outer_fold,
                        seed=seed,
                        cost_lambda=value,
                    ))
    if len(jobs) != SECOND_BATCH_RUNS or len(set(jobs)) != SECOND_BATCH_RUNS:
        raise RuntimeError("TMLR V6 phase-two 225-job matrix drift")
    return tuple(jobs)


def expected_phase2_job_counts() -> dict[str, int]:
    return {
        "second_batch": SECOND_BATCH_RUNS,
        "f0_f1": F0_F1_RUNS,
        "phase2_total": PHASE2_TOTAL_RUNS,
    }


def second_batch_run_path(root: Path, job: Phase2Job) -> Path:
    return job.run_dir(root)


def _repository_path(repository: Path, path: Path) -> Path:
    """Resolve one FIT-only path only after lexical/symlink checks."""

    root = repository.absolute()
    if root.is_symlink() or root.resolve() != root:
        raise RuntimeError("TMLR V6 phase-two repository root is symlinked")
    if ".." in path.parts:
        raise RuntimeError("TMLR V6 phase-two path escapes repository")
    requested = path if path.is_absolute() else root / path
    requested = Path(os.path.abspath(requested))
    try:
        relative = requested.relative_to(root)
    except ValueError as error:
        raise RuntimeError("TMLR V6 phase-two path escapes repository") from error
    normalized = relative.as_posix().lower()
    if any(item in normalized for item in _FORBIDDEN_PATH_FRAGMENTS):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("TMLR V6 phase-two path contains a symlink")
    result = requested.resolve(strict=False)
    try:
        resolved_relative = result.relative_to(root)
    except ValueError as error:
        raise RuntimeError("TMLR V6 phase-two path escapes repository") from error
    if any(
        item in resolved_relative.as_posix().lower()
        for item in _FORBIDDEN_PATH_FRAGMENTS
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")
    return result


def _logical_path(repository: Path, path: Path) -> str:
    return _repository_path(repository, path).relative_to(
        repository.resolve()
    ).as_posix()


def _safe_file(repository: Path, path: Path, *, description: str) -> Path:
    actual = _repository_path(repository, path)
    try:
        metadata = actual.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"unsafe or absent TMLR V6 {description}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"unsafe or absent TMLR V6 {description}")
    return actual


def safe_phase2_repository_path(repository: Path, path: Path) -> Path:
    """Public no-read path boundary for phase-two entry points."""

    return _repository_path(repository, path)


def verify_exact_rebuild_directory(
    existing: Path, generated: Path, *, ignore_names: frozenset[str] = frozenset(),
) -> None:
    """Require a deterministic flat rebuild to match an existing bundle."""

    def regular_files(root: Path) -> dict[str, Path]:
        metadata = root.lstat() if root.exists() else None
        if metadata is None or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("TMLR V6 rebuild output directory is absent or unsafe")
        result: dict[str, Path] = {}
        for item in root.iterdir():
            item_metadata = item.lstat()
            if (
                not stat.S_ISREG(item_metadata.st_mode)
                or item_metadata.st_nlink != 1
                or item.name in result
            ):
                raise RuntimeError("TMLR V6 rebuild output tree is not exact")
            result[item.name] = item
        return result

    observed = regular_files(existing)
    expected = regular_files(generated)
    if set(observed) != set(expected) or not ignore_names <= set(expected):
        raise RuntimeError("TMLR V6 rebuilt output name set drift")
    for name in sorted(set(expected) - set(ignore_names)):
        if observed[name].read_bytes() != expected[name].read_bytes():
            raise RuntimeError(f"TMLR V6 rebuilt output byte drift: {name}")


def file_identity(repository: Path, path: Path) -> dict[str, Any]:
    actual = _safe_file(repository, path, description=str(path))
    return {
        "path": _logical_path(repository, actual),
        "bytes": actual.stat().st_size,
        "sha256": sha256_file(actual),
    }


def _validate_file_identity(
    repository: Path, value: Any, *, expected_path: Path | None,
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FILE_IDENTITY_KEYS:
        raise RuntimeError(f"TMLR V6 {description} identity schema drift")
    result = {
        "path": str(value.get("path", "")),
        "bytes": int(value.get("bytes", -1)),
        "sha256": str(value.get("sha256", "")),
    }
    logical = Path(result["path"])
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or result["bytes"] <= 0
        or _SHA256.fullmatch(result["sha256"]) is None
        or (
            expected_path is not None
            and logical.as_posix() != expected_path.as_posix()
        )
    ):
        raise RuntimeError(f"TMLR V6 {description} identity is invalid")
    actual = _safe_file(repository, logical, description=description)
    if (
        actual.stat().st_size != result["bytes"]
        or sha256_file(actual) != result["sha256"]
    ):
        raise RuntimeError(f"TMLR V6 {description} identity drift")
    return result


def _read_json(repository: Path, path: Path, *, description: str) -> dict[str, Any]:
    actual = _safe_file(repository, path, description=description)
    value = json.loads(actual.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"TMLR V6 {description} must be a JSON object")
    return dict(value)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_phase2_addendum_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen semantic fields needed by all phase-two code."""

    if (
        value.get("schema_version") != PHASE2_ADDENDUM_SCHEMA
        or value.get("status") != PHASE2_ADDENDUM_STATUS
        or value.get("freeze_declaration") != {
            "first_batch_metrics_inspected_before_freeze": False,
            "data_or_artifact_content_read_to_construct_addendum": False,
            "protected_input_read": False,
            "semantic_changes_after_first_batch_metric_inspection_forbidden": True,
            "result_conditioned_threshold_or_predicate_change_forbidden": True,
            "parent_protocol_modified_by_this_addendum": False,
        }
    ):
        raise RuntimeError("TMLR V6 phase-two semantic addendum is not frozen")
    parent = value.get("parent_protocol", {})
    if (
        not isinstance(parent, Mapping)
        or parent.get("path") != PROTOCOL_PATH.as_posix()
        or parent.get("schema_version")
        != "rail3.tmlr-v6-p0-prospective-capacity.v1"
        or _SHA256.fullmatch(str(parent.get("sha256", ""))) is None
        or int(parent.get("bytes", -1)) <= 0
    ):
        raise RuntimeError("TMLR V6 phase-two parent protocol binding drift")
    matrix = value.get("job_matrix", {})
    second = matrix.get("second_batch", {}) if isinstance(matrix, Mapping) else {}
    if (
        second.get("scale") != "S1364"
        or second.get("bundle") != "F0_P"
        or int(second.get("residual_and_gain_jobs", -1)) != 45
        or int(second.get("direct_lambda_specific_jobs", -1)) != 180
        or int(second.get("total_jobs", -1)) != SECOND_BATCH_RUNS
        or int(matrix.get("f0_f1", {}).get("total_jobs", -1)) != F0_F1_RUNS
        or int(matrix.get("phase2_total_jobs", -1)) != PHASE2_TOTAL_RUNS
        or matrix.get("extra_or_replacement_job_forbidden") is not True
    ):
        raise RuntimeError("TMLR V6 phase-two job matrix drift")
    objectives = value.get("objectives", {})
    required_objectives = {
        "R2_P", "R3_P", "Q2_P", "LL4TTA_P", "L2D_P", "SPO_PLUS_P",
    }
    if (
        not isinstance(objectives, Mapping)
        or not required_objectives <= set(objectives)
        or objectives["R2_P"].get("total") != "L_atom + L_abs + 0.5*L_PIDR"
        or tuple(objectives["R2_P"].get("lambda_grid_inside_one_objective", ()))
        != PHASE2_LAMBDAS
        or objectives["R3_P"].get("total")
        != "L_atom + L_abs + L_rel + 0.5*L_sign_weighted + 0.5*L_rank"
        or objectives["Q2_P"].get("scientific_stop_prediction") != 0.0
        or objectives["Q2_P"].get("scientific_stop_override_timing")
        != "after continuous seed averaging and before every metric, action selection, or budget ranking"
        or objectives["LL4TTA_P"].get("alias_of") != "R0_SMALL_P"
        or int(objectives["LL4TTA_P"].get("new_training_jobs", -1)) != 0
        or tuple(objectives["L2D_P"].get("lambda_grid", ())) != PHASE2_LAMBDAS
        or tuple(objectives["SPO_PLUS_P"].get("lambda_grid", ()))
        != PHASE2_LAMBDAS
        or objectives["L2D_P"].get("parameters") != 16_517
        or objectives["SPO_PLUS_P"].get("parameters") != 16_517
    ):
        raise RuntimeError("TMLR V6 phase-two objective registry drift")
    protected = value.get("protected_boundary", {})
    if (
        protected.get("fit_only") is not True
        or protected.get("protected_reader_invocations") != 0
        or protected.get("protected_evaluator_invocations") != 0
        or protected.get("ledger_writes") != 0
        or protected.get("sam_inference") is not False
        or protected.get("new_prompt_or_trajectory_generation") is not False
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")
    return dict(value)


def load_phase2_addendum(repository: Path) -> dict[str, Any]:
    path = _safe_file(
        repository, PHASE2_ADDENDUM_PATH, description="phase-two addendum",
    )
    if (
        path.stat().st_size != FROZEN_ADDENDUM_BYTES
        or sha256_file(path) != FROZEN_ADDENDUM_SHA256
    ):
        raise RuntimeError(
            "TMLR V6 phase-two semantic addendum changed after pre-metric freeze"
        )
    addendum = validate_phase2_addendum_payload(_read_json(
        repository, PHASE2_ADDENDUM_PATH, description="phase-two addendum",
    ))
    parent = _safe_file(repository, PROTOCOL_PATH, description="parent protocol")
    binding = addendum["parent_protocol"]
    if (
        parent.stat().st_size != int(binding["bytes"])
        or sha256_file(parent) != str(binding["sha256"])
    ):
        raise RuntimeError("TMLR V6 phase-two parent protocol identity drift")
    return addendum


def phase2_implementation_identity(repository: Path) -> dict[str, Any]:
    """Return a clean, committed identity for the complete phase-two code."""

    repository = repository.resolve()
    _require_formal_git_state(repository)
    logical = tuple(path.as_posix() for path in PHASE2_IMPLEMENTATION_PATHS)
    for path in logical:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=repository, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"TMLR V6 phase-two implementation is not committed: {path}"
            )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *logical],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 phase-two implementation paths are not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("TMLR V6 phase-two implementation commit is invalid")
    source = {
        path: sha256_file(_safe_file(
            repository, Path(path), description=f"implementation {path}",
        ))
        for path in logical
    }
    return {
        "commit": commit,
        "combined_source_sha256": _canonical_sha(source),
        "source_sha256": source,
    }


def _require_formal_git_state(
    repository: Path, *, expected_commit: str | None = None,
) -> str:
    """Exclude untracked import shadows and any post-lock code commit."""

    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "main":
        raise RuntimeError("TMLR V6 formal phase-two worktree is not on main")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 formal phase-two worktree is not fully clean")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if _COMMIT.fullmatch(head) is None:
        raise RuntimeError("TMLR V6 formal phase-two HEAD is invalid")
    if expected_commit is not None and head != expected_commit:
        raise RuntimeError("TMLR V6 phase-two HEAD changed after launch lock")
    return head


def _validate_implementation_identity(
    repository: Path, value: Any,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"commit", "combined_source_sha256", "source_sha256"}
        or _COMMIT.fullmatch(str(value.get("commit", ""))) is None
        or _SHA256.fullmatch(str(value.get("combined_source_sha256", ""))) is None
        or not isinstance(value.get("source_sha256"), Mapping)
        or set(value["source_sha256"])
        != {path.as_posix() for path in PHASE2_IMPLEMENTATION_PATHS}
        or any(
            _SHA256.fullmatch(str(digest)) is None
            for digest in value["source_sha256"].values()
        )
        or _canonical_sha(dict(value["source_sha256"]))
        != value["combined_source_sha256"]
    ):
        raise RuntimeError("TMLR V6 phase-two implementation identity drift")
    commit = str(value["commit"])
    _require_formal_git_state(repository, expected_commit=commit)
    logical = tuple(path.as_posix() for path in PHASE2_IMPLEMENTATION_PATHS)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *logical],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 phase-two implementation paths are not clean")
    declared = dict(value["source_sha256"])
    for relative in logical:
        current = _safe_file(
            repository, Path(relative), description=f"implementation {relative}",
        )
        if sha256_file(current) != declared[relative]:
            raise RuntimeError("TMLR V6 current phase-two source identity drift")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=repository,
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != declared[relative]:
            raise RuntimeError("TMLR V6 committed phase-two source identity drift")
    return dict(value)


def _validate_fold_identities(
    repository: Path, value: Any, addendum: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(PHASE2_SCALES):
        raise RuntimeError("TMLR V6 phase-two fold identity set drift")
    frozen = addendum["bound_parent_registries"]["fold_registry"]
    result: dict[str, Any] = {}
    for scale, expected_path in PHASE2_FOLD_PATHS.items():
        identity = _validate_file_identity(
            repository, value[scale], expected_path=expected_path,
            description=f"{scale} fold manifest",
        )
        if (
            identity["path"] != str(frozen[scale]["path"])
            or identity["sha256"] != str(frozen[scale]["sha256"])
            or (
                "bytes" in frozen[scale]
                and identity["bytes"] != int(frozen[scale]["bytes"])
            )
        ):
            raise RuntimeError(f"TMLR V6 {scale} fold identity drift")
        result[scale] = identity
    return result


def load_scale_cost_profile_registry(
    repository: Path, lock: Phase2LaunchLock | Mapping[str, Any],
) -> dict[str, Any]:
    """Load the lock-bound 4x5x5 seed-invariant cost registry."""

    payload = lock.payload if isinstance(lock, Phase2LaunchLock) else lock
    registration = payload.get("phase2_scale_cost_profiles", {})
    if not isinstance(registration, Mapping) or set(registration) != {"csv", "json"}:
        raise RuntimeError("TMLR V6 phase-two scale-cost identity schema drift")
    csv_identity = _validate_file_identity(
        repository, registration["csv"], expected_path=PHASE2_SCALE_COST_CSV_PATH,
        description="phase-two scale cost CSV",
    )
    json_identity = _validate_file_identity(
        repository, registration["json"], expected_path=PHASE2_SCALE_COST_JSON_PATH,
        description="phase-two scale cost JSON",
    )
    result = _read_json(
        repository, PHASE2_SCALE_COST_JSON_PATH,
        description="phase-two scale cost JSON",
    )
    from rail3.models.tmlr_v6.phase2_cost_profiles import (
        build_phase2_scale_cost_profiles,
        load_registered_phase2_scale_cost_sources,
        phase2_scale_cost_csv_bytes,
        validate_phase2_scale_cost_payload,
    )

    validate_phase2_scale_cost_payload(result)
    rows = result.get("rows")
    checks = result.get("checks")
    if (
        result.get("schema_version") != SCALE_COST_SCHEMA
        or result.get("status") != SCALE_COST_STATUS
        or result.get("actions") != list(ACTIONS)
        or result.get("scales") != list(PHASE2_SCALES)
        or result.get("profile_seed_dependency") is not False
        or not isinstance(rows, list)
        or len(rows) != 100
        or checks != _SCALE_COST_CHECKS
    ):
        raise RuntimeError("TMLR V6 phase-two scale cost registry drift")
    access = result.get("heldout_access", result.get("protected_access", {}))
    assert_zero_external_access(access)
    if result.get("implementation") != payload.get("phase2_implementation"):
        raise RuntimeError(
            "TMLR V6 phase-two scale cost implementation identity drift"
        )
    sources = result.get("source_identities", {})
    if not isinstance(sources, Mapping) or set(sources) != {
        "fit_and_trajectory_sources", "fold_manifests", "semantic_addendum",
    }:
        raise RuntimeError("TMLR V6 phase-two scale cost source registry drift")
    locked_folds = payload.get("fold_manifests", {})
    locked_addendum = payload.get("phase2_semantic_addendum", {})
    if (
        sources.get("fold_manifests") != locked_folds
        or not isinstance(locked_addendum, Mapping)
        or sources.get("semantic_addendum") != {
            key: locked_addendum.get(key) for key in _FILE_IDENTITY_KEYS
        }
    ):
        raise RuntimeError("TMLR V6 phase-two scale cost frozen-source drift")
    s1364_registration = payload.get("s1364_cost_profiles", {})
    if (
        not isinstance(s1364_registration, Mapping)
        or set(s1364_registration) != {"csv", "json"}
    ):
        raise RuntimeError("TMLR V6 phase-two parent cost source is unbound")
    _validate_file_identity(
        repository, s1364_registration["json"],
        expected_path=S1364_COST_JSON_PATH,
        description="scale-cost parent S1364 cost JSON",
    )
    s1364_payload = _read_json(
        repository, S1364_COST_JSON_PATH,
        description="scale-cost parent S1364 cost JSON",
    )
    if sources.get("fit_and_trajectory_sources") != s1364_payload.get(
        "source_identities"
    ):
        raise RuntimeError("TMLR V6 phase-two scale/parent cost source drift")
    required_fields = {
        "scale", "outer_fold", "action", "n", "median_seconds",
        "mean_seconds", "q25_seconds", "q75_seconds", "normalizer_seconds",
        "normalized_profiled_cost", "outer_training_group_ids_sha256",
        "heldout_access",
    }
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not required_fields <= set(row):
            raise RuntimeError("TMLR V6 phase-two scale cost row schema drift")
        key = (
            str(row.get("scale")), str(row.get("outer_fold")),
            str(row.get("action")),
        )
        if key in indexed:
            raise RuntimeError("TMLR V6 phase-two scale cost row is duplicated")
        if (
            key[0] not in PHASE2_SCALES
            or key[1] not in FOLDS
            or key[2] not in ACTIONS
            or _SHA256.fullmatch(str(row.get(
                "outer_training_group_ids_sha256", "",
            ))) is None
        ):
            raise RuntimeError("TMLR V6 phase-two scale cost key drift")
        row_access = row.get("heldout_access", {})
        if not isinstance(row_access, Mapping):
            raise RuntimeError("TMLR V6 phase-two scale cost access schema drift")
        assert_zero_external_access(row_access)
        numeric = [
            float(row[name]) for name in (
                "median_seconds", "mean_seconds", "q25_seconds",
                "q75_seconds", "normalizer_seconds", "normalized_profiled_cost",
            )
        ]
        if not all(math.isfinite(item) and item >= 0.0 for item in numeric):
            raise RuntimeError("TMLR V6 phase-two scale cost is invalid")
        if key[2] == "STOP" and (
            float(row["median_seconds"]) != 0.0
            or float(row["normalized_profiled_cost"]) != 0.0
        ):
            raise RuntimeError("TMLR V6 phase-two STOP cost is not zero")
        if key[2] != "STOP" and (
            int(row["n"]) <= 0
            or float(row["normalizer_seconds"]) <= 0.0
            or float(row["normalized_profiled_cost"]) <= 0.0
        ):
            raise RuntimeError("TMLR V6 phase-two adaptive cost is invalid")
        indexed[key] = row
    expected = {
        (scale, fold, action)
        for scale in PHASE2_SCALES for fold in FOLDS for action in ACTIONS
    }
    if set(indexed) != expected:
        raise RuntimeError("TMLR V6 phase-two scale cost coverage drift")
    csv_path = _safe_file(
        repository, PHASE2_SCALE_COST_CSV_PATH,
        description="phase-two scale cost CSV",
    )
    if csv_path.read_bytes() != phase2_scale_cost_csv_bytes(result):
        raise RuntimeError("TMLR V6 phase-two scale cost CSV/JSON drift")

    # Identities and algebra alone are not a scientific provenance root: a
    # synchronously altered JSON/CSV pair could otherwise retain the right
    # 100-row shape.  Rebuild all rows from the registered FIT-only reports
    # and four frozen fold manifests, then require byte equality.
    fit, fold_payloads, reports, source_identities = (
        load_registered_phase2_scale_cost_sources(repository)
    )
    rebuilt = build_phase2_scale_cost_profiles(
        fit_manifest=fit,
        fold_manifests=fold_payloads,
        trajectory_reports=reports,
        source_identities=source_identities,
    )
    rebuilt["implementation"] = dict(payload["phase2_implementation"])
    json_path = _safe_file(
        repository, PHASE2_SCALE_COST_JSON_PATH,
        description="phase-two scale cost JSON",
    )
    if json_path.read_bytes() != canonical_json_bytes(rebuilt) + b"\n":
        raise RuntimeError(
            "TMLR V6 phase-two scale costs do not reproduce from frozen sources"
        )
    if csv_path.read_bytes() != phase2_scale_cost_csv_bytes(rebuilt):
        raise RuntimeError(
            "TMLR V6 phase-two scale cost CSV does not reproduce from frozen sources"
        )
    return {
        **result,
        "artifact_identity": {"csv": csv_identity, "json": json_identity},
    }


def load_s1364_fold_cost_profile(
    repository: Path, lock: Phase2LaunchLock | Mapping[str, Any],
    outer_fold: str,
) -> np.ndarray:
    """Return the lock-bound, seed-invariant S1364 cost vector for one fold."""

    if outer_fold not in FOLDS:
        raise ValueError("unknown TMLR V6 outer fold")
    payload = lock.payload if isinstance(lock, Phase2LaunchLock) else lock
    registration = payload.get("s1364_cost_profiles", {})
    if not isinstance(registration, Mapping) or set(registration) != {"csv", "json"}:
        raise RuntimeError("TMLR V6 S1364 cost identity schema drift")
    csv_identity = _validate_file_identity(
        repository, registration["csv"], expected_path=S1364_COST_CSV_PATH,
        description="S1364 cost CSV",
    )
    _validate_file_identity(
        repository, registration["json"], expected_path=S1364_COST_JSON_PATH,
        description="S1364 cost JSON",
    )
    cost = _read_json(
        repository, S1364_COST_JSON_PATH, description="S1364 cost JSON",
    )
    if (
        cost.get("schema_version")
        != "rail3.tmlr-v6.action-type-cost-profiles.v1"
        or cost.get("status") != "TMLR_V6_FIT_ONLY_COST_PROFILES_PASS"
        or cost.get("actions") != list(ACTIONS)
        or cost.get("profile_seed_dependency") is not False
        or cost.get("checks") != _S1364_COST_CHECKS
    ):
        raise RuntimeError("TMLR V6 S1364 cost contract drift")
    assert_zero_external_access(cost.get("protected_access", {}))
    from rail3.models.tmlr_v6.cost_profiles import csv_bytes as cost_csv_bytes

    if _safe_file(
        repository, Path(csv_identity["path"]), description="S1364 cost CSV",
    ).read_bytes() != cost_csv_bytes(cost):
        raise RuntimeError("TMLR V6 S1364 cost CSV/JSON drift")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in cost.get("rows", ()):
        key = (str(row.get("fold")), str(row.get("action")))
        if key in indexed:
            raise RuntimeError("TMLR V6 S1364 cost row duplication")
        indexed[key] = row
    expected = {
        (fold, action) for fold in (*FOLDS, "FULLFIT") for action in ACTIONS
    }
    if set(indexed) != expected:
        raise RuntimeError("TMLR V6 S1364 cost row coverage drift")
    rows = [indexed[(outer_fold, action)] for action in ACTIONS]
    if (
        any(str(row.get("seed")) != "ALL" for row in rows)
        or len({str(row.get("profile_sha256")) for row in rows}) != 1
        or float(rows[0].get("normalized_median", math.nan)) != 0.0
    ):
        raise RuntimeError("TMLR V6 S1364 fold cost profile drift")
    vector = np.asarray(
        [float(row["normalized_median"]) for row in rows], dtype=np.float32,
    )
    if (
        vector.shape != (5,)
        or not np.isfinite(vector).all()
        or np.any(vector < 0.0)
        or vector[0] != 0.0
        or np.any(vector[1:] <= 0.0)
    ):
        raise RuntimeError("TMLR V6 S1364 normalized cost is invalid")
    return vector


def _launch_lock_id_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in sorted(_LAUNCH_LOCK_KEYS - {
            "schema_version", "status", "created_at_utc", "lock_id", "checks",
        })
    }


def phase2_launch_lock_id(value: Mapping[str, Any]) -> str:
    return stable_id(
        "tmlr_v6_phase2_launch_lock", _launch_lock_id_payload(value),
    )


@lru_cache(maxsize=8)
def _recomputed_first_batch_gate_bytes(
    repository_text: str, head: str, gate_sha256: str,
) -> bytes:
    """Re-audit the 75-run gate once per process/HEAD/identity."""

    del head, gate_sha256
    from rail3.models.tmlr_v6.inventory import build_first_batch_gate

    rebuilt = build_first_batch_gate(repository=Path(repository_text))
    return canonical_json_bytes(rebuilt) + b"\n"


def validate_phase2_launch_lock(repository: Path) -> Phase2LaunchLock:
    """Authenticate the immutable identity-only launch lock, with no bypass."""

    repository = repository.resolve()
    path = _safe_file(
        repository, PHASE2_LAUNCH_LOCK_PATH, description="phase-two launch lock",
    )
    lock_sha = sha256_file(path)
    addendum = load_phase2_addendum(repository)
    payload = _read_json(
        repository, PHASE2_LAUNCH_LOCK_PATH,
        description="phase-two launch lock",
    )
    if (
        set(payload) != _LAUNCH_LOCK_KEYS
        or payload.get("schema_version") != LAUNCH_LOCK_SCHEMA
        or payload.get("status") != LAUNCH_LOCK_STATUS
        or not isinstance(payload.get("created_at_utc"), str)
        or not payload["created_at_utc"].endswith("Z")
        or payload.get("checks") != {name: True for name in sorted(_LAUNCH_CHECKS)}
        or payload.get("job_counts") != expected_phase2_job_counts()
        or payload.get("lock_id") != phase2_launch_lock_id(payload)
    ):
        raise RuntimeError("TMLR V6 phase-two launch lock schema drift")
    assert_zero_external_access(payload.get("heldout_access", {}))

    parent = _validate_file_identity(
        repository, payload["parent_protocol"], expected_path=PROTOCOL_PATH,
        description="launch-bound parent protocol",
    )
    if (
        parent["bytes"] != int(addendum["parent_protocol"]["bytes"])
        or parent["sha256"] != str(addendum["parent_protocol"]["sha256"])
    ):
        raise RuntimeError("TMLR V6 launch-bound parent protocol drift")

    addendum_identity = payload["phase2_semantic_addendum"]
    if (
        not isinstance(addendum_identity, Mapping)
        or set(addendum_identity) != _FILE_IDENTITY_KEYS | {"commit"}
        or _COMMIT.fullmatch(str(addendum_identity.get("commit", ""))) is None
    ):
        raise RuntimeError("TMLR V6 launch-bound addendum identity schema drift")
    _validate_file_identity(
        repository,
        {key: addendum_identity[key] for key in _FILE_IDENTITY_KEYS},
        expected_path=PHASE2_ADDENDUM_PATH,
        description="launch-bound phase-two addendum",
    )
    addendum_commit = str(addendum_identity["commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", addendum_commit, "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("TMLR V6 phase-two addendum commit is not an ancestor")
    committed_addendum = subprocess.run(
        ["git", "show", f"{addendum_commit}:{PHASE2_ADDENDUM_PATH.as_posix()}"],
        cwd=repository, check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(committed_addendum).hexdigest() != addendum_identity["sha256"]:
        raise RuntimeError("TMLR V6 phase-two addendum committed identity drift")

    gate_identity = payload["first_batch_completion_gate"]
    if (
        not isinstance(gate_identity, Mapping)
        or set(gate_identity) != _FILE_IDENTITY_KEYS | {"gate_id"}
    ):
        raise RuntimeError("TMLR V6 first-batch gate identity schema drift")
    _validate_file_identity(
        repository,
        {key: gate_identity[key] for key in _FILE_IDENTITY_KEYS},
        expected_path=FIRST_BATCH_GATE_PATH,
        description="first-batch completion gate",
    )
    gate = _read_json(
        repository, FIRST_BATCH_GATE_PATH, description="first-batch gate",
    )
    frozen_gate = addendum[
        "first_batch_gate_and_phase2_launch_lock"
    ]["first_batch_gate"]
    if (
        set(gate) != _FIRST_GATE_KEYS
        or gate.get("schema_version") != frozen_gate["schema_version"]
        or gate.get("status") != frozen_gate["required_status"]
        or gate.get("protocol_sha256") != parent["sha256"]
        or int(gate.get("completed_valid_runs", -1)) != 75
        or gate.get("second_batch_authorized_by_gate") is not True
        or gate.get("checks") != _FIRST_GATE_CHECKS
        or gate.get("gate_id") != gate_identity.get("gate_id")
    ):
        raise RuntimeError("TMLR V6 first-batch gate is not safety/completeness-only")
    assert_zero_external_access(gate.get("heldout_access", {}))

    inventory_identity = _validate_file_identity(
        repository, payload["first_batch_run_inventory"],
        expected_path=FIRST_BATCH_INVENTORY_PATH,
        description="first-batch run inventory",
    )
    if gate.get("inventory") != inventory_identity:
        raise RuntimeError("TMLR V6 first-batch gate/inventory identity drift")
    inventory = _read_json(
        repository, FIRST_BATCH_INVENTORY_PATH,
        description="first-batch run inventory",
    )
    if (
        inventory.get("schema_version")
        != "rail3.tmlr-v6.first-batch-run-inventory.v1"
        or inventory.get("status")
        != "TMLR_V6_FIRST_BATCH_RUN_INVENTORY_PASS"
        or int(inventory.get("completed_valid_runs", -1)) != 75
        or not isinstance(inventory.get("checks"), Mapping)
        or not all(bool(value) for value in inventory["checks"].values())
    ):
        raise RuntimeError("TMLR V6 first-batch inventory is incomplete")
    assert_zero_external_access(inventory.get("heldout_access", {}))

    evaluation_identity = _validate_file_identity(
        repository, payload["first_batch_evaluation"],
        expected_path=FIRST_BATCH_EVALUATION_PATH,
        description="first-batch evaluation",
    )
    gate_evaluation = gate.get("first_batch_evaluation", {})
    if (
        not isinstance(gate_evaluation, Mapping)
        or gate_evaluation.get("path") != evaluation_identity["path"]
        or gate_evaluation.get("sha256") != evaluation_identity["sha256"]
        or gate_evaluation.get("status")
        != "TMLR_V6_FIRST_BATCH_EVALUATION_PASS"
    ):
        raise RuntimeError("TMLR V6 first-batch gate/evaluation identity drift")
    evaluation = _read_json(
        repository, FIRST_BATCH_EVALUATION_PATH,
        description="first-batch evaluation",
    )
    if (
        evaluation.get("schema_version")
        != "rail3.tmlr-v6.first-batch-fit-evaluation.v1"
        or evaluation.get("status") != "TMLR_V6_FIRST_BATCH_EVALUATION_PASS"
        or evaluation.get("metric_pipeline_pass") is not True
        or not isinstance(evaluation.get("checks"), Mapping)
        or not all(bool(value) for value in evaluation["checks"].values())
    ):
        raise RuntimeError("TMLR V6 first-batch metric pipeline is incomplete")
    assert_zero_external_access(
        evaluation.get("heldout_access", evaluation.get("protected_access", {}))
    )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if path.parent.joinpath("first_batch_completion_gate.json").resolve() != (
        repository / FIRST_BATCH_GATE_PATH
    ).resolve():
        raise RuntimeError("TMLR V6 first-batch gate fixed path drift")
    gate_path = _safe_file(
        repository, FIRST_BATCH_GATE_PATH,
        description="first-batch completion gate",
    )
    if gate_path.read_bytes() != _recomputed_first_batch_gate_bytes(
        str(repository), head, str(gate_identity["sha256"]),
    ):
        raise RuntimeError("TMLR V6 first-batch gate is not reproducible")

    bound = addendum["bound_parent_registries"]
    if payload["prospective_bundle_registry_canonical_sha256"] != bound[
        "prospective_bundle_identities"
    ]["canonical_json_sha256"]:
        raise RuntimeError("TMLR V6 launch-bound prospective registry drift")
    historical = _validate_file_identity(
        repository, payload["historical_fit_oof_inventory"],
        expected_path=Path(bound["historical_fit_oof_inventory"]["path"]),
        description="historical FIT OOF inventory",
    )
    if any(
        historical[name] != bound["historical_fit_oof_inventory"][name]
        for name in _FILE_IDENTITY_KEYS
    ):
        raise RuntimeError("TMLR V6 historical FIT OOF identity drift")

    s1364 = payload["s1364_cost_profiles"]
    if not isinstance(s1364, Mapping) or set(s1364) != {"csv", "json"}:
        raise RuntimeError("TMLR V6 S1364 cost launch identity schema drift")
    for kind, path_key in (("csv", S1364_COST_CSV_PATH), ("json", S1364_COST_JSON_PATH)):
        identity = _validate_file_identity(
            repository, s1364[kind], expected_path=path_key,
            description=f"S1364 cost {kind}",
        )
        expected = bound["cost_profile"][kind]
        if any(identity[name] != expected[name] for name in _FILE_IDENTITY_KEYS):
            raise RuntimeError("TMLR V6 S1364 cost identity drift")

    _validate_fold_identities(repository, payload["fold_manifests"], addendum)
    load_scale_cost_profile_registry(repository, payload)
    load_s1364_fold_cost_profile(repository, payload, FOLDS[0])
    _validate_implementation_identity(repository, payload["phase2_implementation"])

    return Phase2LaunchLock(
        payload=payload,
        path=path,
        bytes=path.stat().st_size,
        sha256=lock_sha,
    )


__all__ = [
    "F0_F1_RUNS",
    "FIRST_BATCH_EVALUATION_PATH",
    "FIRST_BATCH_GATE_PATH",
    "FIRST_BATCH_INVENTORY_PATH",
    "FROZEN_ADDENDUM_BYTES",
    "FROZEN_ADDENDUM_SHA256",
    "LAUNCH_LOCK_SCHEMA",
    "LAUNCH_LOCK_STATUS",
    "PHASE2_ADDENDUM_PATH",
    "PHASE2_FOLD_PATHS",
    "PHASE2_IMPLEMENTATION_PATHS",
    "PHASE2_INTEGRITY_TEST_PATHS",
    "PHASE2_LAUNCH_LOCK_PATH",
    "PHASE2_SCALE_COST_CSV_PATH",
    "PHASE2_SCALE_COST_JSON_PATH",
    "PHASE2_SCALES",
    "PHASE2_TOTAL_RUNS",
    "S1364_COST_CSV_PATH",
    "S1364_COST_JSON_PATH",
    "SECOND_BATCH_RUNS",
    "Phase2Job",
    "Phase2LaunchLock",
    "expected_phase2_job_counts",
    "expected_second_batch_jobs",
    "file_identity",
    "load_phase2_addendum",
    "load_s1364_fold_cost_profile",
    "load_scale_cost_profile_registry",
    "phase2_implementation_identity",
    "phase2_launch_lock_id",
    "safe_phase2_repository_path",
    "verify_exact_rebuild_directory",
    "second_batch_run_path",
    "validate_phase2_addendum_payload",
    "validate_phase2_launch_lock",
]
