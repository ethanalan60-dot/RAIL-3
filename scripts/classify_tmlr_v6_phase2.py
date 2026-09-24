#!/usr/bin/env python3
"""Mechanically classify the complete TMLR V6 prospective FIT evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.evaluate_tmlr_v6_phase2_second_batch_fit import (
    DEFAULT_OUTPUT_ROOT as SECOND_OUTPUT_ROOT,
)
from scripts.evaluate_tmlr_v6_phase2_f0_f1_fit import (
    DEFAULT_OUTPUT_ROOT as F0_F1_OUTPUT_ROOT,
)
from scripts.evaluate_tmlr_v6_phase2_sensitivity_fit import (
    DEFAULT_OUTPUT_ROOT as SENSITIVITY_OUTPUT_ROOT,
    SENSITIVITY_REPORT_CHECKS,
)
from scripts.build_tmlr_v6_phase2_qualitative_atlas import (
    DEFAULT_OUTPUT_ROOT as QUALITATIVE_OUTPUT_ROOT,
    QUALITATIVE_REPORT_CHECKS,
    SOURCE_FIELDS as QUALITATIVE_SOURCE_FIELDS,
)
from scripts.evaluate_tmlr_v6_first_batch_fit import load_first_batch_surfaces
from rail3.contracts import canonical_json_bytes
from rail3.analysis.tmlr_v6_phase2 import (
    BUDGETS,
    BUDGET_SEMANTICS,
    EXTERNAL_SPLITS,
    LAMBDAS,
    PRIMARY_SPARSE_BUDGETS,
    Phase2Surface,
    assert_zero_external_access,
    classify_prospective_core,
    derive_mechanical_predicates,
    direct_decision_family_interpretation,
    paired_bootstrap_group_counts,
    paired_ratio_bootstrap,
    residual_ratio_sufficient,
    validate_prediction_metric_row,
    validate_f0_f1_report_contract,
    validate_second_batch_report_contract,
)
from rail3.analysis.tmlr_v6_sensitivity import (
    SUBSET_NAMES,
    UTILITY_NAMES,
    sensitivity_reversal_predicate,
)
from rail3.analysis.tmlr_v6_qualitative import validate_selection_manifest
from rail3.models.tmlr_v6.phase2_contract import (
    FIRST_BATCH_EVALUATION_PATH,
    FIRST_BATCH_GATE_PATH,
    PHASE2_INTEGRITY_TEST_PATHS,
    validate_phase2_launch_lock,
)
from rail3.models.tmlr_v6.inventory import (
    DEFAULT_INVENTORY_PATH as FIRST_BATCH_INVENTORY_PATH,
    build_first_batch_gate,
)
from rail3.models.tmlr_v6.phase2_inventory import (
    DEFAULT_SECOND_BATCH_INVENTORY_PATH,
    SECOND_BATCH_INVENTORY_STATUS,
    audit_phase2_second_batch_runs,
)
from rail3.models.tmlr_v6.phase2_f0_f1 import (
    F0_F1_INVENTORY_PATH,
    INVENTORY_STATUS as F0_F1_INVENTORY_STATUS,
    audit_f0_f1_runs,
    f0_f1_inventory_artifact_payload,
)
from rail3.models.tmlr_v6.training import (
    load_protocol,
    sha256_file,
    validate_fold_manifest,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/paper/source_data/tmlr_v6/phase2_classification"
)
_FORBIDDEN_PATH_FRAGMENTS = (
    "artifacts/protected",
    "validation40",
    "locked40",
    "calibration30",
    "pilot-test30",
    "pilot_test30",
    "official-voc",
    "official_voc",
    "segppd",
    "uav-iap",
    "uav_iap",
)


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 classification must run at worktree root")
    return root


def _inside(repository: Path, path: Path) -> Path:
    root = repository.absolute()
    requested = path if path.is_absolute() else root / path
    requested = Path(os.path.abspath(requested))
    try:
        relative = requested.relative_to(root)
    except ValueError as error:
        raise RuntimeError("TMLR V6 classification path escapes repository") from error
    normalized = relative.as_posix().lower()
    if any(fragment in normalized for fragment in _FORBIDDEN_PATH_FRAGMENTS):
        raise RuntimeError("TMLR V6 classification path enters a forbidden boundary")
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("TMLR V6 classification path uses a symlink")
    result = requested.resolve(strict=False)
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError("TMLR V6 classification path resolves outside repository") from error
    return result


def _load_identity_json(
    *, repository: Path, identity: Mapping[str, Any], description: str,
    expected_path: Path,
) -> dict[str, Any]:
    declared = Path(str(identity.get("path", "")))
    if declared.is_absolute() or declared.as_posix() != expected_path.as_posix():
        raise RuntimeError(f"TMLR V6 {description} logical path drift")
    path = _inside(repository, declared)
    metadata = path.lstat() if path.exists() else None
    if (
        metadata is None
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != str(identity.get("sha256", ""))
    ):
        raise RuntimeError(f"TMLR V6 {description} identity drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"TMLR V6 {description} is not a JSON object")
    return payload


def _single_link_regular_bytes(path: Path, *, description: str) -> bytes:
    """Read a fixed artifact only after a non-aliasing regular-file gate."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"TMLR V6 {description} is absent") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(
            f"TMLR V6 {description} is not one regular single-link file"
        )
    return path.read_bytes()


def _verify_output_identities(
    *, repository: Path, report: Mapping[str, Any], description: str,
    expected_root: Path, expected_rows: Mapping[str, int] | None = None,
    expected_names: set[str] | None = None,
) -> None:
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise RuntimeError(f"TMLR V6 {description} has no bound outputs")
    if expected_rows is not None and set(outputs) != set(expected_rows):
        raise RuntimeError(f"TMLR V6 {description} output registry drift")
    if expected_names is not None and set(outputs) != expected_names:
        raise RuntimeError(f"TMLR V6 {description} output registry drift")
    for name, identity in outputs.items():
        if not isinstance(name, str) or not isinstance(identity, Mapping):
            raise RuntimeError(f"TMLR V6 {description} output registry drift")
        declared = Path(str(identity.get("path", "")))
        expected = expected_root / name
        if declared.is_absolute() or declared.as_posix() != expected.as_posix():
            raise RuntimeError(
                f"TMLR V6 {description} output logical path drift: {name}"
            )
        path = _inside(repository, declared)
        metadata = path.lstat() if path.exists() else None
        if (
            path.name != name
            or metadata is None
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(path) != str(identity.get("sha256", ""))
            or (
                expected_rows is not None
                and identity.get("rows") != expected_rows[name]
            )
        ):
            raise RuntimeError(f"TMLR V6 {description} output identity drift: {name}")


def _load_report_and_metrics(
    *, repository: Path, report_path: Path, metrics_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _inside(repository, report_path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"TMLR V6 required report is absent: {report_path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report.get("outputs", {}).get(metrics_name)
    if not isinstance(identity, Mapping):
        raise RuntimeError("TMLR V6 report does not bind its machine metrics")
    metrics = _load_identity_json(
        repository=repository, identity=identity,
        description=f"{metrics_name} machine metrics",
        expected_path=report_path.parent / metrics_name,
    )
    return report, metrics


def _first_metrics(
    *, repository: Path, launch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _load_identity_json(
        repository=repository,
        identity=launch.payload["first_batch_evaluation"],
        description="first-batch evaluation",
        expected_path=FIRST_BATCH_EVALUATION_PATH,
    )
    metrics = _load_identity_json(
        repository=repository,
        identity=report["outputs"]["first_batch_metrics.json"],
        description="first-batch machine metrics",
        expected_path=FIRST_BATCH_EVALUATION_PATH.parent / "first_batch_metrics.json",
    )
    return report, metrics


def _mechanical_comparisons(repository: Path) -> list[dict[str, Any]]:
    """Recompute all seven addendum-frozen contrasts under one exact bootstrap."""

    protocol = load_protocol(repository)
    folds = validate_fold_manifest(repository, protocol)
    first, _ = load_first_batch_surfaces(
        repository=repository, protocol=protocol, fold_manifest=folds,
    )
    converted = {
        model: Phase2Surface(
            model_id=model,
            prediction_kind="residual",
            prediction=surface.prediction,
            target=surface.target,
            feasible=surface.feasible,
            groups=surface.groups,
            state_ids=surface.state_ids,
        ).validated()
        for model, surface in first.items()
    }
    reference = converted["R1_P"]
    for surface in converted.values():
        if (
            surface.state_ids != reference.state_ids
            or not np.array_equal(surface.groups, reference.groups)
            or not np.array_equal(surface.feasible, reference.feasible)
            or not np.array_equal(surface.target, reference.target, equal_nan=True)
        ):
            raise RuntimeError("TMLR V6 mechanical comparison alignment drift")
    _, shared_bootstrap_counts = paired_bootstrap_group_counts(
        reference.groups, expected_groups=1364,
    )
    result = []
    for left, right in (
        ("R1_P", "R0_SMALL_P"),
        ("R1_P", "R0_CM_P"),
        ("R1_P", "RECT_P"),
        ("R1_P", "UNION_P"),
        ("R0_CM_P", "R0_SMALL_P"),
        ("RECT_P", "R0_CM_P"),
        ("UNION_P", "R0_CM_P"),
    ):
        left_values = residual_ratio_sufficient(converted[left])
        right_values = residual_ratio_sufficient(converted[right])
        for metric in ("absolute_residual_mae", "drre"):
            left_num, left_den = left_values[metric]
            right_num, right_den = right_values[metric]
            result.append({
                **paired_ratio_bootstrap(
                    groups=converted[left].groups,
                    left_numerator=left_num,
                    left_denominator=left_den,
                    right_numerator=right_num,
                    right_denominator=right_den,
                    contrast_id=f"{left}_MINUS_{right}",
                    metric=metric,
                    expected_groups=1364,
                    bootstrap_counts=shared_bootstrap_counts,
                ),
                "left_model": left,
                "right_model": right,
            })
    return result


def _integrity_checks(repository: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = "src:."

    def run(
        command: Sequence[str], *, require_empty_stdout: bool = False,
    ) -> dict[str, Any]:
        result = subprocess.run(
            list(command), cwd=repository, capture_output=True, text=True,
            env=environment,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
            "pass": (
                result.returncode == 0
                and (not require_empty_stdout or not result.stdout.strip())
            ),
        }

    return {
        "rebuild_first_batch_outputs": run([
            sys.executable, "scripts/evaluate_tmlr_v6_first_batch_fit.py",
            "--check",
        ]),
        "rebuild_second_batch_outputs": run([
            sys.executable,
            "scripts/evaluate_tmlr_v6_phase2_second_batch_fit.py", "--check",
        ]),
        "rebuild_f0_f1_outputs": run([
            sys.executable,
            "scripts/evaluate_tmlr_v6_phase2_f0_f1_fit.py", "--check",
        ]),
        "rebuild_sensitivity_outputs": run([
            sys.executable,
            "scripts/evaluate_tmlr_v6_phase2_sensitivity_fit.py", "--check",
        ]),
        "rebuild_qualitative_outputs": run([
            sys.executable,
            "scripts/build_tmlr_v6_phase2_qualitative_atlas.py", "--check",
        ]),
        "named_integrity_tests": run([
            sys.executable,
            "-m",
            "unittest",
            *(
                path.with_suffix("").as_posix().replace("/", ".")
                for path in PHASE2_INTEGRITY_TEST_PATHS
            ),
        ]),
        "compileall": run([
            sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests",
        ]),
        "pip_check": run([sys.executable, "-m", "pip", "check"]),
        "git_tests_clean": run([
            "git", "status", "--porcelain", "--untracked-files=all", "--", "tests",
        ], require_empty_stdout=True),
        "git_diff_check": run(["git", "diff", "--check"]),
        "git_fsck_full": run(["git", "fsck", "--full"]),
    }


def _verify_recomputed_workload_inventories(
    repository: Path,
) -> dict[str, dict[str, Any]]:
    """Recompute all 420 training-run gates before a terminal label exists."""

    first_gate = build_first_batch_gate(repository=repository)
    first_gate_path = _inside(repository, FIRST_BATCH_GATE_PATH)
    if _single_link_regular_bytes(
        first_gate_path, description="first-batch completion gate",
    ) != canonical_json_bytes(first_gate) + b"\n":
        raise RuntimeError("TMLR V6 first-batch gate is not reproducible")
    first_inventory_path = _inside(repository, FIRST_BATCH_INVENTORY_PATH)
    first_inventory_bytes = _single_link_regular_bytes(
        first_inventory_path, description="first-batch run inventory",
    )
    first_inventory = json.loads(first_inventory_bytes)
    first_checks = {
        "exact_cross_product_75",
        "physical_feature_target_separation",
        "prospective_schema_only",
        "runtime_feature_count_zero",
        "post_action_diagnostic_access_zero",
        "ground_truth_target_weights_absent_from_prediction_graph",
        "same_state_target_across_roles",
        "same_action_feasibility_across_roles",
        "exact_state_oof_coverage",
        "exact_atom_oof_coverage_for_atomic_models",
        "saved_additive_reconstruction_exact",
        "all_declared_files_hash_valid",
        "all_run_ids_deterministic_and_unique",
        "all_training_states_finite",
        "single_training_implementation_commit",
        "single_training_source_contract",
        "exact_cuda_determinism_environment",
        "protected_access_zero",
    }
    if (
        first_gate.get("status") != "TMLR_V6_P0_FIRST_BATCH_GATE_PASS"
        or first_gate.get("inventory", {}).get("path")
        != FIRST_BATCH_INVENTORY_PATH.as_posix()
        or first_gate.get("inventory", {}).get("bytes") != len(first_inventory_bytes)
        or first_gate.get("inventory", {}).get("sha256")
        != sha256_file(first_inventory_path)
        or first_inventory.get("status")
        != "TMLR_V6_FIRST_BATCH_RUN_INVENTORY_PASS"
        or first_inventory.get("expected_runs") != 75
        or first_inventory.get("completed_valid_runs") != 75
        or first_inventory.get("missing_runs") != 0
        or first_inventory.get("unexpected_runs") != 0
        or set(first_inventory.get("checks", {})) != first_checks
        or any(value is not True for value in first_inventory["checks"].values())
    ):
        raise RuntimeError("TMLR V6 first-batch workload proof drift")
    assert_zero_external_access(first_inventory.get("heldout_access", {}))

    second = audit_phase2_second_batch_runs(repository=repository)
    second_path = _inside(repository, DEFAULT_SECOND_BATCH_INVENTORY_PATH)
    if _single_link_regular_bytes(
        second_path, description="second-batch run inventory",
    ) != canonical_json_bytes(second) + b"\n":
        raise RuntimeError("TMLR V6 second-batch inventory is not reproducible")
    second_checks = {
        "launch_lock_no_bypass", "exact_cross_product_225",
        "ll4tta_is_zero_job_logical_alias", "immutable_three_file_run_tree",
        "all_artifact_hashes_valid", "all_run_ids_unique",
        "single_committed_clean_implementation", "predictor_cost_input_zero",
        "exact_state_oof_coverage", "exact_atom_oof_coverage_for_r2_r3",
        "q2_raw_stop_saved_but_not_scientifically_consumed_before_ensemble",
        "all_training_states_finite", "resource_limits_pass",
        "protected_access_zero",
    }
    if (
        second.get("status") != SECOND_BATCH_INVENTORY_STATUS
        or second.get("expected_runs") != 225
        or second.get("completed_valid_runs") != 225
        or second.get("missing_runs") != 0
        or second.get("unexpected_runs") != 0
        or set(second.get("checks", {})) != second_checks
        or any(value is not True for value in second["checks"].values())
    ):
        raise RuntimeError("TMLR V6 second-batch workload proof drift")
    assert_zero_external_access(second.get("heldout_access", {}))

    f0_f1 = f0_f1_inventory_artifact_payload(audit_f0_f1_runs(repository))
    f0_f1_path = _inside(repository, F0_F1_INVENTORY_PATH)
    if _single_link_regular_bytes(
        f0_f1_path, description="F0/F1 run inventory",
    ) != canonical_json_bytes(f0_f1) + b"\n":
        raise RuntimeError("TMLR V6 F0/F1 inventory is not reproducible")
    f0_checks = {
        "exact_job_product_120", "exact_three_files_per_run",
        "a3_prediction_nan", "remaining_predictions_finite",
        "outer_heldout_coverage_exact", "scales_complete",
        "filtrations_complete", "protected_access_zero",
    }
    if (
        f0_f1.get("status") != F0_F1_INVENTORY_STATUS
        or f0_f1.get("expected_jobs") != 120
        or f0_f1.get("validated_runs") != 120
        or len(f0_f1.get("rows", ())) != 120
        or set(f0_f1.get("checks", {})) != f0_checks
        or any(value is not True for value in f0_f1["checks"].values())
    ):
        raise RuntimeError("TMLR V6 F0/F1 workload proof drift")
    assert_zero_external_access(f0_f1.get("heldout_access", {}))

    return {
        "first_batch_gate": {
            "path": FIRST_BATCH_GATE_PATH.as_posix(),
            "bytes": first_gate_path.stat().st_size,
            "sha256": sha256_file(first_gate_path),
        },
        "first_batch_inventory": {
            "path": FIRST_BATCH_INVENTORY_PATH.as_posix(),
            "bytes": first_inventory_path.stat().st_size,
            "sha256": sha256_file(first_inventory_path),
            "completed_valid_runs": 75,
        },
        "second_batch_inventory": {
            "path": DEFAULT_SECOND_BATCH_INVENTORY_PATH.as_posix(),
            "bytes": second_path.stat().st_size,
            "sha256": sha256_file(second_path),
            "completed_valid_runs": int(second["completed_valid_runs"]),
        },
        "f0_f1_inventory": {
            "path": F0_F1_INVENTORY_PATH.as_posix(),
            "bytes": f0_f1_path.stat().st_size,
            "sha256": sha256_file(f0_f1_path),
            "completed_valid_runs": int(f0_f1["validated_runs"]),
        },
    }


def _exact_row_registry(
    rows: Any, *, fields: Sequence[str], expected: set[tuple[Any, ...]],
) -> bool:
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    try:
        observed = [tuple(row[field] for field in fields) for row in rows]
    except (KeyError, TypeError):
        return False
    return len(set(observed)) == len(observed) and set(observed) == expected


_SENSITIVITY_OUTPUT_ROWS = {
    "phase2_utility_audit.csv": 4,
    "phase2_utility_sparse_budget_metrics.csv": 24,
    "phase2_target_rich_subset_metrics.csv": 24,
    "phase2_sensitivity_metrics.json": 52,
}
_PHASE2_BINDING_NAMES = {
    "phase2_launch_lock_sha256",
    "first_batch_completion_gate_sha256",
    "first_batch_run_inventory_sha256",
    "first_batch_evaluation_sha256",
}


def _validate_sensitivity_report_contract(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version")
        != "rail3.tmlr-v6.phase2-fit-utility-sensitivity.v1"
        or report.get("status")
        != "TMLR_V6_PHASE2_FIT_UTILITY_SENSITIVITY_PASS"
        or report.get("interpretation") != "FIT_ONLY_POST_HOC_SENSITIVITY"
        or report.get("confirmatory_claim_allowed") is not False
        or report.get("metric_specific_training") is not False
        or report.get("sam_inference") is not False
        or not isinstance(report.get("checks"), Mapping)
        or set(report["checks"]) != set(SENSITIVITY_REPORT_CHECKS)
        or any(value is not True for value in report["checks"].values())
        or not isinstance(report.get("input_bindings"), Mapping)
        or set(report["input_bindings"]) != _PHASE2_BINDING_NAMES
    ):
        raise RuntimeError("TMLR V6 sensitivity report contract drift")
    outputs = report.get("outputs")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != set(_SENSITIVITY_OUTPUT_ROWS)
        or any(
            not isinstance(outputs[name], Mapping)
            or outputs[name].get("rows") != rows
            for name, rows in _SENSITIVITY_OUTPUT_ROWS.items()
        )
    ):
        raise RuntimeError("TMLR V6 sensitivity output registry drift")
    assert_zero_external_access(report.get("heldout_access", {}))


def _validate_qualitative_output_contract(
    *, repository: Path, report: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> set[str]:
    if (
        report.get("schema_version")
        != "rail3.tmlr-v6.phase2-qualitative-atlas.v1"
        or report.get("status") != "TMLR_V6_PHASE2_QUALITATIVE_ATLAS_PASS"
        or report.get("manual_selection") is not False
        or report.get("predicate_relaxation") is not False
        or report.get("sam_inference") is not False
        or report.get("categories") != selection.get("categories")
        or not isinstance(report.get("checks"), Mapping)
        or set(report["checks"]) != set(QUALITATIVE_REPORT_CHECKS)
        or any(value is not True for value in report["checks"].values())
        or not isinstance(report.get("input_bindings"), Mapping)
        or set(report["input_bindings"])
        != _PHASE2_BINDING_NAMES | {"fit_only_input_inventory", "historical_fit_oof"}
    ):
        raise RuntimeError("TMLR V6 qualitative report contract drift")
    assert_zero_external_access(report.get("heldout_access", {}))
    entries = selection.get("categories", ())
    selected = {
        str(item["category"])
        for item in entries
        if item.get("status") == "SELECTED"
    }
    expected_names = {
        "qualitative_cases_source.csv",
        "qualitative_selection_manifest.json",
        *{
            f"{category}.{suffix}"
            for category in selected
            for suffix in ("png", "pdf", "svg")
        },
    }
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != expected_names:
        raise RuntimeError("TMLR V6 qualitative output registry drift")
    _verify_output_identities(
        repository=repository,
        report=report,
        description="qualitative atlas",
        expected_root=QUALITATIVE_OUTPUT_ROOT,
        expected_names=expected_names,
    )
    if set(report.get("case_assets", {})) != selected:
        raise RuntimeError("TMLR V6 qualitative case-asset registry drift")
    source_path = _inside(
        repository, QUALITATIVE_OUTPUT_ROOT / "qualitative_cases_source.csv",
    )
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != tuple(QUALITATIVE_SOURCE_FIELDS):
            raise RuntimeError("TMLR V6 qualitative source CSV field drift")
    observed = [(row.get("category"), row.get("action")) for row in rows]
    expected_rows = {
        (category, action)
        for category in selected
        for action in ("STOP", "A3", "A4", "A5", "A6")
    }
    if len(observed) != len(expected_rows) or set(observed) != expected_rows:
        raise RuntimeError("TMLR V6 qualitative source CSV row drift")
    return expected_names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    repository = _repository()
    launch = validate_phase2_launch_lock(repository)
    first_report, first_metrics = _first_metrics(repository=repository, launch=launch)
    second_report, second_metrics = _load_report_and_metrics(
        repository=repository,
        report_path=SECOND_OUTPUT_ROOT / "phase2_second_batch_evaluation_report.json",
        metrics_name="phase2_second_batch_metrics.json",
    )
    validate_second_batch_report_contract(second_report)
    f0_report, f0_metrics = _load_report_and_metrics(
        repository=repository,
        report_path=F0_F1_OUTPUT_ROOT / "phase2_f0_f1_evaluation_report.json",
        metrics_name="phase2_f0_f1_metrics.json",
    )
    validate_f0_f1_report_contract(f0_report)
    sensitivity_report, sensitivity_metrics = _load_report_and_metrics(
        repository=repository,
        report_path=SENSITIVITY_OUTPUT_ROOT / "phase2_sensitivity_report.json",
        metrics_name="phase2_sensitivity_metrics.json",
    )
    qualitative_path = _inside(
        repository, QUALITATIVE_OUTPUT_ROOT / "qualitative_atlas_manifest.json",
    )
    if qualitative_path.is_symlink() or not qualitative_path.is_file():
        raise RuntimeError("TMLR V6 qualitative atlas manifest is absent or unsafe")
    qualitative = json.loads(qualitative_path.read_text(encoding="utf-8"))
    qualitative_selection = _load_identity_json(
        repository=repository,
        identity=qualitative.get("outputs", {}).get(
            "qualitative_selection_manifest.json", {}
        ) if isinstance(qualitative, Mapping) else {},
        description="qualitative selection manifest",
        expected_path=(
            QUALITATIVE_OUTPUT_ROOT / "qualitative_selection_manifest.json"
        ),
    )
    validate_selection_manifest(qualitative_selection)
    workload_proof = _verify_recomputed_workload_inventories(repository)
    _validate_sensitivity_report_contract(sensitivity_report)
    qualitative_output_names = _validate_qualitative_output_contract(
        repository=repository, report=qualitative,
        selection=qualitative_selection,
    )
    for description, report, expected_root in (
        ("first-batch evaluation", first_report, FIRST_BATCH_EVALUATION_PATH.parent),
        ("second-batch evaluation", second_report, SECOND_OUTPUT_ROOT),
        ("F0/F1 evaluation", f0_report, F0_F1_OUTPUT_ROOT),
        ("sensitivity evaluation", sensitivity_report, SENSITIVITY_OUTPUT_ROOT),
        ("qualitative atlas", qualitative, QUALITATIVE_OUTPUT_ROOT),
    ):
        _verify_output_identities(
            repository=repository, report=report, description=description,
            expected_root=expected_root,
            expected_rows=(
                _SENSITIVITY_OUTPUT_ROWS
                if report is sensitivity_report else None
            ),
            expected_names=(
                qualitative_output_names if report is qualitative else None
            ),
        )
    for row in second_metrics["prediction_metrics"]:
        validate_prediction_metric_row(row, extra_fields=("trained_lambda",))
    for row in f0_metrics["prediction_metrics"]:
        validate_prediction_metric_row(row, extra_fields=("scale",))

    comparison_rows = _mechanical_comparisons(repository)
    decision_rows = [
        *first_metrics["decision_metrics"],
        *second_metrics["decision_metrics"],
    ]
    sparse_rows = [
        *first_metrics["sparse_budget_metrics"],
        *second_metrics["sparse_budget_metrics"],
    ]
    direct_family = direct_decision_family_interpretation(
        decision_rows=decision_rows, sparse_rows=sparse_rows,
    )
    provisional = derive_mechanical_predicates(
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
        sparse_rows=sparse_rows,
        post_hoc_reversal=False,
    )
    reversal = sensitivity_reversal_predicate(
        primary_gap_persists=provisional[
            "PREDICTION_INTERVENTION_GAP_PERSISTS"
        ],
        sensitivity_budget_rows=sensitivity_metrics[
            "utility_sparse_budget_metrics"
        ],
        subset_rows=sensitivity_metrics["target_rich_subset_metrics"],
    )
    predicates = derive_mechanical_predicates(
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
        sparse_rows=sparse_rows,
        post_hoc_reversal=reversal,
    )
    qa = _integrity_checks(repository)
    expected_bindings = {
        "phase2_launch_lock_sha256": launch.sha256,
        "first_batch_completion_gate_sha256": launch.payload[
            "first_batch_completion_gate"
        ]["sha256"],
        "first_batch_run_inventory_sha256": launch.payload[
            "first_batch_run_inventory"
        ]["sha256"],
        "first_batch_evaluation_sha256": launch.payload[
            "first_batch_evaluation"
        ]["sha256"],
    }
    second_models = (
        "R2_P", "R3_P", "Q2_P", "LL4TTA_P", "L2D_P", "SPO_PLUS_P",
    )
    first_models = (
        "R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P",
    )
    first_prediction_keys = {(model,) for model in first_models}
    first_decision_keys = {
        (model, value) for model in first_models for value in LAMBDAS
    }
    first_sparse_keys = {
        (model, value, budget, semantics)
        for model in first_models
        for value in LAMBDAS
        for budget in BUDGETS
        for semantics in BUDGET_SEMANTICS
    }
    first_bootstrap_keys = {
        (left, right, metric)
        for left, right in (
            ("R1_P", "R0_SMALL_P"),
            ("R1_P", "R0_CM_P"),
            ("R1_P", "RECT_P"),
            ("R1_P", "UNION_P"),
        )
        for metric in ("absolute_residual_mae", "drre")
    }
    second_prediction_keys = {
        (model, None) for model in ("R2_P", "R3_P", "Q2_P", "LL4TTA_P")
    } | {
        (model, value)
        for model in ("L2D_P", "SPO_PLUS_P") for value in LAMBDAS
    }
    second_decision_keys = {
        (model, value) for model in second_models for value in LAMBDAS
    }
    second_sparse_keys = {
        (model, value, budget, semantics)
        for model in second_models
        for value in LAMBDAS
        for budget in BUDGETS
        for semantics in BUDGET_SEMANTICS
    }
    scales = ("S250", "S500", "S1000", "S1364")
    filtrations = ("F0_P", "F1_P")
    f0_prediction_keys = {
        (scale, filtration) for scale in scales for filtration in filtrations
    }
    f0_decision_keys = {
        (scale, filtration, value)
        for scale in scales for filtration in filtrations for value in LAMBDAS
    }
    f0_sparse_keys = {
        (scale, filtration, value, budget, semantics)
        for scale in scales
        for filtration in filtrations
        for value in LAMBDAS
        for budget in BUDGETS
        for semantics in BUDGET_SEMANTICS
    }
    f0_bootstrap_keys = {
        (scale, metric)
        for scale in scales
        for metric in ("absolute_residual_mae", "drre")
    }
    row_contracts = {
        "first_batch_valid_runs": (
            first_report.get("schema_version")
            == "rail3.tmlr-v6.first-batch-fit-evaluation.v1"
            and first_report.get("status")
            == "TMLR_V6_FIRST_BATCH_EVALUATION_PASS"
            and first_report.get("completed_valid_runs") == 75
            and first_report.get("metric_pipeline_pass") is True
        ),
        "second_batch_valid_training_jobs": (
            second_report.get("completed_valid_training_jobs") == 225
        ),
        "f0_f1_valid_training_jobs": (
            f0_report.get("completed_valid_training_jobs") == 120
        ),
        "phase2_valid_training_jobs": (
            second_report.get("completed_valid_training_jobs", 0)
            + f0_report.get("completed_valid_training_jobs", 0) == 345
        ),
        "first_metric_rows": (
            first_metrics.get("schema_version")
            == "rail3.tmlr-v6.first-batch-machine-metrics.v1"
            and len(first_metrics["prediction_metrics"]) == 5
            and len(first_metrics["decision_metrics"]) == 30
            and len(first_metrics["sparse_budget_metrics"]) == 360
            and len(first_metrics["primary_group_bootstrap"]) == 8
            and _exact_row_registry(
                first_metrics["prediction_metrics"],
                fields=("model",),
                expected=first_prediction_keys,
            )
            and _exact_row_registry(
                first_metrics["decision_metrics"],
                fields=("model", "lambda"),
                expected=first_decision_keys,
            )
            and _exact_row_registry(
                first_metrics["sparse_budget_metrics"],
                fields=("model", "lambda", "budget", "budget_semantics"),
                expected=first_sparse_keys,
            )
            and _exact_row_registry(
                first_metrics["primary_group_bootstrap"],
                fields=("left_model", "right_model", "metric"),
                expected=first_bootstrap_keys,
            )
        ),
        "mechanical_bootstrap_rows": (
            len(comparison_rows) == 14
            and all(row.get("image_groups") == 1364 for row in comparison_rows)
            and all(row.get("bootstrap_replicates") == 2000 for row in comparison_rows)
            and all(row.get("bootstrap_seed") == 13 for row in comparison_rows)
            and len({
                row.get("bootstrap_group_count_matrix_sha256")
                for row in comparison_rows
            }) == 1
        ),
        "second_metric_rows": (
            second_metrics.get("schema_version")
            == "rail3.tmlr-v6.phase2-second-batch-machine-metrics.v1"
            and len(second_metrics["prediction_metrics"]) == 16
            and len(second_metrics["decision_metrics"]) == 36
            and len(second_metrics["sparse_budget_metrics"]) == 432
            and _exact_row_registry(
                second_metrics["prediction_metrics"],
                fields=("model", "trained_lambda"),
                expected=second_prediction_keys,
            )
            and _exact_row_registry(
                second_metrics["decision_metrics"],
                fields=("model", "lambda"),
                expected=second_decision_keys,
            )
            and _exact_row_registry(
                second_metrics["sparse_budget_metrics"],
                fields=("model", "lambda", "budget", "budget_semantics"),
                expected=second_sparse_keys,
            )
        ),
        "f0_f1_metric_rows": (
            f0_metrics.get("schema_version")
            == "rail3.tmlr-v6.phase2-f0-f1-machine-metrics.v1"
            and len(f0_metrics["prediction_metrics"]) == 8
            and len(f0_metrics["decision_metrics"]) == 48
            and len(f0_metrics["sparse_budget_metrics"]) == 576
            and len(f0_metrics["paired_group_bootstrap"]) == 8
            and _exact_row_registry(
                f0_metrics["prediction_metrics"],
                fields=("scale", "model"),
                expected=f0_prediction_keys,
            )
            and _exact_row_registry(
                f0_metrics["decision_metrics"],
                fields=("scale", "model", "lambda"),
                expected=f0_decision_keys,
            )
            and _exact_row_registry(
                f0_metrics["sparse_budget_metrics"],
                fields=(
                    "scale", "model", "lambda", "budget", "budget_semantics",
                ),
                expected=f0_sparse_keys,
            )
            and _exact_row_registry(
                f0_metrics["paired_group_bootstrap"],
                fields=("scale", "metric"),
                expected=f0_bootstrap_keys,
            )
        ),
        "sensitivity_rows": (
            sensitivity_metrics.get("schema_version")
            == "rail3.tmlr-v6.phase2-sensitivity-machine-metrics.v1"
            and len(sensitivity_metrics["utility_audit"]) == 4
            and len(sensitivity_metrics["utility_sparse_budget_metrics"]) == 24
            and len(sensitivity_metrics["target_rich_subset_metrics"]) == 24
            and _exact_row_registry(
                sensitivity_metrics["utility_audit"],
                fields=("utility",),
                expected={(utility,) for utility in UTILITY_NAMES},
            )
            and _exact_row_registry(
                sensitivity_metrics["utility_sparse_budget_metrics"],
                fields=("utility", "budget", "budget_semantics"),
                expected={
                    (utility, budget, semantics)
                    for utility in UTILITY_NAMES
                    for budget in PRIMARY_SPARSE_BUDGETS
                    for semantics in BUDGET_SEMANTICS
                },
            )
            and _exact_row_registry(
                sensitivity_metrics["target_rich_subset_metrics"],
                fields=("subset", "budget", "budget_semantics"),
                expected={
                    (subset, budget, semantics)
                    for subset in SUBSET_NAMES
                    for budget in PRIMARY_SPARSE_BUDGETS
                    for semantics in BUDGET_SEMANTICS
                },
            )
        ),
        "qualitative_categories": len(qualitative.get("categories", ())) == 6,
        "all_output_report_checks": all(
            value is True
            for report in (
                first_report, second_report, f0_report,
                sensitivity_report, qualitative,
            )
            for value in report.get("checks", {}).values()
        ),
        "all_phase2_input_bindings_exact": all(
            isinstance(report.get("input_bindings"), Mapping)
            and all(
                report["input_bindings"].get(key) == value
                for key, value in expected_bindings.items()
            )
            for report in (
                second_report, f0_report, sensitivity_report, qualitative,
            )
        ),
        "protected_access_zero": all(
            report.get("heldout_access") == {name: 0 for name in EXTERNAL_SPLITS}
            for report in (
                first_report, second_report, f0_report,
                sensitivity_report, qualitative,
            )
        ),
    }
    integrity_complete = (
        all(row_contracts.values())
        and all(item.get("pass") is True for item in qa.values())
    )
    classification = classify_prospective_core(
        integrity_complete=integrity_complete, predicates=predicates,
    )
    if args.output_root != DEFAULT_OUTPUT_ROOT:
        raise RuntimeError("TMLR V6 classification output root is not registered")
    output_root = _inside(repository, args.output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "rail3.tmlr-v6.phase2-mechanical-classification.v1",
        "status": "TMLR_V6_PHASE2_CLASSIFICATION_COMPLETE",
        **classification,
        "predicates": predicates,
        "direct_decision_family_interpretation": direct_family,
        "post_hoc_sensitivity_may_upgrade_to_strong": False,
        "post_hoc_sensitivity_may_downgrade_strong_to_mixed": True,
        "integrity_complete": integrity_complete,
        "classification_requirements": row_contracts,
        "execution_qa": qa,
        "comparison_rows": comparison_rows,
        "input_bindings": {
            "phase2_launch_lock_sha256": launch.sha256,
            "first_batch_evaluation_sha256": launch.payload[
                "first_batch_evaluation"
            ]["sha256"],
            "second_batch_evaluation_sha256": sha256_file(
                _inside(repository, SECOND_OUTPUT_ROOT / "phase2_second_batch_evaluation_report.json")
            ),
            "f0_f1_evaluation_sha256": sha256_file(
                _inside(repository, F0_F1_OUTPUT_ROOT / "phase2_f0_f1_evaluation_report.json")
            ),
            "sensitivity_report_sha256": sha256_file(
                _inside(repository, SENSITIVITY_OUTPUT_ROOT / "phase2_sensitivity_report.json")
            ),
            "qualitative_atlas_manifest_sha256": sha256_file(qualitative_path),
            "recomputed_workload_inventories": workload_proof,
        },
        "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
        "exactly_one_terminal_label": True,
    }
    assert_zero_external_access(report["heldout_access"])
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        path = temporary / "phase2_classification.json"
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_root)
    print(classification["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
