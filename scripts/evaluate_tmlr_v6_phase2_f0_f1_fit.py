#!/usr/bin/env python3
"""Evaluate the exact 120-job prospective F0/F1 remaining-action panel."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.analysis.tmlr_v6_phase2 import (
    EXTERNAL_SPLITS,
    F0_F1_REPORT_SCHEMA,
    F0_F1_REPORT_STATUS,
    LAMBDAS,
    ProfiledCostSurface,
    Phase2Surface,
    decision_metrics,
    paired_bootstrap_group_counts,
    paired_ratio_bootstrap,
    residual_prediction_metrics,
    residual_ratio_sufficient,
    sparse_budget_metrics,
    strict_continuous_seed_mean,
    validate_prediction_metric_row,
    validate_f0_f1_report_contract,
)
from rail3.models.tmlr_v6.phase2_contract import (
    safe_phase2_repository_path,
    validate_phase2_launch_lock,
    verify_exact_rebuild_directory,
)
from rail3.models.tmlr_v6.phase2_cost_profiles import (
    validate_phase2_scale_cost_payload,
)
from rail3.models.tmlr_v6.phase2_f0_f1 import audit_f0_f1_runs
from rail3.models.tmlr_v6.phase2_io import (
    FOLDS,
    REMAINING_PHYSICAL_INDICES,
    SCALES,
    SEEDS,
    Phase2View,
    load_aligned_f0_f1_views,
)
from rail3.models.tmlr_v6.training import ACTIONS, _strings, sha256_file


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/paper/source_data/tmlr_v6/phase2_f0_f1"
)
PREDICTION_KEYS = {"state_index", "state_id", "state_prediction"}


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 F0/F1 evaluation must run at worktree root")
    return root


def _inside(repository: Path, path: Path) -> Path:
    return safe_phase2_repository_path(repository, path)


def _verified_prediction_path(
    repository: Path, report: Mapping[str, Any], run_dir: Path,
) -> Path:
    identity = report.get("prediction")
    if not isinstance(identity, Mapping):
        raise RuntimeError("TMLR V6 F0/F1 prediction identity is absent")
    path = _inside(repository, Path(str(identity.get("path", ""))))
    if (
        path != run_dir / "predictions.npz"
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != str(identity.get("sha256", ""))
    ):
        raise RuntimeError("TMLR V6 F0/F1 prediction identity drift")
    return path


def _cost_surface(
    *, view: Phase2View, payload: Mapping[str, Any],
) -> ProfiledCostSurface:
    rows = {
        (str(row["scale"]), str(row["outer_fold"]), str(row["action"])): row
        for row in payload["rows"]
    }
    fold_by_group = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in view.fold_manifest["groups"]
    }
    groups = _strings(view.features["image_group_ids"], name="F0/F1 group IDs")
    try:
        state_folds = [fold_by_group[str(group)] for group in groups]
        normalized = np.asarray([
            [
                float(rows[(view.scale, fold, action)]["normalized_profiled_cost"])
                for action in ACTIONS
            ]
            for fold in state_folds
        ], dtype=np.float64)
        seconds = np.asarray([
            [
                float(rows[(view.scale, fold, action)]["median_seconds"])
                for action in ACTIONS
            ]
            for fold in state_folds
        ], dtype=np.float64)
    except KeyError as error:
        raise RuntimeError("TMLR V6 F0/F1 cost profile coverage drift") from error
    return ProfiledCostSurface(normalized=normalized, seconds=seconds).validated(
        (len(groups), len(ACTIONS)),
    )


def load_f0_f1_surfaces(
    *, repository: Path,
) -> tuple[
    dict[tuple[str, str], Phase2Surface],
    dict[str, ProfiledCostSurface],
    list[dict[str, Any]],
    Any,
]:
    """Use the common 120-run audit, then stitch/average registered views."""

    launch = validate_phase2_launch_lock(repository)
    audit = audit_f0_f1_runs(repository, launch)
    if (
        audit.get("expected_jobs") != 120
        or audit.get("validated_runs") != 120
        or len(audit.get("runs", ())) != 120
        or any(value is not True for value in audit.get("checks", {}).values())
    ):
        raise RuntimeError("TMLR V6 F0/F1 120-run inventory is incomplete")
    scale_identity = launch.payload["phase2_scale_cost_profiles"]["json"]
    cost_path = _inside(repository, Path(str(scale_identity["path"])))
    if (
        cost_path.is_symlink()
        or not cost_path.is_file()
        or cost_path.stat().st_size != int(scale_identity["bytes"])
        or sha256_file(cost_path) != str(scale_identity["sha256"])
    ):
        raise RuntimeError("TMLR V6 F0/F1 scale cost identity drift")
    cost_payload = validate_phase2_scale_cost_payload(
        json.loads(cost_path.read_text(encoding="utf-8"))
    )
    reports = {
        (
            str(item["job"]["scale"]),
            str(item["job"]["filtration_id"]),
            str(item["job"]["outer_fold"]),
            int(item["job"]["seed"]),
        ): item
        for item in audit["runs"]
    }
    if len(reports) != 120:
        raise RuntimeError("TMLR V6 F0/F1 run-key duplication")
    surfaces: dict[tuple[str, str], Phase2Surface] = {}
    costs: dict[str, ProfiledCostSurface] = {}
    inventory_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        f0, f1 = load_aligned_f0_f1_views(repository, scale=scale)
        costs[scale] = _cost_surface(view=f0, payload=cost_payload)
        for view in (f0, f1):
            state_ids = _strings(view.features["state_ids"], name="F0/F1 state IDs")
            groups = _strings(view.features["image_group_ids"], name="F0/F1 group IDs")
            group_to_fold = {
                str(item["image_group_id"]): str(item["fold_id"])
                for item in view.fold_manifest["groups"]
            }
            state_fold = np.asarray([group_to_fold[str(group)] for group in groups])
            by_seed = {
                seed: np.full((len(state_ids), len(ACTIONS)), np.nan, dtype=np.float64)
                for seed in SEEDS
            }
            covered = {seed: np.zeros(len(state_ids), dtype=bool) for seed in SEEDS}
            for fold in FOLDS:
                local_index = np.flatnonzero(state_fold == fold).astype(np.int32)
                global_index = np.asarray(view.global_state_indices)[local_index].astype(np.int32)
                for seed in SEEDS:
                    item = reports[(scale, view.filtration_id, fold, seed)]
                    report = item["report"]
                    manifest_path = _inside(repository, Path(item["manifest_path"]))
                    run_dir = manifest_path.parent
                    path = _verified_prediction_path(repository, report, run_dir)
                    with np.load(path, allow_pickle=False) as payload:
                        if set(payload.files) != PREDICTION_KEYS:
                            raise RuntimeError("TMLR V6 F0/F1 prediction schema drift")
                        observed_index = np.asarray(payload["state_index"], dtype=np.int32)
                        observed_ids = _strings(
                            payload["state_id"], name="F0/F1 prediction state IDs",
                        )
                        prediction = np.asarray(
                            payload["state_prediction"], dtype=np.float64,
                        )
                        if (
                            not np.array_equal(observed_index, global_index)
                            or not np.array_equal(observed_ids, state_ids[local_index])
                            or prediction.shape != (len(local_index), len(ACTIONS))
                            or not np.isnan(prediction[:, 1]).all()
                            or not np.isfinite(
                                prediction[:, REMAINING_PHYSICAL_INDICES]
                            ).all()
                            or covered[seed][local_index].any()
                        ):
                            raise RuntimeError("TMLR V6 F0/F1 OOF stitching drift")
                        by_seed[seed][local_index] = prediction
                        covered[seed][local_index] = True
                    inventory_rows.append({
                        "model": view.filtration_id,
                        "filtration": view.filtration_id,
                        "scale": scale,
                        "outer_fold": fold,
                        "seed": seed,
                        "run_id": str(report["run_id"]),
                        "manifest_path": manifest_path.relative_to(repository).as_posix(),
                        "manifest_sha256": sha256_file(manifest_path),
                        "prediction_path": str(report["prediction"]["path"]),
                        "prediction_sha256": str(report["prediction"]["sha256"]),
                        "checkpoint_path": str(report["checkpoint"]["path"]),
                        "checkpoint_sha256": str(report["checkpoint"]["sha256"]),
                    })
            if any(not mask.all() for mask in covered.values()):
                raise RuntimeError("TMLR V6 F0/F1 five-fold coverage is incomplete")
            prediction = strict_continuous_seed_mean(
                by_seed, action_indices=REMAINING_PHYSICAL_INDICES,
            )
            surfaces[(scale, view.filtration_id)] = Phase2Surface(
                model_id=view.filtration_id,
                prediction_kind="residual",
                prediction=prediction,
                target=np.asarray(view.targets["state_target"], dtype=np.float64),
                feasible=np.asarray(view.targets["feasible"], dtype=bool),
                groups=groups,
                state_ids=tuple(state_ids.tolist()),
                action_indices=REMAINING_PHYSICAL_INDICES,
            ).validated()
    return surfaces, costs, inventory_rows, launch


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize empty TMLR V6 F0/F1 table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value
            )
            for key, value in row.items()
        })
    return output.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repository = _repository()
    surfaces, costs, run_rows, launch = load_f0_f1_surfaces(repository=repository)
    prediction_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    sparse_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for filtration in ("F0_P", "F1_P"):
            surface = surfaces[(scale, filtration)]
            prediction_rows.append({
                **residual_prediction_metrics(surface), "scale": scale,
            })
            for value in LAMBDAS:
                decision_rows.append({
                    **decision_metrics(surface, costs[scale], cost_lambda=value),
                    "scale": scale,
                })
                sparse_rows.extend({
                    **row, "scale": scale,
                } for row in sparse_budget_metrics(
                    surface, costs[scale], cost_lambda=value,
                ))
        left = surfaces[(scale, "F1_P")]
        right = surfaces[(scale, "F0_P")]
        if (
            left.state_ids != right.state_ids
            or not np.array_equal(left.groups, right.groups)
            or not np.array_equal(left.feasible, right.feasible)
            or not np.array_equal(left.target, right.target, equal_nan=True)
        ):
            raise RuntimeError("TMLR V6 F0/F1 paired surface alignment drift")
        left_sufficient = residual_ratio_sufficient(left)
        right_sufficient = residual_ratio_sufficient(right)
        expected_group_count = int(scale.removeprefix("S"))
        _, shared_bootstrap_counts = paired_bootstrap_group_counts(
            left.groups, expected_groups=expected_group_count,
        )
        for metric in ("absolute_residual_mae", "drre"):
            left_num, left_den = left_sufficient[metric]
            right_num, right_den = right_sufficient[metric]
            bootstrap_rows.append({
                **paired_ratio_bootstrap(
                    groups=left.groups,
                    left_numerator=left_num,
                    left_denominator=left_den,
                    right_numerator=right_num,
                    right_denominator=right_den,
                    contrast_id=f"{scale}_F1_P_MINUS_F0_P",
                    metric=metric,
                    expected_groups=expected_group_count,
                    bootstrap_counts=shared_bootstrap_counts,
                ),
                "scale": scale,
                "left_model": "F1_P",
                "right_model": "F0_P",
            })
    for row in prediction_rows:
        validate_prediction_metric_row(row, extra_fields=("scale",))
    if (
        len(run_rows) != 120
        or len(prediction_rows) != 8
        or len(decision_rows) != 48
        or len(sparse_rows) != 576
        or len(bootstrap_rows) != 8
    ):
        raise RuntimeError("TMLR V6 F0/F1 metric row cardinality drift")
    for scale in SCALES:
        rows = [row for row in bootstrap_rows if row["scale"] == scale]
        if len({row["bootstrap_group_count_matrix_sha256"] for row in rows}) != 1:
            raise RuntimeError("TMLR V6 F0/F1 bootstrap matrix was not reused")

    if args.output_root != DEFAULT_OUTPUT_ROOT:
        raise RuntimeError("TMLR V6 F0/F1 output root is not registered")
    output_root = _inside(repository, args.output_root)
    if args.check and not output_root.exists():
        raise FileNotFoundError(output_root)
    if not args.check and output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        tables = {
            "phase2_f0_f1_prediction_metrics.csv": prediction_rows,
            "phase2_f0_f1_decision_metrics.csv": decision_rows,
            "phase2_f0_f1_sparse_budget_metrics.csv": sparse_rows,
            "phase2_f0_f1_group_bootstrap.csv": bootstrap_rows,
            "phase2_f0_f1_run_inventory.csv": run_rows,
        }
        outputs = {}
        for name, rows in tables.items():
            path = temporary / name
            _write(path, _csv_bytes(rows))
            outputs[name] = {
                "path": (args.output_root / name).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        metrics = {
            "schema_version": "rail3.tmlr-v6.phase2-f0-f1-machine-metrics.v1",
            "prediction_metrics": prediction_rows,
            "decision_metrics": decision_rows,
            "sparse_budget_metrics": sparse_rows,
            "paired_group_bootstrap": bootstrap_rows,
        }
        metrics_path = temporary / "phase2_f0_f1_metrics.json"
        _write(metrics_path, canonical_json_bytes(metrics) + b"\n")
        outputs[metrics_path.name] = {
            "path": (args.output_root / metrics_path.name).as_posix(),
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
            "rows": 8 + 48 + 576 + 8,
        }
        payload = launch.payload
        report = {
            "schema_version": F0_F1_REPORT_SCHEMA,
            "status": F0_F1_REPORT_STATUS,
            "completed_valid_training_jobs": 120,
            "metric_pipeline_pass": True,
            "run_counts": {"F0_P": 60, "F1_P": 60},
            "remaining_physical_indices": list(REMAINING_PHYSICAL_INDICES),
            "scientific_estimand": (
                "conditional remaining-action prediction and utility after "
                "the declared filtration; not end-to-end A3 cost-effectiveness"
            ),
            "a3_sunk_cost_added_to_remaining_argmin": False,
            "same_profile_for_f0_and_f1": True,
            "input_bindings": {
                "phase2_launch_lock_sha256": launch.sha256,
                "first_batch_completion_gate_sha256": payload[
                    "first_batch_completion_gate"
                ]["sha256"],
                "first_batch_run_inventory_sha256": payload[
                    "first_batch_run_inventory"
                ]["sha256"],
                "first_batch_evaluation_sha256": payload[
                    "first_batch_evaluation"
                ]["sha256"],
            },
            "outputs": outputs,
            "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
            "checks": {
                "exact_cross_product_120": True,
                "common_launch_lock_and_run_audit": True,
                "a3_prediction_nan_both_filtrations": True,
                "remaining_action_set_only": True,
                "continuous_seed_mean_before_selection": True,
                "scale_specific_cost_profiles": True,
                "whole_state_complete_case_remaining": True,
                "f0_f1_targets_and_feasibility_aligned": True,
                "within_scale_paired_group_bootstrap": True,
                "zero_headroom_is_null": True,
                "metric_row_cardinality_exact": True,
                "protected_access_zero": True,
            },
        }
        validate_f0_f1_report_contract(report)
        _write(
            temporary / "phase2_f0_f1_evaluation_report.json",
            canonical_json_bytes(report) + b"\n",
        )
        if args.check:
            verify_exact_rebuild_directory(output_root, temporary)
        else:
            os.replace(temporary, output_root)
    print(json.dumps({
        "status": F0_F1_REPORT_STATUS,
        "completed_valid_training_jobs": 120,
        "prediction_rows": 8,
        "decision_rows": 48,
        "sparse_rows": 576,
        "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
        "check": args.check,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
