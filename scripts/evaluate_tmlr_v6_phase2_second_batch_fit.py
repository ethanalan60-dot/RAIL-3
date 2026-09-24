#!/usr/bin/env python3
"""Evaluate the exact 225-job TMLR V6 phase-two second batch on FIT OOF."""

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
from scripts.evaluate_tmlr_v6_first_batch_fit import (
    load_first_batch_surfaces,
    load_cost_surface,
)
from rail3.analysis.tmlr_v6_phase2 import (
    EXTERNAL_SPLITS,
    LAMBDAS,
    SECOND_BATCH_REPORT_SCHEMA,
    SECOND_BATCH_REPORT_STATUS,
    ProfiledCostSurface,
    Phase2Surface,
    decision_metrics,
    q2_scientific_gain_after_seed_mean,
    residual_prediction_metrics,
    sparse_budget_metrics,
    strict_continuous_seed_mean,
    validate_prediction_metric_row,
    validate_second_batch_report_contract,
)
from rail3.models.tmlr_v6.phase2_contract import (
    expected_second_batch_jobs,
    safe_phase2_repository_path,
    verify_exact_rebuild_directory,
)
from rail3.models.tmlr_v6.phase2_inventory import (
    audit_phase2_second_batch_runs,
    build_second_batch_validation_context,
    validate_second_batch_run_manifest,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    SEEDS,
    _strings,
    load_protocol,
    sha256_file,
    validate_fold_manifest,
)


DEFAULT_RUN_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/phase2-second-batch-crossfit"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/paper/source_data/tmlr_v6/phase2_second_batch"
)
PREDICTION_KEYS = {
    "R2_P": {
        "state_index", "state_id", "state_prediction",
        "atom_index", "atom_id", "atom_prediction",
    },
    "R3_P": {
        "state_index", "state_id", "state_prediction",
        "atom_index", "atom_id", "atom_prediction",
    },
    "Q2_P": {"state_index", "state_id", "raw_gain_prediction"},
    "L2D_P": {"state_index", "state_id", "action_logits"},
    "SPO_PLUS_P": {
        "state_index", "state_id", "action_value_prediction",
    },
}
PREDICTION_ARRAY = {
    "R2_P": "state_prediction",
    "R3_P": "state_prediction",
    "Q2_P": "raw_gain_prediction",
    "L2D_P": "action_logits",
    "SPO_PLUS_P": "action_value_prediction",
}
PREDICTION_KIND = {
    "R2_P": "residual",
    "R3_P": "residual",
    "Q2_P": "gain",
    "L2D_P": "logit",
    "SPO_PLUS_P": "value",
}


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 phase-two evaluation must run at worktree root")
    return root


def _inside(repository: Path, path: Path) -> Path:
    return safe_phase2_repository_path(repository, path)


def _prediction_path(
    repository: Path, manifest: Mapping[str, Any], run_dir: Path,
) -> Path:
    identity = manifest.get("prediction")
    if not isinstance(identity, Mapping):
        raise RuntimeError("TMLR V6 second-batch prediction identity is absent")
    path = _inside(repository, Path(str(identity.get("path", ""))))
    if path != run_dir / "predictions.npz":
        raise RuntimeError("TMLR V6 second-batch prediction path drift")
    # The common manifest verifier has already authenticated bytes/SHA.  The
    # local check protects this read against a replacement between calls.
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != str(identity.get("sha256", ""))
    ):
        raise RuntimeError("TMLR V6 second-batch prediction identity drift")
    return path


def load_second_batch_surfaces(
    *, repository: Path, run_root: Path,
) -> tuple[
    dict[tuple[str, float | None], Phase2Surface],
    ProfiledCostSurface,
    list[dict[str, Any]],
    Any,
]:
    """Authenticate, stitch five folds, then average continuous seed outputs."""

    context = build_second_batch_validation_context(repository)
    launch = context.lock
    audit = audit_phase2_second_batch_runs(
        repository=repository, root=run_root,
    )
    if (
        audit.get("launch_lock", {}).get("sha256") != launch.sha256
        or audit.get("completed_valid_runs") != 225
        or audit.get("missing_runs") != 0
        or audit.get("duplicate_runs") != 0
        or audit.get("unexpected_runs") != 0
        or any(value is not True for value in audit.get("checks", {}).values())
    ):
        raise RuntimeError("TMLR V6 second-batch 225-run inventory is incomplete")
    protocol = context.protocol
    fold_manifest = context.fold_manifest
    bundle = context.bundle
    state_ids = _strings(bundle.features["state_ids"], name="phase2 state_ids")
    groups = _strings(
        bundle.features["image_group_ids"], name="phase2 image_group_ids",
    )
    group_to_fold = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in fold_manifest["groups"]
    }
    state_fold = np.asarray([group_to_fold[str(group)] for group in groups])
    first_cost, _ = load_cost_surface(
        repository=repository,
        protocol=protocol,
        groups=groups,
        fold_manifest=fold_manifest,
    )
    costs = ProfiledCostSurface(
        normalized=np.asarray(first_cost.normalized, dtype=np.float64),
        seconds=np.asarray(first_cost.seconds, dtype=np.float64),
    ).validated((len(state_ids), len(ACTIONS)))
    expected_jobs = tuple(expected_second_batch_jobs())
    if len(expected_jobs) != 225:
        raise RuntimeError("TMLR V6 expected second-batch job matrix is not 225")
    keys = sorted({
        (job.model_id, job.cost_lambda) for job in expected_jobs
    }, key=lambda item: (item[0], -1.0 if item[1] is None else float(item[1])))
    by_key_seed = {
        key: {
            seed: np.full((len(state_ids), len(ACTIONS)), np.nan, dtype=np.float64)
            for seed in SEEDS
        }
        for key in keys
    }
    covered = {
        key: {seed: np.zeros(len(state_ids), dtype=bool) for seed in SEEDS}
        for key in keys
    }
    inventory_rows: list[dict[str, Any]] = []
    for job in expected_jobs:
        key = (job.model_id, job.cost_lambda)
        run_dir = job.run_dir(_inside(repository, run_root))
        manifest_path = run_dir / "run-manifest.json"
        manifest = validate_second_batch_run_manifest(
            repository,
            manifest_path,
            expected_job=job,
            launch_lock=launch,
            verify_artifacts=True,
            context=context,
        )
        expected_index = np.flatnonzero(state_fold == job.outer_fold).astype(np.int32)
        path = _prediction_path(repository, manifest, run_dir)
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != PREDICTION_KEYS[job.model_id]:
                raise RuntimeError("TMLR V6 second-batch prediction schema drift")
            observed_index = np.asarray(payload["state_index"], dtype=np.int32)
            observed_ids = _strings(payload["state_id"], name="phase2 prediction state_id")
            prediction = np.asarray(
                payload[PREDICTION_ARRAY[job.model_id]], dtype=np.float64,
            )
            if (
                not np.array_equal(observed_index, expected_index)
                or not np.array_equal(observed_ids, state_ids[expected_index])
                or prediction.shape != (len(expected_index), len(ACTIONS))
                or not np.isfinite(prediction).all()
                or covered[key][job.seed][observed_index].any()
            ):
                raise RuntimeError("TMLR V6 second-batch OOF stitching drift")
            by_key_seed[key][job.seed][observed_index] = prediction
            covered[key][job.seed][observed_index] = True
        inventory_rows.append({
            "model": job.model_id,
            "scale": job.scale,
            "lambda": job.cost_lambda,
            "outer_fold": job.outer_fold,
            "seed": job.seed,
            "run_id": str(manifest["run_id"]),
            "manifest_path": manifest_path.relative_to(repository).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction_path": str(manifest["prediction"]["path"]),
            "prediction_sha256": str(manifest["prediction"]["sha256"]),
            "checkpoint_path": str(manifest["checkpoint"]["path"]),
            "checkpoint_sha256": str(manifest["checkpoint"]["sha256"]),
        })
    if any(
        not mask.all()
        for by_seed in covered.values() for mask in by_seed.values()
    ):
        raise RuntimeError("TMLR V6 second-batch five-fold coverage is incomplete")
    surfaces: dict[tuple[str, float | None], Phase2Surface] = {}
    for key in keys:
        model, cost_lambda = key
        if model == "Q2_P":
            prediction = q2_scientific_gain_after_seed_mean(by_key_seed[key])
        else:
            prediction = strict_continuous_seed_mean(
                by_key_seed[key], action_indices=tuple(range(len(ACTIONS))),
            )
        surfaces[key] = Phase2Surface(
            model_id=model,
            prediction_kind=PREDICTION_KIND[model],
            prediction=prediction,
            target=np.asarray(bundle.targets["state_target"], dtype=np.float64),
            feasible=np.asarray(bundle.targets["feasible"], dtype=bool),
            groups=groups,
            state_ids=tuple(state_ids.tolist()),
            trained_cost_lambda=(
                float(cost_lambda) if model in {"L2D_P", "SPO_PLUS_P"} else None
            ),
        ).validated()
    return surfaces, costs, inventory_rows, launch


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize empty TMLR V6 phase-two table")
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


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _identity(logical: Path, physical: Path, rows: int) -> dict[str, Any]:
    return {
        "path": logical.as_posix(),
        "bytes": physical.stat().st_size,
        "sha256": sha256_file(physical),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repository = _repository()
    surfaces, costs, run_rows, launch = load_second_batch_surfaces(
        repository=repository, run_root=args.run_root,
    )
    # LL4TTA is a zero-job logical alias of the first-batch R0_SMALL_P surface.
    protocol = load_protocol(repository)
    folds = validate_fold_manifest(repository, protocol)
    first, _ = load_first_batch_surfaces(
        repository=repository, protocol=protocol, fold_manifest=folds,
    )
    alias_source = first["R0_SMALL_P"]
    alias = Phase2Surface(
        model_id="LL4TTA_P",
        prediction_kind="residual",
        prediction=np.asarray(alias_source.prediction),
        target=np.asarray(alias_source.target),
        feasible=np.asarray(alias_source.feasible),
        groups=np.asarray(alias_source.groups),
        state_ids=alias_source.state_ids,
    ).validated()
    reference = surfaces[("R2_P", None)]
    if (
        alias.state_ids != reference.state_ids
        or not np.array_equal(alias.groups, reference.groups)
        or not np.array_equal(alias.feasible, reference.feasible)
        or not np.array_equal(alias.target, reference.target, equal_nan=True)
    ):
        raise RuntimeError("TMLR V6 LL4TTA alias/F0 population alignment drift")

    prediction_rows = [
        {
            **residual_prediction_metrics(surface),
            "trained_lambda": key[1],
        }
        for key, surface in sorted(
            surfaces.items(), key=lambda item: str(item[0]),
        )
    ]
    prediction_rows.append({
        **residual_prediction_metrics(alias),
        "trained_lambda": None,
    })
    for row in prediction_rows:
        validate_prediction_metric_row(row, extra_fields=("trained_lambda",))
    decision_rows: list[dict[str, Any]] = []
    sparse_rows: list[dict[str, Any]] = []
    for key, surface in sorted(surfaces.items(), key=lambda item: str(item[0])):
        lambdas = (
            (float(key[1]),)
            if surface.prediction_kind in {"logit", "value"} else LAMBDAS
        )
        for value in lambdas:
            decision_rows.append(decision_metrics(
                surface, costs, cost_lambda=value,
            ))
            sparse_rows.extend(sparse_budget_metrics(
                surface, costs, cost_lambda=value,
            ))
    for value in LAMBDAS:
        decision_rows.append(decision_metrics(alias, costs, cost_lambda=value))
        sparse_rows.extend(sparse_budget_metrics(alias, costs, cost_lambda=value))
    if (
        len(run_rows) != 225
        or len(prediction_rows) != 16
        or len(decision_rows) != 36
        or len(sparse_rows) != 432
    ):
        raise RuntimeError("TMLR V6 second-batch metric row cardinality drift")

    if args.output_root != DEFAULT_OUTPUT_ROOT:
        raise RuntimeError("TMLR V6 second-batch output root is not registered")
    output_root = _inside(repository, args.output_root)
    if args.check and not output_root.exists():
        raise FileNotFoundError(output_root)
    if not args.check and output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    logical_root = args.output_root
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        tables = {
            "phase2_second_batch_prediction_metrics.csv": prediction_rows,
            "phase2_second_batch_decision_metrics.csv": decision_rows,
            "phase2_second_batch_sparse_budget_metrics.csv": sparse_rows,
            "phase2_second_batch_run_inventory.csv": run_rows,
        }
        outputs: dict[str, Any] = {}
        for name, rows in tables.items():
            path = temporary / name
            _write(path, _csv_bytes(rows))
            outputs[name] = _identity(logical_root / name, path, len(rows))
        metrics = {
            "schema_version": "rail3.tmlr-v6.phase2-second-batch-machine-metrics.v1",
            "prediction_metrics": prediction_rows,
            "decision_metrics": decision_rows,
            "sparse_budget_metrics": sparse_rows,
        }
        metrics_path = temporary / "phase2_second_batch_metrics.json"
        _write(metrics_path, canonical_json_bytes(metrics) + b"\n")
        outputs[metrics_path.name] = _identity(
            logical_root / metrics_path.name,
            metrics_path,
            len(prediction_rows) + len(decision_rows) + len(sparse_rows),
        )
        payload = launch.payload
        report = {
            "schema_version": SECOND_BATCH_REPORT_SCHEMA,
            "status": SECOND_BATCH_REPORT_STATUS,
            "completed_valid_training_jobs": 225,
            "metric_pipeline_pass": True,
            "run_counts": {
                "R2_P": 15, "R3_P": 15, "Q2_P": 15,
                "LL4TTA_P": 0, "L2D_P": 90, "SPO_PLUS_P": 90,
            },
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
            "continuous_seed_aggregation": "mean_13_37_71_before_selection",
            "q2_stop_override": "after_seed_mean_before_all_scientific_use",
            "ll4tta": {"alias_of": "R0_SMALL_P", "training_jobs": 0},
            "outputs": outputs,
            "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
            "checks": {
                "exact_cross_product_225": True,
                "launch_lock_exact": True,
                "run_manifests_common_verifier": True,
                "five_fold_oof_coverage_per_seed": True,
                "continuous_seed_mean_before_selection": True,
                "q2_stop_exact_zero_after_mean": True,
                "whole_state_complete_case": True,
                "zero_headroom_is_null": True,
                "metric_row_cardinality_exact": True,
                "protected_access_zero": True,
            },
        }
        validate_second_batch_report_contract(report)
        _write(
            temporary / "phase2_second_batch_evaluation_report.json",
            canonical_json_bytes(report) + b"\n",
        )
        if args.check:
            verify_exact_rebuild_directory(output_root, temporary)
        else:
            os.replace(temporary, output_root)
    print(json.dumps({
        "status": SECOND_BATCH_REPORT_STATUS,
        "completed_valid_training_jobs": 225,
        "decision_rows": 36,
        "sparse_rows": 432,
        "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
        "check": args.check,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
