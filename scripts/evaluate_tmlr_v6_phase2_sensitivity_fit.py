#!/usr/bin/env python3
"""Run frozen FIT-only utility sensitivity and target-rich subset analyses."""

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

from scripts.evaluate_tmlr_v6_first_batch_fit import load_first_batch_surfaces
from rail3.contracts import canonical_json_bytes
from rail3.analysis.tmlr_v6_phase2 import (
    EXTERNAL_SPLITS,
    ProfiledCostSurface,
    Phase2Surface,
    assert_zero_external_access,
    formal_evaluable_state_mask,
)
from rail3.analysis.tmlr_v6_sensitivity import (
    UTILITY_NAMES,
    sensitivity_budget_rows,
    target_rich_subset_masks,
    target_rich_subset_rows,
    utility_audit_rows,
    utility_losses_from_confusion,
)
from rail3.models.tmlr_v6.phase2_contract import (
    safe_phase2_repository_path,
    validate_phase2_launch_lock,
    verify_exact_rebuild_directory,
)
from rail3.models.tmlr_v6.training import (
    _verify_fit_only_input_inventory,
    _strings,
    load_protocol,
    load_registered_bundle,
    sha256_file,
    validate_fold_manifest,
)


SCHEMA_VERSION = "rail3.tmlr-v6.phase2-fit-utility-sensitivity.v1"
STATUS = "TMLR_V6_PHASE2_FIT_UTILITY_SENSITIVITY_PASS"
SENSITIVITY_REPORT_CHECKS = frozenset({
    "launch_lock_exact",
    "confusion_inputs_inventory_registered",
    "confusion_target_definition_equal",
    "confusion_mismatch_target_equal",
    "four_registered_utilities_only",
    "mismatch_selector_fixed",
    "metric_specific_training_false",
    "reference_positive_native_populations",
    "foreground_error_unclipped",
    "subset_raw_feature_index_exact",
    "within_subset_rerank",
    "zero_headroom_is_null",
    "row_cardinality_exact",
    "protected_access_zero",
    "sam_inference_false",
})
CONFUSION_REPORT = Path(
    "artifacts/voc2012/tmlr-reviewer-revision-v1/fit-confusion-v1/"
    "fit-confusion-report.json"
)
CONFUSION_ARRAY = Path(
    "artifacts/voc2012/tmlr-reviewer-revision-v1/fit-confusion-v1/"
    "confusion-counts.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/paper/source_data/tmlr_v6/phase2_sensitivity"
)
EXPECTED_CONFUSION_KEYS = {
    "state_defined", "state_tp", "state_fp", "state_fn", "state_tn",
    "state_predicted_positive", "state_reference_positive", "state_valid",
    "atom_defined", "atom_tp", "atom_fp", "atom_fn", "atom_tn",
    "atom_predicted_positive", "atom_reference_positive", "atom_valid",
    "state_lengths",
}


def _repository() -> Path:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("TMLR V6 sensitivity must run at worktree root")
    return root


def _inside(repository: Path, path: Path) -> Path:
    return safe_phase2_repository_path(repository, path)


def _inventory_registered_confusion(
    *, repository: Path, protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registration = protocol["prospective_bundle_identities"][
        "fit_only_input_inventory"
    ]
    inventory = _verify_fit_only_input_inventory(repository, registration)
    inventory_path = _inside(repository, Path(str(registration["path"])))
    if (
        inventory_path.is_symlink()
        or not inventory_path.is_file()
        or inventory_path.stat().st_size != int(registration["bytes"])
        or sha256_file(inventory_path) != str(registration["sha256"])
    ):
        raise RuntimeError("TMLR V6 sensitivity fit-input inventory drift")
    # Re-read only after the canonical verifier so the subsequent indexed
    # file lookup is tied to the exact authenticated inventory bytes.
    reread = json.loads(inventory_path.read_text(encoding="utf-8"))
    if reread != inventory:
        raise RuntimeError("TMLR V6 sensitivity fit inventory changed during read")
    indexed = {str(row["path"]): row for row in inventory.get("files", ())}
    try:
        report_identity = indexed[CONFUSION_REPORT.as_posix()]
        array_identity = indexed[CONFUSION_ARRAY.as_posix()]
    except KeyError as error:
        raise RuntimeError("TMLR V6 confusion inputs are not inventory registered") from error
    for logical, identity in (
        (CONFUSION_REPORT, report_identity),
        (CONFUSION_ARRAY, array_identity),
    ):
        path = _inside(repository, logical)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or sha256_file(path) != str(identity["sha256"])
        ):
            raise RuntimeError("TMLR V6 inventory-registered confusion identity drift")
    return inventory, dict(report_identity), dict(array_identity)


def _load_confusion(
    *, repository: Path, protocol: Mapping[str, Any], target: np.ndarray,
    feasible: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    inventory, report_identity, array_identity = _inventory_registered_confusion(
        repository=repository, protocol=protocol,
    )
    report_path = _inside(repository, CONFUSION_REPORT)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "rail3.tmlr-reviewer-fit-confusion.v1"
        or report.get("status") != "TMLR_REVIEWER_FIT_CONFUSION_PASS"
        or report.get("heldout_access") != {
            "validation40": 0,
            "calibration30": 0,
            "pilot_test30": 0,
            "official_voc_val": 0,
        }
        or report.get("fit_label_access") != {"images": 1364, "processes": 1}
        or report.get("sam_inference_used") is not False
        or report.get("prompt_or_trajectory_generation_used") is not False
        or report.get("model_training_used") is not False
        or report.get("gpu_used") is not False
        or not report.get("checks")
        or any(value is not True for value in report["checks"].values())
        or report.get("output", {}).get("path") != CONFUSION_ARRAY.as_posix()
        or report.get("output", {}).get("sha256") != array_identity["sha256"]
        or report.get("output", {}).get("bytes") != array_identity["bytes"]
    ):
        raise RuntimeError("TMLR V6 inventory-registered confusion report drift")
    array_path = _inside(repository, CONFUSION_ARRAY)
    with np.load(array_path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_CONFUSION_KEYS:
            raise RuntimeError("TMLR V6 confusion NPZ schema drift")
        defined = np.asarray(payload["state_defined"], dtype=bool)
        tp = np.asarray(payload["state_tp"], dtype=np.int64)
        fp = np.asarray(payload["state_fp"], dtype=np.int64)
        fn = np.asarray(payload["state_fn"], dtype=np.int64)
        tn = np.asarray(payload["state_tn"], dtype=np.int64)
        predicted_positive = np.asarray(
            payload["state_predicted_positive"], dtype=np.int64,
        )
        stored_reference = np.asarray(
            payload["state_reference_positive"], dtype=np.int64,
        )
        valid = np.asarray(payload["state_valid"], dtype=np.int64)
    expected_defined = np.asarray(feasible, dtype=bool) & np.isfinite(target)
    if (
        defined.shape != np.asarray(target).shape
        or any(value.shape != defined.shape for value in (tp, fp, fn, tn, predicted_positive))
        or stored_reference.shape != (len(defined),)
        or valid.shape != (len(defined),)
        or not np.array_equal(defined, expected_defined)
    ):
        raise RuntimeError("TMLR V6 confusion/PROSPECTIVE target definition drift")
    broadcast_reference = np.broadcast_to(stored_reference[:, None], defined.shape)
    broadcast_valid = np.broadcast_to(valid[:, None], defined.shape)
    if (
        np.any(valid <= 0)
        or np.any(stored_reference < 0)
        or any(
            np.any(value[~defined] != -1)
            for value in (tp, fp, fn, tn, predicted_positive)
        )
        or any(np.any(value[defined] < 0) for value in (tp, fp, fn, tn))
        or np.any(tp[defined] + fn[defined] != broadcast_reference[defined])
        or np.any(tp[defined] + fp[defined] != predicted_positive[defined])
        or np.any(
            tp[defined] + fp[defined] + fn[defined] + tn[defined]
            != broadcast_valid[defined]
        )
    ):
        raise RuntimeError("TMLR V6 confusion sufficient-statistic conservation drift")
    utilities, reference = utility_losses_from_confusion(
        defined=defined,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        valid_pixels=valid,
        reference_foreground_pixels=stored_reference,
    )
    observed_mismatch = utilities["full_image_pixel_mismatch"]
    if not np.allclose(
        observed_mismatch[defined], np.asarray(target)[defined],
        rtol=0.0, atol=2e-7,
    ):
        raise RuntimeError("TMLR V6 confusion mismatch differs from V6 target")
    identity = {
        "fit_only_input_inventory": {
            "path": protocol["prospective_bundle_identities"][
                "fit_only_input_inventory"
            ]["path"],
            "bytes": protocol["prospective_bundle_identities"][
                "fit_only_input_inventory"
            ]["bytes"],
            "sha256": protocol["prospective_bundle_identities"][
                "fit_only_input_inventory"
            ]["sha256"],
        },
        "confusion_report": report_identity,
        "confusion_array": array_identity,
    }
    return utilities, reference, identity


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize empty TMLR V6 sensitivity table")
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
    launch = validate_phase2_launch_lock(repository)
    protocol = load_protocol(repository)
    fold_manifest = validate_fold_manifest(repository, protocol)
    first, _ = load_first_batch_surfaces(
        repository=repository, protocol=protocol, fold_manifest=fold_manifest,
    )
    r1_source = first["R1_P"]
    selector = Phase2Surface(
        model_id="R1_P",
        prediction_kind="residual",
        prediction=np.asarray(r1_source.prediction),
        target=np.asarray(r1_source.target),
        feasible=np.asarray(r1_source.feasible),
        groups=np.asarray(r1_source.groups),
        state_ids=r1_source.state_ids,
    ).validated()
    bundle = load_registered_bundle(repository, "F0_P")
    bundle_state_ids = _strings(
        bundle.features["state_ids"], name="sensitivity F0 state IDs",
    )
    bundle_groups = _strings(
        bundle.features["image_group_ids"], name="sensitivity F0 group IDs",
    )
    if (
        not np.array_equal(bundle_state_ids, np.asarray(selector.state_ids))
        or not np.array_equal(bundle_groups, np.asarray(selector.groups).astype(str))
        or not np.array_equal(
            np.asarray(bundle.targets["state_target"]),
            np.asarray(selector.target),
            equal_nan=True,
        )
        or not np.array_equal(
            np.asarray(bundle.targets["feasible"], dtype=bool),
            np.asarray(selector.feasible, dtype=bool),
        )
    ):
        raise RuntimeError("TMLR V6 sensitivity F0/selector alignment drift")
    utilities, reference_pixels, confusion_identity = _load_confusion(
        repository=repository,
        protocol=protocol,
        target=np.asarray(selector.target),
        feasible=np.asarray(selector.feasible),
    )
    zero_costs = ProfiledCostSurface(
        normalized=np.zeros_like(selector.target, dtype=np.float64),
        seconds=np.zeros_like(selector.target, dtype=np.float64),
    ).validated(np.asarray(selector.target).shape)
    audit_rows = utility_audit_rows(
        mismatch_selector=selector, utilities=utilities,
    )
    budget_rows = sensitivity_budget_rows(
        mismatch_selector=selector, utilities=utilities, zero_costs=zero_costs,
    )
    subset_masks = target_rich_subset_masks(
        formal_mask=formal_evaluable_state_mask(selector),
        reference_foreground_pixels=reference_pixels,
        raw_f0_action_x16=np.asarray(bundle.features["action_x16"]),
    )
    subset_rows = target_rich_subset_rows(
        mismatch_selector=selector,
        zero_costs=zero_costs,
        subset_masks=subset_masks,
    )
    if len(audit_rows) != 4 or len(budget_rows) != 24 or len(subset_rows) != 24:
        raise RuntimeError("TMLR V6 sensitivity output row cardinality drift")
    if args.output_root != DEFAULT_OUTPUT_ROOT:
        raise RuntimeError("TMLR V6 sensitivity output root is not registered")
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
            "phase2_utility_audit.csv": audit_rows,
            "phase2_utility_sparse_budget_metrics.csv": budget_rows,
            "phase2_target_rich_subset_metrics.csv": subset_rows,
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
        machine = {
            "schema_version": "rail3.tmlr-v6.phase2-sensitivity-machine-metrics.v1",
            "utility_audit": audit_rows,
            "utility_sparse_budget_metrics": budget_rows,
            "target_rich_subset_metrics": subset_rows,
        }
        machine_path = temporary / "phase2_sensitivity_metrics.json"
        _write(machine_path, canonical_json_bytes(machine) + b"\n")
        outputs[machine_path.name] = {
            "path": (args.output_root / machine_path.name).as_posix(),
            "bytes": machine_path.stat().st_size,
            "sha256": sha256_file(machine_path),
            "rows": 52,
        }
        launch_payload = launch.payload
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "split": "FIT_OOF_ONLY",
            "interpretation": "FIT_ONLY_POST_HOC_SENSITIVITY",
            "confirmatory_claim_allowed": False,
            "selector": "R1_P_FULL_IMAGE_PIXEL_MISMATCH_OOF",
            "metric_specific_training": False,
            "sam_inference": False,
            "utilities": list(UTILITY_NAMES),
            "confusion_inputs": confusion_identity,
            "input_bindings": {
                "phase2_launch_lock_sha256": launch.sha256,
                "first_batch_completion_gate_sha256": launch_payload[
                    "first_batch_completion_gate"
                ]["sha256"],
                "first_batch_run_inventory_sha256": launch_payload[
                    "first_batch_run_inventory"
                ]["sha256"],
                "first_batch_evaluation_sha256": launch_payload[
                    "first_batch_evaluation"
                ]["sha256"],
            },
            "outputs": outputs,
            "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
            "checks": {
                "launch_lock_exact": True,
                "confusion_inputs_inventory_registered": True,
                "confusion_target_definition_equal": True,
                "confusion_mismatch_target_equal": True,
                "four_registered_utilities_only": True,
                "mismatch_selector_fixed": True,
                "metric_specific_training_false": True,
                "reference_positive_native_populations": True,
                "foreground_error_unclipped": True,
                "subset_raw_feature_index_exact": True,
                "within_subset_rerank": True,
                "zero_headroom_is_null": True,
                "row_cardinality_exact": True,
                "protected_access_zero": True,
                "sam_inference_false": True,
            },
        }
        if (
            set(report["checks"]) != set(SENSITIVITY_REPORT_CHECKS)
            or any(value is not True for value in report["checks"].values())
        ):
            raise RuntimeError("TMLR V6 sensitivity report checks are incomplete")
        assert_zero_external_access(report["heldout_access"])
        _write(
            temporary / "phase2_sensitivity_report.json",
            canonical_json_bytes(report) + b"\n",
        )
        if args.check:
            verify_exact_rebuild_directory(output_root, temporary)
        else:
            os.replace(temporary, output_root)
    print(json.dumps({
        "status": STATUS,
        "utility_rows": 4,
        "utility_budget_rows": 24,
        "subset_rows": 24,
        "heldout_access": {name: 0 for name in EXTERNAL_SPLITS},
        "check": args.check,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
