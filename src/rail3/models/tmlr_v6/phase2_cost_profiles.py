"""Four-scale outer-training action-type costs for phase-2 F0/F1.

Costs are derived only from label-free FIT trajectory reports.  They are
fold/scale decision inputs and are never concatenated to predictor features.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.models.tmlr_v6.cost_profiles import (
    ADAPTIVE_ACTIONS,
    COST_SOURCE_IDENTITY_SCHEMA,
    PROFILE_ACTIONS,
    TRAJECTORY_REPORT_KEYS,
    _fit_index,
    _linear_quantile,
    _runtime_index,
    load_fit_only_json,
    lower_empirical_median,
    protocol_cost_source_registrations,
    verify_registered_cost_sources,
)
from rail3.models.tmlr_v6.phase2_io import (
    FOLDS,
    PHASE2_ADDENDUM_PATH,
    SCALES,
    SCALE_GROUP_COUNTS,
    SEEDS,
    ZERO_EXTERNAL_ACCESS,
    load_phase2_addendum,
    load_registered_scale_fold_manifest,
    split_groups,
)
from rail3.models.tmlr_v6.training import load_protocol


SCALE_COST_JSON_PATH = Path(
    "artifacts/source_data/tmlr_v6/phase2_scale_action_type_cost_profiles.json"
)
SCALE_COST_CSV_PATH = Path(
    "artifacts/source_data/tmlr_v6/phase2_scale_action_type_cost_profiles.csv"
)
SCALE_COST_SCHEMA = "rail3.tmlr-v6.phase2-scale-cost-profiles.v1"
SCALE_COST_STATUS = "TMLR_V6_PHASE2_SCALE_COST_PROFILES_PASS"
EXPECTED_CHECKS = {
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
CSV_FIELDS = (
    "scale",
    "outer_fold",
    "action",
    "n",
    "n_total",
    "n_infeasible",
    "median_seconds",
    "mean_seconds",
    "q25_seconds",
    "q75_seconds",
    "normalizer_seconds",
    "normalized_profiled_cost",
    "outer_training_group_count",
    "outer_training_group_ids_sha256",
    "profile_sha256",
    "heldout_access",
)


def _valid_file_identity(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        return False
    try:
        byte_count = int(value.get("bytes", -1))
    except (TypeError, ValueError):
        return False
    path = Path(str(value.get("path", "")))
    digest = str(value.get("sha256", ""))
    return (
        bool(path.as_posix())
        and not path.is_absolute()
        and ".." not in path.parts
        and byte_count > 0
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _validate_source_identities(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "fit_and_trajectory_sources", "fold_manifests", "semantic_addendum",
    }:
        raise RuntimeError("TMLR V6 phase-2 scale cost source identity set drift")
    fit = value["fit_and_trajectory_sources"]
    folds = value["fold_manifests"]
    if (
        not isinstance(fit, Mapping)
        or fit.get("schema_version") != COST_SOURCE_IDENTITY_SCHEMA
        or set(fit) != {
            "schema_version", "origin_protocol", "fit_manifest",
            "fold_manifest", "trajectory_reports",
        }
        or not _valid_file_identity(fit.get("fit_manifest"))
        or not _valid_file_identity(fit.get("fold_manifest"))
        or not isinstance(fit.get("trajectory_reports"), Mapping)
        or set(fit["trajectory_reports"]) != set(TRAJECTORY_REPORT_KEYS)
        or any(
            not _valid_file_identity(fit["trajectory_reports"][key])
            for key in TRAJECTORY_REPORT_KEYS
        )
        or not isinstance(folds, Mapping)
        or set(folds) != set(SCALES)
        or any(not _valid_file_identity(folds[scale]) for scale in SCALES)
        or not _valid_file_identity(value["semantic_addendum"])
        or value["semantic_addendum"]["path"] != PHASE2_ADDENDUM_PATH.as_posix()
    ):
        raise RuntimeError("TMLR V6 phase-2 scale cost source identity drift")
    origin = fit["origin_protocol"]
    if not isinstance(origin, Mapping) or set(origin) != {
        "path", "schema_version", "protocol_id", "registered_source_set_sha256",
    }:
        raise RuntimeError("TMLR V6 phase-2 cost origin protocol identity drift")
    return dict(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _groups_by_fold(
    fold_manifest: Mapping[str, Any], *, expected_groups: set[str], scale: str,
    expected_group_count: int,
) -> dict[str, set[str]]:
    records = list(fold_manifest.get("groups", ()))
    observed: set[str] = set()
    result = {fold: set() for fold in FOLDS}
    for item in records:
        group = str(item.get("image_group_id", ""))
        fold = str(item.get("fold_id", ""))
        if not group or group in observed or fold not in result:
            raise ValueError(f"{scale} fold manifest identity/assignment drift")
        observed.add(group)
        result[fold].add(group)
    expected_scale = "S_MAX" if scale == "S1364" else scale
    if (
        fold_manifest.get("scale") != expected_scale
        or int(fold_manifest.get("group_count", -1)) != expected_group_count
        or int(fold_manifest.get("fold_count", -1)) != 5
        or len(observed) != expected_group_count
        or not observed <= expected_groups
        or any(not value for value in result.values())
    ):
        raise ValueError(f"{scale} cost-profile fold population drift")
    return result


def _profile_rows(
    *, scale: str, outer_fold: str, outer_training: set[str],
    runtime_index: Mapping[str, Mapping[str, Sequence[tuple[bool, float]]]],
) -> list[dict[str, Any]]:
    action_values: dict[str, list[float]] = {}
    action_totals: dict[str, int] = {}
    action_infeasible: dict[str, int] = {}
    for action in ADAPTIVE_ACTIONS:
        observations = [
            observation
            for group in sorted(outer_training)
            for observation in runtime_index[group][action]
        ]
        values = [cost for feasible, cost in observations if feasible]
        if not values:
            raise ValueError(f"{scale}/{outer_fold} has no positive feasible {action}")
        action_values[action] = values
        action_totals[action] = len(observations)
        action_infeasible[action] = sum(not feasible for feasible, _ in observations)
    pooled = [value for action in ADAPTIVE_ACTIONS for value in action_values[action]]
    normalizer = lower_empirical_median(pooled)
    group_sha = _canonical_sha256(sorted(outer_training))
    statistics = []
    for action in ADAPTIVE_ACTIONS:
        values = action_values[action]
        statistics.append({
            "action": action,
            "n": len(values),
            "n_total": action_totals[action],
            "n_infeasible": action_infeasible[action],
            "median_seconds": _linear_quantile(values, 0.50),
            "mean_seconds": float(np.mean(np.asarray(values, dtype=np.float64))),
            "q25_seconds": _linear_quantile(values, 0.25),
            "q75_seconds": _linear_quantile(values, 0.75),
        })
    profile_sha = _canonical_sha256({
        "scale": scale,
        "outer_fold": outer_fold,
        "outer_training_group_ids": sorted(outer_training),
        "normalizer_seconds": normalizer,
        "actions": statistics,
    })
    common = {
        "scale": scale,
        "outer_fold": outer_fold,
        "normalizer_seconds": normalizer,
        "outer_training_group_count": len(outer_training),
        "outer_training_group_ids_sha256": group_sha,
        "profile_sha256": profile_sha,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
    }
    rows: list[dict[str, Any]] = [{
        **common,
        "action": "STOP",
        "n": 0,
        "n_total": 0,
        "n_infeasible": 0,
        "median_seconds": 0.0,
        "mean_seconds": 0.0,
        "q25_seconds": 0.0,
        "q75_seconds": 0.0,
        "normalized_profiled_cost": 0.0,
    }]
    rows.extend({
        **common,
        **item,
        "normalized_profiled_cost": item["median_seconds"] / normalizer,
    } for item in statistics)
    return rows


def build_phase2_scale_cost_profiles(
    *, fit_manifest: Mapping[str, Any],
    fold_manifests: Mapping[str, Mapping[str, Any]],
    trajectory_reports: Sequence[Mapping[str, Any]],
    source_identities: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact 4 x 5 x 5 scale/fold/action registry."""

    if set(fold_manifests) != set(SCALES):
        raise ValueError("phase-2 cost profiles require all four scales exactly")
    group_counts = dict(SCALE_GROUP_COUNTS)
    asset_to_group, fit_groups = _fit_index(fit_manifest)
    runtime_index = _runtime_index(
        trajectory_reports=trajectory_reports,
        asset_to_group=asset_to_group,
    )
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        by_fold = _groups_by_fold(
            fold_manifests[scale],
            expected_groups=fit_groups,
            scale=scale,
            expected_group_count=int(group_counts[scale]),
        )
        scale_groups = set().union(*by_fold.values())
        for outer_fold in FOLDS:
            outer_training = scale_groups - by_fold[outer_fold]
            if not outer_training:
                raise ValueError("phase-2 outer-training group set is empty")
            for seed in SEEDS:
                inner_train, inner_stop, heldout = split_groups(
                    fold_manifests[scale], outer_fold=outer_fold, seed=seed,
                )
                if inner_train | inner_stop != outer_training or heldout != by_fold[outer_fold]:
                    raise RuntimeError("phase-2 cost profile is not seed invariant")
            rows.extend(_profile_rows(
                scale=scale,
                outer_fold=outer_fold,
                outer_training=outer_training,
                runtime_index=runtime_index,
            ))
    if (
        len(rows) != 100
        or {(row["scale"], row["outer_fold"], row["action"]) for row in rows}
        != {(scale, fold, action) for scale in SCALES for fold in FOLDS for action in PROFILE_ACTIONS}
    ):
        raise RuntimeError("phase-2 scale cost registry is not the exact 100-row product")
    payload = {
        "schema_version": SCALE_COST_SCHEMA,
        "status": SCALE_COST_STATUS,
        "scales": list(SCALES),
        "outer_folds": list(FOLDS),
        "actions": list(PROFILE_ACTIONS),
        "required_rows": 100,
        "per_action_estimator": "numpy_linear_quantile_0.5",
        "normalizer_estimator": "lower_empirical_median_pooled_feasible_positive",
        "profile_seed_dependency": False,
        "same_profile_for_f0_and_f1": True,
        "a3_recorded_for_audit_but_unused_in_f0_f1": True,
        "source_identities": _validate_source_identities(source_identities),
        "rows": rows,
        "checks": dict(EXPECTED_CHECKS),
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
    }
    validate_phase2_scale_cost_payload(payload)
    return payload


def validate_phase2_scale_cost_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on row/cardinality/statistical-contract drift."""

    if (
        payload.get("schema_version") != SCALE_COST_SCHEMA
        or payload.get("status") != SCALE_COST_STATUS
        or tuple(payload.get("scales", ())) != SCALES
        or tuple(payload.get("outer_folds", ())) != FOLDS
        or tuple(payload.get("actions", ())) != PROFILE_ACTIONS
        or int(payload.get("required_rows", -1)) != 100
        or payload.get("profile_seed_dependency") is not False
        or payload.get("same_profile_for_f0_and_f1") is not True
        or not isinstance(payload.get("source_identities"), Mapping)
        or payload.get("heldout_access") != ZERO_EXTERNAL_ACCESS
        or payload.get("checks") != EXPECTED_CHECKS
    ):
        raise RuntimeError("TMLR V6 phase-2 scale cost payload drift")
    _validate_source_identities(payload["source_identities"])
    rows = list(payload.get("rows", ()))
    expected = {(scale, fold, action) for scale in SCALES for fold in FOLDS for action in PROFILE_ACTIONS}
    observed = {(row.get("scale"), row.get("outer_fold"), row.get("action")) for row in rows}
    if len(rows) != 100 or observed != expected:
        raise RuntimeError("TMLR V6 phase-2 scale cost row product drift")
    for row in rows:
        values = [
            float(row[key]) for key in (
                "median_seconds", "mean_seconds", "q25_seconds", "q75_seconds",
                "normalizer_seconds", "normalized_profiled_cost",
            )
        ]
        digests = (
            str(row.get("outer_training_group_ids_sha256", "")),
            str(row.get("profile_sha256", "")),
        )
        if (
            not np.isfinite(values).all()
            or float(row["normalizer_seconds"]) <= 0
            or row.get("heldout_access") != ZERO_EXTERNAL_ACCESS
            or any(len(value) != 64 for value in digests)
            or any(
                any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or int(row.get("n", -1)) < 0
            or int(row.get("n_total", -1)) < int(row.get("n", -1))
            or int(row.get("n_infeasible", -1))
            != int(row.get("n_total", -1)) - int(row.get("n", -1))
        ):
            raise RuntimeError("TMLR V6 phase-2 scale cost row is invalid")
        if row["action"] == "STOP":
            if any(float(row[key]) != 0.0 for key in (
                "median_seconds", "mean_seconds", "q25_seconds", "q75_seconds",
                "normalized_profiled_cost",
            )):
                raise RuntimeError("TMLR V6 phase-2 STOP cost is not exact zero")
        elif (
            float(row["median_seconds"]) <= 0
            or float(row["normalized_profiled_cost"]) <= 0
            or not np.isclose(
                float(row["normalized_profiled_cost"]),
                float(row["median_seconds"]) / float(row["normalizer_seconds"]),
                rtol=1e-12, atol=0.0,
            )
        ):
            raise RuntimeError("TMLR V6 phase-2 adaptive cost is not positive/exact")
    return dict(payload)


def phase2_scale_cost_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    validated = validate_phase2_scale_cost_payload(payload)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in validated["rows"]:
        values = dict(row)
        values["heldout_access"] = canonical_json_bytes(values["heldout_access"]).decode("utf-8")
        writer.writerow({field: values[field] for field in CSV_FIELDS})
    return handle.getvalue().encode("utf-8")


def load_registered_phase2_scale_cost_sources(
    repository: Path,
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Verify every source identity before opening scientific JSON bytes."""

    repository = repository.resolve()
    addendum = load_phase2_addendum(repository)
    protocol = load_protocol(repository)
    registrations = protocol_cost_source_registrations(protocol)
    report_registrations = registrations["trajectory_reports"]
    verified = verify_registered_cost_sources(
        repository=repository,
        protocol=protocol,
        fit_manifest_path=Path(str(registrations["fit_manifest"]["path"])),
        fold_manifest_path=Path(str(registrations["fold_manifest"]["path"])),
        trajectory_report_paths={
            key: Path(str(report_registrations[key]["path"]))
            for key in TRAJECTORY_REPORT_KEYS
        },
    )
    fold_payloads: dict[str, dict[str, Any]] = {}
    fold_identities: dict[str, Any] = {}
    for scale in SCALES:
        payload, digest = load_registered_scale_fold_manifest(
            repository, scale=scale, addendum=addendum,
        )
        fold_payloads[scale] = payload
        path = addendum["bound_parent_registries"]["fold_registry"][scale]["path"]
        actual = repository / path
        fold_identities[scale] = {
            "path": path,
            "bytes": actual.stat().st_size,
            "sha256": digest,
        }
    fit = load_fit_only_json(repository / verified["fit_manifest"]["path"])
    reports = [
        load_fit_only_json(repository / verified["trajectory_reports"][key]["path"])
        for key in TRAJECTORY_REPORT_KEYS
    ]
    source_identities = {
        "fit_and_trajectory_sources": verified,
        "fold_manifests": fold_identities,
        "semantic_addendum": {
            "path": PHASE2_ADDENDUM_PATH.as_posix(),
            "bytes": (repository / PHASE2_ADDENDUM_PATH).stat().st_size,
            "sha256": hashlib.sha256(
                (repository / PHASE2_ADDENDUM_PATH).read_bytes()
            ).hexdigest(),
        },
    }
    return fit, fold_payloads, reports, source_identities


def profiled_cost_vector(
    payload: Mapping[str, Any], *, scale: str, outer_fold: str,
) -> np.ndarray:
    """Return STOP,A3,A4,A5,A6 normalized costs for decision use only."""

    validated = validate_phase2_scale_cost_payload(payload)
    selected = {
        str(row["action"]): float(row["normalized_profiled_cost"])
        for row in validated["rows"]
        if row["scale"] == scale and row["outer_fold"] == outer_fold
    }
    if tuple(action for action in PROFILE_ACTIONS if action in selected) != PROFILE_ACTIONS:
        raise RuntimeError("TMLR V6 phase-2 requested cost profile is incomplete")
    result = np.asarray([selected[action] for action in PROFILE_ACTIONS], dtype=np.float32)
    if result.shape != (5,) or result[0] != 0.0 or np.any(result[1:] <= 0):
        raise RuntimeError("TMLR V6 phase-2 requested cost vector is invalid")
    return result


__all__ = [
    "CSV_FIELDS",
    "SCALE_COST_CSV_PATH",
    "SCALE_COST_JSON_PATH",
    "SCALE_COST_SCHEMA",
    "SCALE_COST_STATUS",
    "build_phase2_scale_cost_profiles",
    "load_registered_phase2_scale_cost_sources",
    "phase2_scale_cost_csv_bytes",
    "profiled_cost_vector",
    "validate_phase2_scale_cost_payload",
]
