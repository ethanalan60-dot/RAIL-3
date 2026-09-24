"""Frozen M04/M05 V2A validation gates and label-free inventory helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from rail3.contracts import canonical_json_bytes


V2A_PROTOCOL_PATH = Path("configs/experiments/voc_m04_m05_v2a_sparse_action.json")
V2A_PROTOCOL_SHA256 = "3c1903c1fee765a1281e7011f19067c3e71ae8f1c9f83b56bb072462cfb8bc0b"
ACTION_PROTOCOL_SHA256 = "c558595b6be151307f04ae189ff0b8c74f1de2adbf7e0c046db57048fd12f985"
ACTION_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
NONCANONICAL_ACTIONS = ACTION_ORDER[1:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v2a_protocol(path: Path = V2A_PROTOCOL_PATH) -> dict[str, Any]:
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != V2A_PROTOCOL_SHA256:
        raise ValueError("V2A protocol content hash drift")
    payload = json.loads(encoded)
    if payload.get("schema_version") != "rail3.voc-m04-m05-v2a-sparse-action.v1":
        raise ValueError("V2A protocol schema drift")
    if tuple(payload["action_space"]["deployed_action_codes"]) != ACTION_ORDER:
        raise ValueError("V2A action ordering drift")
    return payload


def mask_sha256(mask: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    if array.ndim != 2:
        raise ValueError("V2A output mask must be two-dimensional")
    header = f"{array.shape[0]}x{array.shape[1]}:".encode("ascii")
    return hashlib.sha256(
        header + np.packbits(array, bitorder="little").tobytes()
    ).hexdigest()


def cache_inventory_digest(records: Iterable[Mapping[str, Any]]) -> str:
    inventory = [
        {
            "cache_key": str(record["cache_key"]),
            "bytes": int(record["cache_bytes"]),
            "sha256": str(record["cache_sha256"]),
        }
        for record in records
    ]
    inventory.sort(key=lambda item: item["cache_key"])
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _require_exact_action_records(records: list[Mapping[str, Any]]) -> None:
    if len(records) != 5600:
        raise ValueError("V2A evaluation must contain exactly 5,600 records")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["state_id"])].append(record)
    if len(grouped) != 800:
        raise ValueError("V2A evaluation must contain exactly 800 states")
    assets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for state_id, state_records in grouped.items():
        codes = tuple(sorted(
            (str(record["action_code"]) for record in state_records),
            key=ACTION_ORDER.index,
        ))
        if codes != ACTION_ORDER:
            raise ValueError(f"state {state_id} does not contain exactly A0-A6")
        if len({str(record["action_id"]) for record in state_records}) != len(ACTION_ORDER):
            raise ValueError(f"state {state_id} repeats an action ID")
        if len({bool(record["gt_presence"]) for record in state_records}) != 1:
            raise ValueError(f"state {state_id} has inconsistent GT presence")
        state_assets = {str(record["asset_id"]) for record in state_records}
        state_classes = {int(record["class_id"]) for record in state_records}
        if len(state_assets) != 1 or len(state_classes) != 1:
            raise ValueError(f"state {state_id} has inconsistent asset/class identity")
        assets[next(iter(state_assets))].add((state_id, next(iter(state_classes))))
    if len(assets) != 40:
        raise ValueError("V2A evaluation must contain exactly 40 actual assets")
    for asset_id, state_classes in assets.items():
        if {class_id for _, class_id in state_classes} != set(range(1, 21)):
            raise ValueError(f"asset {asset_id} does not contain exactly classes 1-20")


def _best(state_records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        record for record in state_records
        if record.get("action_iou") is not None
        and record.get("outcome") not in {"failure", "infeasible"}
    ]
    if not eligible:
        raise ValueError("V2A oracle choice set is empty")
    return min(eligible, key=lambda record: (
        -float(record["action_iou"]),
        float(record["action_cost_seconds"]),
        ACTION_ORDER.index(str(record["action_code"])),
    ))


def _positive_gain_concentration(
    gains: list[float], *, fraction: float
) -> tuple[int, float | None]:
    positive = sorted((gain for gain in gains if gain > 0.0), reverse=True)
    if not positive:
        return 0, None
    count = max(1, math.ceil(len(positive) * fraction))
    return count, sum(positive[:count]) / sum(positive)


def _evaluation_integrity(
    evaluation: Mapping[str, Any], generation_audit: Mapping[str, Any]
) -> dict[str, bool]:
    records = list(evaluation.get("records", ()))
    generation_finished = str(
        generation_audit.get("generation_source", {}).get("finished_at", "")
    )
    evaluation_created = str(evaluation.get("created_at", ""))
    evaluation_action_ids = {
        str(record.get("action_id"))
        for record in records if record.get("action_code") != "A0"
    }
    audit_action_ids = {
        str(record.get("action_id")) for record in generation_audit.get("actions", ())
    }
    return {
        "evaluation_status_pass": evaluation.get("status") == "PASS",
        "evaluation_stage_final": evaluation.get("stage") == "FINAL_EVALUATION",
        "evaluation_split_validation": evaluation.get("pilot_split") == "validation",
        "action_protocol_exact": evaluation.get("protocol_sha256") == ACTION_PROTOCOL_SHA256,
        "generation_audit_pass": generation_audit.get("status") == "PASS",
        "generation_stage_auto_annotation": generation_audit.get("stage") == "AUTO_ANNOTATION",
        "v2a_protocol_exact": generation_audit.get("v2a_protocol_sha256")
        == V2A_PROTOCOL_SHA256,
        "generation_action_protocol_exact": generation_audit.get("action_protocol_sha256")
        == ACTION_PROTOCOL_SHA256,
        "generation_precedes_evaluation": bool(generation_finished)
        and bool(evaluation_created)
        and generation_finished <= evaluation_created,
        "image_count_exact": int(evaluation.get("image_count", -1)) == 40,
        "state_count_exact": len({str(record.get("state_id")) for record in records}) == 800,
        "evaluation_record_count_exact": len(records) == 5600,
        "evaluation_action_inventory_matches_generation": len(evaluation_action_ids) == 4800
        and evaluation_action_ids == audit_action_ids,
        "generation_image_count_exact": int(
            generation_audit.get("counts", {}).get("completed_images", -1)
        ) == 40,
        "generation_state_count_exact": int(
            generation_audit.get("counts", {}).get("class_states", -1)
        ) == 800,
        "generation_action_count_exact": int(
            generation_audit.get("counts", {}).get("noncanonical_action_records", -1)
        ) == 4800,
        "zero_unstructured_failures": int(
            generation_audit.get("counts", {}).get("unstructured_failures", -1)
        ) == 0,
        "generation_integrity_checks": all(
            bool(value) for value in generation_audit.get("checks", {}).values()
        ),
    }


def evaluate_v2a_validation(
    evaluation: Mapping[str, Any],
    generation_audit: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the precommitted conjunctive G1--G8 contract to one evaluation."""

    protocol = load_v2a_protocol() if protocol is None else protocol
    records = list(evaluation.get("records", ()))
    _require_exact_action_records(records)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["state_id"])].append(record)

    positive_states = [
        sorted(state_records, key=lambda item: ACTION_ORDER.index(str(item["action_code"])))
        for state_records in grouped.values()
        if bool(state_records[0]["gt_presence"])
    ]
    if not positive_states:
        raise ValueError("V2A validation contains no GT-positive states")

    state_results = []
    for state_records in positive_states:
        canonical = next(item for item in state_records if item["action_code"] == "A0")
        best = _best(state_records)
        canonical_iou = float(canonical["action_iou"])
        oracle_iou = float(best["action_iou"])
        state_results.append({
            "state_id": str(canonical["state_id"]),
            "asset_id": str(canonical["asset_id"]),
            "sample_id": str(canonical["sample_id"]),
            "class_id": int(canonical["class_id"]),
            "canonical_iou": canonical_iou,
            "canonical_no_result": canonical.get("outcome") == "no_result",
            "oracle_iou": oracle_iou,
            "oracle_gain": oracle_iou - canonical_iou,
            "oracle_action": str(best["action_code"]),
            "oracle_action_id": str(best["action_id"]),
            "oracle_action_cost_seconds": float(best["action_cost_seconds"]),
        })

    canonical_mean = sum(item["canonical_iou"] for item in state_results) / len(state_results)
    oracle_mean = sum(item["oracle_iou"] for item in state_results) / len(state_results)
    absolute_gain = oracle_mean - canonical_mean
    normalized = None if canonical_mean >= 1.0 else absolute_gain / (1.0 - canonical_mean)
    gains = [float(item["oracle_gain"]) for item in state_results]
    gates = protocol["validation_gates"]
    concentration_count, concentration = _positive_gain_concentration(
        gains, fraction=float(gates["G3_sparse_gain_concentration"]["top_fraction"])
    )
    practical_count = sum(
        gain >= float(gates["G4_practically_improved_pair_support"]["gain_threshold"])
        for gain in gains
    )
    practical_fraction = practical_count / len(gains)

    tolerance = float(protocol["metrics"]["unique_best_tolerance"])
    unique_best = Counter()
    unique_best_states: dict[str, list[str]] = defaultdict(list)
    tie_sets = Counter()
    for state_records in positive_states:
        eligible = [
            record for record in state_records
            if record.get("action_iou") is not None
            and record.get("outcome") not in {"failure", "infeasible"}
        ]
        maximum = max(float(record["action_iou"]) for record in eligible)
        winners = [
            record for record in eligible
            if abs(float(record["action_iou"]) - maximum) <= tolerance
        ]
        canonical = next(item for item in state_records if item["action_code"] == "A0")
        if len(winners) > 1:
            tie_sets["+".join(sorted(
                (str(item["action_code"]) for item in winners), key=ACTION_ORDER.index
            ))] += 1
        if (
            len(winners) == 1
            and winners[0]["action_code"] != "A0"
            and maximum - float(canonical["action_iou"]) > tolerance
        ):
            code = str(winners[0]["action_code"])
            unique_best[code] += 1
            unique_best_states[code].append(str(winners[0]["state_id"]))
    diverse_actions = sorted(
        code for code in NONCANONICAL_ACTIONS
        if unique_best[code] >= int(gates["G5_action_diversity"]["minimum_unique_best_pairs_per_action"])
    )

    nonhomogeneity = generation_audit.get("non_homogeneity", {})
    unequal_counts = {
        str(code): int(count) for code, count in
        nonhomogeneity.get("distinct_outputs_unequal_canonical_by_action", {}).items()
    }
    nonhomogeneous_actions = sorted(
        code for code in NONCANONICAL_ACTIONS
        if unequal_counts.get(code, 0) >= int(
            gates["G6_non_homogeneity"]["minimum_distinct_noncanonical_outputs_per_counted_action"]
        )
    )
    identity_fraction = nonhomogeneity.get(
        "feasible_noncanonical_mask_equals_canonical_fraction"
    )
    action_inventory = {
        str(item["action_id"]): item for item in generation_audit.get("actions", ())
    }
    improvement_references = []
    for record in records:
        improvement = record.get("improvement_over_canonical")
        if record.get("action_code") == "A0" or improvement is None or float(improvement) <= 0.0:
            continue
        action = action_inventory.get(str(record["action_id"]))
        improvement_references.append((
            str(record["state_id"]),
            None if action is None else str(action["cache_key"]),
        ))
    improvements_have_distinct_refs = (
        all(cache_key is not None for _, cache_key in improvement_references)
        and len(improvement_references) == len(set(improvement_references))
    )

    integrity = _evaluation_integrity(evaluation, generation_audit)
    g6_checks = {
        "identity_fraction_within_limit": identity_fraction is not None
        and float(identity_fraction) <= float(
            gates["G6_non_homogeneity"]["maximum_feasible_noncanonical_mask_equals_canonical_fraction"]
        ),
        "multiple_actions_have_distinct_outputs": len(nonhomogeneous_actions) >= int(
            gates["G6_non_homogeneity"]["minimum_noncanonical_actions_with_three_distinct_noncanonical_outputs"]
        ),
        "unique_cache_key_per_action_record": bool(
            nonhomogeneity.get("unique_cache_key_per_action_record", False)
        ),
        "positive_improvements_have_distinct_state_cache_records": improvements_have_distinct_refs,
    }
    resource = generation_audit.get("resource_accounting", {})
    gate_results = {
        "G1": absolute_gain >= float(gates["G1_absolute_oracle_headroom"]["threshold"]),
        "G2": normalized is not None and normalized >= float(
            gates["G2_normalized_remaining_error_reduction"]["threshold"]
        ),
        "G3": concentration is not None and concentration >= float(
            gates["G3_sparse_gain_concentration"]["threshold"]
        ),
        "G4": practical_fraction >= float(
            gates["G4_practically_improved_pair_support"]["threshold"]
        ),
        "G5": len(diverse_actions) >= int(
            gates["G5_action_diversity"]["minimum_distinct_noncanonical_actions"]
        ),
        "G6": all(g6_checks.values()),
        "G7": all(integrity.values()),
        "G8": float(resource.get("generation_wall_seconds", math.inf)) <= float(
            gates["G8_cost"]["maximum_generation_wall_seconds"]
        ) and int(resource.get("new_trajectory_cache_bytes", -1)) >= 0
        and int(resource.get("new_trajectory_cache_bytes", -1)) <= int(
            gates["G8_cost"]["maximum_new_trajectory_cache_bytes"]
        ),
    }
    status = gates["pass_status"] if all(gate_results.values()) else gates["failure_status"]

    gain_thresholds = {}
    for threshold in (0.01, 0.03, 0.05, 0.10):
        count = sum(gain >= threshold for gain in gains)
        gain_thresholds[f"ge_{threshold:.2f}"] = {
            "count": count,
            "fraction": count / len(gains),
        }
    concentration_points = {}
    for fraction in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        count, captured = _positive_gain_concentration(gains, fraction=fraction)
        concentration_points[f"top_{int(fraction * 100)}_percent"] = {
            "positive_pair_count": count,
            "positive_gain_captured_fraction": captured,
        }
    redundancy = {}
    audit_actions = list(generation_audit.get("actions", ()))
    for code in NONCANONICAL_ACTIONS:
        selected = [item for item in audit_actions if item.get("action_code") == code]
        feasible_selected = [
            item for item in selected
            if item.get("feasible") and item.get("mask_equals_canonical") is not None
        ]
        identical = sum(bool(item["mask_equals_canonical"]) for item in feasible_selected)
        redundancy[code] = {
            "record_count": len(selected),
            "feasible_evaluated_output_count": len(feasible_selected),
            "outcome_counts": dict(sorted(Counter(
                str(item.get("outcome")) for item in selected
            ).items())),
            "mask_equals_canonical_count": identical,
            "mask_equals_canonical_fraction": (
                None if not feasible_selected else identical / len(feasible_selected)
            ),
            "distinct_output_count": len({
                str(item["output_sha256"])
                for item in feasible_selected if item.get("output_sha256") is not None
            }),
            "distinct_outputs_unequal_canonical": unequal_counts.get(code, 0),
        }
    canonical_no_result_states = [
        item for item in state_results if item["canonical_no_result"]
    ]
    recovered_no_result_states = [
        item for item in canonical_no_result_states
        if item["oracle_action"] != "A0" and item["oracle_gain"] > 0.0
    ]

    gate_audit = {
        "G1": {
            "expected": ">= 0.03 absolute oracle headroom",
            "actual": absolute_gain,
            "passed": gate_results["G1"],
        },
        "G2": {
            "expected": ">= 0.15 aggregate normalized remaining-error reduction",
            "actual": normalized,
            "passed": gate_results["G2"],
        },
        "G3": {
            "expected": ">= 0.80 positive gain captured by top ceil(20%) positive-gain pairs",
            "actual": concentration,
            "selected_positive_pair_count": concentration_count,
            "passed": gate_results["G3"],
        },
        "G4": {
            "expected": ">= 0.10 of GT-positive pairs gain >= 0.05",
            "actual": practical_fraction,
            "actual_count": practical_count,
            "passed": gate_results["G4"],
        },
        "G5": {
            "expected": ">= 2 noncanonical actions uniquely best on >= 3 GT-positive pairs each",
            "actual": len(diverse_actions),
            "actual_qualifying_action_count": len(diverse_actions),
            "actual_qualifying_actions": diverse_actions,
            "actual_unique_best_counts": dict(sorted(unique_best.items())),
            "passed": gate_results["G5"],
        },
        "G6": {
            "expected": "all four frozen non-homogeneity checks pass",
            "actual": g6_checks,
            "passed": gate_results["G6"],
        },
        "G7": {
            "expected": "all frozen execution-integrity checks pass",
            "actual": integrity,
            "passed": gate_results["G7"],
        },
        "G8": {
            "expected": {
                "generation_wall_seconds": "<= 14400",
                "new_trajectory_cache_bytes": "<= 21474836480",
            },
            "actual": {
                "generation_wall_seconds": resource.get("generation_wall_seconds"),
                "new_trajectory_cache_bytes": resource.get("new_trajectory_cache_bytes"),
            },
            "passed": gate_results["G8"],
        },
    }

    per_class = {}
    for class_id in range(1, 21):
        selected = [item for item in state_results if item["class_id"] == class_id]
        per_class[str(class_id)] = {
            "positive_pair_count": len(selected),
            "canonical_present_pair_macro_iou": (
                None if not selected else sum(item["canonical_iou"] for item in selected) / len(selected)
            ),
            "oracle_present_pair_macro_iou": (
                None if not selected else sum(item["oracle_iou"] for item in selected) / len(selected)
            ),
            "oracle_mean_gain": (
                None if not selected else sum(item["oracle_gain"] for item in selected) / len(selected)
            ),
        }
    return {
        "status": status,
        "gate_results": gate_results,
        "gate_audit": gate_audit,
        "measurements": {
            "positive_pair_count": len(state_results),
            "canonical_present_pair_macro_iou": canonical_mean,
            "oracle_present_pair_macro_iou": oracle_mean,
            "absolute_oracle_headroom": absolute_gain,
            "canonical_remaining_error": 1.0 - canonical_mean,
            "aggregate_normalized_error_reduction": normalized,
            "strictly_positive_gain_pair_count": sum(gain > 0.0 for gain in gains),
            "top_positive_gain_pair_count": concentration_count,
            "top_20_percent_positive_gain_capture_fraction": concentration,
            "gain_at_least_0_05_count": practical_count,
            "gain_at_least_0_05_fraction": practical_fraction,
            "gain_distribution": {
                "quantiles": _quantiles(gains),
                "thresholds": gain_thresholds,
            },
            "positive_gain_concentration": concentration_points,
            "unique_best_counts": dict(sorted(unique_best.items())),
            "qualifying_unique_best_actions": diverse_actions,
            "oracle_tie_pair_count": sum(tie_sets.values()),
            "oracle_tie_action_sets": dict(sorted(tie_sets.items())),
            "action_redundancy": redundancy,
            "no_result_recovery": {
                "canonical_no_result_gt_positive_count": len(canonical_no_result_states),
                "recovered_by_noncanonical_action_count": len(recovered_no_result_states),
                "still_unrecovered_count": (
                    len(canonical_no_result_states) - len(recovered_no_result_states)
                ),
                "recovery_action_distribution": dict(sorted(Counter(
                    item["oracle_action"] for item in recovered_no_result_states
                ).items())),
            },
            "non_homogeneity": {
                **dict(nonhomogeneity),
                "qualifying_distinct_output_actions": nonhomogeneous_actions,
                "positive_improvement_record_count": len(improvement_references),
                "checks": g6_checks,
            },
            "execution_integrity": integrity,
            "resource_accounting": dict(resource),
            "oracle_action_distribution": dict(sorted(Counter(
                item["oracle_action"] for item in state_results
            ).items())),
            "canonical_no_result_positive_pair_count": sum(
                bool(item["canonical_no_result"]) for item in state_results
            ),
            "per_class": per_class,
        },
        "state_results": state_results,
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0))
    return dict(zip(
        ("min", "q25", "median", "q75", "p90", "p95", "max"),
        (float(value) for value in quantiles),
    ))
