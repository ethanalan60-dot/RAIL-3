"""Pure Track A metrics on identity-validated, already frozen input arrays.

There are no readers, model calls, or output writes in this module. The caller
owns all provenance and GT-access gates. V9C action-level missingness retains
available MAE cells and STOP-relative pairs. Complete-oracle metrics exclude
the entire state after any plan-feasible action execution is unavailable.

Value, budget, and bootstrap formulas follow analysis/tmlr_v6_phase2.py.
V9C's current missingness contract supersedes its whole-state fidelity mask.
Selection validation separately masks the pre-GT technical-missing actions,
as the frozen producer did; their original plan feasibility stays unchanged.
E4 consumes FORCED_K only. POSITIVE_GAIN_CAP is reported as secondary sparse
sensitivity; this does not authorize sensitivity to alternative target losses.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rail3.evaluation.tmlr_v9c_e4 import evaluate_e4


FAMILIES = ("R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P", "R3_P")
ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
BUDGETS = (0.01, 0.02, 0.05)
BUDGET_SEMANTICS = ("FORCED_K", "POSITIVE_GAIN_CAP")
E4_PRIMARY_BUDGET_SEMANTICS = "FORCED_K"
SPARSE_ANALYSIS_ROLES = {"FORCED_K": "PRIMARY", "POSITIVE_GAIN_CAP": "SECONDARY_SENSITIVITY"}
_COMPARE = {"<": operator.lt, ">": operator.gt}


def array_sha256(array: np.ndarray) -> str:
    """Hash a typed little-endian C-order array without native-endian drift."""
    value = np.asarray(array)
    if value.dtype.kind == "f":
        value = np.array(value, dtype="<f8", order="C", copy=True)
        value[np.isnan(value)] = np.nan
    elif value.dtype.kind in "iu":
        value = np.asarray(value, dtype="<i8", order="C")
    else:
        raise ValueError("array digest requires numeric data")
    header = f"{value.dtype.str}:{','.join(map(str, value.shape))}:".encode("ascii")
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def bootstrap_group_counts(image_group_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(image_group_ids, dtype=str)
    universe, inverse = np.unique(groups, return_inverse=True)
    if not len(universe):
        raise ValueError("image-group population is empty")
    generator = np.random.Generator(np.random.PCG64(13))
    draws = generator.integers(0, len(universe), size=(2000, len(universe)))
    counts = np.stack([np.bincount(row, minlength=len(universe)) for row in draws]).astype(np.int32)
    return universe, inverse, counts


def _ratio_summary(numerator: np.ndarray, denominator: np.ndarray,
                   group_index: np.ndarray, counts: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    grouped_num = np.bincount(group_index, weights=numerator, minlength=counts.shape[1])
    grouped_den = np.bincount(group_index, weights=denominator, minlength=counts.shape[1])
    sampled_den = counts @ grouped_den
    sampled_num = counts @ grouped_num
    samples = np.full(len(counts), np.nan, dtype=np.float64)
    np.divide(sampled_num, sampled_den, out=samples, where=sampled_den > 0)
    total_den = float(grouped_den.sum())
    point = float(grouped_num.sum() / total_den) if total_den > 0 else None
    return {"point": point, "denominator": total_den, **_interval(samples)}, samples


def _interval(samples: np.ndarray) -> dict[str, Any]:
    undefined = int(np.sum(~np.isfinite(samples)))
    if undefined:
        return {"ci": {"lower": None, "upper": None}, "undefined_bootstrap_replicates": undefined}
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {"ci": {"lower": float(lower), "upper": float(upper)}, "undefined_bootstrap_replicates": 0}


def _status_from_ci(spec: Mapping[str, Any], ci: Mapping[str, Any]) -> str | None:
    if any(ci.get(key) is None for key in ("lower", "upper")):
        return None
    for rule_key, status_key in (("supported_if", "supported_status"), ("opposite_if", "opposite_status")):
        rule = spec[rule_key]
        if _COMPARE[rule["operator"]](ci[rule["endpoint"]], rule["value"]):
            return str(spec[status_key])
    return str(spec["otherwise"])


def practical_tie(left: float | None, right: float | None,
                  ci: Mapping[str, float | None]) -> bool | None:
    """Inherited CI containment using the right model's full-sample value."""
    low, high = ci.get("lower"), ci.get("upper")
    if left is None or right is None or low is None or high is None:
        return None
    if not all(math.isfinite(value) for value in (left, right, low, high)) or min(left, right) < 0 or low > high:
        raise ValueError("practical-equivalence inputs must be valid nonnegative metrics and an ordered finite CI")
    if right == 0:
        return left == 0 and low == 0 and high == 0
    return low >= -0.05 * right and high <= 0.05 * right


def evaluate_e5(contract: Mapping[str, Any], e1_status: str, e2_status: str,
                e4_supported: bool) -> dict[str, Any]:
    """Interpret the frozen first-match E5 tree; reject unavailable inputs."""
    domain = contract["decision_tree"]["input_domain"]
    if (e1_status not in domain["E1"] or e2_status not in domain["E2"]
            or type(e4_supported) is not bool):
        raise ValueError("E5 requires valid E1/E2 statuses and a boolean E4 result")
    if contract["decision_tree"]["evaluation"] != "FIRST_MATCH_IN_LIST_ORDER":
        raise ValueError("E5 decision ordering drift")
    inputs = {e1_status: True, e2_status: True, "E4_SUPPORTED": e4_supported}
    predicates = {}
    for key, definition in contract["predicates"].items():
        if "all" in definition:
            predicates[key] = all(inputs.get(item, False) for item in definition["all"])
        elif "any" in definition:
            predicates[key] = any(inputs.get(item, False) for item in definition["any"])
        elif "ref" in definition:
            predicates[key] = inputs[definition["ref"]]
        else:
            raise ValueError("unsupported E5 predicate")
    for rule in contract["decision_tree"]["rules"]:
        if all(predicates[key] is expected for key, expected in rule["when"].items()):
            status = rule["result"]
            return {"status": status, "predicates": predicates,
                    "interpretation": contract["interpretations"][status]}
    raise ValueError("E5 contract has no matching rule")


def e4_primary_sparse_rows(sparse_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Bind E4 only to R3 FORCED_K; secondary sensitivity never supplies D."""
    primary = [row for row in sparse_rows
               if row.get("model") == "R3_P" and row.get("budget_semantics") == E4_PRIMARY_BUDGET_SEMANTICS]
    if len(primary) != len(BUDGETS) or {row.get("budget") for row in primary} != set(BUDGETS):
        raise ValueError("E4 primary requires exactly one FORCED_K row at each frozen budget")
    return primary


def _validate(predictions: Mapping[str, np.ndarray], targets: np.ndarray,
              feasible: np.ndarray, selected_actions: Mapping[str, np.ndarray],
              state_ids: Sequence[str], group_ids: Sequence[str]) -> np.ndarray:
    if set(predictions) != set(FAMILIES) or set(selected_actions) != set(FAMILIES):
        raise ValueError("Track A requires exactly the six frozen model families")
    if (not isinstance(targets, np.ndarray) or targets.dtype != np.float64
            or targets.ndim != 2 or targets.shape[1] != 5 or not len(targets)):
        raise ValueError("targets must be nonempty float64 N x 5")
    if (not isinstance(feasible, np.ndarray) or feasible.dtype != bool or feasible.shape != targets.shape):
        raise ValueError("plan feasibility must be aligned boolean N x 5")
    if np.isinf(targets).any() or np.any(np.isfinite(targets) & ~feasible):
        raise ValueError("target definedness contradicts plan feasibility")
    if np.any((targets[np.isfinite(targets)] < 0) | (targets[np.isfinite(targets)] > 1)):
        raise ValueError("pixel-mismatch targets must lie in [0,1]")
    if (len(state_ids) != len(targets) or len(group_ids) != len(targets)
            or len(set(state_ids)) != len(state_ids)
            or any(not isinstance(item, str) or not item for item in (*state_ids, *group_ids))):
        raise ValueError("state and image-group identities are invalid or misaligned")
    common_available = None
    for family in FAMILIES:
        pred, chosen = predictions[family], selected_actions[family]
        if (not isinstance(pred, np.ndarray) or pred.dtype != np.float64 or pred.shape != targets.shape
                or np.isinf(pred).any()):
            raise ValueError("prediction must be aligned float64 without infinities")
        available = np.isfinite(pred).all(axis=1)
        if np.any(~available & ~np.isnan(pred).all(axis=1)):
            raise ValueError("prediction missingness must affect the entire state")
        if common_available is not None and not np.array_equal(common_available, available):
            raise ValueError("frozen model prediction missingness differs")
        common_available = available
        if (not isinstance(chosen, np.ndarray) or chosen.shape != (len(targets),)
                or chosen.dtype.kind not in "iu" or np.any(chosen[~available] != -1)):
            raise ValueError("selected actions must be aligned integers with missing=-1")
        if np.any(~available & np.isfinite(targets).any(axis=1)):
            raise ValueError("missing predictor state has defined trajectory targets")
        if np.any(available & ~feasible[:, 0]):
            raise ValueError("defined predictions require feasible STOP")
        # Root's identity/missingness gate guarantees that undefined targets
        # here correspond only to frozen pre-GT execution missingness or
        # plan-infeasible actions. This reproduces the producer's selection
        # availability mask, without consulting any numerical GT residual.
        # Keep `feasible` untouched: failed A4 remains plan-feasible and must
        # therefore exclude its state from complete-oracle evaluation.
        selection_available = feasible[available] & np.isfinite(targets[available])
        if np.any(~selection_available.any(axis=1)):
            raise ValueError("defined prediction has no frozen selection-available action")
        expected = np.where(selection_available, pred[available], np.inf).argmin(axis=1)
        if not np.array_equal(chosen[available], expected):
            raise ValueError("frozen selected actions differ from first selection-available residual argmin")
    return common_available


def evaluate_metrics(predictions: Mapping[str, np.ndarray], targets: np.ndarray,
                     plan_feasible: np.ndarray, selected_actions: Mapping[str, np.ndarray],
                     state_ids: Sequence[str], image_group_ids: Sequence[str],
                     e4_contract: Mapping[str, Any], e5_contract: Mapping[str, Any], *,
                     budget_semantics: str) -> dict[str, Any]:
    """Compute frozen Track A metrics and decisions, returning JSON-safe values.

    ``budget_semantics`` is required and must be FORCED_K, the authorized E4
    primary binding. POSITIVE_GAIN_CAP remains a secondary sensitivity output
    and cannot supply E4 or E5. NaN input targets are unavailable/infeasible,
    never zero. Undefined inferential inputs return an explicit hold and do not
    become negative scientific findings.
    """
    if budget_semantics != E4_PRIMARY_BUDGET_SEMANTICS:
        raise ValueError("E4 primary budget semantics must be FORCED_K; POSITIVE_GAIN_CAP is SECONDARY_SENSITIVITY only")
    if tuple(e4_contract["sparse_budgets"]) != BUDGETS:
        raise ValueError("E4 budget registry drift")
    available = _validate(predictions, targets, plan_feasible, selected_actions, state_ids, image_group_ids)
    universe, group_index, counts = bootstrap_group_counts(image_group_ids)
    defined = plan_feasible & np.isfinite(targets) & available[:, None]
    relative_defined = defined[:, :1] & defined[:, 1:]
    incomplete = np.any(plan_feasible & ~np.isfinite(targets), axis=1)
    complete = available & ~incomplete
    oracle_mask = complete & plan_feasible[:, 0] & plan_feasible[:, 1:].any(axis=1)
    n = len(targets)
    ids = np.asarray(state_ids, dtype=str)
    selected_rows = np.flatnonzero(oracle_mask)
    truth = np.where(plan_feasible[oracle_mask], targets[oracle_mask], np.inf)
    oracle_action = truth.argmin(axis=1) if len(truth) else np.empty(0, dtype=np.int64)
    local_rows = np.arange(len(truth))
    oracle_gain = np.maximum(0.0, truth[:, 0] - truth[local_rows, oracle_action])
    total_oracle = float(oracle_gain.sum())
    oracle_beneficial = oracle_gain > 0
    core_rows: list[dict[str, Any]] = []
    family_metrics: dict[str, Any] = {}
    samples_by_family: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    sparse_rows: list[dict[str, Any]] = []
    digest_arrays = {"bootstrap_counts": array_sha256(counts)}
    for family in FAMILIES:
        pred = predictions[family]
        absolute = np.where(defined, np.abs(pred - targets), 0.0)
        relative = np.where(relative_defined, np.abs(
            (pred[:, 1:] - pred[:, :1]) - (targets[:, 1:] - targets[:, :1])), 0.0)
        family_metrics[family], samples_by_family[family] = {}, {}
        for metric, numerator, denominator in (
            ("absolute_residual_mae", absolute.sum(axis=1), defined.sum(axis=1)),
            ("drre", relative.sum(axis=1), relative_defined.sum(axis=1)),
        ):
            result, samples = _ratio_summary(numerator, denominator, group_index, counts)
            family_metrics[family][metric] = result
            samples_by_family[family][metric] = samples
            digest_arrays[f"{family}.{metric}"] = array_sha256(samples)
        core_rows.append({"row_type": "prediction", "model": family, "split": "COCO_EXTERNAL",
                          "absolute_residual_mae": family_metrics[family]["absolute_residual_mae"]["point"],
                          "drre": family_metrics[family]["drre"]["point"],
                          "defined_state_actions": int(defined.sum()),
                          "stop_relative_pairs": int(relative_defined.sum()),
                          "defined_actions_by_action": {action: int(defined[:, index].sum()) for index, action in enumerate(ACTIONS)},
                          "defined_pairs_by_nonstop_action": {action: int(relative_defined[:, index].sum()) for index, action in enumerate(ACTIONS[1:])},
                          "mae_evaluable_states": int(defined.any(axis=1).sum()),
                          "drre_evaluable_states": int(relative_defined.any(axis=1).sum()),
                          "mask_rule": "DEFINED_ACTION_CELLS_AND_DEFINED_STOP_RELATIVE_PAIRS"})
        chosen = selected_actions[family][oracle_mask]
        realized = truth[:, 0] - truth[local_rows, chosen]
        regret = truth[local_rows, chosen] - truth[local_rows, oracle_action]
        nonstop = chosen != 0
        selected_count = int(nonstop.sum())
        false_stop = int(np.sum(~nonstop & oracle_beneficial))
        false_intervention = int(np.sum(nonstop & ~oracle_beneficial))
        negative = int(np.sum(nonstop & (realized < 0)))
        realized_total = float(realized.sum())
        all_prediction_count = int(available.sum())
        all_nonstop = int(np.sum(selected_actions[family][available] != 0))
        row = {"row_type": "intervention", "model": family, "split": "COCO_EXTERNAL", "lambda": 0.0,
               "evaluable_states": int(oracle_mask.sum()), "realized_gain": realized_total,
               "oracle_gain": total_oracle, "oracle_gain_denominator": total_oracle,
               "normalized_regret": float(regret.sum() / total_oracle) if total_oracle > 0 else None,
               "gain_capture": realized_total / total_oracle if total_oracle > 0 else None,
               "selected_nonstop_states": selected_count,
               "predicted_non_STOP_count": all_nonstop,
               "defined_prediction_states": all_prediction_count,
               "missing_prediction_states": int(n - all_prediction_count),
               "predicted_non_STOP_fraction": all_nonstop / all_prediction_count if all_prediction_count else None,
               "false_stop_states": false_stop, "false_intervention_states": false_intervention,
               "negative_intervention_states": negative,
               "false_stop_rate": false_stop / len(chosen) if len(chosen) else None,
               "false_intervention_rate": false_intervention / len(chosen) if len(chosen) else None,
               "negative_intervention_rate": negative / selected_count if selected_count else None}
        core_rows.append(row)
        summaries[family] = row
        num, den = np.zeros(n), np.zeros(n)
        num[selected_rows], den[selected_rows] = regret, oracle_gain
        interval, sampled_regret = _ratio_summary(num, den, group_index, counts)
        family_metrics[family]["normalized_regret"] = interval
        samples_by_family[family]["normalized_regret"] = sampled_regret
        digest_arrays[f"{family}.normalized_regret"] = array_sha256(sampled_regret)
        pred_masked = np.where(plan_feasible[oracle_mask], pred[oracle_mask], np.inf)
        best_nonstop = pred_masked[:, 1:].argmin(axis=1) + 1 if len(truth) else np.empty(0, dtype=np.int64)
        score = pred_masked[:, 0] - pred_masked[local_rows, best_nonstop]
        action_gain = truth[:, 0] - truth[local_rows, best_nonstop]
        order = np.lexsort((ids[oracle_mask], -score))
        for budget in BUDGETS:
            cap = max(1, math.ceil(budget * len(order)))
            denominator = float(np.sort(oracle_gain)[::-1][:cap].sum())
            for semantics in BUDGET_SEMANTICS:
                selected = order[:cap]
                if semantics == "POSITIVE_GAIN_CAP":
                    selected = selected[score[selected] > 0]
                gains = action_gain[selected]
                selection = [[str(ids[selected_rows[index]]), ACTIONS[int(best_nonstop[index])]] for index in selected]
                # Canonical JSON array framing matches inherited budget hashes.
                selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                sparse = {"row_type": "sparse", "model": family, "split": "COCO_EXTERNAL", "lambda": 0.0,
                          "budget": budget, "budget_semantics": semantics, "eligible_states": len(order),
                          "analysis_role": SPARSE_ANALYSIS_ROLES[semantics],
                          "supplies_E4": family == "R3_P" and semantics == E4_PRIMARY_BUDGET_SEMANTICS,
                          "supplies_E5_directly": False,
                          "cap": cap, "used": len(selected), "unused": cap - len(selected),
                          "selected_true_positive_states": int(np.sum(gains > 0)),
                          "selected_negative_states": int(np.sum(gains < 0)),
                          "selected_zero_gain_states": int(np.sum(gains == 0)),
                          "realized_gain": float(gains.sum()) if len(order) else None,
                          "oracle_gain": denominator if len(order) else None,
                          "oracle_gain_denominator": denominator if len(order) else None,
                          "gain_capture": float(gains.sum() / denominator) if denominator > 0 else None,
                          "selection_sha256": selection_hash}
                sparse_rows.append(sparse)
                core_rows.append(sparse)
    contrasts = {}
    comparisons = [("E1", "R1_P", "R0_CM_P", "absolute_residual_mae"),
                   ("E2", "R1_P", "R0_CM_P", "drre")]
    comparisons += [(f"E3_{right}_{metric}", "R1_P", right, metric)
                    for right in ("RECT_P", "UNION_P") for metric in ("absolute_residual_mae", "drre")]
    for key, left, right, metric in comparisons:
        left_point = family_metrics[left][metric]["point"]
        right_point = family_metrics[right][metric]["point"]
        contrast_samples = samples_by_family[left][metric] - samples_by_family[right][metric]
        entry = {"left": left, "right": right, "metric": metric,
                 "left_point": left_point, "right_point": right_point,
                 "difference": left_point - right_point if left_point is not None and right_point is not None else None,
                 **_interval(contrast_samples)}
        if key.startswith("E3"):
            tie = practical_tie(left_point, right_point, entry["ci"])
            entry.update({"practical_tie": tie, "relative_margin": 0.05,
                          "margin_reference": "right_full_sample_point",
                          "status": None if tie is None else "PRACTICAL_TIE" if tie else "NOT_PRACTICAL_TIE"})
        contrasts[key] = entry
        digest_arrays[key] = array_sha256(contrast_samples)
    improvement_samples = 1.0 - samples_by_family["R3_P"]["normalized_regret"]
    r3 = summaries["R3_P"]
    improvement = {"point": 1.0 - r3["normalized_regret"] if r3["normalized_regret"] is not None else None,
                   "definition": "1 - normalized_regret(R3_P)", **_interval(improvement_samples)}
    digest_arrays["E4.utility_improvement"] = array_sha256(improvement_samples)
    holds = []
    registry: dict[str, Any] = {}
    for hypothesis in ("E1", "E2"):
        status = _status_from_ci(e5_contract["primitive_statuses"][hypothesis], contrasts[hypothesis]["ci"])
        registry[hypothesis] = {"status": status, **contrasts[hypothesis]}
        if status is None:
            holds.append(f"{hypothesis}_INTERVAL_UNDEFINED")
    registry["E3"] = {"status": "PAIRWISE_COMPARISONS_REPORTED", "overall_support_criterion_added": False,
                      "comparisons": {key: value for key, value in contrasts.items() if key.startswith("E3")}}
    e4_metrics = {"normalized_regret": r3["normalized_regret"], "utility_improvement_ci": improvement["ci"],
                  "predicted_non_STOP_count": r3["predicted_non_STOP_count"],
                  "sparse": e4_primary_sparse_rows(sparse_rows)}
    if r3["normalized_regret"] is None or improvement["ci"]["lower"] is None:
        registry["E4"] = {"status": None, "E4_SUPPORTED": None, "E5_U": None,
                          "reason": "UNDEFINED_AGGREGATE_OR_BOOTSTRAP"}
        holds.append("E4_AGGREGATE_OR_INTERVAL_UNDEFINED")
    else:
        registry["E4"] = evaluate_e4(e4_contract, e4_metrics)
    registry["E4"].update({"metrics": e4_metrics, "budget_semantics": budget_semantics,
                            "analysis_role": "PRIMARY",
                            "utility_improvement": improvement})
    if holds:
        registry["E5"] = {"status": None, "reason": "REQUIRED_PRIMITIVE_UNAVAILABLE"}
    else:
        registry["E5"] = evaluate_e5(e5_contract, registry["E1"]["status"], registry["E2"]["status"], registry["E4"]["E4_SUPPORTED"])
    return {"schema_version": "rail3.tmlr-v9c-track-a-metrics.v1",
            "evaluation_status": "EVALUATION_INPUT_HOLD" if holds else "COMPLETE", "holds": holds,
            "core_rows": core_rows,
            "sparse_budget_binding": {"e4_primary": E4_PRIMARY_BUDGET_SEMANTICS,
                                      "secondary": "POSITIVE_GAIN_CAP",
                                      "secondary_role": "SECONDARY_SENSITIVITY",
                                      "secondary_supplies_E4": False, "secondary_supplies_E5": False},
            "bootstrap_results": {"unit": "COCO image_id", "image_groups": len(universe), "replicates": 2000,
                                  "seed": 13, "generator": "numpy.random.PCG64", "quantile_method": "linear",
                                  "group_order": "ascending", "family_metrics": family_metrics,
                                  "contrasts": contrasts, "e4_utility_improvement": improvement,
                                  "count_matrix_sha256": hashlib.sha256(counts.astype("<i4").tobytes()).hexdigest()},
            "e1_e5_registry": registry,
            "missingness_evaluation": {"panel_states": n, "retained_image_groups": len(universe),
                                       "prediction_unavailable_states": int((~available).sum()),
                                       "plan_feasible_target_undefined_actions": int(np.sum(plan_feasible & ~np.isfinite(targets))),
                                       "complete_oracle_excluded_states": int((~oracle_mask).sum()),
                                       "technical_incomplete_or_prediction_missing_states": int((~complete).sum()),
                                       "complete_states_without_feasible_nonstop": int((complete & ~plan_feasible[:, 1:].any(axis=1)).sum()),
                                       "complete_oracle_evaluable_states": int(oracle_mask.sum()),
                                       "primary_mae_defined_state_actions": int(defined.sum()),
                                       "primary_drre_defined_stop_pairs": int(relative_defined.sum()),
                                       "primary_mae_evaluable_states": int(defined.any(axis=1).sum()),
                                       "primary_drre_evaluable_states": int(relative_defined.any(axis=1).sum()),
                                       "rule": "V9C_ACTION_LEVEL_MISSINGNESS_PRIMARY_AND_COMPLETE_ORACLE_UTILITY"},
            "reproducibility": {"array_sha256": digest_arrays},
            "track_b_status": "NO_GO", "e6": None,
            "secondary_utility_sensitivity": "NOT_AUTHORIZED_NOT_RUN",
            "secondary_utility_sensitivity_scope": "ALTERNATIVE_TARGET_LOSSES_ONLY_NOT_SPARSE_BUDGET_SEMANTICS"}
