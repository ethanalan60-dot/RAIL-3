"""Frozen FIT-only estimators for the TMLR V6 first batch.

The independent sampling unit is an image group.  Class-conditioned states,
outer folds, and training seeds are nested technical observations and are
never treated as independent replicates.  This module has no data readers and
is deliberately unaware of protected evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.models.tmlr_v6.training import ACTIONS, EXTERNAL_SPLITS, SEEDS


PREDICTION_METRICS = (
    "absolute_residual_mae",
    "drre",
    "beneficial_sign_accuracy",
    "stop_vs_best_accuracy",
    "informative_action_order_accuracy",
)
DECISION_METRICS = (
    "normalized_regret",
    "predicted_beneficial_rate",
    "false_stop_rate",
    "false_intervention_rate",
    "negative_intervention_rate",
    "beneficial_detection_rate",
    "gain_capture",
)
PRIMARY_METRICS = ("absolute_residual_mae", "drre")
BUDGET_SEMANTICS = ("FORCED_K", "POSITIVE_GAIN_CAP")
PRIMARY_CONTRASTS = (
    ("R1_P", "R0_SMALL_P", "R1_P_MINUS_R0_SMALL_P"),
    ("R1_P", "R0_CM_P", "R1_P_MINUS_R0_CM_P"),
    ("R1_P", "RECT_P", "R1_P_MINUS_RECT_P"),
    ("R1_P", "UNION_P", "R1_P_MINUS_UNION_P"),
)
HISTORICAL_CONTRASTS = (
    ("R0_SMALL_P", "RETRO_R0_FIT_OOF", "C1"),
    ("R1_P", "RETRO_R1_FIT_OOF", "C2"),
)
HISTORICAL_COMPARISON_METADATA = {
    "C1": {
        "comparison_scope": (
            "whole_pipeline_RETRO_V4_R0_to_PROSPECTIVE_V1_R0_SMALL"
        ),
        "changed_components": (
            "forbidden_runtime_columns_removed; input_dimension_changed; "
            "parameter_count_8065_to_7809"
        ),
        "shared_components": "state_level_training_objective",
        "pure_runtime_ablation": False,
        "label_isolated_causal_effect": False,
    },
    "C2": {
        "comparison_scope": (
            "multi_component_whole_pipeline_RETRO_V4_R1_to_PROSPECTIVE_V1_R1"
        ),
        "changed_components": (
            "forbidden_runtime_columns_removed; action_dimension_18_to_16; "
            "parameter_count_103649_to_103521; target_valid_to_full_raster_"
            "inference_aggregation; atom_plus_state_to_state_only_objective"
        ),
        "shared_components": "R1_shared_atomic_model_family",
        "pure_runtime_ablation": False,
        "label_isolated_causal_effect": False,
    },
}
FIRST_BATCH_REPORT_CHECKS = frozenset({
    "exact_five_model_panel", "exact_cross_product_75",
    "five_folds_stitched_per_seed", "continuous_seed_mean_before_selection",
    "cost_profiles_fold_specific_seed_invariant", "cost_profiles_outside_predictor",
    "all_registered_prediction_metrics", "all_registered_lambda_decision_metrics",
    "all_lambda_budget_semantics", "paired_image_group_bootstrap_2000_seed13",
    "image_group_is_independent_unit", "class_states_not_independent_replicates",
    "folds_not_independent_replicates", "group_win_fractions_reported",
    "historical_c1_c2_complete_safe_fit_oof",
    "historical_c1_c2_not_label_isolated_causal_effect",
    "historical_aggregate_fallback_not_used",
    "output_artifact_identities_exhaustive",
    "whole_state_complete_case_applied_to_all_formal_metrics",
    "protected_access_zero", "training_not_invoked", "sam_inference_not_invoked",
})
FIRST_BATCH_OUTPUT_ROWS = {
    "first_batch_prediction_metrics.csv": 5,
    "historical_c1_c2_prediction_metrics.csv": 2,
    "first_batch_decision_metrics.csv": 30,
    "first_batch_sparse_budget_metrics.csv": 360,
    "first_batch_primary_group_bootstrap.csv": 8,
    "historical_c1_c2_group_bootstrap.csv": 4,
    "primary_metric_per_group.csv": 19_096,
    "first_batch_run_inventory.csv": 75,
    "historical_c1_c2_run_inventory.csv": 30,
    "first_batch_metrics.json": 409,
}


def annotate_historical_comparison(
    rows: Sequence[Mapping[str, Any]], *, comparison_id_key: str = "contrast_id",
) -> list[dict[str, Any]]:
    """Attach the frozen non-causal C1/C2 interpretation to machine rows."""

    result: list[dict[str, Any]] = []
    for source in rows:
        comparison_id = str(source.get(comparison_id_key, ""))
        if comparison_id not in HISTORICAL_COMPARISON_METADATA:
            raise RuntimeError(
                f"unregistered TMLR V6 historical comparison: {comparison_id}"
            )
        row = dict(source)
        row["comparison_id"] = comparison_id
        row.update(HISTORICAL_COMPARISON_METADATA[comparison_id])
        result.append(row)
    return result


@dataclass(frozen=True)
class ResidualSurface:
    """One continuous seed-ensemble OOF residual surface."""

    model_id: str
    prediction: np.ndarray
    target: np.ndarray
    feasible: np.ndarray
    groups: np.ndarray
    state_ids: tuple[str, ...]

    def validated(self) -> "ResidualSurface":
        prediction = np.asarray(self.prediction)
        target = np.asarray(self.target)
        feasible = np.asarray(self.feasible)
        groups = np.asarray(self.groups)
        if prediction.ndim != 2 or prediction.shape[1] != len(ACTIONS):
            raise ValueError("TMLR V6 residual prediction must be N x 5")
        if target.shape != prediction.shape or feasible.shape != prediction.shape:
            raise ValueError("TMLR V6 prediction/target/feasibility drift")
        if groups.shape != (len(prediction),) or len(self.state_ids) != len(prediction):
            raise ValueError("TMLR V6 state identities are not aligned")
        if not np.isfinite(prediction).all():
            raise ValueError("TMLR V6 residual prediction is not finite")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("TMLR V6 state IDs are not unique")
        if feasible.dtype.kind != "b" or np.any(np.isfinite(target) & ~feasible):
            raise ValueError("TMLR V6 defined target is marked pre-action infeasible")
        return self


def complete_case_state_mask(
    surface: ResidualSurface,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude a whole state after any plan-feasible execution failure."""

    surface.validated()
    target = np.asarray(surface.target, dtype=np.float64)
    plan_feasible = np.asarray(surface.feasible, dtype=bool)
    excluded = np.any(plan_feasible & ~np.isfinite(target), axis=1)
    excluded_ids = sorted(
        state_id for state_id, drop in zip(surface.state_ids, excluded) if drop
    )
    complete = ~excluded
    return complete, {
        "complete_case_states": int(complete.sum()),
        "plan_feasible_target_undefined_excluded_states": int(excluded.sum()),
        "plan_feasible_target_undefined_state_ids_sha256": hashlib.sha256(
            canonical_json_bytes(excluded_ids)
        ).hexdigest(),
    }


@dataclass(frozen=True)
class CostSurface:
    """Fold-profiled cost arrays kept outside every residual predictor."""

    normalized: np.ndarray
    seconds: np.ndarray

    def validated(self, shape: tuple[int, int]) -> "CostSurface":
        normalized = np.asarray(self.normalized, dtype=np.float64)
        seconds = np.asarray(self.seconds, dtype=np.float64)
        if normalized.shape != shape or seconds.shape != shape:
            raise ValueError("TMLR V6 cost profile is not state/action aligned")
        if (
            not np.isfinite(normalized).all()
            or not np.isfinite(seconds).all()
            or np.any(normalized < 0.0)
            or np.any(seconds < 0.0)
            or not np.array_equal(normalized[:, 0], np.zeros(shape[0]))
            or not np.array_equal(seconds[:, 0], np.zeros(shape[0]))
        ):
            raise ValueError("TMLR V6 cost profile violates its nonnegative STOP-zero contract")
        return self


def validate_evaluation_protocol(protocol: Mapping[str, Any]) -> None:
    """Fail closed if any registered V6 evaluation/statistics field drifts."""

    evaluation = protocol.get("evaluation", {})
    statistics = protocol.get("statistics", {})
    expected_evaluation = {
        "lambdas": [0.0, 0.01, 0.025, 0.05, 0.1, 0.2],
        "budgets": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        "budget_semantics": list(BUDGET_SEMANTICS),
        "prediction_metrics": list(PREDICTION_METRICS),
        "decision_metrics": list(DECISION_METRICS),
        "tie_order": list(ACTIONS),
        "seed_aggregation": (
            "stitch five outer-held-out folds per seed, then elementwise "
            "arithmetic-mean the three continuous prediction surfaces before selection"
        ),
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise RuntimeError(f"TMLR V6 evaluation protocol drift: {key}")
    expected_statistics = {
        "unit": "image_group_id",
        "paired": True,
        "replicates": 2000,
        "seed": 13,
        "confidence_interval": [0.025, 0.975],
        "primary_contrasts": [
            "R1_P-R0_SMALL_P", "R1_P-R0_CM_P",
            "R1_P-RECT_P", "R1_P-UNION_P",
        ],
        "primary_metrics": list(PRIMARY_METRICS),
        "report_group_win_fraction": True,
        "folds_as_independent_replicates": False,
        "class_states_as_independent_replicates": False,
    }
    for key, value in expected_statistics.items():
        if statistics.get(key) != value:
            raise RuntimeError(f"TMLR V6 statistics protocol drift: {key}")


def strict_seed_ensemble(
    seed_predictions: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Average three complete continuous surfaces in the frozen seed order."""

    if set(seed_predictions) != set(SEEDS):
        raise RuntimeError("TMLR V6 seed ensemble is not exactly 13/37/71")
    arrays = [np.asarray(seed_predictions[seed], dtype=np.float64) for seed in SEEDS]
    if (
        arrays[0].ndim != 2
        or arrays[0].shape[1] != len(ACTIONS)
        or any(item.shape != arrays[0].shape for item in arrays)
        or any(not np.isfinite(item).all() for item in arrays)
    ):
        raise RuntimeError("TMLR V6 seed surface coverage/shape is incomplete")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def sorted_group_universe(groups: Sequence[str]) -> np.ndarray:
    result = np.asarray(sorted(set(str(item) for item in groups)), dtype=str)
    if result.size == 0:
        raise ValueError("TMLR V6 group universe is empty")
    return result


def group_index(groups: Sequence[str], universe: Sequence[str]) -> np.ndarray:
    ordered = tuple(str(item) for item in universe)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("TMLR V6 group universe is empty or duplicated")
    lookup = {value: index for index, value in enumerate(ordered)}
    try:
        return np.asarray([lookup[str(item)] for item in groups], dtype=np.int32)
    except KeyError as error:
        raise ValueError("TMLR V6 surface contains an unregistered group") from error


def bootstrap_group_counts(
    group_count: int, *, replicates: int, seed: int,
) -> np.ndarray:
    """Draw deterministic paired image-group bootstrap multiplicities."""

    if group_count <= 0 or replicates <= 0:
        raise ValueError("TMLR V6 bootstrap dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, group_count, size=(replicates, group_count))
    counts = np.stack([
        np.bincount(row, minlength=group_count) for row in draws
    ]).astype(np.int32)
    if not np.all(counts.sum(axis=1) == group_count):
        raise RuntimeError("TMLR V6 group-bootstrap replicate size drift")
    return counts


def _sum_by_group(
    values: np.ndarray, state_group: np.ndarray, group_count: int,
) -> np.ndarray:
    return np.bincount(
        np.asarray(state_group, dtype=np.int64),
        weights=np.asarray(values, dtype=np.float64),
        minlength=group_count,
    ).astype(np.float64)


def _prediction_sufficient(
    surface: ResidualSurface, universe: Sequence[str],
) -> dict[str, np.ndarray]:
    surface.validated()
    prediction = np.asarray(surface.prediction, dtype=np.float64)
    target = np.asarray(surface.target, dtype=np.float64)
    plan_feasible = np.asarray(surface.feasible, dtype=bool)
    complete, _ = complete_case_state_mask(surface)
    defined = complete[:, None] & plan_feasible
    state_group = group_index(surface.groups, universe)
    group_count = len(universe)
    evaluable = defined[:, 0] & defined[:, 1:].any(axis=1)

    absolute = np.where(defined, np.abs(prediction - target), 0.0)
    pair_defined = evaluable[:, None] & defined[:, :1] & defined[:, 1:]
    predicted_delta = prediction[:, 1:] - prediction[:, :1]
    true_delta = target[:, 1:] - target[:, :1]
    relative_absolute = np.where(
        pair_defined, np.abs(predicted_delta - true_delta), 0.0,
    )

    masked_prediction = np.where(defined, prediction, np.inf)
    masked_target = np.where(defined, target, np.inf)
    pred_best_nonstop = np.min(masked_prediction[:, 1:], axis=1)
    true_best_nonstop = np.min(masked_target[:, 1:], axis=1)
    stop_best_correct = evaluable & (
        ((pred_best_nonstop < masked_prediction[:, 0])
         == (true_best_nonstop < masked_target[:, 0]))
    )

    left, right = np.triu_indices(len(ACTIONS), k=1)
    informative = (
        evaluable[:, None]
        & defined[:, left]
        & defined[:, right]
        & (np.abs(target[:, left] - target[:, right]) > 1e-12)
    )
    order_correct = informative & (
        np.sign(prediction[:, left] - prediction[:, right])
        == np.sign(target[:, left] - target[:, right])
    )
    return {
        "absolute_error_sum": _sum_by_group(
            absolute.sum(axis=1), state_group, group_count,
        ),
        "absolute_error_count": _sum_by_group(
            defined.sum(axis=1), state_group, group_count,
        ),
        "relative_error_sum": _sum_by_group(
            relative_absolute.sum(axis=1), state_group, group_count,
        ),
        "relative_pair_count": _sum_by_group(
            pair_defined.sum(axis=1), state_group, group_count,
        ),
        "beneficial_sign_correct": _sum_by_group(
            np.where(
                pair_defined,
                (predicted_delta < 0.0) == (true_delta < 0.0),
                False,
            ).sum(axis=1),
            state_group, group_count,
        ),
        "stop_best_correct": _sum_by_group(
            stop_best_correct, state_group, group_count,
        ),
        "evaluable_state_count": _sum_by_group(
            evaluable, state_group, group_count,
        ),
        "action_order_correct": _sum_by_group(
            order_correct.sum(axis=1), state_group, group_count,
        ),
        "informative_pair_count": _sum_by_group(
            informative.sum(axis=1), state_group, group_count,
        ),
    }


def _ratio(numerator: float, denominator: float, *, metric: str) -> float:
    if denominator <= 0.0:
        raise RuntimeError(f"TMLR V6 {metric} has no defined observations")
    return float(numerator / denominator)


def prediction_metrics(surface: ResidualSurface) -> dict[str, Any]:
    """Compute all registered cost-free prediction metrics."""

    universe = sorted_group_universe(surface.groups)
    sufficient = _prediction_sufficient(surface, universe)
    total = {name: float(value.sum()) for name, value in sufficient.items()}
    _, complete_case = complete_case_state_mask(surface)
    return {
        "model": surface.model_id,
        "split": "FIT_OOF",
        "seed_aggregation": "continuous_mean_13_37_71_before_selection",
        "image_groups": len(universe),
        "states": len(surface.state_ids),
        **complete_case,
        "defined_state_actions": int(total["absolute_error_count"]),
        "evaluable_states": int(total["evaluable_state_count"]),
        "stop_relative_pairs": int(total["relative_pair_count"]),
        "informative_action_pairs": int(total["informative_pair_count"]),
        "absolute_residual_mae": _ratio(
            total["absolute_error_sum"], total["absolute_error_count"],
            metric="absolute residual MAE",
        ),
        "drre": _ratio(
            total["relative_error_sum"], total["relative_pair_count"],
            metric="DRRE",
        ),
        "beneficial_sign_accuracy": _ratio(
            total["beneficial_sign_correct"], total["relative_pair_count"],
            metric="beneficial-sign accuracy",
        ),
        "stop_vs_best_accuracy": _ratio(
            total["stop_best_correct"], total["evaluable_state_count"],
            metric="STOP-versus-best accuracy",
        ),
        "informative_action_order_accuracy": _ratio(
            total["action_order_correct"], total["informative_pair_count"],
            metric="informative action-order accuracy",
        ),
    }


def _decision_state_arrays(
    surface: ResidualSurface, costs: CostSurface, cost_lambda: float,
) -> dict[str, Any]:
    surface.validated()
    costs.validated(np.asarray(surface.target).shape)
    value_lambda = float(cost_lambda)
    if not np.isfinite(value_lambda) or value_lambda < 0.0:
        raise ValueError("TMLR V6 lambda must be finite and nonnegative")
    prediction = np.asarray(surface.prediction, dtype=np.float64)
    target = np.asarray(surface.target, dtype=np.float64)
    plan_feasible = np.asarray(surface.feasible, dtype=bool)
    complete, complete_case = complete_case_state_mask(surface)
    evaluable = complete & plan_feasible[:, 0] & plan_feasible[:, 1:].any(axis=1)
    if not evaluable.any():
        raise RuntimeError("TMLR V6 decision surface has no evaluable state")
    pred_value = prediction + value_lambda * np.asarray(costs.normalized)
    true_value = target + value_lambda * np.asarray(costs.normalized)
    pred_value = np.where(
        plan_feasible[evaluable], pred_value[evaluable], np.inf,
    )
    true_value = np.where(
        plan_feasible[evaluable], true_value[evaluable], np.inf,
    )
    selected = pred_value.argmin(axis=1)
    oracle = true_value.argmin(axis=1)
    rows = np.arange(len(selected))
    realized = true_value[:, 0] - true_value[rows, selected]
    oracle_gain = np.maximum(0.0, true_value[:, 0] - true_value[rows, oracle])
    return {
        "evaluable": evaluable,
        "prediction_value": pred_value,
        "true_value": true_value,
        "selected": selected,
        "oracle": oracle,
        "realized_gain": realized,
        "oracle_gain": oracle_gain,
        "regret": true_value[rows, selected] - true_value[rows, oracle],
        "complete_case": complete_case,
    }


def decision_metrics(
    surface: ResidualSurface, costs: CostSurface, *, cost_lambda: float,
) -> dict[str, Any]:
    """Compute registered value metrics under one profiled-cost lambda."""

    arrays = _decision_state_arrays(surface, costs, cost_lambda)
    selected = arrays["selected"]
    oracle = arrays["oracle"]
    realized = arrays["realized_gain"]
    oracle_beneficial = oracle != 0
    selected_nonstop = selected != 0
    count = len(selected)
    oracle_count = int(oracle_beneficial.sum())
    selected_count = int(selected_nonstop.sum())
    oracle_total = float(arrays["oracle_gain"].sum())
    realized_total = float(realized.sum())
    headroom_positive = oracle_total > 0.0
    false_stop = int(np.sum((selected == 0) & oracle_beneficial))
    false_intervention = int(np.sum(selected_nonstop & ~oracle_beneficial))
    negative = int(np.sum(selected_nonstop & (realized < 0.0)))
    # A beneficial state is detected only when the selected action actually
    # realizes strictly positive net gain.  Merely choosing any non-STOP
    # action in an oracle-beneficial state can still be harmful or neutral.
    detected = int(np.sum(oracle_beneficial & (realized > 0.0)))
    return {
        "model": surface.model_id,
        "split": "FIT_OOF",
        "lambda": float(cost_lambda),
        **arrays["complete_case"],
        "evaluable_states": count,
        "oracle_beneficial_states": oracle_count,
        "selected_nonstop_states": selected_count,
        "false_stop_states": false_stop,
        "false_intervention_states": false_intervention,
        "negative_intervention_states": negative,
        "beneficial_detected_states": detected,
        "realized_gain": realized_total,
        "oracle_gain": oracle_total,
        "oracle_headroom_positive": headroom_positive,
        "oracle_headroom_status": (
            "POSITIVE_ORACLE_HEADROOM"
            if headroom_positive else "ZERO_ORACLE_HEADROOM_METRIC_UNDEFINED"
        ),
        "normalized_regret_defined": headroom_positive,
        "normalized_regret": (
            float(arrays["regret"].sum()) / oracle_total
            if headroom_positive else None
        ),
        "predicted_beneficial_rate": float(selected_count / count),
        # The two false-event rates retain the established panel convention:
        # both use all evaluable states as denominator.  Detection is the
        # conditional recall among oracle-beneficial states.
        "false_stop_rate": float(false_stop / count),
        "false_intervention_rate": float(false_intervention / count),
        "negative_intervention_rate": (
            float(negative / selected_count) if selected_count else None
        ),
        "beneficial_detection_rate": (
            float(detected / oracle_count) if oracle_count else None
        ),
        "gain_capture_defined": headroom_positive,
        "gain_capture": (
            realized_total / oracle_total if headroom_positive else None
        ),
    }


def budget_metrics(
    surface: ResidualSurface,
    costs: CostSurface,
    *,
    cost_lambda: float,
    budgets: Sequence[float],
    semantics: Sequence[str] = BUDGET_SEMANTICS,
) -> list[dict[str, Any]]:
    """Evaluate both registered sparse-selection semantics at every budget."""

    if tuple(semantics) != BUDGET_SEMANTICS:
        raise RuntimeError("TMLR V6 sparse-budget semantics drift")
    arrays = _decision_state_arrays(surface, costs, cost_lambda)
    prediction = arrays["prediction_value"]
    truth = arrays["true_value"]
    evaluable = arrays["evaluable"]
    state_ids = np.asarray(surface.state_ids, dtype=str)[evaluable]
    rows = np.arange(len(state_ids))
    best_nonstop = prediction[:, 1:].argmin(axis=1) + 1
    score = prediction[:, 0] - prediction[rows, best_nonstop]
    realized = truth[:, 0] - truth[rows, best_nonstop]
    oracle = np.maximum(0.0, truth[:, 0] - truth.min(axis=1))
    normalized_cost = np.asarray(costs.normalized)[evaluable][rows, best_nonstop]
    seconds = np.asarray(costs.seconds)[evaluable][rows, best_nonstop]
    id_order = np.argsort(state_ids, kind="mergesort")
    id_rank = np.empty(len(state_ids), dtype=np.int64)
    id_rank[id_order] = np.arange(len(state_ids))
    order = np.lexsort((id_rank, -score))
    ideal = np.sort(oracle)[::-1]
    beneficial_total = int(np.sum(oracle > 0.0))
    result: list[dict[str, Any]] = []
    for budget in budgets:
        value = float(budget)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid TMLR V6 sparse budget: {value}")
        cap = max(1, math.ceil(value * len(state_ids)))
        oracle_at_cap = float(ideal[:cap].sum())
        headroom_positive = oracle_at_cap > 0.0
        ranked = order[:cap]
        for budget_semantics in semantics:
            chosen = ranked
            if budget_semantics == "POSITIVE_GAIN_CAP":
                chosen = ranked[score[ranked] > 0.0]
            gains = realized[chosen]
            selected_ids = state_ids[chosen]
            selected_actions = [ACTIONS[int(best_nonstop[index])] for index in chosen]
            positive = int(np.sum(gains > 0.0))
            negative = int(np.sum(gains < 0.0))
            result.append({
                "model": surface.model_id,
                "split": "FIT_OOF",
                "lambda": float(cost_lambda),
                "budget": value,
                "budget_semantics": budget_semantics,
                **arrays["complete_case"],
                "eligible_states": len(state_ids),
                "cap": cap,
                "used": int(len(chosen)),
                "unused": int(cap - len(chosen)),
                "selected_true_positive_states": positive,
                "selected_negative_states": negative,
                "selected_zero_gain_states": int(np.sum(gains == 0.0)),
                "realized_gain": float(gains.sum()),
                "oracle_gain": oracle_at_cap,
                "oracle_headroom_positive": headroom_positive,
                "oracle_headroom_status": (
                    "POSITIVE_ORACLE_HEADROOM"
                    if headroom_positive
                    else "ZERO_ORACLE_HEADROOM_METRIC_UNDEFINED"
                ),
                "gain_capture_defined": headroom_positive,
                "gain_capture": (
                    float(gains.sum()) / oracle_at_cap
                    if headroom_positive else None
                ),
                "negative_selected_rate": (
                    float(negative / len(chosen)) if len(chosen) else 0.0
                ),
                "beneficial_detection_rate": (
                    float(positive / beneficial_total) if beneficial_total else None
                ),
                "profiled_cost_common_units": float(normalized_cost[chosen].sum()),
                "profiled_cost_seconds": float(seconds[chosen].sum()),
                "selection_sha256": hashlib.sha256(canonical_json_bytes([
                    [str(state_id), str(action)]
                    for state_id, action in zip(selected_ids, selected_actions)
                ])).hexdigest(),
            })
    return result


def assert_aligned_surfaces(surfaces: Mapping[str, ResidualSurface]) -> None:
    """Require a common state/target surface before paired comparisons."""

    if not surfaces:
        raise ValueError("TMLR V6 surface panel is empty")
    reference = next(iter(surfaces.values())).validated()
    for surface in surfaces.values():
        surface.validated()
        if (
            surface.state_ids != reference.state_ids
            or not np.array_equal(surface.groups, reference.groups)
            or not np.array_equal(surface.feasible, reference.feasible)
            or not np.array_equal(surface.target, reference.target, equal_nan=True)
        ):
            raise RuntimeError("TMLR V6 paired surface alignment drift")


def paired_primary_statistics(
    surfaces: Mapping[str, ResidualSurface],
    *,
    contrasts: Sequence[tuple[str, str, str]],
    replicates: int,
    seed: int,
    interval: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bootstrap fixed MAE/DRRE contrasts and report group win fractions."""

    assert_aligned_surfaces(surfaces)
    reference = next(iter(surfaces.values()))
    _, complete_case = complete_case_state_mask(reference)
    universe = sorted_group_universe(reference.groups)
    counts = bootstrap_group_counts(
        len(universe), replicates=replicates, seed=seed,
    )
    lower, upper = (float(interval[0]), float(interval[1]))
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("TMLR V6 confidence interval quantiles are invalid")
    sufficient = {
        model: _prediction_sufficient(surface, universe)
        for model, surface in surfaces.items()
    }
    group_rows: list[dict[str, Any]] = []
    group_values: dict[str, dict[str, np.ndarray]] = {}
    for model in sorted(surfaces):
        item = sufficient[model]
        values = {
            "absolute_residual_mae": (
                item["absolute_error_sum"] / item["absolute_error_count"]
            ),
            "drre": item["relative_error_sum"] / item["relative_pair_count"],
        }
        if any(not np.isfinite(value).all() for value in values.values()):
            raise RuntimeError("TMLR V6 primary metric is undefined for an image group")
        group_values[model] = values
        for metric in PRIMARY_METRICS:
            for index, group in enumerate(universe):
                group_rows.append({
                    "split": "FIT_OOF",
                    "model": model,
                    "image_group_id": str(group),
                    "metric": metric,
                    "value": float(values[metric][index]),
                    **complete_case,
                })

    def aggregate(model: str, metric: str) -> tuple[float, np.ndarray]:
        item = sufficient[model]
        if metric == "absolute_residual_mae":
            numerator, denominator = (
                item["absolute_error_sum"], item["absolute_error_count"],
            )
        elif metric == "drre":
            numerator, denominator = (
                item["relative_error_sum"], item["relative_pair_count"],
            )
        else:
            raise ValueError(f"unregistered TMLR V6 primary metric: {metric}")
        point = float(numerator.sum() / denominator.sum())
        sampled = (counts @ numerator) / (counts @ denominator)
        return point, sampled

    rows: list[dict[str, Any]] = []
    for left, right, contrast_id in contrasts:
        if left not in surfaces or right not in surfaces:
            raise RuntimeError(f"TMLR V6 fixed contrast input is missing: {contrast_id}")
        for metric in PRIMARY_METRICS:
            left_point, left_sample = aggregate(left, metric)
            right_point, right_sample = aggregate(right, metric)
            difference = left_sample - right_sample
            left_group = group_values[left][metric]
            right_group = group_values[right][metric]
            tie = np.isclose(left_group, right_group, rtol=0.0, atol=1e-15)
            win = (left_group < right_group) & ~tie
            loss = (left_group > right_group) & ~tie
            sampled_win = (counts @ win.astype(np.float64)) / len(universe)
            rows.append({
                "row_type": "paired_contrast",
                "split": "FIT_OOF",
                "contrast_id": contrast_id,
                "left_model": left,
                "right_model": right,
                "metric": metric,
                "direction": "lower_is_better",
                "left_point": left_point,
                "left_bootstrap_ci_lower": float(np.quantile(left_sample, lower)),
                "left_bootstrap_ci_upper": float(np.quantile(left_sample, upper)),
                "right_point": right_point,
                "right_bootstrap_ci_lower": float(np.quantile(right_sample, lower)),
                "right_bootstrap_ci_upper": float(np.quantile(right_sample, upper)),
                "effect_left_minus_right": left_point - right_point,
                "bootstrap_ci_lower": float(np.quantile(difference, lower)),
                "bootstrap_ci_upper": float(np.quantile(difference, upper)),
                "group_win_fraction": float(np.mean(win)),
                "group_tie_fraction": float(np.mean(tie)),
                "group_loss_fraction": float(np.mean(loss)),
                "group_win_fraction_ci_lower": float(np.quantile(sampled_win, lower)),
                "group_win_fraction_ci_upper": float(np.quantile(sampled_win, upper)),
                "image_groups": len(universe),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                **complete_case,
            })
    return rows, group_rows


def validate_report_contract(
    report: Mapping[str, Any], *, inventory_sha256: str,
) -> None:
    """Validate the exact report fields consumed by the first-batch gate."""

    required = {
        "schema_version": "rail3.tmlr-v6.first-batch-fit-evaluation.v1",
        "status": "TMLR_V6_FIRST_BATCH_EVALUATION_PASS",
        "first_batch_run_inventory_sha256": inventory_sha256,
        "completed_valid_runs": 75,
        "metric_pipeline_pass": True,
    }
    drift = {
        key: (report.get(key), value)
        for key, value in required.items()
        if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"TMLR V6 first-batch report contract drift: {drift}")
    access = report.get("heldout_access", {})
    if (
        not isinstance(access, Mapping)
        or set(access) != set(EXTERNAL_SPLITS)
        or any(
            type(access[name]) is not int or access[name] != 0
            for name in EXTERNAL_SPLITS
        )
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")
    population = report.get("population", {})
    state_count = population.get("states") if isinstance(population, Mapping) else None
    complete_count = (
        population.get("complete_case_states")
        if isinstance(population, Mapping) else None
    )
    excluded_count = (
        population.get("plan_feasible_target_undefined_excluded_states")
        if isinstance(population, Mapping) else None
    )
    excluded_sha = (
        population.get("plan_feasible_target_undefined_state_ids_sha256")
        if isinstance(population, Mapping) else None
    )
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (state_count, complete_count, excluded_count)
        )
        or state_count != 27_280
        or complete_count < 0
        or excluded_count < 0
        or complete_count + excluded_count != state_count
        or not isinstance(excluded_sha, str)
        or len(excluded_sha) != 64
        or any(character not in "0123456789abcdef" for character in excluded_sha)
    ):
        raise RuntimeError("TMLR V6 complete-case report contract drift")
    checks = report.get("checks", {})
    if (
        not isinstance(checks, Mapping)
        or set(checks) != FIRST_BATCH_REPORT_CHECKS
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError("TMLR V6 first-batch metric checks are incomplete")
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(FIRST_BATCH_OUTPUT_ROWS):
        raise RuntimeError("TMLR V6 first-batch output registry drift")
    for name, rows in FIRST_BATCH_OUTPUT_ROWS.items():
        identity = outputs[name]
        expected_path = (
            "artifacts/paper/source_data/tmlr_v6/first_batch/" + name
        )
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"path", "bytes", "sha256", "rows"}
            or identity.get("path") != expected_path
            or type(identity.get("bytes")) is not int
            or int(identity["bytes"]) <= 0
            or type(identity.get("rows")) is not int
            or identity["rows"] != rows
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in identity["sha256"]
            )
        ):
            raise RuntimeError("TMLR V6 first-batch output registry drift")


__all__ = [
    "BUDGET_SEMANTICS",
    "DECISION_METRICS",
    "HISTORICAL_COMPARISON_METADATA",
    "HISTORICAL_CONTRASTS",
    "PREDICTION_METRICS",
    "PRIMARY_CONTRASTS",
    "PRIMARY_METRICS",
    "FIRST_BATCH_OUTPUT_ROWS",
    "FIRST_BATCH_REPORT_CHECKS",
    "CostSurface",
    "ResidualSurface",
    "annotate_historical_comparison",
    "assert_aligned_surfaces",
    "bootstrap_group_counts",
    "budget_metrics",
    "complete_case_state_mask",
    "decision_metrics",
    "paired_primary_statistics",
    "prediction_metrics",
    "strict_seed_ensemble",
    "validate_evaluation_protocol",
    "validate_report_contract",
]
