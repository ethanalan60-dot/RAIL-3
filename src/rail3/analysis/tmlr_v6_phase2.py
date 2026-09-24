"""Pure FIT-only analysis primitives for the frozen TMLR V6 phase two.

The independent resampling unit is ``image_group_id``.  Outer folds, class
states, and training seeds are nested technical observations.  This module is
intentionally free of file readers and protected-evaluation imports: artifact
identity and launch-lock validation live in
``rail3.models.tmlr_v6.phase2_contract`` and must happen before callers create
the arrays accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes


ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
REMAINING_ACTION_INDICES = (0, 2, 3, 4)
SEEDS = (13, 37, 71)
FOLDS = tuple(f"fold_{index}" for index in range(5))
LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
PRIMARY_SPARSE_BUDGETS = (0.01, 0.02, 0.05)
BUDGET_SEMANTICS = ("FORCED_K", "POSITIVE_GAIN_CAP")
PREDICTION_KINDS = ("residual", "gain", "logit", "value")
ZERO_HEADROOM_STATUS = "ZERO_ORACLE_HEADROOM_METRIC_UNDEFINED"
POSITIVE_HEADROOM_STATUS = "POSITIVE_ORACLE_HEADROOM"
EXTERNAL_SPLITS = (
    "validation40",
    "calibration30",
    "pilot_test30",
    "official_voc_val",
    "segppd",
    "uav_iap",
)

SECOND_BATCH_MODELS = (
    "R2_P", "R3_P", "Q2_P", "L2D_P", "SPO_PLUS_P",
)
SECOND_BATCH_RESIDUAL_MODELS = ("R2_P", "R3_P")
SECOND_BATCH_DIRECT_MODELS = ("L2D_P", "SPO_PLUS_P")
F0_F1_MODELS = ("F0_P", "F1_P")

SECOND_BATCH_REPORT_SCHEMA = "rail3.tmlr-v6.phase2-second-batch-evaluation.v1"
F0_F1_REPORT_SCHEMA = "rail3.tmlr-v6.phase2-f0-f1-evaluation.v1"
SECOND_BATCH_REPORT_STATUS = "TMLR_V6_PHASE2_SECOND_BATCH_EVALUATION_PASS"
F0_F1_REPORT_STATUS = "TMLR_V6_PHASE2_F0_F1_EVALUATION_PASS"
PHASE2_REPORT_INPUT_BINDINGS = frozenset({
    "phase2_launch_lock_sha256",
    "first_batch_completion_gate_sha256",
    "first_batch_run_inventory_sha256",
    "first_batch_evaluation_sha256",
})
SECOND_BATCH_REPORT_CHECKS = frozenset({
    "exact_cross_product_225",
    "launch_lock_exact",
    "run_manifests_common_verifier",
    "five_fold_oof_coverage_per_seed",
    "continuous_seed_mean_before_selection",
    "q2_stop_exact_zero_after_mean",
    "whole_state_complete_case",
    "zero_headroom_is_null",
    "metric_row_cardinality_exact",
    "protected_access_zero",
})
F0_F1_REPORT_CHECKS = frozenset({
    "exact_cross_product_120",
    "common_launch_lock_and_run_audit",
    "a3_prediction_nan_both_filtrations",
    "remaining_action_set_only",
    "continuous_seed_mean_before_selection",
    "scale_specific_cost_profiles",
    "whole_state_complete_case_remaining",
    "f0_f1_targets_and_feasibility_aligned",
    "within_scale_paired_group_bootstrap",
    "zero_headroom_is_null",
    "metric_row_cardinality_exact",
    "protected_access_zero",
})
REGISTERED_PREDICTION_METRICS = (
    "absolute_residual_mae",
    "drre",
    "beneficial_sign_accuracy",
    "stop_vs_best_accuracy",
    "informative_action_order_accuracy",
)
PREDICTION_METRIC_ROW_FIELDS = frozenset({
    "model",
    "split",
    "prediction_kind",
    "prediction_metric_applicability",
    "residual_prediction_metrics_defined",
    "absolute_residual_mae_defined",
    *REGISTERED_PREDICTION_METRICS,
    "defined_state_actions",
    "stop_relative_pairs",
    "evaluable_states",
    "informative_action_pairs",
    "formal_complete_case_states",
    "plan_feasible_target_undefined_excluded_states",
    "excluded_state_ids_sha256",
})


def assert_zero_external_access(access: Mapping[str, Any]) -> None:
    """Require the exact six-domain zero-access schema."""

    if set(access) != set(EXTERNAL_SPLITS):
        raise RuntimeError("TMLR V6 external-access schema drift")
    if any(
        not isinstance(access[name], int)
        or isinstance(access[name], bool)
        or access[name] != 0
        for name in EXTERNAL_SPLITS
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")


def _physical_action_indices(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or result[0] != 0
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
        or any(index < 0 or index >= len(ACTIONS) for index in result)
    ):
        raise ValueError("TMLR V6 registered action set is invalid")
    return result


@dataclass(frozen=True)
class Phase2Surface:
    """One seed-averaged, five-fold-stitched FIT OOF surface.

    ``prediction`` always retains the five physical slots.  Inactive slots may
    be NaN, but every active slot must be finite.  ``target`` may be NaN only
    for a plan-feasible execution failure; such a failure excludes the whole
    state from formal metrics over this surface's registered action set.
    """

    model_id: str
    prediction_kind: str
    prediction: np.ndarray
    target: np.ndarray
    feasible: np.ndarray
    groups: np.ndarray
    state_ids: tuple[str, ...]
    action_indices: tuple[int, ...] = tuple(range(len(ACTIONS)))
    trained_cost_lambda: float | None = None

    def validated(self) -> "Phase2Surface":
        prediction = np.asarray(self.prediction, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        feasible = np.asarray(self.feasible)
        groups = np.asarray(self.groups)
        indices = _physical_action_indices(self.action_indices)
        if self.prediction_kind not in PREDICTION_KINDS:
            raise ValueError("TMLR V6 phase-two prediction kind is unregistered")
        if prediction.ndim != 2 or prediction.shape[1] != len(ACTIONS):
            raise ValueError("TMLR V6 phase-two prediction must be N x 5")
        if target.shape != prediction.shape or feasible.shape != prediction.shape:
            raise ValueError("TMLR V6 phase-two surface geometry drift")
        if feasible.dtype.kind != "b":
            raise ValueError("TMLR V6 plan feasibility must be boolean")
        if groups.shape != (len(prediction),) or len(self.state_ids) != len(prediction):
            raise ValueError("TMLR V6 phase-two state identities are not aligned")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("TMLR V6 phase-two state IDs are not unique")
        if any(not str(value) for value in self.state_ids):
            raise ValueError("TMLR V6 phase-two state ID is empty")
        if not np.isfinite(prediction[:, indices]).all():
            raise ValueError("TMLR V6 active phase-two predictions are not finite")
        inactive = tuple(index for index in range(len(ACTIONS)) if index not in indices)
        if inactive and np.isfinite(prediction[:, inactive]).any():
            raise ValueError("TMLR V6 inactive action prediction must be NaN")
        if np.any(np.isfinite(target) & ~feasible):
            raise ValueError("TMLR V6 defined target is marked plan-infeasible")
        if np.any(feasible[:, inactive]) if inactive else False:
            raise ValueError("TMLR V6 inactive action is marked plan-feasible")
        if self.trained_cost_lambda is not None:
            value = float(self.trained_cost_lambda)
            if value not in LAMBDAS:
                raise ValueError("TMLR V6 direct model lambda is unregistered")
        if self.prediction_kind in {"logit", "value"}:
            if self.trained_cost_lambda is None:
                raise ValueError("TMLR V6 direct decision surface lacks its lambda")
        elif self.trained_cost_lambda is not None:
            raise ValueError("TMLR V6 non-direct surface carries a trained lambda")
        if self.prediction_kind == "gain" and not np.array_equal(
            prediction[:, 0], np.zeros(len(prediction), dtype=np.float64),
        ):
            raise ValueError("TMLR V6 scientific Q2 STOP gain is not exactly zero")
        return self


@dataclass(frozen=True)
class ProfiledCostSurface:
    """Fold-profiled costs that remain physically outside every predictor."""

    normalized: np.ndarray
    seconds: np.ndarray

    def validated(self, shape: tuple[int, int]) -> "ProfiledCostSurface":
        normalized = np.asarray(self.normalized, dtype=np.float64)
        seconds = np.asarray(self.seconds, dtype=np.float64)
        if normalized.shape != shape or seconds.shape != shape:
            raise ValueError("TMLR V6 phase-two cost surface is not state aligned")
        if (
            not np.isfinite(normalized).all()
            or not np.isfinite(seconds).all()
            or np.any(normalized < 0.0)
            or np.any(seconds < 0.0)
            or not np.array_equal(normalized[:, 0], np.zeros(shape[0]))
            or not np.array_equal(seconds[:, 0], np.zeros(shape[0]))
        ):
            raise ValueError("TMLR V6 phase-two profiled costs violate STOP-zero")
        return self


def strict_continuous_seed_mean(
    seed_predictions: Mapping[int, np.ndarray],
    *,
    action_indices: Sequence[int],
) -> np.ndarray:
    """Average exact 13/37/71 continuous surfaces before any selection.

    Inactive physical slots must be NaN in every seed and remain NaN.  This
    prevents per-seed action selection, majority voting, and accidental reuse
    of the masked F0/F1 A3 column.
    """

    indices = _physical_action_indices(action_indices)
    if set(seed_predictions) != set(SEEDS):
        raise RuntimeError("TMLR V6 seed ensemble is not exactly 13/37/71")
    arrays = [
        np.asarray(seed_predictions[seed], dtype=np.float64) for seed in SEEDS
    ]
    if (
        arrays[0].ndim != 2
        or arrays[0].shape[1] != len(ACTIONS)
        or any(array.shape != arrays[0].shape for array in arrays)
        or any(not np.isfinite(array[:, indices]).all() for array in arrays)
    ):
        raise RuntimeError("TMLR V6 seed surface coverage/shape is incomplete")
    inactive = tuple(index for index in range(len(ACTIONS)) if index not in indices)
    if inactive and any(np.isfinite(array[:, inactive]).any() for array in arrays):
        raise RuntimeError("TMLR V6 inactive seed prediction is not NaN")
    result = np.full_like(arrays[0], np.nan, dtype=np.float64)
    result[:, indices] = np.mean(
        np.stack([array[:, indices] for array in arrays], axis=0), axis=0,
    )
    return result


def q2_scientific_gain_after_seed_mean(
    seed_raw_gain_predictions: Mapping[int, np.ndarray],
    *,
    action_indices: Sequence[int] = tuple(range(len(ACTIONS))),
) -> np.ndarray:
    """Average raw Q2 heads, then apply the frozen scientific STOP override."""

    result = strict_continuous_seed_mean(
        seed_raw_gain_predictions, action_indices=action_indices,
    )
    result[:, 0] = 0.0
    return result


def complete_case_state_mask(
    surface: Phase2Surface,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply whole-state complete-case exclusion over the registered actions."""

    surface.validated()
    indices = _physical_action_indices(surface.action_indices)
    target = np.asarray(surface.target, dtype=np.float64)[:, indices]
    feasible = np.asarray(surface.feasible, dtype=bool)[:, indices]
    excluded = np.any(feasible & ~np.isfinite(target), axis=1)
    excluded_ids = sorted(
        state_id for state_id, drop in zip(surface.state_ids, excluded) if drop
    )
    complete = ~excluded
    return complete, {
        "formal_complete_case_states": int(complete.sum()),
        "plan_feasible_target_undefined_excluded_states": int(excluded.sum()),
        "excluded_state_ids_sha256": hashlib.sha256(
            canonical_json_bytes(excluded_ids)
        ).hexdigest(),
    }


def formal_evaluable_state_mask(surface: Phase2Surface) -> np.ndarray:
    """Require complete case, feasible STOP, and a feasible non-STOP action."""

    complete, _ = complete_case_state_mask(surface)
    indices = _physical_action_indices(surface.action_indices)
    nonstop = tuple(index for index in indices if index != 0)
    feasible = np.asarray(surface.feasible, dtype=bool)
    result = complete & feasible[:, 0] & feasible[:, nonstop].any(axis=1)
    return result


def _validate_lambda(surface: Phase2Surface, cost_lambda: float) -> float:
    value = float(cost_lambda)
    if value not in LAMBDAS:
        raise ValueError("TMLR V6 evaluation lambda is unregistered")
    if surface.prediction_kind in {"logit", "value"} and (
        surface.trained_cost_lambda != value
    ):
        raise ValueError("TMLR V6 direct surface/evaluation lambda mismatch")
    return value


def action_selection_arrays(
    surface: Phase2Surface,
    costs: ProfiledCostSurface,
    *,
    cost_lambda: float,
    population_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select actions only after continuous seed averaging.

    The first physical optimum is chosen by NumPy's stable first-index
    ``argmin``/``argmax`` behavior after infeasible actions are masked.
    """

    surface.validated()
    prediction = np.asarray(surface.prediction, dtype=np.float64)
    target = np.asarray(surface.target, dtype=np.float64)
    feasible = np.asarray(surface.feasible, dtype=bool)
    costs.validated(target.shape)
    value_lambda = _validate_lambda(surface, cost_lambda)
    active = np.zeros(len(ACTIONS), dtype=bool)
    active[list(surface.action_indices)] = True
    evaluable = formal_evaluable_state_mask(surface)
    if population_mask is not None:
        population = np.asarray(population_mask)
        if population.shape != (len(target),) or population.dtype.kind != "b":
            raise ValueError("TMLR V6 population mask is not state aligned")
        evaluable &= population
    if not evaluable.any():
        raise RuntimeError("TMLR V6 phase-two surface has no evaluable state")
    eligible = feasible[evaluable] & active[None, :]
    normalized_cost = np.asarray(costs.normalized, dtype=np.float64)[evaluable]
    true_value = target[evaluable] + value_lambda * normalized_cost
    masked_true = np.where(eligible, true_value, np.inf)
    oracle = masked_true.argmin(axis=1)
    rows = np.arange(len(oracle))

    if surface.prediction_kind == "residual":
        predicted_value = prediction[evaluable] + value_lambda * normalized_cost
        masked_prediction = np.where(eligible, predicted_value, np.inf)
        selected = masked_prediction.argmin(axis=1)
        nonstop_masked = masked_prediction.copy()
        nonstop_masked[:, 0] = np.inf
        best_nonstop = nonstop_masked.argmin(axis=1)
        dispatch_score = (
            masked_prediction[:, 0] - nonstop_masked[rows, best_nonstop]
        )
    elif surface.prediction_kind == "gain":
        net_gain = prediction[evaluable] - value_lambda * normalized_cost
        masked_prediction = np.where(eligible, net_gain, -np.inf)
        selected = masked_prediction.argmax(axis=1)
        nonstop_masked = masked_prediction.copy()
        nonstop_masked[:, 0] = -np.inf
        best_nonstop = nonstop_masked.argmax(axis=1)
        dispatch_score = nonstop_masked[rows, best_nonstop]
    elif surface.prediction_kind == "logit":
        masked_prediction = np.where(eligible, prediction[evaluable], -np.inf)
        selected = masked_prediction.argmax(axis=1)
        nonstop_masked = masked_prediction.copy()
        nonstop_masked[:, 0] = -np.inf
        best_nonstop = nonstop_masked.argmax(axis=1)
        dispatch_score = (
            nonstop_masked[rows, best_nonstop] - masked_prediction[:, 0]
        )
    else:
        masked_prediction = np.where(eligible, prediction[evaluable], np.inf)
        selected = masked_prediction.argmin(axis=1)
        nonstop_masked = masked_prediction.copy()
        nonstop_masked[:, 0] = np.inf
        best_nonstop = nonstop_masked.argmin(axis=1)
        dispatch_score = (
            masked_prediction[:, 0] - nonstop_masked[rows, best_nonstop]
        )
    if np.any(selected == np.iinfo(np.int64).max):  # defensive, unreachable
        raise RuntimeError("TMLR V6 phase-two action selection failed")
    return {
        "evaluable_mask": evaluable,
        "eligible": eligible,
        "true_value": masked_true,
        "prediction_score": masked_prediction,
        "selected": selected.astype(np.int64),
        "oracle": oracle.astype(np.int64),
        "best_nonstop": best_nonstop.astype(np.int64),
        "dispatch_score": np.asarray(dispatch_score, dtype=np.float64),
        "realized_gain_selected": (
            masked_true[:, 0] - masked_true[rows, selected]
        ),
        "realized_gain_best_nonstop": (
            masked_true[:, 0] - masked_true[rows, best_nonstop]
        ),
        "oracle_gain": np.maximum(
            0.0, masked_true[:, 0] - masked_true[rows, oracle],
        ),
        "regret": masked_true[rows, selected] - masked_true[rows, oracle],
    }


def decision_metrics(
    surface: Phase2Surface,
    costs: ProfiledCostSurface,
    *,
    cost_lambda: float,
    population_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the frozen action-set-aware decision metrics."""

    arrays = action_selection_arrays(
        surface, costs, cost_lambda=cost_lambda, population_mask=population_mask,
    )
    selected = arrays["selected"]
    oracle = arrays["oracle"]
    realized = arrays["realized_gain_selected"]
    oracle_gain = arrays["oracle_gain"]
    selected_nonstop = selected != 0
    oracle_beneficial = oracle_gain > 0.0
    total_oracle = float(oracle_gain.sum())
    total_realized = float(realized.sum())
    headroom = total_oracle > 0.0
    count = len(selected)
    false_stop = int(np.sum((selected == 0) & oracle_beneficial))
    false_intervention = int(np.sum(selected_nonstop & ~oracle_beneficial))
    negative = int(np.sum(selected_nonstop & (realized < 0.0)))
    detected = int(np.sum(oracle_beneficial & (realized > 0.0)))
    _, complete_case = complete_case_state_mask(surface)
    return {
        "model": surface.model_id,
        "split": "FIT_OOF",
        "prediction_kind": surface.prediction_kind,
        "lambda": float(cost_lambda),
        "action_indices": list(surface.action_indices),
        **complete_case,
        "evaluable_states": count,
        "oracle_beneficial_states": int(oracle_beneficial.sum()),
        "selected_nonstop_states": int(selected_nonstop.sum()),
        "false_stop_states": false_stop,
        "false_intervention_states": false_intervention,
        "negative_intervention_states": negative,
        "beneficial_detected_states": detected,
        "realized_gain": total_realized,
        "oracle_gain": total_oracle,
        "oracle_headroom_positive": headroom,
        "oracle_headroom_status": (
            POSITIVE_HEADROOM_STATUS if headroom else ZERO_HEADROOM_STATUS
        ),
        "normalized_regret_defined": headroom,
        "normalized_regret": (
            float(arrays["regret"].sum()) / total_oracle if headroom else None
        ),
        "predicted_beneficial_rate": float(selected_nonstop.sum() / count),
        "false_stop_rate": float(false_stop / count),
        "false_intervention_rate": float(false_intervention / count),
        "negative_intervention_rate": (
            float(negative / selected_nonstop.sum())
            if selected_nonstop.any() else None
        ),
        "beneficial_detection_rate": (
            float(detected / oracle_beneficial.sum())
            if oracle_beneficial.any() else None
        ),
        "gain_capture_defined": headroom,
        "gain_capture": total_realized / total_oracle if headroom else None,
    }


def residual_prediction_metrics(surface: Phase2Surface) -> dict[str, Any]:
    """Report residual applicability exactly for residual and direct families."""

    surface.validated()
    complete, complete_case = complete_case_state_mask(surface)
    if surface.prediction_kind in {"logit", "value"}:
        return {
            "model": surface.model_id,
            "split": "FIT_OOF",
            "prediction_kind": surface.prediction_kind,
            "prediction_metric_applicability": "NOT_APPLICABLE_DIRECT_DECISION_FAMILY",
            "residual_prediction_metrics_defined": False,
            "absolute_residual_mae_defined": False,
            "absolute_residual_mae": None,
            "drre": None,
            "beneficial_sign_accuracy": None,
            "stop_vs_best_accuracy": None,
            "informative_action_order_accuracy": None,
            "defined_state_actions": None,
            "stop_relative_pairs": None,
            "evaluable_states": None,
            "informative_action_pairs": None,
            **complete_case,
        }
    prediction = np.asarray(surface.prediction, dtype=np.float64)
    if surface.prediction_kind == "gain":
        residual = np.full_like(prediction, np.nan)
        residual[:, surface.action_indices] = -prediction[:, surface.action_indices]
        residual[:, 0] = 0.0
        absolute_defined = False
    else:
        residual = prediction
        absolute_defined = True
    target = np.asarray(surface.target, dtype=np.float64)
    feasible = np.asarray(surface.feasible, dtype=bool)
    active = np.zeros(len(ACTIONS), dtype=bool)
    active[list(surface.action_indices)] = True
    defined = complete[:, None] & feasible & active[None, :]
    nonstop = tuple(index for index in surface.action_indices if index != 0)
    evaluable = defined[:, 0] & defined[:, nonstop].any(axis=1)
    relative_defined = evaluable[:, None] & defined[:, :1] & defined[:, 1:]
    predicted_delta = residual[:, 1:] - residual[:, :1]
    true_delta = target[:, 1:] - target[:, :1]
    relative_error = np.abs(
        predicted_delta - true_delta
    )
    relative_count = int(relative_defined.sum())
    if relative_count <= 0:
        raise RuntimeError("TMLR V6 DRRE has no defined STOP-relative pair")
    masked_prediction = np.where(defined, residual, np.inf)
    masked_target = np.where(defined, target, np.inf)
    predicted_best_nonstop = np.min(masked_prediction[:, nonstop], axis=1)
    true_best_nonstop = np.min(masked_target[:, nonstop], axis=1)
    stop_best_correct = evaluable & (
        ((predicted_best_nonstop < masked_prediction[:, 0])
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
        np.sign(residual[:, left] - residual[:, right])
        == np.sign(target[:, left] - target[:, right])
    )
    informative_count = int(informative.sum())
    if informative_count <= 0:
        raise RuntimeError("TMLR V6 action-order metric has no informative pair")
    return {
        "model": surface.model_id,
        "split": "FIT_OOF",
        "prediction_kind": surface.prediction_kind,
        "prediction_metric_applicability": (
            "STOP_RELATIVE_ONLY_Q2_FIXED_ZERO"
            if surface.prediction_kind == "gain"
            else "ALL_REGISTERED_RESIDUAL_METRICS"
        ),
        "residual_prediction_metrics_defined": True,
        "absolute_residual_mae_defined": absolute_defined,
        "absolute_residual_mae": (
            float(np.abs(residual - target)[defined].mean())
            if absolute_defined else None
        ),
        "drre": float(relative_error[relative_defined].mean()),
        "beneficial_sign_accuracy": float(np.mean(
            ((predicted_delta < 0.0) == (true_delta < 0.0))[relative_defined]
        )),
        "stop_vs_best_accuracy": float(stop_best_correct.sum() / evaluable.sum()),
        "informative_action_order_accuracy": float(
            order_correct.sum() / informative_count
        ),
        "defined_state_actions": int(defined.sum()),
        "stop_relative_pairs": relative_count,
        "evaluable_states": int(evaluable.sum()),
        "informative_action_pairs": informative_count,
        **complete_case,
    }


def validate_prediction_metric_row(
    row: Mapping[str, Any], *, extra_fields: Sequence[str] = (),
) -> None:
    """Require the exact frozen five-metric row, including explicit N/A."""

    expected = PREDICTION_METRIC_ROW_FIELDS | frozenset(extra_fields)
    if set(row) != expected:
        raise RuntimeError("TMLR V6 registered prediction-metric field drift")
    if (
        not isinstance(row["model"], str)
        or row["split"] != "FIT_OOF"
        or row["prediction_kind"] not in PREDICTION_KINDS
        or not isinstance(row["excluded_state_ids_sha256"], str)
        or len(row["excluded_state_ids_sha256"]) != 64
    ):
        raise RuntimeError("TMLR V6 prediction-metric identity drift")
    defined = row["residual_prediction_metrics_defined"]
    absolute_defined = row["absolute_residual_mae_defined"]
    if not isinstance(defined, bool) or not isinstance(absolute_defined, bool):
        raise RuntimeError("TMLR V6 prediction-metric applicability drift")
    expected_applicability = {
        "residual": "ALL_REGISTERED_RESIDUAL_METRICS",
        "gain": "STOP_RELATIVE_ONLY_Q2_FIXED_ZERO",
        "logit": "NOT_APPLICABLE_DIRECT_DECISION_FAMILY",
        "value": "NOT_APPLICABLE_DIRECT_DECISION_FAMILY",
    }[str(row["prediction_kind"])]
    if row["prediction_metric_applicability"] != expected_applicability:
        raise RuntimeError("TMLR V6 prediction-metric applicability drift")
    if not defined:
        nullable = (*REGISTERED_PREDICTION_METRICS,
                    "defined_state_actions", "stop_relative_pairs",
                    "evaluable_states", "informative_action_pairs")
        if absolute_defined or any(row[name] is not None for name in nullable):
            raise RuntimeError("TMLR V6 direct-family N/A metric drift")
        return
    if row["prediction_kind"] not in {"residual", "gain"}:
        raise RuntimeError("TMLR V6 residual metric kind drift")
    if row["prediction_kind"] == "gain":
        if absolute_defined or row["absolute_residual_mae"] is not None:
            raise RuntimeError("TMLR V6 Q2 absolute residual MAE must be N/A")
    elif not absolute_defined or row["absolute_residual_mae"] is None:
        raise RuntimeError("TMLR V6 residual absolute MAE is undefined")
    nonnegative = (
        ("absolute_residual_mae", "drre")
        if absolute_defined else ("drre",)
    )
    accuracies = (
        "beneficial_sign_accuracy", "stop_vs_best_accuracy",
        "informative_action_order_accuracy",
    )
    if any(
        row[name] is None or not np.isfinite(float(row[name]))
        or float(row[name]) < 0.0
        for name in nonnegative
    ) or any(
        row[name] is None or not np.isfinite(float(row[name]))
        or not 0.0 <= float(row[name]) <= 1.0
        for name in accuracies
    ):
        raise RuntimeError("TMLR V6 prediction-metric value drift")
    if any(
        not isinstance(row[name], int) or isinstance(row[name], bool)
        or row[name] <= 0
        for name in (
            "defined_state_actions", "stop_relative_pairs",
            "evaluable_states", "informative_action_pairs",
        )
    ):
        raise RuntimeError("TMLR V6 prediction-metric denominator drift")


def residual_ratio_sufficient(
    surface: Phase2Surface,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return per-state numerator/denominator arrays for paired MAE/DRRE."""

    surface.validated()
    if surface.prediction_kind != "residual":
        raise ValueError("TMLR V6 primary sufficient statistics require residuals")
    prediction = np.asarray(surface.prediction, dtype=np.float64)
    target = np.asarray(surface.target, dtype=np.float64)
    feasible = np.asarray(surface.feasible, dtype=bool)
    complete, _ = complete_case_state_mask(surface)
    active = np.zeros(len(ACTIONS), dtype=bool)
    active[list(surface.action_indices)] = True
    defined = complete[:, None] & feasible & active[None, :]
    relative_defined = defined[:, :1] & defined[:, 1:]
    absolute = np.where(defined, np.abs(prediction - target), 0.0)
    relative = np.where(
        relative_defined,
        np.abs(
            (prediction[:, 1:] - prediction[:, :1])
            - (target[:, 1:] - target[:, :1])
        ),
        0.0,
    )
    return {
        "absolute_residual_mae": (
            absolute.sum(axis=1), defined.sum(axis=1).astype(np.float64),
        ),
        "drre": (
            relative.sum(axis=1),
            relative_defined.sum(axis=1).astype(np.float64),
        ),
    }


def _ranked_state_indices(
    score: np.ndarray, state_ids: Sequence[str], mask: np.ndarray,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    return np.asarray(
        sorted(indices.tolist(), key=lambda index: (-float(score[index]), str(state_ids[index]))),
        dtype=np.int64,
    )


def sparse_budget_metrics(
    surface: Phase2Surface,
    costs: ProfiledCostSurface,
    *,
    cost_lambda: float,
    budgets: Sequence[float] = BUDGETS,
    population_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Re-rank within the current population for both frozen budget semantics."""

    arrays = action_selection_arrays(
        surface, costs, cost_lambda=cost_lambda, population_mask=population_mask,
    )
    evaluable = arrays["evaluable_mask"]
    source_indices = np.flatnonzero(evaluable)
    state_ids = np.asarray(surface.state_ids, dtype=str)[source_indices]
    score = np.asarray(arrays["dispatch_score"], dtype=np.float64)
    order = _ranked_state_indices(
        score, state_ids, np.ones(len(score), dtype=bool),
    )
    gains = np.asarray(arrays["realized_gain_best_nonstop"], dtype=np.float64)
    oracle_gains = np.asarray(arrays["oracle_gain"], dtype=np.float64)
    best_nonstop = np.asarray(arrays["best_nonstop"], dtype=np.int64)
    normalized = np.asarray(costs.normalized, dtype=np.float64)[evaluable]
    seconds = np.asarray(costs.seconds, dtype=np.float64)[evaluable]
    _, complete_case = complete_case_state_mask(surface)
    result: list[dict[str, Any]] = []
    for budget_value in budgets:
        budget = float(budget_value)
        if budget not in BUDGETS:
            raise ValueError("TMLR V6 sparse budget is unregistered")
        cap = max(1, int(math.ceil(budget * len(order))))
        oracle_denominator = float(np.sort(oracle_gains)[::-1][:cap].sum())
        headroom = oracle_denominator > 0.0
        for semantics in BUDGET_SEMANTICS:
            chosen = order[:cap]
            if semantics == "POSITIVE_GAIN_CAP":
                chosen = chosen[score[chosen] > 0.0]
            chosen_gains = gains[chosen]
            chosen_actions = best_nonstop[chosen]
            selected_ids = state_ids[chosen]
            result.append({
                "model": surface.model_id,
                "split": "FIT_OOF",
                "lambda": float(cost_lambda),
                "budget": budget,
                "budget_semantics": semantics,
                "action_indices": list(surface.action_indices),
                **complete_case,
                "eligible_states": len(order),
                "cap": cap,
                "used": int(len(chosen)),
                "unused": int(cap - len(chosen)),
                "selected_true_positive_states": int(np.sum(chosen_gains > 0.0)),
                "selected_negative_states": int(np.sum(chosen_gains < 0.0)),
                "selected_zero_gain_states": int(np.sum(chosen_gains == 0.0)),
                "realized_gain": float(chosen_gains.sum()),
                "oracle_gain": oracle_denominator,
                "oracle_headroom_positive": headroom,
                "oracle_headroom_status": (
                    POSITIVE_HEADROOM_STATUS if headroom else ZERO_HEADROOM_STATUS
                ),
                "gain_capture_defined": headroom,
                "gain_capture": (
                    float(chosen_gains.sum()) / oracle_denominator
                    if headroom else None
                ),
                "negative_selected_rate": (
                    float(np.mean(chosen_gains < 0.0)) if len(chosen) else 0.0
                ),
                "profiled_cost_common_units": float(
                    normalized[chosen, chosen_actions].sum()
                ),
                "profiled_cost_seconds": float(
                    seconds[chosen, chosen_actions].sum()
                ),
                "selection_sha256": hashlib.sha256(canonical_json_bytes([
                    [str(state_id), ACTIONS[int(action)]]
                    for state_id, action in zip(selected_ids, chosen_actions)
                ])).hexdigest(),
            })
    return result


def sorted_group_universe(groups: Sequence[str]) -> np.ndarray:
    result = np.asarray(sorted(set(str(group) for group in groups)), dtype=str)
    if result.size == 0:
        raise ValueError("TMLR V6 phase-two group universe is empty")
    return result


def paired_bootstrap_group_counts(
    groups: Sequence[str], *, expected_groups: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen PCG64(13) 2,000 x G paired count matrix."""

    universe = sorted_group_universe(groups)
    if expected_groups is not None and len(universe) != int(expected_groups):
        raise RuntimeError("TMLR V6 bootstrap image-group cardinality drift")
    generator = np.random.Generator(np.random.PCG64(13))
    draws = generator.integers(0, len(universe), size=(2000, len(universe)))
    counts = np.stack([
        np.bincount(row, minlength=len(universe)) for row in draws
    ]).astype(np.int32)
    if not np.all(counts.sum(axis=1) == len(universe)):
        raise RuntimeError("TMLR V6 phase-two bootstrap count matrix drift")
    return universe, counts


def _group_index(groups: Sequence[str], universe: Sequence[str]) -> np.ndarray:
    lookup = {str(group): index for index, group in enumerate(universe)}
    try:
        return np.asarray([lookup[str(group)] for group in groups], dtype=np.int32)
    except KeyError as error:
        raise ValueError("TMLR V6 surface contains an unregistered group") from error


def _group_sums(
    values: np.ndarray, state_group: np.ndarray, group_count: int,
) -> np.ndarray:
    return np.bincount(
        state_group,
        weights=np.asarray(values, dtype=np.float64),
        minlength=group_count,
    ).astype(np.float64)


def paired_ratio_bootstrap(
    *,
    groups: Sequence[str],
    left_numerator: np.ndarray,
    left_denominator: np.ndarray,
    right_numerator: np.ndarray,
    right_denominator: np.ndarray,
    contrast_id: str,
    metric: str,
    expected_groups: int | None = None,
    bootstrap_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired lower-is-better ratio contrast at the image-group level."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            left_numerator, left_denominator,
            right_numerator, right_denominator,
        )
    ]
    if any(array.shape != (len(groups),) for array in arrays):
        raise ValueError("TMLR V6 paired sufficient statistics are misaligned")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("TMLR V6 paired sufficient statistic is non-finite")
    if np.any(arrays[1] < 0.0) or np.any(arrays[3] < 0.0):
        raise ValueError("TMLR V6 paired denominator is negative")
    universe = sorted_group_universe(groups)
    if expected_groups is not None and len(universe) != int(expected_groups):
        raise RuntimeError("TMLR V6 bootstrap image-group cardinality drift")
    if bootstrap_counts is None:
        _, counts = paired_bootstrap_group_counts(
            groups, expected_groups=expected_groups,
        )
    else:
        counts = np.asarray(bootstrap_counts)
        if (
            counts.dtype != np.dtype(np.int32)
            or counts.shape != (2000, len(universe))
            or np.any(counts < 0)
            or not np.all(counts.sum(axis=1) == len(universe))
        ):
            raise RuntimeError("TMLR V6 supplied bootstrap count matrix drift")
    state_group = _group_index(groups, universe)
    grouped = [
        _group_sums(value, state_group, len(universe)) for value in arrays
    ]
    left_num, left_den, right_num, right_den = grouped
    if left_den.sum() <= 0.0 or right_den.sum() <= 0.0:
        raise RuntimeError("TMLR V6 paired point metric is undefined")
    sampled_left_den = counts @ left_den
    sampled_right_den = counts @ right_den
    if np.any(sampled_left_den <= 0.0) or np.any(sampled_right_den <= 0.0):
        raise RuntimeError("TMLR V6 paired bootstrap replicate is undefined")
    left_point = float(left_num.sum() / left_den.sum())
    right_point = float(right_num.sum() / right_den.sum())
    sampled_left = (counts @ left_num) / sampled_left_den
    sampled_right = (counts @ right_num) / sampled_right_den
    effect = sampled_left - sampled_right
    pair_defined = (left_den > 0.0) & (right_den > 0.0)
    if pair_defined.any():
        left_group = left_num[pair_defined] / left_den[pair_defined]
        right_group = right_num[pair_defined] / right_den[pair_defined]
        exact_equal = left_group == right_group
        win_fraction: float | None = float(np.mean(
            (left_group < right_group) & ~exact_equal
        ))
        loss_fraction: float | None = float(np.mean(
            (left_group > right_group) & ~exact_equal
        ))
        tie_fraction: float | None = float(np.mean(exact_equal))
    else:
        win_fraction = loss_fraction = tie_fraction = None
    row = {
        "row_type": "paired_contrast",
        "split": "FIT_OOF",
        "contrast_id": contrast_id,
        "metric": metric,
        "direction": "lower_is_better",
        "left_point": left_point,
        "right_point": right_point,
        "effect_left_minus_right": left_point - right_point,
        "bootstrap_ci_lower": float(np.quantile(effect, 0.025, method="linear")),
        "bootstrap_ci_upper": float(np.quantile(effect, 0.975, method="linear")),
        "group_win_fraction": win_fraction,
        "group_loss_fraction": loss_fraction,
        "group_tie_fraction": tie_fraction,
        "paired_defined_image_groups": int(pair_defined.sum()),
        "image_groups": len(universe),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 13,
        "bootstrap_generator": "numpy.random.Generator(numpy.random.PCG64(13))",
        "bootstrap_group_count_matrix_sha256": hashlib.sha256(
            counts.astype("<i4", copy=False).tobytes(order="C")
        ).hexdigest(),
        "percentile_quantile_method": "numpy_linear",
    }
    row["mechanical_status"] = lower_better_status(row)
    return row


def lower_better_status(row: Mapping[str, Any]) -> str:
    """Apply the frozen five-percent/paired-CI mechanical comparison rule."""

    left = float(row["left_point"])
    right = float(row["right_point"])
    lower = float(row["bootstrap_ci_lower"])
    upper = float(row["bootstrap_ci_upper"])
    win = row.get("group_win_fraction")
    loss = row.get("group_loss_fraction")
    if min(left, right) < 0.0 or not all(np.isfinite([left, right, lower, upper])):
        raise ValueError("TMLR V6 lower-is-better metric must be finite/nonnegative")
    clear_better = (
        right > 0.0
        and upper < 0.0
        and left <= 0.95 * right
        and win is not None
        and float(win) > 0.5
    )
    magnitude_worse = left > 0.0 if right == 0.0 else left >= 1.05 * right
    clear_worse = (
        lower > 0.0
        and magnitude_worse
        and loss is not None
        and float(loss) > 0.5
    )
    practical_tie = (
        left == 0.0 and lower == 0.0 and upper == 0.0
        if right == 0.0
        else lower >= -0.05 * right and upper <= 0.05 * right
    )
    if clear_better:
        return "CLEAR_BETTER"
    if clear_worse:
        return "CLEAR_WORSE"
    if practical_tie:
        return "PRACTICAL_TIE"
    return "UNCERTAIN"


def _validate_report_common(
    report: Mapping[str, Any], *, schema: str, status: str,
    expected_jobs: int, expected_checks: frozenset[str],
) -> None:
    required = {
        "schema_version": schema,
        "status": status,
        "completed_valid_training_jobs": expected_jobs,
        "metric_pipeline_pass": True,
    }
    drift = {
        key: (report.get(key), value)
        for key, value in required.items()
        if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"TMLR V6 phase-two report contract drift: {drift}")
    assert_zero_external_access(report.get("heldout_access", {}))
    bindings = report.get("input_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != set(PHASE2_REPORT_INPUT_BINDINGS)
        or any(
            not isinstance(bindings[key], str)
            or len(bindings[key]) != 64
            or any(character not in "0123456789abcdef" for character in bindings[key])
            for key in PHASE2_REPORT_INPUT_BINDINGS
        )
    ):
        raise RuntimeError("TMLR V6 phase-two report input binding drift")
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(expected_checks)
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError("TMLR V6 phase-two report checks are incomplete")


def _validate_output_registry(
    report: Mapping[str, Any], expected_rows: Mapping[str, int],
) -> None:
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected_rows):
        raise RuntimeError("TMLR V6 phase-two report output registry drift")
    for name, rows in expected_rows.items():
        identity = outputs[name]
        path = Path(str(identity.get("path", ""))) if isinstance(identity, Mapping) else Path()
        digest = identity.get("sha256") if isinstance(identity, Mapping) else None
        if (
            not isinstance(identity, Mapping)
            or path.is_absolute()
            or ".." in path.parts
            or path.name != name
            or not isinstance(identity.get("bytes"), int)
            or isinstance(identity.get("bytes"), bool)
            or int(identity["bytes"]) <= 0
            or digest is None
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or identity.get("rows") != rows
        ):
            raise RuntimeError("TMLR V6 phase-two output identity/row drift")


def validate_second_batch_report_contract(report: Mapping[str, Any]) -> None:
    _validate_report_common(
        report,
        schema=SECOND_BATCH_REPORT_SCHEMA,
        status=SECOND_BATCH_REPORT_STATUS,
        expected_jobs=225,
        expected_checks=SECOND_BATCH_REPORT_CHECKS,
    )
    counts = report.get("run_counts")
    if counts != {
        "R2_P": 15,
        "R3_P": 15,
        "Q2_P": 15,
        "LL4TTA_P": 0,
        "L2D_P": 90,
        "SPO_PLUS_P": 90,
    }:
        raise RuntimeError("TMLR V6 exact 225-job report matrix drift")
    _validate_output_registry(report, {
        "phase2_second_batch_prediction_metrics.csv": 16,
        "phase2_second_batch_decision_metrics.csv": 36,
        "phase2_second_batch_sparse_budget_metrics.csv": 432,
        "phase2_second_batch_run_inventory.csv": 225,
        "phase2_second_batch_metrics.json": 484,
    })


def validate_f0_f1_report_contract(report: Mapping[str, Any]) -> None:
    _validate_report_common(
        report,
        schema=F0_F1_REPORT_SCHEMA,
        status=F0_F1_REPORT_STATUS,
        expected_jobs=120,
        expected_checks=F0_F1_REPORT_CHECKS,
    )
    counts = report.get("run_counts")
    if counts != {"F0_P": 60, "F1_P": 60}:
        raise RuntimeError("TMLR V6 exact 120-job report matrix drift")
    if report.get("remaining_physical_indices") != list(REMAINING_ACTION_INDICES):
        raise RuntimeError("TMLR V6 F0/F1 remaining action set drift")
    if (
        report.get("a3_sunk_cost_added_to_remaining_argmin") is not False
        or report.get("same_profile_for_f0_and_f1") is not True
    ):
        raise RuntimeError("TMLR V6 F0/F1 conditional estimand drift")
    _validate_output_registry(report, {
        "phase2_f0_f1_prediction_metrics.csv": 8,
        "phase2_f0_f1_decision_metrics.csv": 48,
        "phase2_f0_f1_sparse_budget_metrics.csv": 576,
        "phase2_f0_f1_group_bootstrap.csv": 8,
        "phase2_f0_f1_run_inventory.csv": 120,
        "phase2_f0_f1_metrics.json": 640,
    })


def meaningful_utility(
    decision_row: Mapping[str, Any], sparse_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Evaluate CLEAR_UTILITY_TRANSLATION without changing frozen thresholds."""

    regret = decision_row.get("normalized_regret")
    if (
        decision_row.get("normalized_regret_defined") is not True
        or regret is None
        or not float(regret) < 0.95
    ):
        return False
    expected = {
        (budget, semantics)
        for budget in PRIMARY_SPARSE_BUDGETS
        for semantics in BUDGET_SEMANTICS
    }
    observed: dict[tuple[float, str], Mapping[str, Any]] = {}
    for row in sparse_rows:
        key = (float(row["budget"]), str(row["budget_semantics"]))
        if key in expected:
            if key in observed:
                raise RuntimeError("TMLR V6 duplicate primary sparse utility cell")
            observed[key] = row
    if set(observed) != expected:
        raise RuntimeError("TMLR V6 primary sparse utility cells are incomplete")
    return all(
        row.get("oracle_headroom_positive") is True
        and row.get("gain_capture_defined") is True
        and row.get("gain_capture") is not None
        and float(row["gain_capture"]) > 0.05
        and float(row["realized_gain"]) > 0.0
        for row in observed.values()
    )


def direct_decision_family_interpretation(
    *, decision_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen PERSISTS/WEAKENS/NOT_SUPPORTED family rule."""

    models = ("R1_P", "R2_P", "R3_P", "Q2_P", "L2D_P", "SPO_PLUS_P")
    clear: dict[str, bool] = {}
    some_meaningful: dict[str, bool] = {}
    for model in models:
        candidates = [
            row for row in decision_rows
            if row.get("model") == model and float(row.get("lambda", -1.0)) == 0.0
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"TMLR V6 lambda-zero decision row is not unique: {model}"
            )
        model_sparse = [
            row for row in sparse_rows
            if row.get("model") == model and float(row.get("lambda", -1.0)) == 0.0
        ]
        clear[model] = meaningful_utility(candidates[0], model_sparse)
        meaningful_regret = (
            candidates[0].get("normalized_regret_defined") is True
            and candidates[0].get("normalized_regret") is not None
            and float(candidates[0]["normalized_regret"]) < 0.95
        )
        meaningful_cell = any(
            float(row.get("budget", -1.0)) in PRIMARY_SPARSE_BUDGETS
            and row.get("budget_semantics") in BUDGET_SEMANTICS
            and row.get("oracle_headroom_positive") is True
            and row.get("gain_capture_defined") is True
            and row.get("gain_capture") is not None
            and float(row["gain_capture"]) > 0.05
            and float(row.get("realized_gain", 0.0)) > 0.0
            for row in model_sparse
        )
        some_meaningful[model] = meaningful_regret or meaningful_cell
    non_alias = ("R2_P", "R3_P", "Q2_P", "L2D_P", "SPO_PLUS_P")
    independent_clear = sum(clear[model] for model in non_alias)
    if clear["R1_P"] or independent_clear >= 2:
        status = "NOT_SUPPORTED"
    elif any(some_meaningful[model] for model in non_alias):
        status = "WEAKENS"
    else:
        status = "PERSISTS"
    return {
        "status": status,
        "clear_utility_translation": clear,
        "some_meaningful_utility_evidence": some_meaningful,
        "independent_non_alias_clear_family_count": independent_clear,
        "ll4tta_alias_excluded_from_independent_family_count": True,
        "lambda": 0.0,
    }


def _comparison_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    inverse = {
        "CLEAR_BETTER": "CLEAR_WORSE",
        "CLEAR_WORSE": "CLEAR_BETTER",
        "PRACTICAL_TIE": "PRACTICAL_TIE",
        "UNCERTAIN": "UNCERTAIN",
    }
    for row in rows:
        left = str(row.get("left_model", ""))
        right = str(row.get("right_model", ""))
        metric = str(row.get("metric", ""))
        status = str(row.get("mechanical_status", ""))
        if not left or not right or metric not in {
            "absolute_residual_mae", "drre",
        } or status not in inverse:
            raise RuntimeError("TMLR V6 mechanical comparison row drift")
        key = (left, right, metric)
        reverse = (right, left, metric)
        if key in result or reverse in result:
            raise RuntimeError("TMLR V6 duplicate mechanical comparison")
        result[key] = status
        result[reverse] = inverse[status]
    return result


def derive_mechanical_predicates(
    *,
    comparison_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
    post_hoc_reversal: bool,
) -> dict[str, bool]:
    """Derive every terminal-classification predicate from machine rows."""

    lookup = _comparison_lookup(comparison_rows)
    metrics = ("absolute_residual_mae", "drre")

    def statuses(left: str, right: str) -> tuple[str, str]:
        try:
            return tuple(lookup[(left, right, metric)] for metric in metrics)  # type: ignore[return-value]
        except KeyError as error:
            raise RuntimeError(
                f"TMLR V6 classification contrast missing: {left}-{right}"
            ) from error

    r1_small = statuses("R1_P", "R0_SMALL_P")
    r1_capacity = statuses("R1_P", "R0_CM_P")
    r0_capacity_small = statuses("R0_CM_P", "R0_SMALL_P")
    localized_capacity = {
        model: statuses(model, "R0_CM_P")
        for model in ("R1_P", "RECT_P", "UNION_P")
    }
    r1_generic = {
        model: statuses("R1_P", model) for model in ("RECT_P", "UNION_P")
    }
    fidelity_strong = all(value == "CLEAR_BETTER" for value in r1_small)
    fidelity_partial = (
        sum(value == "CLEAR_BETTER" for value in r1_small) == 1
        and all(value != "CLEAR_WORSE" for value in r1_small)
    )
    beyond_capacity = all(value == "CLEAR_BETTER" for value in r1_capacity)
    generic_localization = (
        all(
            all(value == "CLEAR_BETTER" for value in values)
            for values in localized_capacity.values()
        )
        and all(
            all(value == "PRACTICAL_TIE" for value in values)
            for values in r1_generic.values()
        )
    )
    capacity_dominant = (
        all(value == "CLEAR_BETTER" for value in r0_capacity_small)
        and all(value == "PRACTICAL_TIE" for value in r1_capacity)
    )
    partial_localization = (
        any(
            all(value == "CLEAR_BETTER" for value in values)
            for values in localized_capacity.values()
        )
        or (
            sum(value == "CLEAR_BETTER" for value in r1_capacity) == 1
            and all(value != "CLEAR_WORSE" for value in r1_capacity)
        )
    )

    models = (
        "R1_P", "R2_P", "R3_P", "Q2_P", "L2D_P", "SPO_PLUS_P",
    )
    translation: dict[str, bool] = {}
    decision_by_model: dict[str, Mapping[str, Any]] = {}
    sparse_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for model in models:
        selected_decision = [
            row for row in decision_rows
            if row.get("model") == model and float(row.get("lambda", -1.0)) == 0.0
        ]
        if len(selected_decision) != 1:
            raise RuntimeError(f"TMLR V6 lambda-zero decision row drift: {model}")
        selected_sparse = [
            row for row in sparse_rows
            if row.get("model") == model and float(row.get("lambda", -1.0)) == 0.0
        ]
        decision_by_model[model] = selected_decision[0]
        sparse_by_model[model] = selected_sparse
        translation[model] = meaningful_utility(
            selected_decision[0], selected_sparse,
        )

    r1_sparse_primary = {
        (float(row["budget"]), str(row["budget_semantics"])): row
        for row in sparse_by_model["R1_P"]
        if float(row["budget"]) in PRIMARY_SPARSE_BUDGETS
        and str(row["budget_semantics"]) in BUDGET_SEMANTICS
    }
    expected_primary = {
        (budget, semantics)
        for budget in PRIMARY_SPARSE_BUDGETS
        for semantics in BUDGET_SEMANTICS
    }
    if set(r1_sparse_primary) != expected_primary:
        raise RuntimeError("TMLR V6 R1 primary sparse cells drift")
    fidelity_signal = fidelity_strong or fidelity_partial
    all_headroom = (
        decision_by_model["R1_P"].get("oracle_headroom_positive") is True
        and all(
            row.get("oracle_headroom_positive") is True
            for row in r1_sparse_primary.values()
        )
    )
    regret = decision_by_model["R1_P"].get("normalized_regret")
    regret_meaningful = (
        decision_by_model["R1_P"].get("normalized_regret_defined") is True
        and regret is not None
        and float(regret) < 0.95
    )
    cell_meaningful = [
        row.get("gain_capture_defined") is True
        and row.get("gain_capture") is not None
        and float(row["gain_capture"]) > 0.05
        and float(row["realized_gain"]) > 0.0
        for row in r1_sparse_primary.values()
    ]
    gap_persists = (
        fidelity_signal
        and all_headroom
        and not regret_meaningful
        and not any(cell_meaningful)
    )
    positive_headroom_exists = (
        decision_by_model["R1_P"].get("oracle_headroom_positive") is True
        or any(
            row.get("oracle_headroom_positive") is True
            for row in r1_sparse_primary.values()
        )
    )
    gap_weakens = (
        fidelity_signal
        and positive_headroom_exists
        and any([regret_meaningful, *cell_meaningful])
        and not all([regret_meaningful, *cell_meaningful])
        and not translation["R1_P"]
    )
    gap_not_supported = (
        not fidelity_signal
        or not positive_headroom_exists
        or translation["R1_P"]
    )
    equally_new = translation["R1_P"] or sum(
        translation[model]
        for model in ("R2_P", "R3_P", "Q2_P", "L2D_P", "SPO_PLUS_P")
    ) >= 2
    return {
        "FIDELITY_STRONG": fidelity_strong,
        "FIDELITY_PARTIAL": fidelity_partial,
        "R1_BEYOND_CAPACITY": beyond_capacity,
        "GENERIC_LOCALIZATION_CLEAR": generic_localization,
        "CAPACITY_DOMINANT": capacity_dominant,
        "PARTIAL_LOCALIZATION": partial_localization,
        "PREDICTION_INTERVENTION_GAP_PERSISTS": gap_persists,
        "PREDICTION_INTERVENTION_GAP_WEAKENS": gap_weakens,
        "PREDICTION_INTERVENTION_GAP_NOT_SUPPORTED": gap_not_supported,
        "EQUALLY_MEANINGFUL_NEW_RESULT": equally_new,
        "ANY_NON_ALIAS_CLEAR_UTILITY_TRANSLATION": any(translation.values()),
        "POST_HOC_REVERSAL": bool(post_hoc_reversal),
    }


def classify_prospective_core(
    *,
    integrity_complete: bool,
    predicates: Mapping[str, bool],
) -> dict[str, Any]:
    """Return exactly one terminal label under the frozen precedence rules."""

    required = {
        "FIDELITY_STRONG",
        "FIDELITY_PARTIAL",
        "R1_BEYOND_CAPACITY",
        "GENERIC_LOCALIZATION_CLEAR",
        "CAPACITY_DOMINANT",
        "PARTIAL_LOCALIZATION",
        "PREDICTION_INTERVENTION_GAP_PERSISTS",
        "PREDICTION_INTERVENTION_GAP_WEAKENS",
        "PREDICTION_INTERVENTION_GAP_NOT_SUPPORTED",
        "EQUALLY_MEANINGFUL_NEW_RESULT",
        "ANY_NON_ALIAS_CLEAR_UTILITY_TRANSLATION",
        "POST_HOC_REVERSAL",
    }
    if set(predicates) != required or any(
        not isinstance(value, bool) for value in predicates.values()
    ):
        raise RuntimeError("TMLR V6 mechanical predicate registry drift")
    if not integrity_complete:
        raise RuntimeError(
            "TMLR V6 scientific classification requires complete integrity; "
            "an integrity failure is not scientific evidence"
        )
    strong = (
        predicates["FIDELITY_STRONG"]
        and (
            predicates["R1_BEYOND_CAPACITY"]
            or predicates["GENERIC_LOCALIZATION_CLEAR"]
        )
        and (
            predicates["PREDICTION_INTERVENTION_GAP_PERSISTS"]
            or predicates["EQUALLY_MEANINGFUL_NEW_RESULT"]
        )
        and not predicates["POST_HOC_REVERSAL"]
    )
    if strong:
        return {
            "label": "TMLR_V6_PROSPECTIVE_CORE_STRONG",
            "classification_case": "STRONG",
            "scientific_negative_effect_claim": None,
            "post_hoc_upgrade_applied": False,
        }
    stable = any(
        predicates[name]
        for name in (
            "FIDELITY_STRONG",
            "FIDELITY_PARTIAL",
            "CAPACITY_DOMINANT",
            "GENERIC_LOCALIZATION_CLEAR",
            "PARTIAL_LOCALIZATION",
            "PREDICTION_INTERVENTION_GAP_WEAKENS",
            "ANY_NON_ALIAS_CLEAR_UTILITY_TRANSLATION",
            "POST_HOC_REVERSAL",
        )
    )
    return {
        "label": (
            "TMLR_V6_PROSPECTIVE_CORE_MIXED"
            if stable else "TMLR_V6_PROSPECTIVE_CORE_NOT_SUPPORTED"
        ),
        "classification_case": "MIXED" if stable else "NOT_SUPPORTED",
        "scientific_negative_effect_claim": None,
        "post_hoc_upgrade_applied": False,
    }


__all__ = [
    "ACTIONS",
    "ACTION_INDEX",
    "BUDGETS",
    "BUDGET_SEMANTICS",
    "EXTERNAL_SPLITS",
    "F0_F1_REPORT_SCHEMA",
    "F0_F1_REPORT_CHECKS",
    "F0_F1_REPORT_STATUS",
    "LAMBDAS",
    "PRIMARY_SPARSE_BUDGETS",
    "ProfiledCostSurface",
    "Phase2Surface",
    "REMAINING_ACTION_INDICES",
    "SECOND_BATCH_REPORT_SCHEMA",
    "SECOND_BATCH_REPORT_CHECKS",
    "SECOND_BATCH_REPORT_STATUS",
    "SEEDS",
    "ZERO_HEADROOM_STATUS",
    "action_selection_arrays",
    "assert_zero_external_access",
    "classify_prospective_core",
    "complete_case_state_mask",
    "decision_metrics",
    "direct_decision_family_interpretation",
    "derive_mechanical_predicates",
    "formal_evaluable_state_mask",
    "lower_better_status",
    "meaningful_utility",
    "paired_bootstrap_group_counts",
    "paired_ratio_bootstrap",
    "q2_scientific_gain_after_seed_mean",
    "residual_prediction_metrics",
    "residual_ratio_sufficient",
    "sparse_budget_metrics",
    "strict_continuous_seed_mean",
    "validate_f0_f1_report_contract",
    "validate_prediction_metric_row",
    "validate_second_batch_report_contract",
]
