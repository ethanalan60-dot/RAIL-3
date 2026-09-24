"""Frozen FIT-only utility sensitivity and target-rich subset diagnostics.

No model is retrained here.  The full-image pixel-mismatch R1 OOF prediction
surface is the only selector for all four utilities.  TP/FP/FN sufficient
statistics are target-side inputs and never become predictor features.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rail3.models.tmlr_v6.schema import ACTION_FEATURE_NAMES
from rail3.analysis.tmlr_v6_phase2 import (
    ACTIONS,
    BUDGET_SEMANTICS,
    PRIMARY_SPARSE_BUDGETS,
    ProfiledCostSurface,
    Phase2Surface,
    decision_metrics,
    formal_evaluable_state_mask,
    sparse_budget_metrics,
)


UTILITY_NAMES = (
    "full_image_pixel_mismatch",
    "one_minus_iou",
    "one_minus_dice",
    "foreground_normalized_symmetric_pixel_error",
)
SEGMENTATION_NATIVE_UTILITIES = (
    "one_minus_iou",
    "one_minus_dice",
    "foreground_normalized_symmetric_pixel_error",
)
SUBSET_NAMES = (
    "ALL_STATES",
    "REFERENCE_POSITIVE",
    "A0_SOURCE_PRESENT",
    "REFERENCE_POSITIVE_AND_A0_SOURCE_PRESENT",
)
NO_EVALUABLE_STATES = "NO_EVALUABLE_STATES"


def utility_losses_from_confusion(
    *,
    defined: np.ndarray,
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    valid_pixels: np.ndarray,
    reference_foreground_pixels: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Construct the four preregistered loss surfaces without clipping."""

    is_defined = np.asarray(defined)
    tp = np.asarray(true_positive)
    fp = np.asarray(false_positive)
    fn = np.asarray(false_negative)
    valid = np.asarray(valid_pixels)
    if (
        is_defined.ndim != 2
        or is_defined.shape[1] != len(ACTIONS)
        or is_defined.dtype.kind != "b"
        or any(value.shape != is_defined.shape for value in (tp, fp, fn))
        or valid.shape != (len(is_defined),)
        or any(value.dtype.kind not in "iu" for value in (tp, fp, fn, valid))
    ):
        raise ValueError("TMLR V6 confusion sufficient-statistic geometry drift")
    if np.any(valid <= 0):
        raise ValueError("TMLR V6 valid-pixel denominator must be positive")
    supplied_reference: np.ndarray | None = None
    if reference_foreground_pixels is not None:
        supplied_reference = np.asarray(reference_foreground_pixels)
        if (
            supplied_reference.shape != (len(is_defined),)
            or supplied_reference.dtype.kind not in "iu"
            or np.any(supplied_reference < 0)
            or np.any(supplied_reference > valid)
        ):
            raise ValueError("TMLR V6 reference foreground geometry drift")
    for value in (tp, fp, fn):
        if np.any(value[is_defined] < 0) or np.any(value[~is_defined] != -1):
            raise ValueError("TMLR V6 confusion undefined sentinel/count drift")
    safe_tp = np.where(is_defined, tp, 0).astype(np.float64)
    safe_fp = np.where(is_defined, fp, 0).astype(np.float64)
    safe_fn = np.where(is_defined, fn, 0).astype(np.float64)
    reference_by_action = safe_tp + safe_fn
    # Reference foreground is action invariant wherever an action is defined.
    reference_pixels = (
        np.asarray(supplied_reference, dtype=np.int64).copy()
        if supplied_reference is not None
        else np.zeros(len(is_defined), dtype=np.int64)
    )
    for row in range(len(is_defined)):
        values = reference_by_action[row, is_defined[row]]
        if values.size == 0:
            if supplied_reference is None:
                raise ValueError("TMLR V6 confusion state has no defined action")
            continue
        if not np.all(values == values[0]):
            raise ValueError("TMLR V6 reference foreground differs by action")
        derived_reference = int(values[0])
        if (
            supplied_reference is not None
            and reference_pixels[row] != derived_reference
        ):
            raise ValueError("TMLR V6 supplied reference foreground differs by action")
        reference_pixels[row] = derived_reference
    valid_matrix = np.broadcast_to(valid[:, None], is_defined.shape).astype(np.float64)
    union = safe_tp + safe_fp + safe_fn
    dice_denominator = 2.0 * safe_tp + safe_fp + safe_fn
    mismatch = np.divide(
        safe_fp + safe_fn,
        valid_matrix,
        out=np.full_like(safe_tp, np.nan),
        where=is_defined,
    )
    iou = np.divide(
        safe_tp,
        union,
        out=np.ones_like(safe_tp),
        where=union > 0.0,
    )
    dice = np.divide(
        2.0 * safe_tp,
        dice_denominator,
        out=np.ones_like(safe_tp),
        where=dice_denominator > 0.0,
    )
    foreground_error = np.divide(
        safe_fp + safe_fn,
        reference_by_action,
        out=np.full_like(safe_tp, np.nan),
        where=reference_by_action > 0.0,
    )
    reference_positive = reference_pixels > 0
    native_defined = is_defined & reference_positive[:, None]
    losses = {
        "full_image_pixel_mismatch": np.where(is_defined, mismatch, np.nan),
        "one_minus_iou": np.where(native_defined, 1.0 - iou, np.nan),
        "one_minus_dice": np.where(native_defined, 1.0 - dice, np.nan),
        "foreground_normalized_symmetric_pixel_error": np.where(
            native_defined, foreground_error, np.nan,
        ),
    }
    for name, values in losses.items():
        finite = np.isfinite(values)
        if np.any(values[finite] < 0.0):
            raise RuntimeError(f"TMLR V6 {name} is negative")
    return losses, reference_pixels


def _first_eligible_argmin(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    masked = np.where(eligible, values, np.inf)
    if np.any(~np.isfinite(masked).any(axis=1)):
        raise RuntimeError("TMLR V6 sensitivity state has no eligible action")
    return masked.argmin(axis=1)


def _utility_population(
    *,
    base_surface: Phase2Surface,
    utility: np.ndarray,
) -> np.ndarray:
    values = np.asarray(utility, dtype=np.float64)
    if values.shape != np.asarray(base_surface.target).shape:
        raise ValueError("TMLR V6 utility surface is not state/action aligned")
    formal = formal_evaluable_state_mask(base_surface)
    feasible = np.asarray(base_surface.feasible, dtype=bool)
    # Every plan-feasible action in the registered action set must have the
    # current utility defined; no action-wise deletion is permitted.
    return formal & ~np.any(feasible & ~np.isfinite(values), axis=1)


def utility_audit_rows(
    *,
    mismatch_selector: Phase2Surface,
    utilities: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Audit prevalence, oracle structure, headroom, and oracle agreement."""

    mismatch_selector.validated()
    if mismatch_selector.prediction_kind != "residual":
        raise ValueError("TMLR V6 utility selector must be residual-valued")
    if tuple(utilities) != UTILITY_NAMES:
        raise RuntimeError("TMLR V6 utility registry/order drift")
    feasible = np.asarray(mismatch_selector.feasible, dtype=bool)
    prediction = np.asarray(mismatch_selector.prediction, dtype=np.float64)
    mismatch = np.asarray(utilities[UTILITY_NAMES[0]], dtype=np.float64)
    mismatch_population = _utility_population(
        base_surface=mismatch_selector, utility=mismatch,
    )
    mismatch_oracle = np.full(len(prediction), -1, dtype=np.int64)
    mismatch_oracle[mismatch_population] = _first_eligible_argmin(
        mismatch[mismatch_population], feasible[mismatch_population],
    )
    result: list[dict[str, Any]] = []
    for name in UTILITY_NAMES:
        values = np.asarray(utilities[name], dtype=np.float64)
        population = _utility_population(
            base_surface=mismatch_selector, utility=values,
        )
        indices = np.flatnonzero(population)
        if len(indices) == 0:
            result.append({
                "utility": name,
                "split": "FIT_OOF_ONLY",
                "status": NO_EVALUABLE_STATES,
                "evaluable_states": 0,
                "metric_specific_training": False,
                "confirmatory_claim_allowed": False,
            })
            continue
        eligible = feasible[indices]
        current = values[indices]
        oracle = _first_eligible_argmin(current, eligible)
        rows = np.arange(len(indices))
        masked_nonstop = np.where(eligible[:, 1:], current[:, 1:], np.inf)
        best_nonstop = np.min(masked_nonstop, axis=1)
        strict_beneficial = best_nonstop < current[:, 0]
        minimum = np.min(np.where(eligible, current, np.inf), axis=1)
        tie_count = np.sum(eligible & (current == minimum[:, None]), axis=1)
        oracle_counts = {
            action: int(np.sum(oracle == action_index))
            for action_index, action in enumerate(ACTIONS)
        }
        intersection = population & mismatch_population
        intersection_indices = np.flatnonzero(intersection)
        current_intersection_oracle = oracle[
            np.searchsorted(indices, intersection_indices)
        ]
        mismatch_intersection_oracle = mismatch_oracle[intersection]
        agreement = (
            float(np.mean(
                current_intersection_oracle == mismatch_intersection_oracle
            ))
            if intersection.any() else None
        )
        pair_counts = {
            mismatch_action: {
                utility_action: int(np.sum(
                    (mismatch_intersection_oracle == mismatch_index)
                    & (current_intersection_oracle == utility_index)
                ))
                for utility_index, utility_action in enumerate(ACTIONS)
            }
            for mismatch_index, mismatch_action in enumerate(ACTIONS)
        }
        total_headroom = float(np.maximum(
            0.0, current[:, 0] - current[rows, oracle],
        ).sum())
        result.append({
            "utility": name,
            "split": "FIT_OOF_ONLY",
            "status": "PASS",
            "evaluable_states": len(indices),
            "stop_zero_error_rate_using_exact_equality": float(np.mean(
                current[:, 0] == 0.0
            )),
            "beneficial_repair_states": int(strict_beneficial.sum()),
            "beneficial_repair_prevalence_using_strict_inequality": float(
                strict_beneficial.mean()
            ),
            "oracle_best_action_counts": oracle_counts,
            "oracle_best_action_fractions": {
                action: count / len(indices) for action, count in oracle_counts.items()
            },
            "oracle_tie_states": int(np.sum(tie_count > 1)),
            "oracle_tie_rate": float(np.mean(tie_count > 1)),
            "total_oracle_headroom": total_headroom,
            "oracle_headroom_positive": total_headroom > 0.0,
            "oracle_action_agreement_with_pixel_mismatch": agreement,
            "oracle_action_pair_counts_with_pixel_mismatch": pair_counts,
            "oracle_agreement_population_intersection_states": int(
                intersection.sum()
            ),
            "metric_specific_training": False,
            "selector": "R1_P_FULL_IMAGE_PIXEL_MISMATCH_OOF",
            "confirmatory_claim_allowed": False,
        })
    return result


def sensitivity_budget_rows(
    *,
    mismatch_selector: Phase2Surface,
    utilities: Mapping[str, np.ndarray],
    zero_costs: ProfiledCostSurface,
) -> list[dict[str, Any]]:
    """Keep mismatch prediction/ranking fixed and rescore true utility at lambda 0."""

    mismatch_selector.validated()
    if mismatch_selector.prediction_kind != "residual":
        raise ValueError("TMLR V6 sensitivity selector must be residual-valued")
    result: list[dict[str, Any]] = []
    base_target = np.asarray(mismatch_selector.target, dtype=np.float64)
    for name in UTILITY_NAMES:
        values = np.asarray(utilities[name], dtype=np.float64)
        population = _utility_population(
            base_surface=mismatch_selector, utility=values,
        )
        if not population.any():
            for budget in PRIMARY_SPARSE_BUDGETS:
                for semantics in BUDGET_SEMANTICS:
                    result.append({
                        "utility": name,
                        "budget": budget,
                        "budget_semantics": semantics,
                        "status": NO_EVALUABLE_STATES,
                        "evaluable_states": 0,
                        "normalized_regret_defined": False,
                        "normalized_regret": None,
                        "gain_capture": None,
                    })
            continue
        # Outside the current utility population values are never consumed.
        # Filling them from the registered mismatch target preserves the base
        # complete-case accounting instead of relabeling reference-negative
        # states as execution failures.
        target = np.where(population[:, None], values, base_target)
        rescored = Phase2Surface(
            model_id=mismatch_selector.model_id,
            prediction_kind="residual",
            prediction=np.asarray(mismatch_selector.prediction),
            target=target,
            feasible=np.asarray(mismatch_selector.feasible),
            groups=np.asarray(mismatch_selector.groups),
            state_ids=mismatch_selector.state_ids,
            action_indices=mismatch_selector.action_indices,
        ).validated()
        decision = decision_metrics(
            rescored,
            zero_costs,
            cost_lambda=0.0,
            population_mask=population,
        )
        rows = sparse_budget_metrics(
            rescored,
            zero_costs,
            cost_lambda=0.0,
            budgets=PRIMARY_SPARSE_BUDGETS,
            population_mask=population,
        )
        for row in rows:
            result.append({
                **row,
                "utility": name,
                "status": "PASS",
                "selector_utility": "full_image_pixel_mismatch",
                "true_action_utility": name,
                "metric_specific_training": False,
                "within_current_population_rerank": True,
                "post_hoc_sensitivity": True,
                "normalized_regret_defined": decision[
                    "normalized_regret_defined"
                ],
                "normalized_regret": decision["normalized_regret"],
            })
    return result


def target_rich_subset_masks(
    *,
    formal_mask: np.ndarray,
    reference_foreground_pixels: np.ndarray,
    raw_f0_action_x16: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the four exact, raw-feature-only target-rich predicates."""

    formal = np.asarray(formal_mask)
    reference = np.asarray(reference_foreground_pixels)
    action = np.asarray(raw_f0_action_x16)
    if (
        formal.ndim != 1
        or formal.dtype.kind != "b"
        or reference.shape != formal.shape
        or action.shape != (len(formal), len(ACTIONS), 16)
    ):
        raise ValueError("TMLR V6 target-rich subset inputs are not aligned")
    if (
        len(ACTION_FEATURE_NAMES) != 16
        or ACTION_FEATURE_NAMES[1] != "source_candidate_present"
    ):
        raise RuntimeError("TMLR V6 source_candidate_present schema index drift")
    source = action[:, (2, 3, 4), 1]
    if not np.isin(source, (0.0, 1.0)).all():
        raise RuntimeError("TMLR V6 source_candidate_present is not binary")
    if not (
        np.array_equal(source[:, 0], source[:, 1])
        and np.array_equal(source[:, 0], source[:, 2])
    ):
        raise RuntimeError("TMLR V6 A4/A5/A6 source-present field drift")
    source_present = source[:, 0] == 1.0
    reference_positive = reference > 0
    return {
        "ALL_STATES": formal.copy(),
        "REFERENCE_POSITIVE": formal & reference_positive,
        "A0_SOURCE_PRESENT": formal & source_present,
        "REFERENCE_POSITIVE_AND_A0_SOURCE_PRESENT": (
            formal & reference_positive & source_present
        ),
    }


def target_rich_subset_rows(
    *,
    mismatch_selector: Phase2Surface,
    zero_costs: ProfiledCostSurface,
    subset_masks: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Within-subset re-rank R1 at 1/2/5% under both budget semantics."""

    mismatch_selector.validated()
    if tuple(subset_masks) != SUBSET_NAMES:
        raise RuntimeError("TMLR V6 target-rich subset registry/order drift")
    target = np.asarray(mismatch_selector.target, dtype=np.float64)
    feasible = np.asarray(mismatch_selector.feasible, dtype=bool)
    result: list[dict[str, Any]] = []
    for subset_name in SUBSET_NAMES:
        mask = np.asarray(subset_masks[subset_name])
        if mask.shape != (len(target),) or mask.dtype.kind != "b":
            raise ValueError("TMLR V6 target-rich subset mask is misaligned")
        indices = np.flatnonzero(mask & formal_evaluable_state_mask(mismatch_selector))
        if len(indices) == 0:
            for budget in PRIMARY_SPARSE_BUDGETS:
                for semantics in BUDGET_SEMANTICS:
                    result.append({
                        "subset": subset_name,
                        "lambda": 0.0,
                        "budget": budget,
                        "budget_semantics": semantics,
                        "status": NO_EVALUABLE_STATES,
                        "n_states": 0,
                        "beneficial_prevalence": None,
                        "cap": 0,
                        "used": 0,
                        "unused": 0,
                        "oracle_gain_denominator": 0.0,
                        "realized_gain": 0.0,
                        "normalized_regret_defined": False,
                        "normalized_regret": None,
                        "gain_capture": None,
                    })
            continue
        masked_target = np.where(feasible[indices], target[indices], np.inf)
        best_nonstop = np.min(masked_target[:, 1:], axis=1)
        prevalence = float(np.mean(best_nonstop < masked_target[:, 0]))
        decision = decision_metrics(
            mismatch_selector,
            zero_costs,
            cost_lambda=0.0,
            population_mask=mask,
        )
        rows = sparse_budget_metrics(
            mismatch_selector,
            zero_costs,
            cost_lambda=0.0,
            budgets=PRIMARY_SPARSE_BUDGETS,
            population_mask=mask,
        )
        for row in rows:
            result.append({
                **row,
                "subset": subset_name,
                "status": "PASS",
                "n_states": len(indices),
                "beneficial_prevalence": prevalence,
                "oracle_gain_denominator": row["oracle_gain"],
                "normalized_regret_defined": decision[
                    "normalized_regret_defined"
                ],
                "normalized_regret": decision["normalized_regret"],
                "within_subset_rerank": True,
                "fallback_to_all_states": False,
            })
    return result


def sensitivity_reversal_predicate(
    *,
    primary_gap_persists: bool,
    sensitivity_budget_rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Compute POST_HOC_REVERSAL; this predicate can only downgrade STRONG."""

    if not primary_gap_persists:
        return False

    def clear(cells: Sequence[Mapping[str, Any]]) -> bool:
        expected = {
            (budget, semantics)
            for budget in PRIMARY_SPARSE_BUDGETS
            for semantics in BUDGET_SEMANTICS
        }
        passing = [row for row in cells if row.get("status") == "PASS"]
        mapped = {
            (float(row["budget"]), str(row["budget_semantics"])): row
            for row in passing
        }
        if (
            len(passing) != len(expected)
            or len(mapped) != len(passing)
            or set(mapped) != expected
        ):
            return False
        regrets = {
            float(row["normalized_regret"])
            for row in mapped.values()
            if row.get("normalized_regret_defined") is True
            and row.get("normalized_regret") is not None
        }
        return len(regrets) == 1 and next(iter(regrets)) < 0.95 and all(
            row.get("oracle_headroom_positive") is True
            and row.get("gain_capture") is not None
            and float(row["gain_capture"]) > 0.05
            and float(row["realized_gain"]) > 0.0
            for row in mapped.values()
        )

    native_reversal = any(
        clear([
            row for row in sensitivity_budget_rows
            if row.get("utility") == utility
        ])
        for utility in SEGMENTATION_NATIVE_UTILITIES
    )
    rich_reversal = clear([
        row for row in subset_rows
        if row.get("subset") == "REFERENCE_POSITIVE_AND_A0_SOURCE_PRESENT"
    ])
    return native_reversal or rich_reversal


__all__ = [
    "NO_EVALUABLE_STATES",
    "SEGMENTATION_NATIVE_UTILITIES",
    "SUBSET_NAMES",
    "UTILITY_NAMES",
    "sensitivity_budget_rows",
    "sensitivity_reversal_predicate",
    "target_rich_subset_masks",
    "target_rich_subset_rows",
    "utility_audit_rows",
    "utility_losses_from_confusion",
]
