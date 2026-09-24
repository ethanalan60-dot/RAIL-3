"""Deterministic FIT-OOF qualitative predicates for TMLR V6 phase two."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.analysis.tmlr_v6_phase2 import (
    ACTIONS,
    Phase2Surface,
    complete_case_state_mask,
    formal_evaluable_state_mask,
)


QUALITATIVE_CATEGORIES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
GOOD_RESIDUAL_THRESHOLD = 0.05
NO_ELIGIBLE_STATE = "NO_ELIGIBLE_STATE"
Q6_COMPARISON_LABEL = "MULTI_COMPONENT_RETRO_V4_TO_PROSPECTIVE_V1_DISAGREEMENT"
Q6_CHANGED_COMPONENTS = (
    "runtime_columns_removed",
    "action_dimension_changed",
    "parameter_count_changed",
    "prediction_aggregation_changed",
    "training_objective_changed",
)


def selection_key(state_id: str) -> tuple[str, str]:
    value = str(state_id)
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), value


def _aligned_residual_surfaces(
    surfaces: Mapping[str, Phase2Surface],
) -> tuple[Phase2Surface, Phase2Surface, Phase2Surface]:
    if set(surfaces) != {"R1_P", "R0_CM_P", "RETRO_R1_FIT_OOF"}:
        raise RuntimeError("TMLR V6 qualitative model panel drift")
    r1 = surfaces["R1_P"].validated()
    r0 = surfaces["R0_CM_P"].validated()
    retro = surfaces["RETRO_R1_FIT_OOF"].validated()
    for surface in (r1, r0, retro):
        if surface.prediction_kind != "residual":
            raise ValueError("TMLR V6 qualitative surface is not residual-valued")
        if surface.action_indices != tuple(range(len(ACTIONS))):
            raise ValueError("TMLR V6 qualitative surface action set drift")
    for surface in (r0, retro):
        if (
            surface.state_ids != r1.state_ids
            or not np.array_equal(surface.groups, r1.groups)
            or not np.array_equal(surface.feasible, r1.feasible)
            or not np.array_equal(surface.target, r1.target, equal_nan=True)
        ):
            raise RuntimeError("TMLR V6 qualitative common-target alignment drift")
    return r1, r0, retro


def qualitative_predicate_masks(
    surfaces: Mapping[str, Phase2Surface],
) -> dict[str, Any]:
    """Evaluate Q1--Q6 literally, without result-conditioned relaxation."""

    r1, r0, retro = _aligned_residual_surfaces(surfaces)
    feasible = np.asarray(r1.feasible, dtype=bool)
    target = np.asarray(r1.target, dtype=np.float64)
    r1_prediction = np.asarray(r1.prediction, dtype=np.float64)
    r0_prediction = np.asarray(r0.prediction, dtype=np.float64)
    retro_prediction = np.asarray(retro.prediction, dtype=np.float64)
    formal = formal_evaluable_state_mask(r1)
    masked_target = np.where(feasible, target, np.inf)
    r1_selected = np.where(feasible, r1_prediction, np.inf).argmin(axis=1)
    r0_selected = np.where(feasible, r0_prediction, np.inf).argmin(axis=1)
    retro_selected = np.where(feasible, retro_prediction, np.inf).argmin(axis=1)
    minimum = np.min(masked_target, axis=1)
    oracle_membership = feasible & (target == minimum[:, None])
    display_oracle = oracle_membership.argmax(axis=1)
    rows = np.arange(len(target))
    eligible_count = feasible.sum(axis=1)
    absolute_error = np.where(
        feasible, np.abs(r1_prediction - target), 0.0,
    )
    mean_error = np.divide(
        absolute_error.sum(axis=1),
        eligible_count,
        out=np.full(len(target), np.inf, dtype=np.float64),
        where=eligible_count > 0,
    )
    good = formal & (mean_error <= GOOD_RESIDUAL_THRESHOLD)
    r1_correct = oracle_membership[rows, r1_selected]
    best_nonstop_true = np.min(masked_target[:, 1:], axis=1)
    masks = {
        "Q1": good & r1_correct,
        "Q2": good & ~r1_correct,
        "Q3": formal & (r1_selected == 0) & (best_nonstop_true < target[:, 0]),
        "Q4": formal & (r1_selected != 0) & (
            target[rows, r1_selected] > target[:, 0]
        ),
        "Q5": formal & (r0_selected != r1_selected),
        "Q6": formal & (retro_selected != r1_selected),
    }
    return {
        "masks": masks,
        "formal_mask": formal,
        "mean_absolute_residual_error": mean_error,
        "good_residual": good,
        "r1_selected": r1_selected,
        "r0_cm_selected": r0_selected,
        "retro_r1_selected": retro_selected,
        "oracle_membership": oracle_membership,
        "display_oracle": display_oracle,
    }


def build_selection_manifest(
    surfaces: Mapping[str, Phase2Surface],
) -> dict[str, Any]:
    """Select the first state under the frozen hash/state-ID key per category."""

    r1, r0, retro = _aligned_residual_surfaces(surfaces)
    analysis = qualitative_predicate_masks(surfaces)
    masks = analysis["masks"]
    state_ids = np.asarray(r1.state_ids, dtype=str)
    target = np.asarray(r1.target, dtype=np.float64)
    feasible = np.asarray(r1.feasible, dtype=bool)
    entries: list[dict[str, Any]] = []
    for category in QUALITATIVE_CATEGORIES:
        mask = np.asarray(masks[category], dtype=bool)
        indices = np.flatnonzero(mask)
        eligible_ids = sorted(str(state_ids[index]) for index in indices)
        eligible_sha = hashlib.sha256(
            canonical_json_bytes(eligible_ids)
        ).hexdigest()
        if len(indices) == 0:
            entries.append({
                "category": category,
                "status": NO_ELIGIBLE_STATE,
                "eligible_states": 0,
                "eligible_state_ids_sha256": eligible_sha,
                "selected_state_id": None,
                "selected_state_index": None,
                "selection_key_sha256": None,
                "predicate_relaxed": False,
            })
            continue
        selected_index = min(
            indices.tolist(), key=lambda index: selection_key(state_ids[index]),
        )
        selected_id = str(state_ids[selected_index])
        entry = {
            "category": category,
            "status": "SELECTED",
            "eligible_states": len(indices),
            "eligible_state_ids_sha256": eligible_sha,
            "selected_state_id": selected_id,
            "selected_state_index": int(selected_index),
            "selection_key_sha256": selection_key(selected_id)[0],
            "selection_key_state_id": selection_key(selected_id)[1],
            "image_group_id": str(r1.groups[selected_index]),
            "r1_selected_action": ACTIONS[int(analysis["r1_selected"][selected_index])],
            "display_oracle_action": ACTIONS[int(analysis["display_oracle"][selected_index])],
            "oracle_action_set": [
                ACTIONS[index]
                for index in range(len(ACTIONS))
                if analysis["oracle_membership"][selected_index, index]
            ],
            "mean_absolute_r1_residual_error": float(
                analysis["mean_absolute_residual_error"][selected_index]
            ),
            "good_residual_threshold": GOOD_RESIDUAL_THRESHOLD,
            "good_residual": bool(analysis["good_residual"][selected_index]),
            "r1_prediction": [
                float(value) for value in np.asarray(r1.prediction)[selected_index]
            ],
            "true_residual": [
                float(value) if np.isfinite(value) else None
                for value in target[selected_index]
            ],
            "plan_feasible": [bool(value) for value in feasible[selected_index]],
            "predicate_relaxed": False,
        }
        if category == "Q5":
            entry.update({
                "comparison_model": "R0_CM_P",
                "comparison_selected_action": ACTIONS[int(
                    analysis["r0_cm_selected"][selected_index]
                )],
                "comparison_prediction": [
                    float(value) for value in np.asarray(r0.prediction)[selected_index]
                ],
            })
        elif category == "Q6":
            entry.update({
                "comparison_model": "RETRO_R1_FIT_OOF",
                "comparison_label": Q6_COMPARISON_LABEL,
                "pure_runtime_ablation_claim_allowed": False,
                "changed_components": list(Q6_CHANGED_COMPONENTS),
                "comparison_selected_action": ACTIONS[int(
                    analysis["retro_r1_selected"][selected_index]
                )],
                "comparison_prediction": [
                    float(value) for value in np.asarray(retro.prediction)[selected_index]
                ],
            })
        entries.append(entry)
    _, complete_case = complete_case_state_mask(r1)
    return {
        "schema_version": "rail3.tmlr-v6.phase2-qualitative-selection.v1",
        "status": "TMLR_V6_PHASE2_QUALITATIVE_SELECTION_PASS",
        "split": "FIT_OOF_ONLY",
        "lambda": 0.0,
        "categories": entries,
        "selection_key": "(sha256(state_id.encode('utf-8')).hexdigest(),state_id)",
        "good_residual_threshold": GOOD_RESIDUAL_THRESHOLD,
        "good_residual_threshold_source": "frozen_state_huber_delta",
        "cross_category_deduplication": False,
        "manual_selection": False,
        "predicate_relaxation": False,
        "q1_nonstop_requirement": False,
        **complete_case,
    }


def validate_selection_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on category omission, relaxation, or manual replacement."""

    if (
        manifest.get("schema_version")
        != "rail3.tmlr-v6.phase2-qualitative-selection.v1"
        or manifest.get("status")
        != "TMLR_V6_PHASE2_QUALITATIVE_SELECTION_PASS"
        or manifest.get("split") != "FIT_OOF_ONLY"
        or manifest.get("lambda") != 0.0
        or manifest.get("cross_category_deduplication") is not False
        or manifest.get("manual_selection") is not False
        or manifest.get("predicate_relaxation") is not False
    ):
        raise RuntimeError("TMLR V6 qualitative selection manifest drift")
    entries = manifest.get("categories")
    if (
        not isinstance(entries, list)
        or tuple(entry.get("category") for entry in entries)
        != QUALITATIVE_CATEGORIES
    ):
        raise RuntimeError("TMLR V6 qualitative category registry drift")
    for entry in entries:
        if entry.get("predicate_relaxed") is not False:
            raise RuntimeError("TMLR V6 qualitative predicate was relaxed")
        if entry.get("status") == NO_ELIGIBLE_STATE:
            if (
                entry.get("eligible_states") != 0
                or entry.get("selected_state_id") is not None
                or entry.get("selected_state_index") is not None
            ):
                raise RuntimeError("TMLR V6 no-eligible qualitative row drift")
        elif entry.get("status") == "SELECTED":
            selected = entry.get("selected_state_id")
            prediction = entry.get("r1_prediction")
            target = entry.get("true_residual")
            feasible = entry.get("plan_feasible")
            oracle_set = entry.get("oracle_action_set")
            if (
                not isinstance(selected, str)
                or entry.get("selection_key_sha256") != selection_key(selected)[0]
                or entry.get("selection_key_state_id") != selected
                or int(entry.get("eligible_states", 0)) <= 0
                or entry.get("r1_selected_action") not in ACTIONS
                or entry.get("display_oracle_action") not in ACTIONS
                or not isinstance(oracle_set, list)
                or not oracle_set
                or any(action not in ACTIONS for action in oracle_set)
                or entry.get("display_oracle_action") not in oracle_set
                or not isinstance(prediction, list)
                or len(prediction) != len(ACTIONS)
                or not np.isfinite(np.asarray(prediction, dtype=float)).all()
                or not isinstance(target, list)
                or len(target) != len(ACTIONS)
                or not isinstance(feasible, list)
                or len(feasible) != len(ACTIONS)
                or any(not isinstance(value, bool) for value in feasible)
            ):
                raise RuntimeError("TMLR V6 deterministic qualitative selection drift")
            category = str(entry["category"])
            if category in {"Q5", "Q6"}:
                comparison = entry.get("comparison_prediction")
                expected_model = (
                    "R0_CM_P" if category == "Q5" else "RETRO_R1_FIT_OOF"
                )
                if (
                    entry.get("comparison_model") != expected_model
                    or entry.get("comparison_selected_action") not in ACTIONS
                    or not isinstance(comparison, list)
                    or len(comparison) != len(ACTIONS)
                    or not np.isfinite(np.asarray(comparison, dtype=float)).all()
                ):
                    raise RuntimeError("TMLR V6 qualitative comparison content drift")
            if category == "Q6" and (
                entry.get("comparison_label") != Q6_COMPARISON_LABEL
                or entry.get("pure_runtime_ablation_claim_allowed") is not False
                or entry.get("changed_components") != list(Q6_CHANGED_COMPONENTS)
            ):
                raise RuntimeError("TMLR V6 Q6 interpretation boundary drift")
        else:
            raise RuntimeError("TMLR V6 qualitative selection status drift")


def error_change_map(
    *,
    truth: np.ndarray,
    stop_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return -1 repaired, 0 unchanged, +1 newly wrong, and NaN for void."""

    target = np.asarray(truth)
    stop = np.asarray(stop_prediction)
    selected = np.asarray(selected_prediction)
    validity = np.asarray(valid)
    if (
        target.shape != stop.shape
        or target.shape != selected.shape
        or target.shape != validity.shape
        or any(value.dtype.kind != "b" for value in (target, stop, selected, validity))
    ):
        raise ValueError("TMLR V6 qualitative error-map geometry/type drift")
    result = np.full(target.shape, np.nan, dtype=np.float32)
    result[validity] = (
        (selected[validity] != target[validity]).astype(np.int8)
        - (stop[validity] != target[validity]).astype(np.int8)
    )
    if not np.isin(result[validity], (-1.0, 0.0, 1.0)).all():
        raise RuntimeError("TMLR V6 qualitative error-change map drift")
    return result


__all__ = [
    "GOOD_RESIDUAL_THRESHOLD",
    "NO_ELIGIBLE_STATE",
    "Q6_CHANGED_COMPONENTS",
    "Q6_COMPARISON_LABEL",
    "QUALITATIVE_CATEGORIES",
    "build_selection_manifest",
    "error_change_map",
    "qualitative_predicate_masks",
    "selection_key",
    "validate_selection_manifest",
]
