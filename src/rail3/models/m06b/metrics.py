"""Fit-only OOF QG/PAL, budget, and group-bootstrap metrics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes


MARGIN = 1e-12


def average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    left = np.asarray(prediction, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return None
    left_rank, right_rank = average_ranks(left[finite]), average_ranks(right[finite])
    if left_rank.std() == 0 or right_rank.std() == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _state_matrix(values: np.ndarray, state: np.ndarray, action: np.ndarray, n_states: int) -> np.ndarray:
    result = np.full((n_states, 7), np.nan, dtype=np.float64)
    result[state, action] = values
    return result


def _selected_actions(scores: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    eligible = np.where(feasible[:, 1:], scores[:, 1:], -np.inf)
    best = np.argmax(eligible, axis=1) + 1
    best_score = eligible[np.arange(scores.shape[0]), best - 1]
    return np.where(best_score > 0.0, best, -1)


def _state_truth(gains: np.ndarray, feasible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.where(feasible[:, 1:], gains[:, 1:], -np.inf)
    oracle = np.maximum(0.0, np.max(eligible, axis=1))
    return eligible, oracle


def _mean_state_error(error: np.ndarray, state: np.ndarray) -> float:
    values = []
    for index in np.unique(state):
        values.append(float(np.mean(error[state == index])))
    return float(np.mean(values)) if values else math.nan


def qg_metrics(
    *, scores: np.ndarray, gains: np.ndarray, feasible: np.ndarray,
    state: np.ndarray, action: np.ndarray, state_ids: Sequence[str],
    state_classes: np.ndarray, state_mask: np.ndarray,
    pair_left: np.ndarray, pair_right: np.ndarray, pair_preference: np.ndarray,
    pair_state: np.ndarray,
) -> dict[str, Any]:
    selected_states = np.flatnonzero(np.asarray(state_mask, dtype=bool))
    if selected_states.size == 0:
        raise ValueError("QG metrics require held-out states")
    state_position = {int(value): index for index, value in enumerate(selected_states)}
    row_mask = np.isin(state, selected_states) & feasible & np.isfinite(gains) & np.isfinite(scores)
    local_state = np.asarray([state_position[int(value)] for value in state[row_mask]], dtype=np.int32)
    n = selected_states.size
    score_matrix = _state_matrix(scores[row_mask], local_state, action[row_mask], n)
    gain_matrix = _state_matrix(gains[row_mask], local_state, action[row_mask], n)
    feasible_matrix = np.isfinite(gain_matrix)
    selected = _selected_actions(score_matrix, feasible_matrix)
    eligible_truth, oracle = _state_truth(gain_matrix, feasible_matrix)
    realized = np.zeros(n, dtype=np.float64)
    active = selected >= 1
    realized[active] = gain_matrix[np.arange(n)[active], selected[active]]
    regret = oracle - realized
    denominator = max(float(oracle.sum()), 1e-12)
    correct = np.zeros(n, dtype=bool)
    stop_truth = oracle <= MARGIN
    correct[stop_truth] = selected[stop_truth] == -1
    action_truth = ~stop_truth
    correct[action_truth] = (
        selected[action_truth] >= 1
    ) & (
        np.abs(realized[action_truth] - oracle[action_truth]) <= MARGIN
    )
    top2_regret = np.empty(n, dtype=np.float64)
    for index in range(n):
        choices = [(-1, 0.0)]
        choices.extend(
            (action_id, score_matrix[index, action_id])
            for action_id in range(1, 7) if feasible_matrix[index, action_id]
        )
        top = sorted(choices, key=lambda item: (-item[1], item[0]))[:2]
        best_true = max(
            0.0 if action_id == -1 else gain_matrix[index, action_id]
            for action_id, _ in top
        )
        top2_regret[index] = oracle[index] - best_true
    high_count = max(1, math.ceil(0.20 * n))
    local_ids = [state_ids[int(value)] for value in selected_states]
    high_order = sorted(range(n), key=lambda i: (-oracle[i], local_ids[i]))[:high_count]
    high_recall = float(np.mean(selected[high_order] >= 1))
    pair_mask = np.isin(pair_state, selected_states)
    pair_correct = []
    for left, right, preference in zip(
        pair_left[pair_mask], pair_right[pair_mask], pair_preference[pair_mask], strict=True,
    ):
        difference = scores[int(left)] - scores[int(right)]
        predicted = 1 if difference > 0 else -1 if difference < 0 else 0
        pair_correct.append(predicted == int(preference))
    classes = state_classes[selected_states]
    result = {
        "state_count": int(n),
        "signed_gain_mae": _mean_state_error(
            np.abs(scores[row_mask] - gains[row_mask]), local_state,
        ),
        "signed_gain_rmse": math.sqrt(_mean_state_error(
            np.square(scores[row_mask] - gains[row_mask]), local_state,
        )),
        "spearman": spearman(scores[row_mask], gains[row_mask]),
        "pairwise_accuracy": float(np.mean(pair_correct)) if pair_correct else None,
        "pairwise_count": len(pair_correct),
        "top1_action_accuracy": float(correct.mean()),
        "top1_regret": float(regret.mean()),
        "normalized_top1_regret": float(regret.sum() / denominator),
        "top2_regret": float(top2_regret.mean()),
        "canonical_stop_accuracy": float(np.mean(selected[stop_truth] == -1)) if stop_truth.any() else None,
        "noncanonical_best_recall": float(np.mean(correct[action_truth])) if action_truth.any() else None,
        "high_gain_state_recall": high_recall,
        "oracle_positive_gain_sum": float(oracle.sum()),
        "realized_true_gain_sum": float(realized.sum()),
        "selected_action_count": int(active.sum()),
        "negative_action_rate": float(np.mean(realized[active] < 0)) if active.any() else 0.0,
        "per_class_normalized_regret": {},
    }
    for class_id in range(1, 21):
        class_mask = classes == class_id
        if class_mask.any():
            result["per_class_normalized_regret"][str(class_id)] = float(
                regret[class_mask].sum() / max(float(oracle[class_mask].sum()), 1e-12)
            )
    result["_state_arrays"] = {
        "state_indices": selected_states, "selected": selected, "oracle": oracle,
        "realized": realized, "regret": regret,
        "high_gain_local_indices": np.asarray(high_order, dtype=np.int32),
        "score_matrix": score_matrix, "gain_matrix": gain_matrix,
        "feasible_matrix": feasible_matrix,
    }
    return result


def budget_curve(
    state_arrays: Mapping[str, np.ndarray], *, cost_matrix: np.ndarray,
    state_ids: Sequence[str], action_rates: Sequence[float], cost_lambda: float,
    scores_are_utility: bool = False,
) -> dict[str, Any]:
    state_indices = state_arrays["state_indices"]
    score_matrix = state_arrays["score_matrix"]
    gain_matrix = state_arrays["gain_matrix"]
    feasible_matrix = state_arrays["feasible_matrix"]
    oracle = state_arrays["oracle"]
    high = set(int(value) for value in state_arrays["high_gain_local_indices"])
    local_cost = cost_matrix[state_indices]
    action_utility = score_matrix[:, 1:].copy()
    if not scores_are_utility:
        action_utility -= cost_lambda * local_cost[:, 1:]
    action_utility[~feasible_matrix[:, 1:]] = -np.inf
    best_action = np.argmax(action_utility, axis=1) + 1
    best_utility = action_utility[np.arange(action_utility.shape[0]), best_action - 1]
    identifiers = [state_ids[int(index)] for index in state_indices]
    order = sorted(range(len(identifiers)), key=lambda i: (-best_utility[i], identifiers[i]))
    total_oracle = max(float(oracle.sum()), 1e-12)
    result = {}
    for rate in action_rates:
        limit = math.ceil(float(rate) * len(order))
        selected = [index for index in order[:limit] if best_utility[index] > 0]
        actions = best_action[selected] if selected else np.asarray([], dtype=np.int64)
        realized = (
            gain_matrix[np.asarray(selected), actions] if selected else np.asarray([], dtype=np.float64)
        )
        captured = float(oracle[selected].sum()) if selected else 0.0
        distribution = Counter(f"A{int(value)}" for value in actions)
        result[str(rate)] = {
            "maximum_action_count": int(limit), "selected_action_count": len(selected),
            "oracle_positive_gain_captured": captured,
            "gain_capture_fraction": captured / total_oracle,
            "realized_true_gain": float(realized.sum()),
            "regret": float(oracle.sum() - realized.sum()),
            "selected_state_precision": float(np.mean(realized > 0)) if selected else 0.0,
            "high_gain_recall": len(high & set(selected)) / max(len(high), 1),
            "action_distribution": dict(sorted(distribution.items())),
            "gpu_cost_seconds": float(local_cost[np.asarray(selected), actions].sum()) if selected else 0.0,
            "negative_action_rate": float(np.mean(realized < 0)) if selected else 0.0,
        }
    return result


def training_derived_gpu_budgets(
    *, cost_matrix: np.ndarray, feasible_matrix: np.ndarray, training_states: np.ndarray,
    heldout_state_count: int, levels: Sequence[float],
) -> dict[str, float]:
    minimums = []
    for state_index in training_states:
        costs = cost_matrix[int(state_index), 1:][feasible_matrix[int(state_index), 1:]]
        if costs.size:
            minimums.append(float(costs.min()))
    if not minimums:
        raise ValueError("training groups contain no feasible noncanonical action costs")
    scale = heldout_state_count / len(training_states)
    total = sum(minimums) * scale
    return {str(level): float(level) * total for level in levels}


def gpu_budget_curve(
    state_arrays: Mapping[str, np.ndarray], *, cost_matrix: np.ndarray,
    state_ids: Sequence[str], budgets: Mapping[str, float], cost_lambda: float,
    scores_are_utility: bool = False,
) -> dict[str, Any]:
    state_indices = state_arrays["state_indices"]
    scores = state_arrays["score_matrix"][:, 1:].copy()
    local_cost = cost_matrix[state_indices, 1:]
    if not scores_are_utility:
        scores -= cost_lambda * local_cost
    scores[~state_arrays["feasible_matrix"][:, 1:]] = -np.inf
    best_action = np.argmax(scores, axis=1) + 1
    best_utility = scores[np.arange(scores.shape[0]), best_action - 1]
    identifiers = [state_ids[int(index)] for index in state_indices]
    order = sorted(range(len(identifiers)), key=lambda i: (-best_utility[i], identifiers[i]))
    oracle = state_arrays["oracle"]
    gains = state_arrays["gain_matrix"]
    high = set(int(value) for value in state_arrays["high_gain_local_indices"])
    total_oracle = max(float(oracle.sum()), 1e-12)
    result = {}
    for label, limit in budgets.items():
        selected, used = [], 0.0
        for index in order:
            if best_utility[index] <= 0:
                continue
            cost = float(local_cost[index, best_action[index] - 1])
            if used + cost <= float(limit) + 1e-12:
                selected.append(index)
                used += cost
        actions = best_action[selected] if selected else np.asarray([], dtype=np.int64)
        realized = gains[np.asarray(selected), actions] if selected else np.asarray([], dtype=np.float64)
        captured = float(oracle[selected].sum()) if selected else 0.0
        result[label] = {
            "budget_seconds": float(limit), "selected_action_count": len(selected),
            "oracle_positive_gain_captured": captured,
            "gain_capture_fraction": captured / total_oracle,
            "realized_true_gain": float(realized.sum()),
            "regret": float(oracle.sum() - realized.sum()),
            "selected_state_precision": float(np.mean(realized > 0)) if selected else 0.0,
            "high_gain_recall": len(high & set(selected)) / max(len(high), 1),
            "action_distribution": dict(sorted(Counter(f"A{int(v)}" for v in actions).items())),
            "gpu_cost_seconds": used,
            "negative_action_rate": float(np.mean(realized < 0)) if selected else 0.0,
        }
    return result


def pal_metrics(
    *, prediction: np.ndarray, target: np.ndarray, valid_pixels: np.ndarray,
    mismatch_pixels: np.ndarray, state_action: np.ndarray, action: np.ndarray,
    class_id: np.ndarray, gt_present: np.ndarray, conflict_type: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(row_mask, dtype=bool) & (valid_pixels > 0) & np.isfinite(target) & np.isfinite(prediction)
    if not mask.any():
        raise ValueError("PAL metrics require valid held-out atoms")
    weights = valid_pixels[mask].astype(np.float64)
    errors = prediction[mask] - target[mask]
    atom_mae = float(np.sum(weights * np.abs(errors)) / weights.sum())
    atom_rmse = float(np.sqrt(np.sum(weights * np.square(errors)) / weights.sum()))
    row_indices = np.flatnonzero(mask)
    order = np.argsort(state_action[row_indices], kind="mergesort")
    ordered_rows = row_indices[order]
    ordered_sa = state_action[ordered_rows]
    starts = np.r_[0, np.flatnonzero(ordered_sa[1:] != ordered_sa[:-1]) + 1]
    selected_sa = ordered_sa[starts]
    totals = np.add.reduceat(valid_pixels[ordered_rows].astype(np.float64), starts)
    pred = np.add.reduceat(
        prediction[ordered_rows] * valid_pixels[ordered_rows], starts,
    ) / totals
    truth = np.add.reduceat(
        mismatch_pixels[ordered_rows].astype(np.float64), starts,
    ) / totals
    first_rows = ordered_rows[starts]
    absolute = np.abs(pred - truth)
    square = np.square(pred - truth)
    result = {
        "atom_area_weighted_mae": atom_mae,
        "atom_area_weighted_rmse": atom_rmse,
        "state_action_count": int(selected_sa.size),
        "state_action_reconstructed_mae": float(absolute.mean()),
        "state_action_reconstructed_rmse": float(np.sqrt(square.mean())),
        "state_action_spearman": spearman(pred, truth),
        "per_action_mae": {}, "per_class_mae": {},
    }
    sa_actions_array = np.asarray(action[first_rows], dtype=np.int8)
    sa_classes_array = np.asarray(class_id[first_rows], dtype=np.int8)
    sa_presence_array = np.asarray(gt_present[first_rows], dtype=bool)
    for value in range(7):
        subset = sa_actions_array == value
        if subset.any():
            result["per_action_mae"][f"A{value}"] = float(absolute[subset].mean())
    for value in range(1, 21):
        subset = sa_classes_array == value
        if subset.any():
            result["per_class_mae"][str(value)] = float(absolute[subset].mean())
    result["positive_state_mae"] = float(absolute[sa_presence_array].mean()) if sa_presence_array.any() else None
    result["negative_state_mae"] = float(absolute[~sa_presence_array].mean()) if (~sa_presence_array).any() else None
    no_result = sa_actions_array == 0
    result["no_result_action_mae"] = float(absolute[no_result].mean()) if no_result.any() else None
    disagreement = mask & (conflict_type == 3)
    consensus = mask & (conflict_type == 1)
    result["cross_action_disagreement_atom_mae"] = (
        float(np.sum(valid_pixels[disagreement] * np.abs(prediction[disagreement] - target[disagreement])) / valid_pixels[disagreement].sum())
        if disagreement.any() else None
    )
    result["all_candidate_consensus_atom_mae"] = (
        float(np.sum(valid_pixels[consensus] * np.abs(prediction[consensus] - target[consensus])) / valid_pixels[consensus].sum())
        if consensus.any() else None
    )
    result["_state_action_arrays"] = {
        "state_action": selected_sa, "prediction": pred, "target": truth,
        "action": sa_actions_array, "class_id": sa_classes_array,
        "gt_present": sa_presence_array,
    }
    return result


def stable_random_scores(state_ids: Sequence[str], action: np.ndarray, state: np.ndarray, seed: int) -> np.ndarray:
    scores = np.zeros(action.shape[0], dtype=np.float64)
    for index, (state_index, action_index) in enumerate(zip(state, action, strict=True)):
        digest = hashlib.sha256(canonical_json_bytes({
            "namespace": "m06b-random-v1", "seed": seed,
            "state_id": state_ids[int(state_index)], "action": int(action_index),
        })).digest()
        scores[index] = int.from_bytes(digest[:8], "big") / float(2**64)
    return scores


def group_bootstrap(
    group_ids: Sequence[str], metric: Callable[[np.ndarray], float], *,
    seed: int = 13, replicates: int = 1000,
) -> dict[str, float]:
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    if unique.size == 0:
        raise ValueError("bootstrap requires groups")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == value) for value in sampled])
        estimates.append(float(metric(indices)))
    return {
        "replicates": replicates, "unit_count": int(unique.size),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "median": float(np.quantile(estimates, 0.5)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def strip_internal_arrays(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}
