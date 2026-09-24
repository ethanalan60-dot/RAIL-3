"""Fit-only diagnostics for residual accuracy and counterfactual utility.

The functions in this module operate only on already materialized out-of-fold
predictions and frozen fit targets.  They do not load checkpoints, fit models,
or access protected splits.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.models.m06b.metrics import spearman
from rail3.models.m06e.data import TensorBundle, sha256_file
from rail3.models.m06e.training import FOLDS, MODEL_IDS, SEEDS


ACTION_CODES = ("STOP", "A3", "A4", "A5", "A6")
DIAGNOSTIC_LABELS = (
    "RELATIVE_RESIDUAL_MISALIGNMENT",
    "COMMON_MODE_STATE_BIAS",
    "SPARSE_BENEFIT_SELECTION_FAILURE",
    "COST_DOMINATED_STOP_BIAS",
    "SMALL_ACTION_MARGIN_REGIME",
    "PIDR_OBJECTIVE_DEGENERACY",
    "DATA_SCALE_SATURATION",
    "MIXED_OR_UNRESOLVED",
)


def validate_exact_oof_indices(
    observed: np.ndarray, expected: np.ndarray, *, kind: str,
) -> None:
    observed = np.asarray(observed, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"M06-F {kind} prediction is not exact outer-held-out coverage")


@dataclass(frozen=True)
class ScaleEvidence:
    """Identity-checked, seed-averaged formal OOF evidence for one scale."""

    scale: str
    scope: Mapping[str, Any]
    prediction: Mapping[str, np.ndarray]
    seed_prediction: Mapping[str, Mapping[int, np.ndarray]]
    atom_prediction: Mapping[str, np.ndarray]
    cost_norm: np.ndarray
    run_ids: Mapping[str, tuple[str, ...]]
    reports: tuple[Mapping[str, Any], ...]


def load_scale_evidence(
    *, bundle_root: Path, fold_manifest_path: Path, run_root: Path,
    expected_git_commit: str,
) -> ScaleEvidence:
    """Load only complete outer-held-out predictions and fail on identity drift."""

    bundle = TensorBundle(bundle_root)
    verify_zero_protected_access((bundle.manifest,))
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if not all(fold_manifest.get("checks", {}).values()):
        raise RuntimeError("M06-F fold manifest checks are not all PASS")
    scale = str(fold_manifest["scale"])
    group_to_fold = {
        str(item["image_group_id"]): str(item["fold_id"])
        for item in fold_manifest["groups"]
    }
    all_groups = np.asarray(bundle.manifest["group_ids"])
    state_global = np.flatnonzero(np.asarray([value in group_to_fold for value in all_groups]))
    lengths_all = bundle.array("state_lengths")
    offsets = np.r_[0, np.cumsum(lengths_all)]
    atom_global = np.concatenate([
        np.arange(offsets[index], offsets[index + 1]) for index in state_global
    ])
    global_to_local_state = {int(value): index for index, value in enumerate(state_global)}
    global_to_local_atom = {int(value): index for index, value in enumerate(atom_global)}
    groups = all_groups[state_global]
    folds = np.asarray([group_to_fold[str(value)] for value in groups])
    scope: dict[str, Any] = {
        "state_global": state_global,
        "atom_global": atom_global,
        "groups": groups,
        "folds": folds,
        "state_ids": [bundle.manifest["state_ids"][index] for index in state_global],
        "target": bundle.array("state_target")[state_global],
        "cost": bundle.array("cost_seconds")[state_global],
        "base_cost": bundle.array("base_cost_seconds")[state_global],
        "a0_cost": bundle.array("a0_cost_seconds")[state_global],
        "class_id": bundle.array("class_id")[state_global],
        "gt_present": bundle.array("gt_present")[state_global],
        "a0_no_result": bundle.array("a0_no_result")[state_global],
        "state_lengths": lengths_all[state_global],
        "atom_x": bundle.array("atom_x")[atom_global],
        "atom_target": bundle.array("atom_target")[atom_global],
        "atom_weights": bundle.array("atom_weights")[atom_global],
    }
    inventory: dict[tuple[str, str, int], tuple[Mapping[str, Any], Path]] = {}
    reports: list[Mapping[str, Any]] = []
    for path in sorted((run_root / scale).rglob("run-manifest.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (str(report["model_id"]), str(report["outer_fold"]), int(report["seed"]))
        if key in inventory:
            raise RuntimeError(f"M06-F duplicate formal run identity: {key}")
        if (
            report.get("status") != "PASS"
            or report.get("scale") != scale
            or report.get("git_commit") != expected_git_commit
            or bool(report.get("validation_row_access", False))
        ):
            raise RuntimeError(f"M06-F run contract drift: {path}")
        verify_zero_protected_access((report,))
        prediction_path = Path(str(report["prediction"]["path"]))
        if (
            not prediction_path.is_file()
            or prediction_path.stat().st_size != int(report["prediction"]["bytes"])
            or sha256_file(prediction_path) != report["prediction"]["sha256"]
        ):
            raise RuntimeError(f"M06-F prediction identity drift: {prediction_path}")
        inventory[key] = report, prediction_path
        reports.append(report)
    expected = {
        (model, fold, seed) for model in MODEL_IDS for fold in FOLDS for seed in SEEDS
    }
    if set(inventory) != expected:
        raise RuntimeError(
            f"M06-F formal run inventory mismatch: missing={sorted(expected-set(inventory))}, "
            f"extra={sorted(set(inventory)-expected)}"
        )

    predictions: dict[str, np.ndarray] = {}
    seed_predictions: dict[str, dict[int, np.ndarray]] = {}
    atom_predictions: dict[str, np.ndarray] = {}
    run_ids: dict[str, tuple[str, ...]] = {}
    cost_norm = np.full_like(scope["cost"], np.nan, dtype=np.float64)
    atomic_models = {"R1_ATOMIC_RESIDUAL", "R2_ATOMIC_PIDR", "P1_LINEAR_PAL"}
    for model in MODEL_IDS:
        seed_state = {
            seed: np.full_like(scope["target"], np.nan, dtype=np.float64) for seed in SEEDS
        }
        seed_atom = {
            seed: np.full_like(scope["atom_target"], np.nan, dtype=np.float64)
            for seed in SEEDS
        } if model in atomic_models else {}
        model_run_ids: list[str] = []
        for fold in FOLDS:
            expected_state = state_global[folds == fold]
            expected_atom = np.concatenate([
                np.arange(offsets[index], offsets[index + 1]) for index in expected_state
            ])
            fold_medians = []
            for seed in SEEDS:
                report, prediction_path = inventory[(model, fold, seed)]
                model_run_ids.append(f"{scale}/{model}/{fold}/seed_{seed}")
                fold_medians.append(float(report["cost_median_seconds"]))
                with np.load(prediction_path, allow_pickle=False) as payload:
                    state_index = np.asarray(payload["state_index"], dtype=np.int64)
                    validate_exact_oof_indices(
                        state_index, expected_state, kind="state",
                    )
                    local_state = np.asarray([global_to_local_state[int(value)] for value in state_index])
                    seed_state[seed][local_state] = payload["state_prediction"]
                    if model in atomic_models:
                        atom_index = np.asarray(payload["atom_index"], dtype=np.int64)
                        validate_exact_oof_indices(
                            atom_index, expected_atom, kind="atomic",
                        )
                        local_atom = np.asarray([global_to_local_atom[int(value)] for value in atom_index])
                        seed_atom[seed][local_atom] = payload["atom_prediction"]
            if not np.allclose(fold_medians, fold_medians[0], rtol=0.0, atol=1e-8):
                raise RuntimeError("M06-F outer-fold cost median changed across seeds")
            use = folds == fold
            cost_norm[use] = scope["cost"][use] / fold_medians[0]
            cost_norm[use, 0] = 0.0
        if any(not np.isfinite(item).all() for item in seed_state.values()):
            raise RuntimeError(f"M06-F {model} OOF state coverage is incomplete")
        seed_predictions[model] = seed_state
        predictions[model] = np.mean(list(seed_state.values()), axis=0)
        if model in atomic_models:
            if any(not np.isfinite(item).all() for item in seed_atom.values()):
                raise RuntimeError(f"M06-F {model} OOF atom coverage is incomplete")
            atom_predictions[model] = np.mean(list(seed_atom.values()), axis=0)
        run_ids[model] = tuple(sorted(model_run_ids))
    if not np.isfinite(cost_norm).all():
        raise RuntimeError("M06-F normalized cost OOF coverage is incomplete")
    return ScaleEvidence(
        scale=scale, scope=scope, prediction=predictions,
        seed_prediction=seed_predictions, atom_prediction=atom_predictions,
        cost_norm=cost_norm, run_ids=run_ids, reports=tuple(reports),
    )


def _as_aligned(
    prediction: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 5:
        raise ValueError("M06-F prediction and target must be aligned Nx5 matrices")
    defined = np.isfinite(prediction) & np.isfinite(target)
    return prediction, target, defined


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ("p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    return {
        key: float(np.quantile(values, probability))
        for key, probability in (
            ("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
            ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99),
        )
    }


def absolute_error_summary(
    prediction: np.ndarray, target: np.ndarray, *, state_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction, target, defined = _as_aligned(prediction, target)
    if state_mask is not None:
        mask = np.asarray(state_mask, dtype=bool)
        if mask.shape != target.shape[:1]:
            raise ValueError("M06-F absolute-error state mask is not aligned")
        defined &= mask[:, None]
    if not defined.any():
        raise ValueError("M06-F absolute-error slice has no defined observations")
    error = prediction[defined] - target[defined]
    result: dict[str, Any] = {
        "count": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_error": float(np.median(error)),
    }
    result.update({f"abs_{key}": value for key, value in quantiles(np.abs(error)).items()})
    return result


def gain_matrices(
    prediction: np.ndarray, target: np.ndarray, *, prediction_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction, target, defined = _as_aligned(prediction, target)
    pair_defined = defined & defined[:, :1]
    true_gain = target[:, :1] - target
    if prediction_kind == "residual":
        predicted_gain = prediction[:, :1] - prediction
    elif prediction_kind == "gain":
        predicted_gain = prediction.copy()
        predicted_gain[:, 0] = 0.0
    else:
        raise ValueError("M06-F prediction kind must be residual or gain")
    true_gain = np.where(pair_defined, true_gain, np.nan)
    predicted_gain = np.where(pair_defined, predicted_gain, np.nan)
    return predicted_gain, true_gain, pair_defined


def relative_residual_summary(
    prediction: np.ndarray, target: np.ndarray, *, prediction_kind: str,
) -> dict[str, Any]:
    predicted_gain, true_gain, defined = gain_matrices(
        prediction, target, prediction_kind=prediction_kind,
    )
    use = defined[:, 1:]
    pred = predicted_gain[:, 1:][use]
    truth = true_gain[:, 1:][use]
    if truth.size == 0:
        raise ValueError("M06-F relative residual has no defined non-STOP action")
    error = pred - truth
    sign_accuracy = np.mean((pred > 0.0) == (truth > 0.0))

    informative_pairs = correct_pairs = 0
    best_correct = stop_best_correct = evaluable_states = 0
    for index in range(true_gain.shape[0]):
        valid = np.flatnonzero(defined[index])
        if valid.size < 2 or 0 not in valid:
            continue
        evaluable_states += 1
        true_values = true_gain[index, valid]
        pred_values = predicted_gain[index, valid]
        if int(valid[np.argmax(true_values)]) == int(valid[np.argmax(pred_values)]):
            best_correct += 1
        true_benefit = bool(np.nanmax(true_gain[index, 1:]) > 0.0)
        pred_benefit = bool(np.nanmax(predicted_gain[index, 1:]) > 0.0)
        stop_best_correct += int(true_benefit == pred_benefit)
        for left in range(valid.size):
            for right in range(left + 1, valid.size):
                truth_difference = true_values[left] - true_values[right]
                if abs(truth_difference) <= 1e-12:
                    continue
                informative_pairs += 1
                predicted_difference = pred_values[left] - pred_values[right]
                correct_pairs += int(np.sign(predicted_difference) == np.sign(truth_difference))
    return {
        "count": int(truth.size),
        "relative_residual_mae": float(np.mean(np.abs(error))),
        "relative_residual_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "gain_mae": float(np.mean(np.abs(error))),
        "gain_spearman": spearman(pred, truth),
        "beneficial_action_sign_accuracy": float(sign_accuracy),
        "pairwise_action_ordering_accuracy": (
            float(correct_pairs / informative_pairs) if informative_pairs else None
        ),
        "informative_pair_count": informative_pairs,
        "best_action_accuracy": float(best_correct / evaluable_states),
        "stop_vs_best_action_accuracy": float(stop_best_correct / evaluable_states),
        "evaluable_state_count": evaluable_states,
    }


def common_mode_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction, target, defined = _as_aligned(prediction, target)
    error = prediction - target
    state_count = defined.sum(axis=1)
    valid_states = state_count > 0
    bias = np.full(target.shape[0], np.nan, dtype=np.float64)
    bias[valid_states] = np.nansum(np.where(defined, error, np.nan), axis=1)[valid_states] / state_count[valid_states]
    centered = error - bias[:, None]
    total = error[defined]
    repeated_bias = np.broadcast_to(bias[:, None], target.shape)[defined]
    centered_values = centered[defined]
    total_variance = float(np.var(total))
    common_variance = float(np.var(repeated_bias))
    centered_variance = float(np.var(centered_values))
    return {
        "state_count": int(valid_states.sum()),
        "observation_count": int(defined.sum()),
        "total_error_variance": total_variance,
        "common_mode_bias_variance": common_variance,
        "centered_action_error_variance": centered_variance,
        "common_mode_variance_fraction": common_variance / max(total_variance, 1e-15),
        "absolute_mae": float(np.mean(np.abs(total))),
        "oracle_centered_mae": float(np.mean(np.abs(centered_values))),
        "mean_state_bias": float(np.nanmean(bias)),
        "diagnostic_status": "OFFLINE_NONDEPLOYABLE_DIAGNOSTIC",
    }


def stop_calibration_summary(
    prediction: np.ndarray, target: np.ndarray, *, bins: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction, target, defined = _as_aligned(prediction, target)
    stop_defined = defined[:, 0]
    if not stop_defined.any():
        raise ValueError("M06-F STOP calibration has no defined STOP rows")
    stop_error = prediction[stop_defined, 0] - target[stop_defined, 0]
    predicted_gain, true_gain, pair_defined = gain_matrices(
        prediction, target, prediction_kind="residual",
    )
    valid_nonstop = pair_defined[:, 1:]
    non_stop_error = (prediction[:, 1:] - target[:, 1:])[valid_nonstop]
    true_best = np.max(np.where(valid_nonstop, true_gain[:, 1:], -np.inf), axis=1)
    predicted_best = np.max(
        np.where(valid_nonstop, predicted_gain[:, 1:], -np.inf), axis=1,
    )
    true_benefit = true_best > 0.0
    pred_benefit = predicted_best > 0.0
    evaluable = stop_defined & valid_nonstop.any(axis=1)
    summary = {
        "state_count": int(evaluable.sum()),
        "stop_bias": float(np.mean(stop_error)),
        "stop_mae": float(np.mean(np.abs(stop_error))),
        "non_stop_residual_bias": float(np.mean(non_stop_error)),
        "predicted_beneficial_rate": float(np.mean(pred_benefit[evaluable])),
        "true_beneficial_rate": float(np.mean(true_benefit[evaluable])),
        "false_stop_rate": float(np.mean((~pred_benefit & true_benefit)[evaluable])),
        "false_intervention_rate": float(np.mean((pred_benefit & ~true_benefit)[evaluable])),
    }
    ordered = np.flatnonzero(stop_defined)[np.argsort(prediction[stop_defined, 0], kind="mergesort")]
    curves = []
    for bin_index, indices in enumerate(np.array_split(ordered, bins)):
        if indices.size == 0:
            continue
        curves.append({
            "bin": bin_index,
            "count": int(indices.size),
            "mean_prediction": float(np.mean(prediction[indices, 0])),
            "mean_truth": float(np.mean(target[indices, 0])),
            "mean_bias": float(np.mean(prediction[indices, 0] - target[indices, 0])),
        })
    return summary, curves


def sparse_policy_decomposition(
    prediction: np.ndarray, target: np.ndarray, *, prediction_kind: str,
    state_ids: Sequence[str], budgets: Sequence[float],
) -> list[dict[str, Any]]:
    predicted_gain, true_gain, defined = gain_matrices(
        prediction, target, prediction_kind=prediction_kind,
    )
    evaluable = defined[:, 0] & defined[:, 1:].any(axis=1)
    predicted_gain = predicted_gain[evaluable]
    true_gain = true_gain[evaluable]
    defined = defined[evaluable]
    local_ids = np.asarray([state_ids[index] for index in np.flatnonzero(evaluable)])
    predicted_gain = np.where(defined, predicted_gain, -np.inf)
    true_gain = np.where(defined, true_gain, -np.inf)
    predicted_gain[:, 0] = 0.0
    true_gain[:, 0] = 0.0
    model_action = np.argmax(predicted_gain, axis=1)
    oracle_action = np.argmax(true_gain, axis=1)
    rows = np.arange(model_action.size)
    model_score = np.max(predicted_gain[:, 1:], axis=1)
    oracle_score = np.maximum(0.0, true_gain[rows, oracle_action])
    id_order = np.argsort(local_ids, kind="mergesort")
    id_rank = np.empty(local_ids.size, dtype=np.int64)
    id_rank[id_order] = np.arange(local_ids.size)
    model_order = np.lexsort((id_rank, -model_score))
    oracle_order = np.lexsort((id_rank, -oracle_score))
    policies = {
        "D0_MODEL_SELECTOR_MODEL_ACTION": (model_order, model_action),
        "D1_ORACLE_SELECTOR_MODEL_ACTION": (oracle_order, model_action),
        "D2_MODEL_SELECTOR_ORACLE_ACTION": (model_order, oracle_action),
        "D3_ORACLE_SELECTOR_ORACLE_ACTION": (oracle_order, oracle_action),
    }
    result: list[dict[str, Any]] = []
    ideal = np.sort(oracle_score)[::-1]
    for budget in budgets:
        limit = max(1, math.ceil(float(budget) * len(local_ids)))
        oracle_at_budget = float(ideal[:limit].sum())
        for policy, (order, actions) in policies.items():
            selected = order[:limit]
            selected_actions = actions[selected]
            realized = true_gain[selected, selected_actions]
            non_stop = selected_actions != 0
            true_positive = realized > 0.0
            negative = realized < 0.0
            item = {
                "policy": policy,
                "diagnostic_status": (
                    "DEPLOYABLE_FORM" if policy == "D0_MODEL_SELECTOR_MODEL_ACTION"
                    else "ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE"
                ),
                "budget": float(budget),
                "eligible_states": int(len(local_ids)),
                "maximum_action_count": limit,
                "selected_states": int(selected.size),
                "selected_stop_states": int((~non_stop).sum()),
                "selected_non_stop_states": int(non_stop.sum()),
                "selected_true_positive_states": int(true_positive.sum()),
                "selected_negative_gain_states": int(negative.sum()),
                "total_oracle_positive_gain": oracle_at_budget,
                "selected_realized_gain": float(realized.sum()),
                "gain_capture": float(realized.sum() / max(oracle_at_budget, 1e-15)),
                "regret": float(oracle_at_budget - realized.sum()),
                "negative_action_rate": float(np.mean(negative[non_stop])) if non_stop.any() else 0.0,
                "predicted_gain_p50": float(np.median(model_score[selected])),
                "predicted_gain_p90": float(np.quantile(model_score[selected], 0.90)),
                "true_gain_p50": float(np.median(oracle_score[selected])),
                "true_gain_p90": float(np.quantile(oracle_score[selected], 0.90)),
            }
            item.update({
                f"selected_predicted_gain_{key}": value
                for key, value in quantiles(model_score[selected]).items()
            })
            item.update({
                f"selected_true_best_gain_{key}": value
                for key, value in quantiles(oracle_score[selected]).items()
            })
            item.update({
                f"eligible_predicted_gain_{key}": value
                for key, value in quantiles(model_score).items()
            })
            item.update({
                f"eligible_true_best_gain_{key}": value
                for key, value in quantiles(oracle_score).items()
            })
            result.append(item)
    return result


def decision_margin_rows(
    target: np.ndarray, cost_norm: np.ndarray, *, lambdas: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    target = np.asarray(target, dtype=np.float64)
    cost_norm = np.asarray(cost_norm, dtype=np.float64)
    if target.shape != cost_norm.shape or target.ndim != 2 or target.shape[1] != 5:
        raise ValueError("M06-F margin inputs must be aligned Nx5 matrices")
    defined = np.isfinite(target)
    evaluable = defined[:, 0] & defined[:, 1:].any(axis=1)
    rows: list[dict[str, Any]] = []
    for value in lambdas:
        cost = np.where(defined[evaluable], target[evaluable] + float(value) * cost_norm[evaluable], np.inf)
        stop = cost[:, 0]
        best_nonstop = np.min(cost[:, 1:], axis=1)
        margin_stop = stop - best_nonstop
        sorted_cost = np.sort(cost, axis=1)
        margin_top2 = sorted_cost[:, 1] - sorted_cost[:, 0]
        for name, margins in (("STOP_VS_BEST", margin_stop), ("BEST_VS_SECOND", margin_top2)):
            item: dict[str, Any] = {
                "lambda": float(value), "margin_type": name,
                "state_count": int(margins.size),
                "beneficial_state_fraction": float(np.mean(margin_stop > 0.0)),
                "mean_absolute_margin": float(np.mean(np.abs(margins))),
            }
            item.update({f"abs_{key}": result for key, result in quantiles(np.abs(margins)).items()})
            for threshold in thresholds:
                item[f"fraction_within_{threshold:g}"] = float(np.mean(np.abs(margins) <= float(threshold)))
            rows.append(item)
    return rows


def pidr_lambda_rows(
    prediction: np.ndarray, target: np.ndarray, cost_norm: np.ndarray, *,
    lambdas: Sequence[float], tau: float,
) -> list[dict[str, Any]]:
    prediction, target, defined = _as_aligned(prediction, target)
    cost_norm = np.asarray(cost_norm, dtype=np.float64)
    if cost_norm.shape != target.shape:
        raise ValueError("M06-F PIDR cost matrix is not aligned")
    evaluable = defined[:, 0] & (defined.sum(axis=1) >= 2)
    prediction, target, defined, cost_norm = (
        value[evaluable] for value in (prediction, target, defined, cost_norm)
    )
    rows: list[dict[str, Any]] = []
    for value in lambdas:
        pred_cost = np.where(defined, prediction + float(value) * cost_norm, np.inf)
        true_cost = np.where(defined, target + float(value) * cost_norm, np.inf)
        logits = np.where(defined, -pred_cost / float(tau), -np.inf)
        shift = np.max(logits, axis=1, keepdims=True)
        probability = np.exp(logits - shift)
        probability /= probability.sum(axis=1, keepdims=True)
        pred_action = np.argmin(pred_cost, axis=1)
        oracle_action = np.argmin(true_cost, axis=1)
        sorted_true = np.sort(true_cost, axis=1)
        true_margin = sorted_true[:, 1] - sorted_true[:, 0]
        log_probability = np.zeros_like(probability)
        positive_probability = probability > 0.0
        log_probability[positive_probability] = np.log(probability[positive_probability])
        entropy = -np.sum(probability * log_probability, axis=1)
        valid_count = defined.sum(axis=1)
        normalized_entropy = entropy / np.log(valid_count)
        oracle_cost = np.min(true_cost, axis=1)
        expected_cost = np.sum(probability * np.where(defined, true_cost, 0.0), axis=1)
        contribution = expected_cost - oracle_cost
        beneficial = oracle_action != 0
        total_contribution = float(contribution.sum())
        item: dict[str, Any] = {
            "lambda": float(value), "tau": float(tau),
            "state_count": int(target.shape[0]),
            "oracle_stop_fraction": float(np.mean(oracle_action == 0)),
            "predicted_stop_fraction": float(np.mean(pred_action == 0)),
            "mean_soft_policy_entropy": float(np.mean(entropy)),
            "mean_normalized_soft_policy_entropy": float(np.mean(normalized_entropy)),
            "mean_true_best_vs_second_margin": float(np.mean(true_margin)),
            "near_tie_fraction_margin_le_tau": float(np.mean(true_margin <= float(tau))),
            "mean_pidr_loss_contribution": float(np.mean(contribution)),
            "non_beneficial_pidr_contribution_fraction": (
                float(contribution[~beneficial].sum() / total_contribution)
                if total_contribution > 1e-15 else 0.0
            ),
        }
        for action_index, action in enumerate(ACTION_CODES):
            item[f"oracle_{action}_fraction"] = float(np.mean(oracle_action == action_index))
            item[f"predicted_{action}_fraction"] = float(np.mean(pred_action == action_index))
        rows.append(item)
    return rows


def verify_zero_protected_access(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("validation40", "calibration30", "pilot_test30", "official_voc_val")
    totals = {key: 0 for key in keys}
    for item in items:
        access = item.get("heldout_access", {})
        for key in keys:
            totals[key] += int(access.get(key, 0))
    if any(totals.values()):
        raise RuntimeError(f"M06-F protected access is nonzero: {totals}")
    return totals
