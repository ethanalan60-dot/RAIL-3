"""Fit-only estimators for the frozen TMLR reviewer-revision protocol.

The independent sampling unit is an image group.  Class-conditioned states
remain attached to their image group throughout every bootstrap replicate.
This module is deliberately unaware of protected-data readers and evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes


ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
EPSILON = 1e-15


def committed_source_identity(
    repository: Path, relative_paths: Sequence[Path],
) -> dict[str, object]:
    """Bind a formal artifact writer to committed, clean implementation files."""

    repository = repository.resolve()
    logical = tuple(path.as_posix() for path in relative_paths)
    if not logical or len(set(logical)) != len(logical):
        raise ValueError("implementation source paths are empty or duplicated")
    for value in logical:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("implementation source path is not repository-relative")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", value], cwd=repository,
            capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"formal implementation source is not committed: {value}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *logical], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("formal implementation source paths are not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    source_sha256: dict[str, str] = {}
    for value in logical:
        current = repository / value
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{value}"], cwd=repository,
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(f"formal implementation commit/file drift: {value}")
        source_sha256[value] = digest
    return {
        "commit": commit,
        "source_sha256": source_sha256,
        "combined_sha256": hashlib.sha256(
            canonical_json_bytes(source_sha256)
        ).hexdigest(),
    }


@dataclass(frozen=True)
class EvaluationSurface:
    """One aligned OOF prediction surface and its frozen fit targets."""

    model_id: str
    prediction_kind: str
    prediction: np.ndarray
    target: np.ndarray
    feasible: np.ndarray
    groups: np.ndarray
    state_ids: tuple[str, ...]

    def validated(self) -> "EvaluationSurface":
        prediction = np.asarray(self.prediction)
        target = np.asarray(self.target)
        feasible = np.asarray(self.feasible)
        groups = np.asarray(self.groups)
        if self.prediction_kind not in {
            "residual", "direct_gain", "legacy_direct_gain_learned_stop",
        }:
            raise ValueError(f"unknown prediction kind: {self.prediction_kind}")
        if prediction.ndim != 2 or prediction.shape[1] != len(ACTIONS):
            raise ValueError("reviewer-revision prediction must be N x 5")
        if target.shape != prediction.shape or feasible.shape != prediction.shape:
            raise ValueError("reviewer-revision prediction/target/feasibility drift")
        if groups.shape != (len(prediction),) or len(self.state_ids) != len(prediction):
            raise ValueError("reviewer-revision state identity arrays are not aligned")
        if not np.isfinite(prediction).all():
            raise ValueError("reviewer-revision prediction contains non-finite values")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("reviewer-revision state IDs are not unique")
        return self

    @property
    def residual_equivalent(self) -> np.ndarray:
        """Return lower-is-better scores in residual coordinates.

        A direct-gain model predicts STOP residual minus action residual, so its
        action ordering and STOP-relative error are represented by ``-gain``.
        This conversion does not make absolute residual MAE meaningful.
        """

        value = np.asarray(self.prediction, dtype=np.float64)
        if self.prediction_kind == "residual":
            return value
        if self.prediction_kind == "legacy_direct_gain_learned_stop":
            # Historical compatibility only.  The frozen fit/Locked40 Q2
            # evaluator treated the learned STOP head as an ordinary gain
            # coordinate.  New scientific results must use ``direct_gain``.
            return -value
        # Direct gain is defined relative to STOP: g(a)=R(STOP)-R(a).
        # The STOP coordinate is therefore the fixed reference zero, not a
        # learned action head whose numerical drift may move every threshold.
        residual = -value.copy()
        residual[:, 0] = 0.0
        return residual


@dataclass(frozen=True)
class BudgetStateArrays:
    """State-level sufficient arrays for exact weighted forced-K evaluation."""

    score: np.ndarray
    realized_gain: np.ndarray
    oracle_gain: np.ndarray
    group_index: np.ndarray
    state_ids: np.ndarray


def sorted_group_universe(groups: Sequence[str]) -> np.ndarray:
    result = np.asarray(sorted(set(str(item) for item in groups)), dtype=str)
    if result.size == 0:
        raise ValueError("group universe is empty")
    return result


def group_index(groups: Sequence[str], universe: Sequence[str]) -> np.ndarray:
    ordered = tuple(str(item) for item in universe)
    if len(set(ordered)) != len(ordered):
        raise ValueError("group universe contains duplicates")
    lookup = {value: index for index, value in enumerate(ordered)}
    try:
        result = np.asarray([lookup[str(item)] for item in groups], dtype=np.int32)
    except KeyError as error:
        raise ValueError(f"state group is absent from the bootstrap universe: {error}") from error
    return result


def bootstrap_group_counts(
    group_count: int, *, replicates: int, seed: int,
) -> np.ndarray:
    """Draw one deterministic nonparametric group-bootstrap count matrix."""

    if group_count <= 0 or replicates <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, group_count, size=(replicates, group_count))
    counts = np.stack([
        np.bincount(row, minlength=group_count) for row in draws
    ]).astype(np.int32)
    if not np.all(counts.sum(axis=1) == group_count):
        raise RuntimeError("group bootstrap replicate size drift")
    return counts


def _sum_by_group(
    values: np.ndarray, indices: np.ndarray, group_count: int,
) -> np.ndarray:
    return np.bincount(
        np.asarray(indices, dtype=np.int64),
        weights=np.asarray(values, dtype=np.float64),
        minlength=group_count,
    ).astype(np.float64)


def grouped_metric_sufficient_statistics(
    surface: EvaluationSurface, *, group_universe: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return additive group-level sufficient statistics for all-state metrics."""

    surface.validated()
    universe = (
        sorted_group_universe(surface.groups)
        if group_universe is None else np.asarray(group_universe, dtype=str)
    )
    state_group = group_index(surface.groups, universe)
    group_count = len(universe)
    prediction = surface.residual_equivalent
    target = np.asarray(surface.target, dtype=np.float64)
    defined = np.asarray(surface.feasible, dtype=bool) & np.isfinite(target)
    evaluable = defined[:, 0] & defined[:, 1:].any(axis=1)

    result: dict[str, np.ndarray] = {}
    absolute_error = np.where(defined, np.abs(prediction - target), 0.0)
    result["absolute_error_sum"] = _sum_by_group(
        absolute_error.sum(axis=1), state_group, group_count,
    )
    result["absolute_error_count"] = _sum_by_group(
        defined.sum(axis=1), state_group, group_count,
    )

    pair_defined = evaluable[:, None] & defined[:, :1] & defined[:, 1:]
    predicted_delta = prediction[:, 1:] - prediction[:, :1]
    true_delta = target[:, 1:] - target[:, :1]
    relative_error = predicted_delta - true_delta
    result["relative_abs_error_sum"] = _sum_by_group(
        np.where(pair_defined, np.abs(relative_error), 0.0).sum(axis=1),
        state_group, group_count,
    )
    result["relative_pair_count"] = _sum_by_group(
        pair_defined.sum(axis=1), state_group, group_count,
    )
    result["beneficial_sign_correct"] = _sum_by_group(
        np.where(
            pair_defined,
            (predicted_delta < 0.0) == (true_delta < 0.0),
            False,
        ).sum(axis=1),
        state_group, group_count,
    )

    left, right = np.triu_indices(len(ACTIONS), k=1)
    pairwise_defined = (
        evaluable[:, None] & defined[:, left] & defined[:, right]
    )
    true_difference = target[:, left] - target[:, right]
    predicted_difference = prediction[:, left] - prediction[:, right]
    informative = pairwise_defined & (np.abs(true_difference) > 1e-12)
    result["action_order_correct"] = _sum_by_group(
        np.where(
            informative,
            np.sign(predicted_difference) == np.sign(true_difference),
            False,
        ).sum(axis=1),
        state_group, group_count,
    )
    result["action_order_count"] = _sum_by_group(
        informative.sum(axis=1), state_group, group_count,
    )

    masked_prediction = np.where(defined[evaluable], prediction[evaluable], np.inf)
    masked_truth = np.where(defined[evaluable], target[evaluable], np.inf)
    selected = masked_prediction.argmin(axis=1)
    oracle = masked_truth.argmin(axis=1)
    rows = np.arange(len(selected))
    selected_truth = masked_truth[rows, selected]
    oracle_truth = masked_truth[rows, oracle]
    realized_gain = masked_truth[:, 0] - selected_truth
    oracle_gain = np.maximum(0.0, masked_truth[:, 0] - oracle_truth)
    regret = selected_truth - oracle_truth
    local_groups = state_group[evaluable]
    result["evaluable_state_count"] = _sum_by_group(
        np.ones(len(selected)), local_groups, group_count,
    )
    result["regret_sum"] = _sum_by_group(regret, local_groups, group_count)
    result["realized_gain_sum"] = _sum_by_group(
        realized_gain, local_groups, group_count,
    )
    result["oracle_gain_sum"] = _sum_by_group(
        oracle_gain, local_groups, group_count,
    )
    result["predicted_beneficial_count"] = _sum_by_group(
        selected != 0, local_groups, group_count,
    )
    result["negative_intervention_count"] = _sum_by_group(
        (selected != 0) & (realized_gain < 0.0), local_groups, group_count,
    )
    return universe, result


def _ratio(numerator: np.ndarray, denominator: np.ndarray, *, name: str) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if np.any(denominator <= 0):
        raise RuntimeError(f"{name} has a zero-denominator bootstrap replicate")
    return numerator / denominator


def metrics_from_group_counts(
    sufficient: Mapping[str, np.ndarray], counts: np.ndarray,
    *, prediction_kind: str,
) -> dict[str, np.ndarray]:
    """Evaluate registered all-state estimators for weighted image groups."""

    weights = np.asarray(counts, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    required = {
        "absolute_error_sum", "absolute_error_count", "relative_abs_error_sum",
        "relative_pair_count", "beneficial_sign_correct", "action_order_correct",
        "action_order_count", "evaluable_state_count", "regret_sum",
        "realized_gain_sum", "oracle_gain_sum", "predicted_beneficial_count",
        "negative_intervention_count",
    }
    if not required <= set(sufficient):
        raise ValueError(f"missing grouped sufficient statistics: {sorted(required - set(sufficient))}")
    if any(np.asarray(sufficient[key]).shape != (weights.shape[1],) for key in required):
        raise ValueError("group count matrix and sufficient statistics are not aligned")

    total = {key: weights @ np.asarray(sufficient[key], dtype=np.float64) for key in required}
    state_count = total["evaluable_state_count"]
    oracle = total["oracle_gain_sum"]
    result = {
        "drre": _ratio(
            total["relative_abs_error_sum"], total["relative_pair_count"], name="DRRE",
        ),
        "beneficial_sign_accuracy": _ratio(
            total["beneficial_sign_correct"], total["relative_pair_count"],
            name="beneficial-sign accuracy",
        ),
        "within_state_action_order_accuracy": _ratio(
            total["action_order_correct"], total["action_order_count"],
            name="within-state action-order accuracy",
        ),
        "normalized_regret": total["regret_sum"] / np.maximum(oracle, EPSILON),
        "raw_realized_gain": total["realized_gain_sum"],
        "predicted_beneficial_rate": _ratio(
            total["predicted_beneficial_count"], state_count,
            name="predicted-beneficial rate",
        ),
        "negative_intervention_rate": _ratio(
            total["negative_intervention_count"], state_count,
            name="negative-intervention rate",
        ),
        "evaluable_state_count": state_count,
        "oracle_gain": oracle,
    }
    if prediction_kind == "residual":
        result["absolute_residual_mae"] = _ratio(
            total["absolute_error_sum"], total["absolute_error_count"],
            name="absolute residual MAE",
        )
    elif prediction_kind not in {
        "direct_gain", "legacy_direct_gain_learned_stop",
    }:
        raise ValueError(f"unknown prediction kind: {prediction_kind}")
    return result


def budget_state_arrays(
    surface: EvaluationSurface, *, group_universe: Sequence[str] | None = None,
) -> tuple[np.ndarray, BudgetStateArrays]:
    """Prepare exact state-level arrays under frozen action and tie semantics."""

    surface.validated()
    universe = (
        sorted_group_universe(surface.groups)
        if group_universe is None else np.asarray(group_universe, dtype=str)
    )
    prediction = surface.residual_equivalent
    target = np.asarray(surface.target, dtype=np.float64)
    defined = np.asarray(surface.feasible, dtype=bool) & np.isfinite(target)
    evaluable = defined[:, 0] & defined[:, 1:].any(axis=1)
    masked_prediction = np.where(defined[evaluable], prediction[evaluable], np.inf)
    masked_truth = np.where(defined[evaluable], target[evaluable], np.inf)
    ids = np.asarray(surface.state_ids, dtype=str)[evaluable]
    state_group = group_index(np.asarray(surface.groups)[evaluable], universe)
    rows = np.arange(len(ids))
    # np.argmin keeps the frozen action-column order A3,A4,A5,A6 on ties.
    best_nonstop = masked_prediction[:, 1:].argmin(axis=1) + 1
    score = masked_prediction[:, 0] - masked_prediction[rows, best_nonstop]
    realized = masked_truth[:, 0] - masked_truth[rows, best_nonstop]
    oracle = np.maximum(0.0, masked_truth[:, 0] - masked_truth.min(axis=1))
    return universe, BudgetStateArrays(
        score=score,
        realized_gain=realized,
        oracle_gain=oracle,
        group_index=state_group,
        state_ids=ids,
    )


def _weighted_top_k_sum(
    counts: np.ndarray, *, sorted_group_index: np.ndarray,
    sorted_values: np.ndarray, limits: np.ndarray, chunk_size: int,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int32)
    limits = np.asarray(limits, dtype=np.int64)
    result = np.empty(len(counts), dtype=np.float64)
    for start in range(0, len(counts), chunk_size):
        stop = min(start + chunk_size, len(counts))
        multiplicity = counts[start:stop, sorted_group_index].astype(np.int64)
        cumulative = np.cumsum(multiplicity, axis=1)
        before = cumulative - multiplicity
        remaining = limits[start:stop, None] - before
        take = np.minimum(multiplicity, np.maximum(remaining, 0))
        if not np.array_equal(take.sum(axis=1), limits[start:stop]):
            raise RuntimeError("weighted forced-K selection did not fill its budget")
        result[start:stop] = take @ np.asarray(sorted_values, dtype=np.float64)
    return result


def forced_k_from_group_counts(
    arrays: BudgetStateArrays, counts: np.ndarray, *, budgets: Sequence[float],
    chunk_size: int = 64,
) -> dict[float, dict[str, np.ndarray]]:
    """Re-rank each bootstrap multiset and recompute its oracle denominator."""

    weights = np.asarray(counts, dtype=np.int32)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[1] <= int(np.max(arrays.group_index, initial=-1)):
        raise ValueError("forced-K group count matrix is not aligned")
    per_group_states = np.bincount(
        arrays.group_index, minlength=weights.shape[1],
    ).astype(np.int64)
    state_count = weights @ per_group_states
    if np.any(state_count <= 0):
        raise RuntimeError("forced-K replicate contains no evaluable state")

    id_order = np.argsort(arrays.state_ids, kind="mergesort")
    id_rank = np.empty(len(id_order), dtype=np.int64)
    id_rank[id_order] = np.arange(len(id_order))
    predicted_order = np.lexsort((id_rank, -arrays.score))
    # Oracle ties do not change a sum, but mergesort makes the implementation
    # deterministic for provenance and synthetic equivalence tests.
    oracle_order = np.argsort(-arrays.oracle_gain, kind="mergesort")
    output: dict[float, dict[str, np.ndarray]] = {}
    for budget in budgets:
        value = float(budget)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid forced-K budget: {value}")
        limit = np.maximum(1, np.ceil(value * state_count).astype(np.int64))
        realized = _weighted_top_k_sum(
            weights,
            sorted_group_index=arrays.group_index[predicted_order],
            sorted_values=arrays.realized_gain[predicted_order],
            limits=limit,
            chunk_size=chunk_size,
        )
        oracle = _weighted_top_k_sum(
            weights,
            sorted_group_index=arrays.group_index[oracle_order],
            sorted_values=arrays.oracle_gain[oracle_order],
            limits=limit,
            chunk_size=chunk_size,
        )
        negative = _weighted_top_k_sum(
            weights,
            sorted_group_index=arrays.group_index[predicted_order],
            sorted_values=(arrays.realized_gain[predicted_order] < 0.0).astype(float),
            limits=limit,
            chunk_size=chunk_size,
        )
        output[value] = {
            "cap": limit.astype(np.float64),
            "raw_realized_gain": realized,
            "oracle_gain": oracle,
            "gain_capture": realized / np.maximum(oracle, EPSILON),
            "negative_selected_rate": negative / limit,
        }
    return output


def percentile_interval(values: np.ndarray, *, lower: float, upper: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap interval requires a finite one-dimensional sample")
    return float(np.quantile(values, lower)), float(np.quantile(values, upper))


def binary_confusion_by_region(
    *, label_map: np.ndarray, region_count: int, valid: np.ndarray,
    truth: np.ndarray, prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute integer TP/FP/FN/TN counts for one cached binary action mask."""

    label_map = np.asarray(label_map)
    valid = np.asarray(valid, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if not (label_map.shape == valid.shape == truth.shape == prediction.shape):
        raise ValueError("confusion-count geometry is not aligned")
    if region_count <= 0 or label_map.min(initial=0) < 0 or label_map.max(initial=-1) >= region_count:
        raise ValueError("confusion-count region label is out of bounds")

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(label_map[mask], minlength=region_count).astype(np.int64)

    tp = count(valid & truth & prediction)
    fp = count(valid & ~truth & prediction)
    fn = count(valid & truth & ~prediction)
    tn = count(valid & ~truth & ~prediction)
    valid_count = count(valid)
    if not np.array_equal(tp + fp + fn + tn, valid_count):
        raise RuntimeError("TP/FP/FN/TN do not conserve valid pixels")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid": valid_count}


def losses_from_confusion(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return zero-is-best binary segmentation losses from sufficient counts."""

    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    if not (tp.shape == fp.shape == fn.shape) or np.any(tp < 0) or np.any(fp < 0) or np.any(fn < 0):
        raise ValueError("confusion sufficient statistics are invalid")
    union = tp + fp + fn
    dice_denominator = 2.0 * tp + fp + fn
    reference = tp + fn
    iou = np.divide(tp, union, out=np.ones_like(tp), where=union > 0)
    dice = np.divide(
        2.0 * tp, dice_denominator, out=np.ones_like(tp),
        where=dice_denominator > 0,
    )
    foreground_mismatch = np.divide(
        fp + fn, reference, out=np.full_like(tp, np.nan), where=reference > 0,
    )
    return {
        "iou_loss": 1.0 - iou,
        "dice_loss": 1.0 - dice,
        "foreground_normalized_mismatch": foreground_mismatch,
    }


def expand_weighted_sample(
    values: np.ndarray, groups: np.ndarray, counts: np.ndarray,
) -> np.ndarray:
    """Small-test helper for proving weighted bootstrap equivalence."""

    values = np.asarray(values)
    groups = np.asarray(groups, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    return np.repeat(values, counts[groups], axis=0)
