"""Frozen state-balanced Decision-Relative Atomic Residual losses."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


MODEL_IDS = (
    "R3A_ABSOLUTE_RELATIVE_ATOMIC_RESIDUAL",
    "R3_DECISION_RELATIVE_ATOMIC_RESIDUAL",
)
RELATIVE_HUBER_DELTA = 0.005
RELATIVE_WEIGHT_SCALE = 0.01
SIGN_TEMPERATURE = 0.005
RANK_MARGIN_MINIMUM = 0.001


def stop_relative_delta(residual: torch.Tensor) -> torch.Tensor:
    """Return non-STOP residuals relative to the STOP column."""

    if residual.ndim != 2 or residual.shape[1] != 5:
        raise ValueError("M06-G residual matrix must be Nx5 with STOP first")
    return residual[:, 1:] - residual[:, :1]


def decision_relevance_weight(delta_true: torch.Tensor) -> torch.Tensor:
    """Frozen target-only action weight; NaNs remain unavailable."""

    return torch.clamp(torch.abs(delta_true) / RELATIVE_WEIGHT_SCALE, max=1.0)


def _state_balanced_action_mean(
    values: torch.Tensor, defined: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != defined.shape or values.ndim != 2:
        raise ValueError("M06-G state-action loss inputs are not aligned")
    if state_mask.shape != values.shape[:1] or state_mask.dtype != torch.bool:
        raise ValueError("M06-G state mask is not aligned boolean data")
    use = defined & state_mask[:, None]
    per_state = torch.where(use, values, torch.zeros_like(values)).sum(1)
    per_state = per_state / use.sum(1).clamp_min(1)
    selected = state_mask & use.any(1)
    if not selected.any():
        raise ValueError("M06-G objective contains no supervised states")
    return per_state[selected].mean()


def stop_relative_huber(
    prediction: torch.Tensor, target: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    """Weighted STOP-relative Huber, action-mean then state-mean."""

    if prediction.shape != target.shape:
        raise ValueError("M06-G prediction and target are not aligned")
    pred_delta = stop_relative_delta(prediction)
    true_delta = stop_relative_delta(target)
    defined = torch.isfinite(target[:, :1]) & torch.isfinite(target[:, 1:])
    safe_true = torch.nan_to_num(true_delta)
    raw = F.huber_loss(pred_delta, safe_true, reduction="none", delta=RELATIVE_HUBER_DELTA)
    weighted = raw * torch.nan_to_num(decision_relevance_weight(true_delta))
    return _state_balanced_action_mean(weighted, defined, state_mask)


def stop_relative_sign_loss(
    prediction: torch.Tensor, target: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth weighted strict-benefit sign loss on residual differences."""

    if prediction.shape != target.shape:
        raise ValueError("M06-G prediction and target are not aligned")
    pred_delta = stop_relative_delta(prediction)
    true_delta = stop_relative_delta(target)
    defined = torch.isfinite(target[:, :1]) & torch.isfinite(target[:, 1:])
    label = torch.where(true_delta < 0.0, torch.ones_like(true_delta), -torch.ones_like(true_delta))
    raw = F.softplus(label * pred_delta / SIGN_TEMPERATURE)
    weighted = raw * torch.nan_to_num(decision_relevance_weight(true_delta))
    return _state_balanced_action_mean(weighted, defined, state_mask)


def within_state_rank_loss(
    prediction: torch.Tensor, target: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    """State-balanced logistic ordering over frozen informative pairs."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 5:
        raise ValueError("M06-G ranking tensors must be aligned Nx5 matrices")
    if state_mask.shape != prediction.shape[:1] or state_mask.dtype != torch.bool:
        raise ValueError("M06-G state mask is not aligned boolean data")
    left, right = torch.triu_indices(5, 5, offset=1, device=prediction.device)
    true_difference = target[:, right] - target[:, left]
    pred_difference = prediction[:, right] - prediction[:, left]
    defined = torch.isfinite(target[:, left]) & torch.isfinite(target[:, right])
    informative = defined & (torch.abs(true_difference) >= RANK_MARGIN_MINIMUM)
    orientation = torch.sign(torch.nan_to_num(true_difference))
    raw = F.softplus(-orientation * pred_difference)
    return _state_balanced_action_mean(raw, informative, state_mask)


def decision_relative_losses(
    prediction: torch.Tensor, target: torch.Tensor, state_mask: torch.Tensor,
    *, model_id: str,
) -> Mapping[str, torch.Tensor]:
    """Return the frozen new objective terms and their added total."""

    if model_id not in MODEL_IDS:
        raise ValueError("unregistered M06-G model objective")
    relative = stop_relative_huber(prediction, target, state_mask)
    zero = relative.new_zeros(())
    sign = zero
    rank = zero
    added = relative
    if model_id == "R3_DECISION_RELATIVE_ATOMIC_RESIDUAL":
        sign = stop_relative_sign_loss(prediction, target, state_mask)
        rank = within_state_rank_loss(prediction, target, state_mask)
        added = relative + 0.5 * sign + 0.5 * rank
    return {"relative": relative, "sign": sign, "rank": rank, "added": added}
