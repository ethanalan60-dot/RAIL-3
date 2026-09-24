"""State-balanced M06-E regression and Pareto-integrated regret losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


LAMBDA_GRID = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
TAU = 0.05


def normalized_action_cost(cost: torch.Tensor, train_state_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost.ndim != 2 or cost.shape[1] != 5 or train_state_mask.shape != cost.shape[:1]:
        raise ValueError("cost normalization inputs are not aligned")
    non_stop = cost[train_state_mask, 1:]
    positive = non_stop[torch.isfinite(non_stop) & (non_stop > 0)]
    if positive.numel() == 0:
        raise ValueError("training fold has no positive non-STOP cost")
    median = positive.median()
    result = cost / median
    result[:, 0] = 0.0
    return result, median


def _state_action_mean(loss: torch.Tensor, defined: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
    use = defined & state_mask[:, None]
    per_state = (loss * use).sum(dim=1) / use.sum(dim=1).clamp_min(1)
    selected = state_mask & use.any(dim=1)
    if not selected.any():
        raise ValueError("loss selection contains no supervised states")
    return per_state[selected].mean()


def atomic_state_losses(
    atom_prediction: torch.Tensor, state_prediction: torch.Tensor,
    atom_target: torch.Tensor, state_target: torch.Tensor,
    atom_weights: torch.Tensor, state_lengths: torch.Tensor,
    state_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if atom_prediction.shape != atom_target.shape or state_prediction.shape != state_target.shape:
        raise ValueError("residual prediction/target tensors are not aligned")
    atom_defined = torch.isfinite(atom_target)
    safe_atom_target = torch.nan_to_num(atom_target)
    raw_atom = F.huber_loss(atom_prediction, safe_atom_target, reduction="none", delta=0.10)
    weighted = raw_atom * atom_weights[:, None] * atom_defined
    atom_sa = torch.segment_reduce(weighted, "sum", lengths=state_lengths)
    valid_weight = torch.segment_reduce(
        atom_weights[:, None].expand(-1, 5) * atom_defined, "sum", lengths=state_lengths,
    )
    atom_sa = atom_sa / valid_weight.clamp_min(1e-12)
    state_defined = torch.isfinite(state_target)
    atom_loss = _state_action_mean(atom_sa, state_defined & (valid_weight > 0), state_mask)
    raw_state = F.huber_loss(
        state_prediction, torch.nan_to_num(state_target), reduction="none", delta=0.05,
    )
    state_loss = _state_action_mean(raw_state, state_defined, state_mask)
    return atom_loss, state_loss


def pareto_integrated_decision_regret(
    state_prediction: torch.Tensor, state_target: torch.Tensor,
    cost_norm: torch.Tensor, state_mask: torch.Tensor,
    *, lambdas: tuple[float, ...] = LAMBDA_GRID, tau: float = TAU,
) -> torch.Tensor:
    if state_prediction.shape != state_target.shape or cost_norm.shape != state_target.shape:
        raise ValueError("PIDR tensors are not aligned")
    defined = torch.isfinite(state_target)
    selected = state_mask & defined[:, 0] & (defined.sum(dim=1) >= 2)
    if not selected.any():
        raise ValueError("PIDR selection contains no evaluable state")
    pred = state_prediction[selected]
    truth = state_target[selected]
    costs = cost_norm[selected]
    valid = defined[selected]
    terms = []
    for value in lambdas:
        pred_cost = pred + float(value) * costs
        true_cost = truth + float(value) * costs
        logits = torch.where(valid, -pred_cost / tau, torch.full_like(pred_cost, -1e9))
        policy = torch.softmax(logits, dim=1)
        safe_true = torch.where(valid, true_cost, torch.zeros_like(true_cost))
        oracle = torch.where(valid, true_cost, torch.full_like(true_cost, torch.inf)).min(dim=1).values
        terms.append((torch.sum(policy * safe_true, dim=1) - oracle).mean())
    return torch.stack(terms).mean()


def derive_query_gain(state_residual: torch.Tensor) -> torch.Tensor:
    if state_residual.ndim != 2 or state_residual.shape[1] != 5:
        raise ValueError("residual matrix must contain STOP,A3,A4,A5,A6")
    return state_residual[:, :1] - state_residual
