"""Frozen phase-two objectives composed from the historical loss APIs.

No loss formula is reimplemented here.  The prospective wrapper validates the
new physical boundary and delegates R2 to :mod:`m06e.losses`, R3 to
:mod:`m06g.losses`, and the direct decision objectives to the pre-existing
finite-action baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from rail3.diagnostic.baselines import l2d_loss, spo_plus_loss
from rail3.models.m06e.losses import (
    LAMBDA_GRID,
    atomic_state_losses,
    derive_query_gain,
    pareto_integrated_decision_regret,
)
from rail3.models.m06g.losses import decision_relative_losses
from rail3.models.tmlr_v6.training import _state_huber


PHASE2_LAMBDAS = tuple(float(value) for value in LAMBDA_GRID)


@dataclass(frozen=True)
class AtomicObjectiveInputs:
    atom_prediction: torch.Tensor
    state_prediction: torch.Tensor
    atom_target: torch.Tensor
    state_target: torch.Tensor
    atom_weights: torch.Tensor
    state_lengths: torch.Tensor
    state_mask: torch.Tensor
    normalized_cost: torch.Tensor


def _validate_cost_vector(cost: torch.Tensor) -> None:
    if (
        cost.ndim != 1
        or cost.shape != (5,)
        or not torch.isfinite(cost).all()
        or cost[0].item() != 0.0
        or torch.any(cost < 0)
    ):
        raise ValueError("TMLR V6 phase-two fold cost must be finite STOP,A3--A6")


def _cost_surface(vector: torch.Tensor, states: int) -> torch.Tensor:
    _validate_cost_vector(vector)
    return vector[None, :].expand(states, -1)


def r2_objective(inputs: AtomicObjectiveInputs) -> Mapping[str, torch.Tensor]:
    """Exact ``L_atom + L_abs + 0.5 L_PIDR`` registered for R2_P."""

    atom, absolute = atomic_state_losses(
        inputs.atom_prediction,
        inputs.state_prediction,
        inputs.atom_target,
        inputs.state_target,
        inputs.atom_weights,
        inputs.state_lengths,
        inputs.state_mask,
    )
    pidr = pareto_integrated_decision_regret(
        inputs.state_prediction,
        inputs.state_target,
        _cost_surface(inputs.normalized_cost, len(inputs.state_target)),
        inputs.state_mask,
        lambdas=PHASE2_LAMBDAS,
        tau=0.05,
    )
    return {
        "atom": atom,
        "absolute": absolute,
        "pidr": pidr,
        "total": atom + absolute + 0.5 * pidr,
    }


def r3_objective(inputs: AtomicObjectiveInputs) -> Mapping[str, torch.Tensor]:
    """Exact historical R3 atomic/absolute/relative/sign/rank objective."""

    atom, absolute = atomic_state_losses(
        inputs.atom_prediction,
        inputs.state_prediction,
        inputs.atom_target,
        inputs.state_target,
        inputs.atom_weights,
        inputs.state_lengths,
        inputs.state_mask,
    )
    relative = decision_relative_losses(
        inputs.state_prediction,
        inputs.state_target,
        inputs.state_mask,
        model_id="R3_DECISION_RELATIVE_ATOMIC_RESIDUAL",
    )
    return {
        "atom": atom,
        "absolute": absolute,
        "relative": relative["relative"],
        "sign": relative["sign"],
        "rank": relative["rank"],
        "total": atom + absolute + relative["added"],
    }


def q2_gain_target(state_target: torch.Tensor) -> torch.Tensor:
    """Construct defined STOP-relative gains, retaining unavailable actions."""

    if state_target.ndim != 2 or state_target.shape[1] != 5:
        raise ValueError("TMLR V6 Q2 target must be a STOP,A3--A6 matrix")
    # Reuse the historical Q2 target transform.  IEEE subtraction preserves
    # missing actions as NaN and makes every defined STOP target exactly zero.
    return derive_query_gain(state_target)


def q2_objective(
    raw_gain_prediction: torch.Tensor,
    state_target: torch.Tensor,
    state_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    target = q2_gain_target(state_target)
    loss = _state_huber(raw_gain_prediction, target, state_mask)
    return {"gain_huber": loss, "total": loss}


def scientific_q2_seed_ensemble(
    raw_seed_predictions: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Mean continuous seed surfaces, then impose scientific STOP=0."""

    if set(raw_seed_predictions) != {13, 37, 71}:
        raise ValueError("TMLR V6 Q2 ensemble requires seeds 13,37,71 exactly")
    ordered = [raw_seed_predictions[seed] for seed in (13, 37, 71)]
    if (
        not ordered
        or any(value.shape != ordered[0].shape for value in ordered)
        or ordered[0].ndim != 2
        or ordered[0].shape[1] != 5
        or any(not torch.isfinite(value).all() for value in ordered)
    ):
        raise ValueError("TMLR V6 Q2 seed surfaces are invalid")
    result = torch.stack(ordered, dim=0).mean(dim=0)
    result = result.clone()
    result[:, 0] = 0.0
    return result


def direct_objective(
    *, model_id: str, output: torch.Tensor, state_target: torch.Tensor,
    feasible: torch.Tensor, state_mask: torch.Tensor,
    normalized_cost: torch.Tensor, cost_lambda: float,
) -> Mapping[str, torch.Tensor]:
    """Delegate L2D or finite-action SPO+ to the frozen implementations."""

    if cost_lambda not in PHASE2_LAMBDAS:
        raise ValueError("unregistered TMLR V6 phase-two lambda")
    cost = _cost_surface(normalized_cost, len(state_target))
    if model_id == "L2D_P":
        loss = l2d_loss(
            output, state_target, cost, feasible, state_mask, cost_lambda,
        )
        return {"cross_entropy": loss, "total": loss}
    if model_id == "SPO_PLUS_P":
        true_value = state_target + float(cost_lambda) * cost
        loss = spo_plus_loss(output, true_value, feasible, state_mask)
        return {"spo_plus": loss, "total": loss}
    raise ValueError("unregistered TMLR V6 direct objective")


__all__ = [
    "AtomicObjectiveInputs",
    "PHASE2_LAMBDAS",
    "direct_objective",
    "q2_gain_target",
    "q2_objective",
    "r2_objective",
    "r3_objective",
    "scientific_q2_seed_ensemble",
]
