"""Architectures and label-blind capacity matching for TMLR V6 P0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rail3.models.m06e.models import SharedAtomicResidual


@dataclass(frozen=True)
class WidthSolution:
    width: int
    parameter_count: int
    target_parameter_count: int

    @property
    def difference(self) -> int:
        return self.parameter_count - self.target_parameter_count

    @property
    def difference_percent(self) -> float:
        return 100.0 * self.difference / self.target_parameter_count


def capacity_matched_parameter_count(input_dim: int, width: int) -> int:
    """Count parameters in input-w-w-1 with first-layer LayerNorm."""

    if input_dim <= 0 or width <= 0:
        raise ValueError("capacity dimensions must be positive")
    return width * width + (input_dim + 5) * width + 1


def solve_capacity_matched_width(
    *, input_dim: int, target_parameter_count: int,
    search_min: int, search_max: int,
) -> WidthSolution:
    """Use only dimensions/counts; smaller width deterministically breaks ties."""

    if target_parameter_count <= 0 or not 1 <= search_min <= search_max:
        raise ValueError("capacity search contract is invalid")
    width = min(
        range(search_min, search_max + 1),
        key=lambda candidate: (
            abs(
                capacity_matched_parameter_count(input_dim, candidate)
                - target_parameter_count
            ),
            candidate,
        ),
    )
    return WidthSolution(
        width=width,
        parameter_count=capacity_matched_parameter_count(input_dim, width),
        target_parameter_count=target_parameter_count,
    )


class CapacityMatchedGlobalResidual(nn.Module):
    """A global residual MLP whose width is supplied by the frozen solver."""

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        if input_dim <= 0 or width <= 0:
            raise ValueError("capacity-matched model dimensions must be positive")
        self.residual_predictor = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.residual_predictor(features).squeeze(-1)


class ProspectiveSharedAtomicResidual(SharedAtomicResidual):
    """Exact R1 modules with a label-free aggregation-weight API."""

    def forward(
        self,
        atom_features: torch.Tensor,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        state_lengths: torch.Tensor,
        inference_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(
            atom_features,
            state_features,
            action_features,
            state_lengths,
            inference_weights,
        )
