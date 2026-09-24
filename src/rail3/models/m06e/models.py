"""Frozen M06-E residual architectures with one prediction head each."""

from __future__ import annotations

import torch
from torch import nn


class GlobalResidual(nn.Module):
    """R0 global residual and Q2 direct-gain baseline architecture."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.residual_predictor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.residual_predictor(features).squeeze(-1)


class DirectGainBaseline(nn.Module):
    """Required Q2 baseline; deliberately not part of the R2 method."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gain_predictor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.gain_predictor(features).squeeze(-1)


class LinearAtomicResidual(nn.Module):
    """Required P1 linear PAL baseline on causal atom/action inputs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.residual_predictor = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.residual_predictor(features).squeeze(-1))


class SharedAtomicResidual(nn.Module):
    """Exact shared R1/R2 atom-action residual network."""

    def __init__(self, atom_dim: int, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.state_context = nn.Sequential(
            nn.Linear(256 + state_dim, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        self.action_id_embedding = nn.Embedding(5, 32)
        self.action_encoder = nn.Sequential(
            nn.Linear(32 + action_dim, 64), nn.SiLU(),
        )
        self.residual_predictor = nn.Sequential(
            nn.Linear(320, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(
        self, atom_features: torch.Tensor, state_features: torch.Tensor,
        action_features: torch.Tensor, state_lengths: torch.Tensor,
        atom_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom = self.atom_encoder(atom_features)
        mean = torch.segment_reduce(atom, "mean", lengths=state_lengths)
        maximum = torch.segment_reduce(atom, "max", lengths=state_lengths)
        context = self.state_context(torch.cat((mean, maximum, state_features), dim=1))
        action_ids = torch.arange(5, device=action_features.device).expand(
            action_features.shape[0], 5
        )
        action = self.action_encoder(torch.cat((
            self.action_id_embedding(action_ids), action_features,
        ), dim=2))
        atom_states = torch.repeat_interleave(
            torch.arange(state_features.shape[0], device=atom.device), state_lengths,
        )
        fused = torch.cat((
            atom[:, None, :].expand(-1, 5, -1),
            context[atom_states, None, :].expand(-1, 5, -1),
            action[atom_states],
        ), dim=2)
        atomic = self.residual_predictor(fused).squeeze(-1)
        return atomic, aggregate_atomic(atomic, atom_weights, state_lengths)


def aggregate_atomic(
    atomic: torch.Tensor, atom_weights: torch.Tensor, state_lengths: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen additive valid-area residual aggregation."""

    if atom_weights.shape != atomic.shape[:1]:
        raise ValueError("atom weights do not align with atomic predictions")
    return torch.segment_reduce(atomic * atom_weights[:, None], "sum", lengths=state_lengths)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
