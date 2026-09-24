"""Outer-training-only scaling for the F0/F1 remaining-action estimand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rail3.models.m06e.features import FrozenScaler, fit_scaler
from rail3.models.tmlr_v6.phase2_io import (
    A3_PHYSICAL_INDEX,
    REMAINING_PHYSICAL_INDICES,
)
from rail3.models.tmlr_v6.schema import ACTION_DIM, ATOM_DIM, STATE_DIM


@dataclass(frozen=True)
class RemainingActionScalers:
    """Scalers fitted without the physical A3 row."""

    state: FrozenScaler
    action: FrozenScaler
    atom: FrozenScaler
    action_fit_rows: tuple[int, ...] = REMAINING_PHYSICAL_INDICES


@dataclass(frozen=True)
class ScaledRemainingFeatures:
    state: np.ndarray
    action: np.ndarray
    atom: np.ndarray
    scalers: RemainingActionScalers


def scale_remaining_predictor_features(
    *, state_x: np.ndarray, action_x16: np.ndarray, atom_x: np.ndarray,
    state_lengths: np.ndarray, outer_train_mask: np.ndarray,
) -> ScaledRemainingFeatures:
    """Fit/transform only STOP/A4/A5/A6 action rows.

    The returned physical A3 row is initialized, not transformed: all sixteen
    coordinates are exact zero.  Therefore arbitrary values in the raw A3 row
    cannot affect fitted statistics or any remaining-action model input.
    """

    state = np.asarray(state_x)
    action = np.asarray(action_x16)
    atom = np.asarray(atom_x)
    lengths = np.asarray(state_lengths, dtype=np.int64)
    train = np.asarray(outer_train_mask, dtype=bool)
    if (
        state.ndim != 2 or state.shape[1] != STATE_DIM
        or action.shape != (len(state), 5, ACTION_DIM)
        or atom.ndim != 2 or atom.shape[1] != ATOM_DIM
        or lengths.shape != (len(state),)
        or np.any(lengths <= 0) or int(lengths.sum()) != len(atom)
        or train.shape != (len(state),) or not train.any() or train.all()
        or not all(np.isfinite(value).all() for value in (state, action, atom))
    ):
        raise ValueError("TMLR V6 F0/F1 scaling geometry is invalid")
    if (
        np.any(action[:, A3_PHYSICAL_INDEX, 0] != 0.0)
        or np.any(action[:, A3_PHYSICAL_INDEX, 7] != 0.0)
    ):
        raise RuntimeError("TMLR V6 F0/F1 scaler received an unmasked A3 row")

    state_scaler = fit_scaler(state[train])
    action_training = action[train][:, REMAINING_PHYSICAL_INDICES, :].reshape(
        -1, ACTION_DIM,
    )
    # This is deliberately the only call that fits action coordinates.
    action_scaler = fit_scaler(action_training)
    atom_train = np.repeat(train, lengths)
    atom_scaler = fit_scaler(atom[atom_train])

    scaled_action = np.zeros(action.shape, dtype=np.float32)
    remaining = action[:, REMAINING_PHYSICAL_INDICES, :].reshape(-1, ACTION_DIM)
    scaled_action[:, REMAINING_PHYSICAL_INDICES, :] = action_scaler.transform(
        remaining,
    ).reshape(len(state), len(REMAINING_PHYSICAL_INDICES), ACTION_DIM)
    scaled_state = state_scaler.transform(state)
    scaled_atom = atom_scaler.transform(atom)
    if (
        np.any(scaled_action[:, A3_PHYSICAL_INDEX, :] != 0.0)
        or not all(
            np.isfinite(value).all()
            for value in (scaled_state, scaled_action, scaled_atom)
        )
    ):
        raise RuntimeError("TMLR V6 F0/F1 scaled feature contract failed")
    return ScaledRemainingFeatures(
        state=scaled_state,
        action=scaled_action,
        atom=scaled_atom,
        scalers=RemainingActionScalers(
            state=state_scaler,
            action=action_scaler,
            atom=atom_scaler,
        ),
    )


def remaining_scaler_payload(
    scalers: RemainingActionScalers,
) -> dict[str, np.ndarray]:
    """Return an allow-pickle-free checkpoint payload with the row contract."""

    if tuple(scalers.action_fit_rows) != REMAINING_PHYSICAL_INDICES:
        raise RuntimeError("TMLR V6 F0/F1 action-scaler row identity drift")
    return {
        "state_mean": scalers.state.mean,
        "state_scale": scalers.state.scale,
        "action_mean": scalers.action.mean,
        "action_scale": scalers.action.scale,
        "atom_mean": scalers.atom.mean,
        "atom_scale": scalers.atom.scale,
        "action_fit_rows": np.asarray(REMAINING_PHYSICAL_INDICES, dtype=np.int8),
        "action_transform_rows": np.asarray(
            REMAINING_PHYSICAL_INDICES, dtype=np.int8,
        ),
        "a3_scaled_row": np.zeros(ACTION_DIM, dtype=np.float32),
    }


__all__ = [
    "RemainingActionScalers",
    "ScaledRemainingFeatures",
    "remaining_scaler_payload",
    "scale_remaining_predictor_features",
]
