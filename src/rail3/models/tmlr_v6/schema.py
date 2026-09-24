"""The frozen PROSPECTIVE_V1 predictor-feature projection.

This module deliberately has no cost argument or cost-array dependency.  The
only runtime-bearing raw action coordinate is removed before a scaler is fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rail3.models.m06e.features import FrozenScaler, fit_scaler


PROSPECTIVE_SCHEMA_ID = "PROSPECTIVE_V1"
PROSPECTIVE_BUILDER_IMPLEMENTATION_PATHS = (
    "configs/experiments/voc_m04_m05_action_protocol_v1.json",
    "scripts/build_tmlr_v6_prospective_bundles.py",
    "src/rail3/__init__.py",
    "src/rail3/cache/__init__.py",
    "src/rail3/cache/atomic_io.py",
    "src/rail3/cache/candidates.py",
    "src/rail3/contracts/__init__.py",
    "src/rail3/contracts/records.py",
    "src/rail3/contracts/serialization.py",
    "src/rail3/data/m06e_sources.py",
    "src/rail3/data/__init__.py",
    "src/rail3/data/voc_taxonomy.py",
    "src/rail3/diagnostic/__init__.py",
    "src/rail3/diagnostic/representations.py",
    "src/rail3/evaluation/__init__.py",
    "src/rail3/evaluation/voc_v2a.py",
    "src/rail3/models/__init__.py",
    "src/rail3/models/m06e/__init__.py",
    "src/rail3/models/m06e/data.py",
    "src/rail3/models/m06e/features.py",
    "src/rail3/models/tmlr_v6/descriptors.py",
    "src/rail3/models/tmlr_v6/__init__.py",
    "src/rail3/models/tmlr_v6/schema.py",
    "src/rail3/regions/atomic.py",
    "src/rail3/regions/__init__.py",
    "src/rail3/regions/m06e_causal.py",
    "src/rail3/regions/voc_state_atomic.py",
    "src/rail3/sam/checkpoint_compat.py",
    "src/rail3/sam/__init__.py",
    "src/rail3/sam/compat.py",
    "src/rail3/sam/protocols.py",
    "src/rail3/sam/sam31_backend.py",
    "src/rail3/sam/strict_builder.py",
    "src/rail3/sam/voc_actions.py",
    "src/rail3/sam/voc_protocol.py",
)
ATOM_DIM = 27
STATE_DIM = 41
RAW_ACTION_DIM = 17
ACTION_DIM = 16
GLOBAL_DIM = STATE_DIM + ACTION_DIM
REMOVED_RAW_ACTION_INDICES = (7,)
RETAINED_RAW_ACTION_INDICES = (
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16,
)
ACTION_FEATURE_NAMES = (
    "feasible",
    "source_candidate_present",
    "action_is_STOP",
    "action_is_A3",
    "action_is_A4",
    "action_is_A5",
    "action_is_A6",
    "action_precondition_available",
    "lineage_hash_0",
    "lineage_hash_1",
    "lineage_hash_2",
    "lineage_hash_3",
    "lineage_hash_4",
    "lineage_hash_5",
    "lineage_hash_6",
    "lineage_hash_7",
)


@dataclass(frozen=True)
class ProspectiveScalers:
    """Outer-training-only scalers for one representation and fold."""

    state: FrozenScaler
    action: FrozenScaler
    atom: FrozenScaler | None = None


@dataclass(frozen=True)
class ProspectiveFeatures:
    """Scaled predictor features with the post-action runtime removed."""

    state: np.ndarray
    action: np.ndarray
    atom: np.ndarray | None
    scalers: ProspectiveScalers


def validate_schema_contract(protocol: Mapping[str, Any]) -> None:
    """Fail closed if the committed config no longer describes this projection."""

    schema = protocol.get("prospective_schema", {})
    expected = {
        "id": PROSPECTIVE_SCHEMA_ID,
        "atom_dimension": ATOM_DIM,
        "state_dimension": STATE_DIM,
        "raw_action_dimension": RAW_ACTION_DIM,
        "removed_raw_action_indices": list(REMOVED_RAW_ACTION_INDICES),
        "retained_raw_action_indices": list(RETAINED_RAW_ACTION_INDICES),
        "action_dimension": ACTION_DIM,
        "global_per_action_dimension": GLOBAL_DIM,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "state_prediction_aggregation_weight": (
            "full-raster area fraction stored in the prospective feature artifact"
        ),
        "target_only_reconstruction_weight": (
            "ground-truth-valid pixel fraction stored only in the target artifact"
        ),
        "target_weight_available_to_inference": False,
        "predictor_cost_input": False,
        "post_action_mask_input": False,
        "ground_truth_inference_input": False,
        "future_action_outcome_input": False,
    }
    drift = {
        key: (schema.get(key), value)
        for key, value in expected.items()
        if schema.get(key) != value
    }
    if drift:
        raise RuntimeError(f"PROSPECTIVE_V1 feature contract drift: {drift}")
    if schema.get("feature_scaler_fit") != (
        "outer-training groups only after runtime-column deletion"
    ):
        raise RuntimeError("PROSPECTIVE_V1 scaler-order contract drift")


def project_action_features(raw_action: np.ndarray) -> np.ndarray:
    """Delete raw coordinate 7 before any scaling or model construction."""

    values = np.asarray(raw_action)
    if values.ndim != 3 or values.shape[1:] != (5, RAW_ACTION_DIM):
        raise ValueError("PROSPECTIVE_V1 action input must have shape [state,5,17]")
    projected = np.take(values, RETAINED_RAW_ACTION_INDICES, axis=2).astype(
        np.float32, copy=True,
    )
    if projected.shape[2] != ACTION_DIM or not np.isfinite(projected).all():
        raise RuntimeError("PROSPECTIVE_V1 action projection failed")
    return projected


def scale_predictor_features(
    *,
    state_x: np.ndarray,
    action_x16: np.ndarray,
    outer_train_mask: np.ndarray,
    atom_x: np.ndarray | None = None,
    state_lengths: np.ndarray | None = None,
) -> ProspectiveFeatures:
    """Scale an already-projected physical feature bundle on outer training.

    The raw 17-column table is deliberately not accepted by the training
    boundary.  :func:`project_action_features` belongs to the upstream bundle
    materializer; only its 16-column output may cross into model training.
    """

    state = np.asarray(state_x)
    train = np.asarray(outer_train_mask, dtype=bool)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError("PROSPECTIVE_V1 state matrix dimension drift")
    if train.shape != (len(state),) or not train.any() or train.all():
        raise ValueError("PROSPECTIVE_V1 outer-training mask is invalid")
    projected_action = np.asarray(action_x16)
    if (
        projected_action.ndim != 3
        or projected_action.shape[1:] != (5, ACTION_DIM)
        or not np.isfinite(projected_action).all()
    ):
        raise ValueError("PROSPECTIVE_V1 projected action input must be [state,5,16]")
    if len(projected_action) != len(state):
        raise ValueError("PROSPECTIVE_V1 state/action row mismatch")

    state_scaler = fit_scaler(state[train])
    action_scaler = fit_scaler(
        projected_action[train].reshape(-1, ACTION_DIM),
    )
    scaled_state = state_scaler.transform(state)
    scaled_action = action_scaler.transform(
        projected_action.reshape(-1, ACTION_DIM),
    ).reshape(projected_action.shape)

    scaled_atom: np.ndarray | None = None
    atom_scaler: FrozenScaler | None = None
    if (atom_x is None) != (state_lengths is None):
        raise ValueError("PROSPECTIVE_V1 atom matrix and state lengths are inseparable")
    if atom_x is not None and state_lengths is not None:
        atom = np.asarray(atom_x)
        lengths = np.asarray(state_lengths, dtype=np.int64)
        if (
            atom.ndim != 2
            or atom.shape[1] != ATOM_DIM
            or lengths.shape != (len(state),)
            or np.any(lengths <= 0)
            or int(lengths.sum()) != len(atom)
        ):
            raise ValueError("PROSPECTIVE_V1 atomic feature geometry drift")
        atom_train = np.repeat(train, lengths)
        atom_scaler = fit_scaler(atom[atom_train])
        scaled_atom = atom_scaler.transform(atom)

    outputs = (scaled_state, scaled_action)
    if scaled_atom is not None:
        outputs += (scaled_atom,)
    if any(not np.isfinite(value).all() for value in outputs):
        raise RuntimeError("PROSPECTIVE_V1 scaled feature contains a non-finite value")
    return ProspectiveFeatures(
        state=scaled_state,
        action=scaled_action,
        atom=scaled_atom,
        scalers=ProspectiveScalers(
            state=state_scaler, action=action_scaler, atom=atom_scaler,
        ),
    )


def scaler_payload(scalers: ProspectiveScalers) -> dict[str, np.ndarray]:
    """Return an allow-pickle-free checkpoint representation of the scalers."""

    result = {
        "state_mean": scalers.state.mean,
        "state_scale": scalers.state.scale,
        "action_mean": scalers.action.mean,
        "action_scale": scalers.action.scale,
    }
    if scalers.atom is not None:
        result.update({
            "atom_mean": scalers.atom.mean,
            "atom_scale": scalers.atom.scale,
        })
    return result
