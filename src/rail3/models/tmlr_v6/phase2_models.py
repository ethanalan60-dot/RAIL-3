"""Frozen model registry for the TMLR V6 phase-two second batch.

This module only binds already-registered architectures to their prospective
identifiers.  In particular, costs never cross a model constructor or forward
signature; R2/R3 use the exact R1 graph, Q2 uses the historical direct-gain
graph, and L2D/SPO+ share the historical finite-action graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from rail3.diagnostic.baselines import MultiActionNetwork
from rail3.models.m06e.models import DirectGainBaseline, parameter_count
from rail3.models.tmlr_v6.models import ProspectiveSharedAtomicResidual
from rail3.models.tmlr_v6.schema import ACTION_DIM, ATOM_DIM, GLOBAL_DIM, STATE_DIM


SECOND_BATCH_MODEL_IDS = (
    "R2_P",
    "R3_P",
    "Q2_P",
    "L2D_P",
    "SPO_PLUS_P",
)
LOGICAL_ALIAS_REGISTRY = {"LL4TTA_P": "R0_SMALL_P"}
DIRECT_MODEL_IDS = ("L2D_P", "SPO_PLUS_P")
ATOMIC_MODEL_IDS = ("R2_P", "R3_P")

DIRECT_INPUT_DIM = STATE_DIM + 5 * ACTION_DIM
EXPECTED_PARAMETER_COUNTS = {
    "R2_P": 103_521,
    "R3_P": 103_521,
    "Q2_P": 24_321,
    "L2D_P": 16_517,
    "SPO_PLUS_P": 16_517,
}
PREDICTION_SEMANTICS = {
    "R2_P": "ABSOLUTE_RESIDUAL",
    "R3_P": "ABSOLUTE_RESIDUAL",
    "Q2_P": "RAW_GAIN_PRE_SEED_AVERAGE_STOP_NOT_OVERRIDDEN",
    "L2D_P": "FEASIBLE_MASKED_SELECTION_LOGITS",
    "SPO_PLUS_P": "PREDICTED_COST_AUGMENTED_VALUE",
}


@dataclass(frozen=True)
class Phase2ModelSpec:
    model_id: str
    kind: str
    parameter_count: int
    prediction_semantics: str
    lambda_specific: bool


MODEL_REGISTRY = {
    "R2_P": Phase2ModelSpec(
        "R2_P", "shared_atomic_pidr", 103_521,
        PREDICTION_SEMANTICS["R2_P"], False,
    ),
    "R3_P": Phase2ModelSpec(
        "R3_P", "shared_atomic_decision_relative", 103_521,
        PREDICTION_SEMANTICS["R3_P"], False,
    ),
    "Q2_P": Phase2ModelSpec(
        "Q2_P", "direct_gain_fixed_zero_stop_evaluation", 24_321,
        PREDICTION_SEMANTICS["Q2_P"], False,
    ),
    "L2D_P": Phase2ModelSpec(
        "L2D_P", "multi_action_task_adaptation", 16_517,
        PREDICTION_SEMANTICS["L2D_P"], True,
    ),
    "SPO_PLUS_P": Phase2ModelSpec(
        "SPO_PLUS_P", "finite_action_spo_plus_task_adaptation", 16_517,
        PREDICTION_SEMANTICS["SPO_PLUS_P"], True,
    ),
}


def build_phase2_model(model_id: str) -> nn.Module:
    """Construct exactly one frozen phase-two model, without a cost input."""

    if model_id in ATOMIC_MODEL_IDS:
        model: nn.Module = ProspectiveSharedAtomicResidual(
            ATOM_DIM, STATE_DIM, ACTION_DIM,
        )
    elif model_id == "Q2_P":
        model = DirectGainBaseline(GLOBAL_DIM)
    elif model_id in DIRECT_MODEL_IDS:
        model = MultiActionNetwork(DIRECT_INPUT_DIM)
    else:
        raise ValueError("unregistered TMLR V6 phase-two model")
    observed = parameter_count(model)
    expected = EXPECTED_PARAMETER_COUNTS[model_id]
    if observed != expected:
        raise RuntimeError(
            f"TMLR V6 phase-two parameter-count drift: "
            f"{model_id}={observed}, expected={expected}"
        )
    return model


__all__ = [
    "ATOMIC_MODEL_IDS",
    "DIRECT_INPUT_DIM",
    "DIRECT_MODEL_IDS",
    "EXPECTED_PARAMETER_COUNTS",
    "LOGICAL_ALIAS_REGISTRY",
    "MODEL_REGISTRY",
    "PREDICTION_SEMANTICS",
    "SECOND_BATCH_MODEL_IDS",
    "Phase2ModelSpec",
    "build_phase2_model",
]
