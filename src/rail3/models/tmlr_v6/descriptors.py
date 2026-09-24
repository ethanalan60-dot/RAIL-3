"""Plan-derived action descriptors for the PROSPECTIVE_V1 predictor.

The historical feature builder accepted an executed ``ActionOutcome`` and
therefore had access to candidate-action runtime and post-action masks.  This
module provides the opposite boundary: adaptive descriptors are constructed
from an ``ActionPlan`` and STOP is constructed without an outcome object.
Neither the descriptor nor its numeric vector has a cost/runtime field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from rail3.sam.voc_actions import ActionPlan, prompt_payload


ACTION_CODES = ("STOP", "A3", "A4", "A5", "A6")
ACTION_DIM = 16
HASH_DIM = 8
FORBIDDEN_SERIALIZED_TOKENS = (
    "action_cost_seconds",
    "incremental_cost_seconds",
    "total_cost_seconds",
    "runtime_seconds",
    "wall_seconds",
    "telemetry",
    "candidate_count",
    "candidates",
    "mask_rle",
    "mask_sha256",
    "outcome",
)


def _hash_vector(values: list[str], *, namespace: str) -> np.ndarray:
    result = np.zeros(HASH_DIM, dtype=np.float32)
    for value in sorted(values):
        digest = hashlib.sha256(f"{namespace}\0{value}".encode()).digest()
        index = int.from_bytes(digest[:2], "big") % HASH_DIM
        result[index] += 1.0 if digest[2] & 1 else -1.0
    return result / max(1.0, math.sqrt(len(values)))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProspectiveActionDescriptor:
    """The complete deployment-visible action description used by a predictor."""

    action_code: str
    action_id: str
    state_id: str
    feasible: bool
    source_candidate_id: str | None
    action_precondition_available: bool
    lineage_json: str

    def __post_init__(self) -> None:
        if self.action_code not in ACTION_CODES:
            raise ValueError("PROSPECTIVE_V1 descriptor has an unknown action")
        if not self.action_id or not self.state_id:
            raise ValueError("PROSPECTIVE_V1 descriptor identity is empty")
        if self.action_precondition_available != self.feasible:
            raise ValueError("action feasibility/precondition encoding disagrees")
        payload = self.to_payload()
        encoded = _canonical_json(payload).lower()
        forbidden = [token for token in FORBIDDEN_SERIALIZED_TOKENS if token in encoded]
        if forbidden:
            raise ValueError(
                "PROSPECTIVE_V1 descriptor contains future-action fields: "
                + ", ".join(forbidden)
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_code": self.action_code,
            "action_id": self.action_id,
            "state_id": self.state_id,
            "feasible": self.feasible,
            "source_candidate_id": self.source_candidate_id,
            "action_precondition_available": self.action_precondition_available,
            "lineage_json": self.lineage_json,
        }


def descriptor_from_plan(plan: ActionPlan) -> ProspectiveActionDescriptor:
    """Build one A3--A6 descriptor from a registered pre-execution plan.

    An exact ``ActionPlan`` instance is required.  In particular, an executed
    ``ActionOutcome`` or a permissive mapping is not accepted accidentally.
    """

    if type(plan) is not ActionPlan:  # exact type is an intentional API guard
        raise TypeError("PROSPECTIVE_V1 action descriptors require ActionPlan")
    if plan.action_code not in ACTION_CODES[1:]:
        raise ValueError("PROSPECTIVE_V1 candidate plan must be A3, A4, A5, or A6")
    lineage = {
        "prompt": None if plan.prompt is None else prompt_payload(plan.prompt),
        "transform": dict(plan.transform),
        "source_candidate_id": plan.source_candidate_id,
    }
    return ProspectiveActionDescriptor(
        action_code=plan.action_code,
        action_id=plan.action_id,
        state_id=plan.state_id,
        feasible=bool(plan.feasible),
        source_candidate_id=plan.source_candidate_id,
        action_precondition_available=bool(plan.feasible),
        lineage_json=_canonical_json(lineage),
    )


def stop_descriptor(*, state_id: str, canonical_action_id: str) -> ProspectiveActionDescriptor:
    """Construct STOP without loading any adaptive outcome."""

    return ProspectiveActionDescriptor(
        action_code="STOP",
        action_id=str(canonical_action_id),
        state_id=str(state_id),
        feasible=True,
        source_candidate_id=None,
        action_precondition_available=True,
        lineage_json=_canonical_json({"kind": "stop", "base_action_code": "A0"}),
    )


def action_vector(descriptor: ProspectiveActionDescriptor) -> np.ndarray:
    """Encode the plan-derived descriptor directly as 16 safe coordinates."""

    if type(descriptor) is not ProspectiveActionDescriptor:
        raise TypeError("PROSPECTIVE_V1 numeric encoding requires a safe descriptor")
    source = (
        [] if descriptor.source_candidate_id is None
        else [str(descriptor.source_candidate_id)]
    )
    hashed = _hash_vector(
        source + [descriptor.lineage_json], namespace="action-lineage",
    )
    prefix = np.asarray([
        float(descriptor.feasible),
        float(descriptor.source_candidate_id is not None),
        *(float(descriptor.action_code == code) for code in ACTION_CODES),
        float(descriptor.action_precondition_available),
    ], dtype=np.float32)
    result = np.concatenate((prefix, hashed)).astype(np.float32, copy=False)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("PROSPECTIVE_V1 action encoding drift")
    return result


def action_matrix(
    *, stop: ProspectiveActionDescriptor,
    adaptive: tuple[ProspectiveActionDescriptor, ...],
) -> np.ndarray:
    """Encode the required STOP,A3,A4,A5,A6 order."""

    descriptors = (stop, *adaptive)
    if tuple(item.action_code for item in descriptors) != ACTION_CODES:
        raise ValueError("PROSPECTIVE_V1 action order must be STOP,A3,A4,A5,A6")
    state_ids = {item.state_id for item in descriptors}
    if len(state_ids) != 1:
        raise ValueError("PROSPECTIVE_V1 action descriptors cross state boundaries")
    return np.stack([action_vector(item) for item in descriptors]).astype(np.float32)


def full_raster_partition_domain(*, height: int, width: int) -> np.ndarray:
    """Return feature geometry for RECT-P/UNION-P without a label/void input."""

    if height <= 0 or width <= 0:
        raise ValueError("prospective partition geometry must be positive")
    return np.ones((height, width), dtype=bool)
