"""Label-free V9B1 feature materialization and frozen-model inference.

The feature boundary accepts the canonical A0 result plus the already-incurred
A1/A2 outcomes.  It does not accept A3--A6 outcomes, targets, annotations, or a
COCO scaler-fit population.  Model inference consumes scaler payloads embedded
in the frozen VOC full-fit checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rail3.cache import CandidateGenerationResult
from rail3.cache.candidates import decode_binary_mask_array
from rail3.contracts import stable_id
from rail3.diagnostic.representations import (
    RegionPartition,
    guillotine_partition,
    observed_partition_feature_matrices,
    regional_feature_matrices,
    union_component_partition,
)
from rail3.models.m06e.features import FrozenScaler
from rail3.models.tmlr_v6.descriptors import (
    action_matrix,
    descriptor_from_plan,
    stop_descriptor,
)
from rail3.models.tmlr_v6.schema import (
    ACTION_DIM,
    ATOM_DIM,
    GLOBAL_DIM,
    PROSPECTIVE_SCHEMA_ID,
    STATE_DIM,
    ProspectiveFeatures,
    ProspectiveScalers,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    MEMORY_BOUNDED_ATOM_THRESHOLD,
    inference_weights_from_features,
    memory_bounded_shared_atomic_forward,
)
from rail3.regions.atomic import CandidateMaskInput
from rail3.regions.m06e_causal import build_m06e_causal_partition
from rail3.sam.coco_v9b1_trajectory import SemanticState
from rail3.sam.voc_actions import ActionOutcome, ActionPlan, PROTOCOL_SHA256, build_action_plans
from rail3.sam.voc_protocol import canonical_voc_prompts, voc_cache_identity


ROLE_ORDER = ("F0_P", "RECT_FULL_RASTER_P", "UNION_FULL_RASTER_P")
FAMILY_ROLE: Mapping[str, str] = {
    "R0_SMALL_P": "F0_P",
    "R0_CM_P": "F0_P",
    "R1_P": "F0_P",
    "RECT_P": "RECT_FULL_RASTER_P",
    "UNION_P": "UNION_FULL_RASTER_P",
    "R3_P": "F0_P",
}
GLOBAL_FAMILIES = frozenset({"R0_SMALL_P", "R0_CM_P"})
ATOMIC_FAMILIES = frozenset({"R1_P", "RECT_P", "UNION_P", "R3_P"})
SEED_ORDER = (13, 37, 71)
DECISION_ACTION_ORDER = ("STOP", "A3", "A4", "A5", "A6")
ENSEMBLE_ID = "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"
_CHECKPOINT_VALIDATION_MARKER = object()
_CHECKPOINT_KEYS = frozenset({
    "schema_version",
    "family",
    "seed",
    "fullfit_epoch",
    "epochs_completed",
    "epoch_rule_id",
    "epoch_lock_sha256",
    "config_sha256",
    "bundle_role",
    "bundle_identities",
    "prospective_schema_id",
    "feature_dimensions",
    "model_class",
    "parameter_count",
    "objective_id",
    "optimizer_recipe",
    "prediction_semantics",
    "ensemble_id",
    "implementation_commit",
    "implementation_sha256",
    "model_state_dict",
    "scalers",
})


@dataclass(frozen=True)
class StateRoleFeatures:
    """Three frozen representations for one semantic state."""

    semantic_state_id: str
    trajectory_state_id: str
    image_group_id: str
    role_atom_x: Mapping[str, np.ndarray]
    role_state_x: Mapping[str, np.ndarray]
    action_x16: np.ndarray
    role_atom_ids: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.semantic_state_id or not self.trajectory_state_id or not self.image_group_id:
            raise ValueError("V9B1 state/trajectory/group identity is empty")
        if set(self.role_atom_x) != set(ROLE_ORDER) or set(self.role_state_x) != set(ROLE_ORDER):
            raise ValueError("V9B1 feature roles are incomplete")
        if set(self.role_atom_ids) != set(ROLE_ORDER):
            raise ValueError("V9B1 atom identity roles are incomplete")
        if self.action_x16.shape != (len(ACTIONS), ACTION_DIM):
            raise ValueError("V9B1 action matrix geometry drift")
        if not np.isfinite(self.action_x16).all():
            raise ValueError("V9B1 action matrix is non-finite")
        for role in ROLE_ORDER:
            atom = np.asarray(self.role_atom_x[role])
            state = np.asarray(self.role_state_x[role])
            ids = np.asarray(self.role_atom_ids[role])
            if (
                atom.ndim != 2
                or atom.shape[1] != ATOM_DIM
                or state.shape != (STATE_DIM,)
                or ids.shape != (len(atom),)
                or len(set(ids.astype(str).tolist())) != len(ids)
                or not np.isfinite(atom).all()
                or not np.isfinite(state).all()
                or len(atom) <= 0
            ):
                raise ValueError(f"V9B1 {role} state feature geometry drift")
            weights = np.asarray(atom[:, 0], dtype=np.float64)
            if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=2e-6):
                raise ValueError(f"V9B1 {role} inference weights do not conserve area")


@dataclass(frozen=True)
class RoleFeatureArrays:
    """One complete representation in frozen semantic-state order."""

    role_id: str
    atom_x: np.ndarray
    state_x: np.ndarray
    action_x16: np.ndarray
    state_lengths: np.ndarray
    state_ids: np.ndarray
    trajectory_state_ids: np.ndarray
    image_group_ids: np.ndarray
    atom_ids: np.ndarray

    def validate(self, expected_states: int | None = None) -> None:
        if self.role_id not in ROLE_ORDER:
            raise ValueError("V9B1 feature role is unregistered")
        n_states = len(self.state_ids)
        raw_lengths = np.asarray(self.state_lengths)
        if raw_lengths.dtype.kind not in {"i", "u"}:
            raise ValueError("V9B1 state lengths must have an integer dtype")
        lengths = raw_lengths.astype(np.int64)
        if expected_states is not None and n_states != expected_states:
            raise ValueError("V9B1 feature state count drift")
        if (
            self.atom_x.shape != (len(self.atom_ids), ATOM_DIM)
            or self.state_x.shape != (n_states, STATE_DIM)
            or self.action_x16.shape != (n_states, len(ACTIONS), ACTION_DIM)
            or lengths.shape != (n_states,)
            or np.any(lengths <= 0)
            or int(lengths.sum()) != len(self.atom_x)
            or self.trajectory_state_ids.shape != (n_states,)
            or self.image_group_ids.shape != (n_states,)
            or len(set(self.state_ids.astype(str).tolist())) != n_states
            or len(set(self.atom_ids.astype(str).tolist())) != len(self.atom_ids)
            or not all(np.isfinite(value).all() for value in (
                self.atom_x, self.state_x, self.action_x16,
            ))
            or np.any(np.asarray(self.atom_x[:, 0], dtype=np.float64) <= 0)
            or not np.array_equal(self.action_x16[:, :, 0], self.action_x16[:, :, 7])
            or not np.all(np.isin(self.action_x16[:, :, 0], (0.0, 1.0)))
            or not np.all(self.action_x16[:, 0, 0] == 1.0)
        ):
            raise ValueError(f"V9B1 {self.role_id} complete feature geometry drift")
        offsets = np.r_[0, np.cumsum(lengths[:-1])]
        sums = np.add.reduceat(np.asarray(self.atom_x[:, 0], dtype=np.float64), offsets)
        if not np.allclose(sums, 1.0, rtol=0, atol=2e-6):
            raise ValueError(f"V9B1 {self.role_id} area weights do not conserve per state")

    def npz_payload(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "atom_x": np.asarray(self.atom_x, dtype=np.float32),
            "state_x": np.asarray(self.state_x, dtype=np.float32),
            "action_x16": np.asarray(self.action_x16, dtype=np.float32),
            "state_lengths": np.asarray(self.state_lengths, dtype=np.int32),
            "state_ids": np.asarray(self.state_ids),
            "trajectory_state_ids": np.asarray(self.trajectory_state_ids),
            "image_group_ids": np.asarray(self.image_group_ids),
            "atom_ids": np.asarray(self.atom_ids),
        }


@dataclass(frozen=True)
class FrozenCheckpointModel:
    """A model/scaler pair created only by the strict checkpoint validator."""

    family: str
    seed: int
    bundle_role: str
    checkpoint_sha256: str
    model: torch.nn.Module
    scalers: Mapping[str, Any]
    _marker: object

    def __post_init__(self) -> None:
        if self._marker is not _CHECKPOINT_VALIDATION_MARKER:
            raise ValueError("V9B1 checkpoint model bypassed strict validation")


@dataclass(frozen=True)
class SeedContinuousPrediction:
    family: str
    bundle_role: str
    seed: int
    checkpoint_sha256: str
    feature_payload_sha256: str
    state_ids: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.state_ids)
        values = np.asarray(self.values)
        if (
            self.family not in FAMILY_ROLE
            or self.bundle_role != FAMILY_ROLE[self.family]
            or self.seed not in SEED_ORDER
            or any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in (self.checkpoint_sha256, self.feature_payload_sha256)
            )
            or ids.ndim != 1
            or ids.dtype.kind not in {"U", "S"}
            or values.shape != (len(ids), len(ACTIONS))
            or values.dtype.kind != "f"
            or len(set(ids.astype(str).tolist())) != len(ids)
            or np.any(ids.astype(str) == "")
            or not np.isfinite(values).all()
        ):
            raise ValueError("V9B1 seed prediction identity/geometry drift")


@dataclass(frozen=True)
class EnsembleContinuousPrediction:
    family: str
    bundle_role: str
    feature_payload_sha256: str
    checkpoint_sha256_by_seed: Mapping[int, str]
    state_ids: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.state_ids)
        values = np.asarray(self.values)
        if (
            self.family not in FAMILY_ROLE
            or self.bundle_role != FAMILY_ROLE[self.family]
            or tuple(self.checkpoint_sha256_by_seed) != SEED_ORDER
            or any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in (
                    self.feature_payload_sha256,
                    *self.checkpoint_sha256_by_seed.values(),
                )
            )
            or len(set(self.checkpoint_sha256_by_seed.values())) != len(SEED_ORDER)
            or ids.ndim != 1
            or ids.dtype.kind not in {"U", "S"}
            or values.shape != (len(ids), len(ACTIONS))
            or values.dtype != np.dtype(np.float64)
            or len(set(ids.astype(str).tolist())) != len(ids)
            or np.any(ids.astype(str) == "")
            or not np.isfinite(values).all()
        ):
            raise ValueError("V9B1 ensemble prediction identity/geometry drift")


def validate_fullfit_checkpoint_for_inference(
    payload: Mapping[str, Any], *, family: str, seed: int,
    protocol: Mapping[str, Any], expected_epoch_lock_sha256: str,
    expected_config_sha256: str,
    expected_bundle_identities: Mapping[str, Mapping[str, Any]],
    expected_implementation_commit: str,
    expected_implementation_sha256: Mapping[str, str],
    expected_checkpoint_sha256: str,
) -> FrozenCheckpointModel:
    """Admit one authenticated full-fit payload and construct its exact model.

    Physical checkpoint bytes are authenticated by the caller against the
    final 18-run lock before deserialization.  This second gate binds every
    scientific field inside that payload to the same lock/config authorities;
    no caller can pair an arbitrary model with an unrelated scaler or family.
    """

    from rail3.models.tmlr_v9b1 import fullfit as fullfit_contract

    if (
        not isinstance(payload, Mapping)
        or set(payload) != _CHECKPOINT_KEYS
        or family not in FAMILY_ROLE
        or type(seed) is not int
        or seed not in SEED_ORDER
        or len(expected_checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_checkpoint_sha256)
    ):
        raise ValueError("V9B1 checkpoint root/family/seed contract drift")
    expected_objective = (
        "R3_FROZEN_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK_NO_COST"
        if family == "R3_P" else "STATE_HUBER_DELTA_0_05"
    )
    expected_dimensions = {
        "atom": ATOM_DIM,
        "state": STATE_DIM,
        "action": ACTION_DIM,
        "global_per_action": GLOBAL_DIM,
    }
    expected_values = {
        "schema_version": "rail3.tmlr-v9b1-fullfit-checkpoint.v1",
        "family": family,
        "seed": seed,
        "fullfit_epoch": fullfit_contract.FULLFIT_EPOCHS[family],
        "epochs_completed": fullfit_contract.FULLFIT_EPOCHS[family],
        "epoch_rule_id": fullfit_contract.EPOCH_RULE_ID,
        "epoch_lock_sha256": expected_epoch_lock_sha256,
        "config_sha256": expected_config_sha256,
        "bundle_role": FAMILY_ROLE[family],
        "bundle_identities": dict(expected_bundle_identities),
        "prospective_schema_id": PROSPECTIVE_SCHEMA_ID,
        "feature_dimensions": expected_dimensions,
        "model_class": fullfit_contract.EXPECTED_MODEL_CLASSES[family],
        "parameter_count": fullfit_contract.EXPECTED_PARAMETER_COUNTS[family],
        "objective_id": expected_objective,
        "optimizer_recipe": dict(fullfit_contract.OPTIMIZER_RECIPE),
        "prediction_semantics": fullfit_contract.PREDICTION_SEMANTICS,
        "ensemble_id": ENSEMBLE_ID,
        "implementation_commit": expected_implementation_commit,
        "implementation_sha256": dict(expected_implementation_sha256),
    }
    if any(payload.get(key) != value for key, value in expected_values.items()):
        raise ValueError("V9B1 checkpoint scientific authority drift")

    atomic = family in ATOMIC_FAMILIES
    expected_scaler_keys = {
        "state_mean", "state_scale", "action_mean", "action_scale",
    } | ({"atom_mean", "atom_scale"} if atomic else set())
    raw_scalers = payload.get("scalers")
    if not isinstance(raw_scalers, Mapping) or set(raw_scalers) != expected_scaler_keys:
        raise ValueError("V9B1 checkpoint scaler registry drift")
    scaler_dimensions = {"state": STATE_DIM, "action": ACTION_DIM, "atom": ATOM_DIM}
    scalers: dict[str, np.ndarray] = {}
    for name in sorted(expected_scaler_keys):
        tensor = raw_scalers[name]
        prefix = name.rsplit("_", 1)[0]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.shape != (scaler_dimensions[prefix],)
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all().item())
            or (name.endswith("_scale") and bool(torch.any(tensor <= 0).item()))
        ):
            raise ValueError("V9B1 checkpoint scaler payload drift")
        copied = tensor.detach().numpy().copy()
        copied.setflags(write=False)
        scalers[name] = copied

    model = fullfit_contract.build_fullfit_model(family, protocol).cpu()
    expected_state = model.state_dict()
    observed_state = payload.get("model_state_dict")
    if not isinstance(observed_state, Mapping) or set(observed_state) != set(expected_state):
        raise ValueError("V9B1 checkpoint model-state registry drift")
    normalized_state: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected_state.items():
        observed_tensor = observed_state[name]
        if (
            not isinstance(observed_tensor, torch.Tensor)
            or observed_tensor.device.type != "cpu"
            or observed_tensor.layout != torch.strided
            or observed_tensor.shape != expected_tensor.shape
            or observed_tensor.dtype != expected_tensor.dtype
            or not bool(torch.isfinite(observed_tensor).all().item())
        ):
            raise ValueError("V9B1 checkpoint model-state tensor drift")
        normalized_state[name] = observed_tensor.detach().clone()
    try:
        model.load_state_dict(normalized_state, strict=True)
    except RuntimeError as error:
        raise ValueError("V9B1 checkpoint strict model load failed") from error
    model.eval()
    return FrozenCheckpointModel(
        family=family,
        seed=seed,
        bundle_role=FAMILY_ROLE[family],
        checkpoint_sha256=expected_checkpoint_sha256,
        model=model,
        scalers=MappingProxyType(scalers),
        _marker=_CHECKPOINT_VALIDATION_MARKER,
    )


def _canonical_runtime(result: CandidateGenerationResult) -> float:
    if result.candidate is not None:
        value = result.candidate.runtime_seconds
    elif result.no_result is not None and result.no_result.telemetry is not None:
        value = result.no_result.telemetry.runtime_seconds
    elif result.failure is not None and result.failure.telemetry is not None:
        value = result.failure.telemetry.runtime_seconds
    else:
        value = 0.0
    runtime = float(value or 0.0)
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError("V9B1 canonical A0 runtime is invalid")
    return runtime


def _outcome_cost(outcome: ActionOutcome) -> float:
    value = outcome.telemetry.get("action_cost_seconds")
    if type(value) not in {int, float}:
        raise ValueError(f"V9B1 {outcome.action_code} incurred cost is absent")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"V9B1 {outcome.action_code} incurred cost is invalid")
    return result


def _candidate_inputs(
    *, canonical: CandidateGenerationResult, canonical_action_id: str,
    outcomes: Sequence[ActionOutcome], class_id: int,
) -> tuple[list[CandidateMaskInput], list[dict[str, Any]]]:
    if tuple(outcome.action_code for outcome in outcomes) != ("A1", "A2"):
        raise ValueError("V9B1 label-free feature boundary accepts only A1/A2 outcomes")
    candidates = [CandidateMaskInput.create(
        candidate_id=candidate.candidate_id,
        action_id=canonical_action_id,
        action_code="A0",
        class_id=class_id,
        prompt_source="CANONICAL_TEXT",
        mask=decode_binary_mask_array(
            candidate.mask_rle, candidate.width, candidate.height,
        ),
    ) for candidate in canonical.all_candidates]
    lineage: list[dict[str, Any]] = [{
        "action_code": "A0",
        "action_id": canonical_action_id,
        "cache_key": None,
        "feasible": True,
        "outcome": (
            "result" if canonical.candidate is not None
            else "no_result" if canonical.no_result is not None else "failure"
        ),
        "reason_code": (
            canonical.no_result.reason_code if canonical.no_result is not None
            else canonical.failure.reason_code if canonical.failure is not None else None
        ),
        "candidate_ids": [candidate.candidate_id for candidate in canonical.all_candidates],
        "prompt_source": "CANONICAL_TEXT",
        "cost_seconds": _canonical_runtime(canonical),
    }]
    for outcome in outcomes:
        if outcome.class_id != class_id or outcome.outcome == "failure" or not outcome.feasible:
            raise ValueError(f"V9B1 {outcome.action_code} is not a legal base observation")
        for candidate in outcome.candidates:
            candidates.append(CandidateMaskInput.create(
                candidate_id=candidate.candidate_id,
                action_id=outcome.action_id,
                action_code=outcome.action_code,
                class_id=class_id,
                prompt_source=outcome.action_code,
                mask=decode_binary_mask_array(
                    candidate.mask_rle, candidate.width, candidate.height,
                ),
            ))
        lineage.append({
            "action_code": outcome.action_code,
            "action_id": outcome.action_id,
            "cache_key": None,
            "feasible": True,
            "outcome": outcome.outcome,
            "reason_code": outcome.reason_code,
            "candidate_ids": [candidate.candidate_id for candidate in outcome.candidates],
            "source_candidate_id": outcome.source_candidate_id,
            "prompt_source": outcome.action_code,
            "prompt": outcome.prompt,
            "transform": outcome.transform,
            "cost_seconds": _outcome_cost(outcome),
        })
    return candidates, lineage


def _feature_partition_view(partition: Mapping[str, Any]) -> dict[str, Any]:
    """Pad causal lineage with inert adaptive entries for the frozen encoder."""

    lineage = [dict(item) for item in partition["base_action_lineage"]]
    if tuple(item["action_code"] for item in lineage) != ("A0", "A1", "A2"):
        raise ValueError("V9B1 causal base lineage is not A0/A1/A2")
    lineage.extend({
        "action_code": code,
        "action_id": f"not-observed-{code}",
        "cache_key": None,
        "feasible": False,
        "outcome": "not_observed",
        "reason_code": "OUTSIDE_PRE_ACTION_FEATURE_BOUNDARY",
        "candidate_ids": [],
        "prompt_source": code,
        "cost_seconds": 0.0,
    } for code in ("A3", "A4", "A5", "A6"))
    return {**dict(partition), "base_action_lineage": lineage}


def _constant_union_partition(height: int, width: int) -> RegionPartition:
    labels = np.zeros((height, width), dtype=np.int32)
    return RegionPartition(
        label_map=labels,
        rectangles=((0, 0, width, height),),
        kind="CANDIDATE_UNION_COMPONENTS",
    )


def _regional_atom_ids(
    *, role: str, semantic_state_id: str, partition: RegionPartition,
) -> np.ndarray:
    values = []
    for index in range(partition.region_count):
        values.append(stable_id("v9b1_label_free_region_atom", {
            "role": role,
            "semantic_state_id": semantic_state_id,
            "region_index": index,
            "rectangle": list(partition.rectangles[index]),
        }))
    return np.asarray(values)


def build_state_role_features(
    *, state: SemanticState, canonical: CandidateGenerationResult,
    a1: ActionOutcome, a2: ActionOutcome, model_spec_id: str,
) -> StateRoleFeatures:
    """Build all three label-free roles without accepting future outcomes."""

    if canonical.failure is not None:
        raise ValueError("V9B1 canonical A0 failure blocks label-free features")
    item = state.image.item_payload()
    prompt, metadata = canonical_voc_prompts()[state.class_id - 1]
    if metadata.class_id != state.class_id or metadata.canonical_text != state.canonical_prompt:
        raise ValueError("V9B1 canonical prompt identity drift")
    identity = voc_cache_identity(item, prompt, model_spec_id=model_spec_id)
    plans = build_action_plans(
        item,
        class_id=state.class_id,
        canonical_result=canonical,
        canonical_cache_key=identity.cache_key,
    )
    if tuple(plan.action_code for plan in plans) != ("A1", "A2", "A3", "A4", "A5", "A6"):
        raise ValueError("V9B1 action plan order drift")
    for plan, outcome in zip(plans[:2], (a1, a2)):
        if (
            outcome.action_id != plan.action_id
            or outcome.state_id != plan.state_id
            or outcome.asset_id != item["asset_id"]
            or outcome.image_sha256 != item["image_sha256"]
            or outcome.class_id != state.class_id
        ):
            raise ValueError("V9B1 base action outcome/plan identity drift")
    trajectory_state_id = plans[0].state_id
    if any(plan.state_id != trajectory_state_id for plan in plans):
        raise ValueError("V9B1 plans cross trajectory-state identities")
    canonical_action_id = stable_id("canonical_action", {
        "protocol_sha256": PROTOCOL_SHA256,
        "state_id": trajectory_state_id,
        "canonical_cache_key": identity.cache_key,
    })
    candidates, lineage = _candidate_inputs(
        canonical=canonical,
        canonical_action_id=canonical_action_id,
        outcomes=(a1, a2),
        class_id=state.class_id,
    )
    partition = build_m06e_causal_partition(
        asset_id=item["asset_id"],
        sample_id=state.image.canonical_image_id,
        image_group_id=item["asset_id"],
        image_sha256=item["image_sha256"],
        state_id=trajectory_state_id,
        class_id=state.class_id,
        width=state.image.width,
        height=state.image.height,
        candidates=candidates,
        base_action_lineage=lineage,
    )
    f0_atom, f0_state = observed_partition_feature_matrices(
        partition=_feature_partition_view(partition),
        canonical_candidate_ids=(candidate.candidate_id for candidate in canonical.all_candidates),
    )
    valid = np.ones((state.image.height, state.image.width), dtype=bool)
    rectangle = guillotine_partition(valid, len(f0_atom))
    observable_union = np.zeros_like(valid)
    for candidate in candidates:
        if candidate.mask.shape != observable_union.shape:
            raise ValueError("V9B1 observable candidate geometry drift")
        observable_union |= np.asarray(candidate.mask, dtype=bool)
    union = (
        _constant_union_partition(state.image.height, state.image.width)
        if not observable_union.any() or observable_union.all()
        else union_component_partition(observable_union, valid)
    )
    base_cost = sum(float(item.get("cost_seconds", 0.0)) for item in lineage)
    rect_atom, rect_state = regional_feature_matrices(
        partition=rectangle,
        class_id=state.class_id,
        base_lineage=lineage,
        base_cost_seconds=base_cost,
    )
    union_atom, union_state = regional_feature_matrices(
        partition=union,
        class_id=state.class_id,
        base_lineage=lineage,
        base_cost_seconds=base_cost,
        observable_union=observable_union,
    )
    matrix = action_matrix(
        stop=stop_descriptor(
            state_id=trajectory_state_id,
            canonical_action_id=canonical_action_id,
        ),
        adaptive=tuple(descriptor_from_plan(plan) for plan in plans[2:]),
    )
    return StateRoleFeatures(
        semantic_state_id=state.semantic_state_id,
        trajectory_state_id=trajectory_state_id,
        image_group_id=item["asset_id"],
        role_atom_x={
            "F0_P": f0_atom.astype(np.float32),
            "RECT_FULL_RASTER_P": rect_atom.astype(np.float32),
            "UNION_FULL_RASTER_P": union_atom.astype(np.float32),
        },
        role_state_x={
            "F0_P": f0_state.astype(np.float32),
            "RECT_FULL_RASTER_P": rect_state.astype(np.float32),
            "UNION_FULL_RASTER_P": union_state.astype(np.float32),
        },
        action_x16=matrix.astype(np.float32),
        role_atom_ids={
            "F0_P": np.asarray([str(atom["causal_atom_id"]) for atom in partition["atoms"]]),
            "RECT_FULL_RASTER_P": _regional_atom_ids(
                role="RECT_FULL_RASTER_P",
                semantic_state_id=state.semantic_state_id,
                partition=rectangle,
            ),
            "UNION_FULL_RASTER_P": _regional_atom_ids(
                role="UNION_FULL_RASTER_P",
                semantic_state_id=state.semantic_state_id,
                partition=union,
            ),
        },
    )


def concatenate_role_features(
    rows: Sequence[StateRoleFeatures], *, expected_states: int | None = None,
) -> dict[str, RoleFeatureArrays]:
    if not rows:
        raise ValueError("V9B1 feature concatenation has no states")
    if expected_states is not None and len(rows) != expected_states:
        raise ValueError("V9B1 feature concatenation state count drift")
    state_ids = np.asarray([row.semantic_state_id for row in rows])
    trajectory_ids = np.asarray([row.trajectory_state_id for row in rows])
    group_ids = np.asarray([row.image_group_id for row in rows])
    actions = np.stack([row.action_x16 for row in rows]).astype(np.float32)
    result: dict[str, RoleFeatureArrays] = {}
    for role in ROLE_ORDER:
        values = RoleFeatureArrays(
            role_id=role,
            atom_x=np.concatenate([row.role_atom_x[role] for row in rows]).astype(np.float32),
            state_x=np.stack([row.role_state_x[role] for row in rows]).astype(np.float32),
            action_x16=actions.copy(),
            state_lengths=np.asarray([len(row.role_atom_x[role]) for row in rows], dtype=np.int32),
            state_ids=state_ids.copy(),
            trajectory_state_ids=trajectory_ids.copy(),
            image_group_ids=group_ids.copy(),
            atom_ids=np.concatenate([row.role_atom_ids[role] for row in rows]),
        )
        values.validate(expected_states)
        result[role] = values
    return result


def _scaler(payload: Mapping[str, Any], prefix: str, dimension: int) -> FrozenScaler:
    mean = np.asarray(payload[f"{prefix}_mean"], dtype=np.float32)
    scale = np.asarray(payload[f"{prefix}_scale"], dtype=np.float32)
    if (
        mean.shape != (dimension,)
        or scale.shape != (dimension,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError(f"V9B1 frozen {prefix} scaler drift")
    return FrozenScaler(mean=mean, scale=scale)


def apply_frozen_voc_scalers(
    features: RoleFeatureArrays, scaler_payload: Mapping[str, Any], *, atomic: bool,
) -> ProspectiveFeatures:
    """Transform COCO features with checkpoint scalers; never fit on COCO."""

    features.validate()
    expected_keys = {"state_mean", "state_scale", "action_mean", "action_scale"}
    if atomic:
        expected_keys |= {"atom_mean", "atom_scale"}
    if set(scaler_payload) != expected_keys:
        raise ValueError("V9B1 checkpoint scaler key set drift")
    state_scaler = _scaler(scaler_payload, "state", STATE_DIM)
    action_scaler = _scaler(scaler_payload, "action", ACTION_DIM)
    atom_scaler = _scaler(scaler_payload, "atom", ATOM_DIM) if atomic else None
    state = state_scaler.transform(features.state_x)
    action = action_scaler.transform(
        features.action_x16.reshape(-1, ACTION_DIM),
    ).reshape(features.action_x16.shape)
    atom = None if atom_scaler is None else atom_scaler.transform(features.atom_x)
    if any(not np.isfinite(value).all() for value in (
        state, action, *(tuple() if atom is None else (atom,)),
    )):
        raise ValueError("V9B1 frozen-scaler transform produced non-finite values")
    return ProspectiveFeatures(
        state=state,
        action=action,
        atom=atom,
        scalers=ProspectiveScalers(
            state=state_scaler, action=action_scaler, atom=atom_scaler,
        ),
    )


def predict_continuous_residuals(
    *, checkpoint: FrozenCheckpointModel, features: RoleFeatureArrays,
    feature_payload_sha256: str, device: torch.device,
) -> SeedContinuousPrediction:
    """Return one seed's full continuous [state, STOP/A3--A6] prediction."""

    family = checkpoint.family
    if (
        family not in FAMILY_ROLE
        or features.role_id != FAMILY_ROLE[family]
        or len(feature_payload_sha256) != 64
        or any(char not in "0123456789abcdef" for char in feature_payload_sha256)
    ):
        raise ValueError("V9B1 family/feature-role registration drift")
    atomic = family in ATOMIC_FAMILIES
    scaled = apply_frozen_voc_scalers(features, checkpoint.scalers, atomic=atomic)
    model = checkpoint.model.to(device)
    model.eval()
    with torch.no_grad():
        state = torch.from_numpy(scaled.state).to(device)
        action = torch.from_numpy(scaled.action).to(device)
        if family in GLOBAL_FAMILIES:
            fused = torch.cat((
                state[:, None, :].expand(-1, len(ACTIONS), -1), action,
            ), dim=2)
            if fused.shape[2] != GLOBAL_DIM:
                raise ValueError("V9B1 global input dimension drift")
            prediction = model(fused.reshape(-1, GLOBAL_DIM)).reshape(-1, len(ACTIONS))
        else:
            if scaled.atom is None:
                raise ValueError("V9B1 atomic model has no scaled atom features")
            atom = torch.from_numpy(scaled.atom).to(device)
            lengths = torch.from_numpy(
                np.asarray(features.state_lengths, dtype=np.int64),
            ).to(device)
            weights = torch.from_numpy(
                inference_weights_from_features(
                    features.atom_x, features.state_lengths,
                ),
            ).to(device)
            arguments = (atom, state, action, lengths, weights)
            if len(atom) >= MEMORY_BOUNDED_ATOM_THRESHOLD:
                _, prediction = memory_bounded_shared_atomic_forward(
                    model, *arguments, checkpoint_activations=False,
                )
            else:
                _, prediction = model(*arguments)
    result = prediction.detach().cpu().numpy().astype(np.float32)
    if result.shape != (len(features.state_ids), len(ACTIONS)) or not np.isfinite(result).all():
        raise ValueError("V9B1 continuous prediction geometry/finite gate failed")
    return SeedContinuousPrediction(
        family=checkpoint.family,
        bundle_role=checkpoint.bundle_role,
        seed=checkpoint.seed,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        feature_payload_sha256=feature_payload_sha256,
        state_ids=np.asarray(features.state_ids).astype(str),
        values=result,
    )


def ensemble_continuous_predictions(
    seed_predictions: Mapping[int, SeedContinuousPrediction],
) -> EnsembleContinuousPrediction:
    if tuple(seed_predictions) != SEED_ORDER:
        raise ValueError("V9B1 seed predictions are not in frozen 13,37,71 order")
    records = [seed_predictions[seed] for seed in SEED_ORDER]
    if any(
        not isinstance(record, SeedContinuousPrediction) or record.seed != seed
        for seed, record in zip(SEED_ORDER, records)
    ):
        raise ValueError("V9B1 seed prediction authority drift")
    reference_ids = np.asarray(records[0].state_ids).astype(str)
    family = records[0].family
    bundle_role = records[0].bundle_role
    feature_payload_sha256 = records[0].feature_payload_sha256
    if any(
        record.family != family
        or record.bundle_role != bundle_role
        or record.feature_payload_sha256 != feature_payload_sha256
        or not np.array_equal(reference_ids, np.asarray(record.state_ids).astype(str))
        for record in records[1:]
    ):
        raise ValueError("V9B1 seed prediction family/feature/state-order drift")
    arrays = [np.asarray(record.values) for record in records]
    if len({array.shape for array in arrays}) != 1 or any(
        array.ndim != 2 or array.shape[1] != len(ACTIONS) or not np.isfinite(array).all()
        for array in arrays
    ):
        raise ValueError("V9B1 seed prediction geometry/finite drift")
    result = np.stack(
        [array.astype(np.float64, copy=False) for array in arrays], axis=0,
    ).mean(axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("V9B1 ensemble prediction is non-finite")
    return EnsembleContinuousPrediction(
        family=family,
        bundle_role=bundle_role,
        feature_payload_sha256=feature_payload_sha256,
        checkpoint_sha256_by_seed=MappingProxyType({
            record.seed: record.checkpoint_sha256 for record in records
        }),
        state_ids=reference_ids.copy(),
        values=result,
    )


def select_lambda_zero_actions(
    ensemble: EnsembleContinuousPrediction, feasibility: np.ndarray, *,
    state_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the minimum predicted residual, with frozen first-action ties."""

    if not isinstance(ensemble, EnsembleContinuousPrediction):
        raise ValueError("V9B1 lambda-zero ensemble bypassed identity validation")
    values = np.asarray(ensemble.values)
    decision_state_ids = np.asarray(state_ids)
    raw_feasibility = np.asarray(feasibility)
    if raw_feasibility.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("V9B1 feasibility dtype is not binary numeric")
    if raw_feasibility.dtype.kind == "f" and not np.isfinite(raw_feasibility).all():
        raise ValueError("V9B1 feasibility contains a non-finite value")
    if not np.all(np.isin(raw_feasibility, (0, 1))):
        raise ValueError("V9B1 feasibility is not exactly binary")
    feasible = raw_feasibility.astype(bool, copy=False)
    if (
        decision_state_ids.ndim != 1
        or decision_state_ids.dtype.kind not in {"U", "S"}
        or not np.array_equal(
            np.asarray(ensemble.state_ids).astype(str),
            decision_state_ids.astype(str),
        )
        or values.ndim != 2
        or values.shape[1] != len(ACTIONS)
        or feasible.shape != values.shape
        or not np.isfinite(values).all()
        or not np.all(feasible[:, 0])
        or np.any(~feasible.any(axis=1))
    ):
        raise ValueError("V9B1 lambda-zero decision input drift")
    masked = np.where(feasible, values, np.inf)
    indices = np.argmin(masked, axis=1).astype(np.int8)
    actions = np.asarray(DECISION_ACTION_ORDER)[indices]
    selected = values[np.arange(len(values)), indices]
    if not np.isfinite(selected).all():
        raise ValueError("V9B1 selected prediction is non-finite")
    return actions, selected


def prediction_payload_sha256(
    *, state_ids: np.ndarray, prediction: np.ndarray,
) -> str:
    """Stable content identity independent of NPZ container metadata."""

    raw_ids = np.asarray(state_ids)
    ids = raw_ids.astype(str)
    values = np.asarray(prediction)
    if (
        raw_ids.ndim != 1
        or raw_ids.dtype.kind not in {"U", "S"}
        or len(set(ids.tolist())) != len(ids)
        or np.any(ids == "")
        or values.shape != (len(ids), len(ACTIONS))
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise ValueError("V9B1 prediction payload identity geometry drift")
    digest = hashlib.sha256()
    digest.update(b"TMLR_V9B1_CONTINUOUS_PREDICTION_V1\0")
    for value in ids:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def state_id_order_sha256(state_ids: np.ndarray) -> str:
    ids = np.asarray(state_ids)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in {"U", "S"}
        or len(set(ids.astype(str).tolist())) != len(ids)
        or np.any(ids.astype(str) == "")
    ):
        raise ValueError("V9B1 state-ID order identity drift")
    digest = hashlib.sha256(b"TMLR_V9B1_STATE_ID_ORDER_V1\0")
    for value in ids.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def feature_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    """Stable identity for one role, independent of NPZ ZIP metadata."""

    expected = {
        "atom_x", "state_x", "action_x16", "state_lengths", "state_ids",
        "trajectory_state_ids", "image_group_ids", "atom_ids",
    }
    if set(payload) != expected:
        raise ValueError("V9B1 feature payload key-set drift")
    digest = hashlib.sha256(b"TMLR_V9B1_LABEL_FREE_FEATURE_ARRAY_V1\0")
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        encoded_name = name.encode("utf-8")
        encoded_dtype = value.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(len(value.shape).to_bytes(4, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


__all__ = [
    "ATOMIC_FAMILIES",
    "DECISION_ACTION_ORDER",
    "ENSEMBLE_ID",
    "EnsembleContinuousPrediction",
    "FAMILY_ROLE",
    "FrozenCheckpointModel",
    "GLOBAL_FAMILIES",
    "ROLE_ORDER",
    "RoleFeatureArrays",
    "SEED_ORDER",
    "SeedContinuousPrediction",
    "StateRoleFeatures",
    "apply_frozen_voc_scalers",
    "build_state_role_features",
    "concatenate_role_features",
    "ensemble_continuous_predictions",
    "feature_payload_sha256",
    "predict_continuous_residuals",
    "prediction_payload_sha256",
    "select_lambda_zero_actions",
    "state_id_order_sha256",
    "validate_fullfit_checkpoint_for_inference",
]
