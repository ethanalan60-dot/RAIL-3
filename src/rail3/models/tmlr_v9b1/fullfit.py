"""Frozen full-fit mechanics for the six TMLR V9B1 residual families.

This module has no COCO, SAM, trajectory, prediction-artifact, or performance
evaluation API.  It accepts only already-authenticated V6 FIT-only bundle
arrays and returns one final CPU model state plus full-population scalers.
Publication and path admission belong to the thin command-line wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np
import torch
from torch import nn

from rail3.models.m06e.features import fit_scaler
from rail3.models.m06e.models import parameter_count
from rail3.models.tmlr_v6.phase2_models import build_phase2_model
from rail3.models.tmlr_v6.phase2_objectives import AtomicObjectiveInputs, r3_objective
from rail3.models.tmlr_v6.schema import (
    ACTION_DIM,
    ATOM_DIM,
    GLOBAL_DIM,
    PROSPECTIVE_SCHEMA_ID,
    STATE_DIM,
    ProspectiveFeatures,
    ProspectiveScalers,
    scaler_payload,
)
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    FEATURE_ARRAY_NAMES,
    MEMORY_BOUNDED_ATOM_THRESHOLD,
    TARGET_ARRAY_NAMES,
    _assert_target_defined_implies_plan_feasible,
    _seed,
    _snapshot,
    _state_huber,
    build_registered_model,
    inference_weights_from_features,
    memory_bounded_shared_atomic_forward,
    reconstruct_target_states,
)


FAMILIES = (
    "R0_SMALL_P",
    "R0_CM_P",
    "R1_P",
    "RECT_P",
    "UNION_P",
    "R3_P",
)
SEEDS = (13, 37, 71)
FULLFIT_EPOCHS: Mapping[str, int] = {
    "R0_SMALL_P": 120,
    "R0_CM_P": 120,
    "R1_P": 120,
    "RECT_P": 120,
    "UNION_P": 120,
    "R3_P": 108,
}
FAMILY_BUNDLE: Mapping[str, str] = {
    "R0_SMALL_P": "F0_P",
    "R0_CM_P": "F0_P",
    "R1_P": "F0_P",
    "RECT_P": "RECT_FULL_RASTER_P",
    "UNION_P": "UNION_FULL_RASTER_P",
    "R3_P": "F0_P",
}
EXPECTED_PARAMETER_COUNTS: Mapping[str, int] = {
    "R0_SMALL_P": 7_809,
    "R0_CM_P": 103_369,
    "R1_P": 103_521,
    "RECT_P": 103_521,
    "UNION_P": 103_521,
    "R3_P": 103_521,
}
EXPECTED_MODEL_CLASSES: Mapping[str, str] = {
    "R0_SMALL_P": "rail3.models.m06e.models.GlobalResidual",
    "R0_CM_P": "rail3.models.tmlr_v6.models.CapacityMatchedGlobalResidual",
    "R1_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "RECT_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "UNION_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
    "R3_P": "rail3.models.tmlr_v6.models.ProspectiveSharedAtomicResidual",
}
ATOMIC_FAMILIES = frozenset({"R1_P", "RECT_P", "UNION_P", "R3_P"})
GLOBAL_FAMILIES = frozenset({"R0_SMALL_P", "R0_CM_P"})
EPOCH_RULE_ID = "V9_FULLFIT_EPOCH_RULE_V1"
ENSEMBLE_ID = "CONTINUOUS_FLOAT64_MEAN_13_37_71_BEFORE_SELECTION"
PREDICTION_SEMANTICS = "ABSOLUTE_RESIDUAL"
OPTIMIZER_RECIPE: Mapping[str, Any] = {
    "class": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.0001,
    "scheduler": None,
    "amp": False,
    "ddp": False,
    "early_stopping": False,
    "warm_start": False,
}
FULLFIT_GROUP_COUNT = 1_364
FULLFIT_STATE_COUNT = 27_280
FULLFIT_ATOM_COUNTS: Mapping[str, int] = {
    "F0_P": 126_377,
    "RECT_FULL_RASTER_P": 126_377,
    "UNION_FULL_RASTER_P": 53_928,
}
FULLFIT_GROUP_IDS_SHA256 = (
    "a420f6e744f4984a60b91c9db22347456e57e0957601119a330cc52c3777ee1d"
)


@dataclass(frozen=True, order=True)
class FullFitJob:
    """One member of the exact six-family by three-seed grid."""

    family: str
    seed: int
    epochs: int
    bundle_role: str

    @property
    def run_id(self) -> str:
        return f"S1364__{self.family}__seed_{self.seed}__epoch_{self.epochs}"


@dataclass(frozen=True)
class FullFitBundle:
    """Validated V6 FIT-only feature/target arrays for one frozen role."""

    role_id: str
    features: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    group_ids_sha256: str


@dataclass(frozen=True)
class FullFitTrainingResult:
    """The only scientific outputs of one fixed-epoch optimization."""

    job: FullFitJob
    model_class: str
    parameter_count: int
    state_dict: Mapping[str, torch.Tensor]
    scalers: Mapping[str, np.ndarray]
    objective_id: str
    epochs_completed: int
    training_loss_finite: bool


def expected_jobs() -> tuple[FullFitJob, ...]:
    """Return the frozen 18 jobs in family-major, seed-minor order."""

    return tuple(
        FullFitJob(
            family=family,
            seed=seed,
            epochs=FULLFIT_EPOCHS[family],
            bundle_role=FAMILY_BUNDLE[family],
        )
        for family in FAMILIES
        for seed in SEEDS
    )


def validate_job(job: FullFitJob) -> None:
    if job not in expected_jobs():
        raise ValueError("TMLR V9B1 job is outside the frozen 18-job grid")


def _load_exact_npz(stream: BinaryIO, expected: frozenset[str]) -> dict[str, np.ndarray]:
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise RuntimeError(
                "TMLR V9B1 physical array schema drift: "
                f"expected={sorted(expected)}, observed={sorted(payload.files)}"
            )
        return {name: np.asarray(payload[name]) for name in sorted(expected)}


def _strings(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise RuntimeError(f"TMLR V9B1 {name} must be a one-dimensional string array")
    result = value.astype(str)
    if np.any(result == ""):
        raise RuntimeError(f"TMLR V9B1 {name} contains an empty identifier")
    return result


def _canonical_group_hash(group_ids: np.ndarray) -> str:
    values = sorted(set(_strings(group_ids, name="image_group_ids").tolist()))
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def validate_fullfit_bundle_arrays(
    *, role_id: str, features: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray], require_frozen_population: bool = True,
) -> FullFitBundle:
    """Validate a physical prospective bundle without opening any other path."""

    if role_id not in set(FAMILY_BUNDLE.values()):
        raise ValueError("TMLR V9B1 bundle role is unregistered")
    if set(features) != FEATURE_ARRAY_NAMES or set(targets) != TARGET_ARRAY_NAMES:
        raise RuntimeError("TMLR V9B1 bundle member allowlist drift")

    atom_x = np.asarray(features["atom_x"])
    state_x = np.asarray(features["state_x"])
    action_x = np.asarray(features["action_x16"])
    lengths_raw = np.asarray(features["state_lengths"])
    lengths = lengths_raw.astype(np.int64)
    state_ids = _strings(features["state_ids"], name="feature state_ids")
    group_ids = _strings(features["image_group_ids"], name="image_group_ids")
    atom_ids = _strings(features["atom_ids"], name="feature atom_ids")
    target_state_ids = _strings(targets["state_ids"], name="target state_ids")
    target_atom_ids = _strings(targets["atom_ids"], name="target atom_ids")
    n_states, n_atoms = len(state_ids), len(atom_ids)
    if (
        atom_x.shape != (n_atoms, ATOM_DIM)
        or state_x.shape != (n_states, STATE_DIM)
        or action_x.shape != (n_states, len(ACTIONS), ACTION_DIM)
        or atom_x.dtype != np.dtype(np.float32)
        or state_x.dtype != np.dtype(np.float32)
        or action_x.dtype != np.dtype(np.float32)
        or lengths_raw.dtype.kind not in {"i", "u"}
        or lengths.shape != (n_states,)
        or group_ids.shape != (n_states,)
        or np.any(lengths <= 0)
        or int(lengths.sum()) != n_atoms
        or not all(np.isfinite(value).all() for value in (atom_x, state_x, action_x))
        or len(set(state_ids.tolist())) != n_states
        or len(set(atom_ids.tolist())) != n_atoms
    ):
        raise RuntimeError("TMLR V9B1 prospective feature geometry drift")

    atom_target = np.asarray(targets["atom_target"])
    atom_weights = np.asarray(targets["atom_weights"])
    state_target = np.asarray(targets["state_target"])
    feasible = np.asarray(targets["feasible"])
    if (
        atom_target.shape != (n_atoms, len(ACTIONS))
        or atom_weights.shape != (n_atoms,)
        or state_target.shape != (n_states, len(ACTIONS))
        or feasible.shape != (n_states, len(ACTIONS))
        or atom_target.dtype != np.dtype(np.float32)
        or atom_weights.dtype != np.dtype(np.float32)
        or state_target.dtype != np.dtype(np.float32)
        or feasible.dtype.kind != "b"
        or not np.isfinite(atom_weights).all()
        or np.isinf(atom_target).any()
        or np.isinf(state_target).any()
        or np.any(atom_weights < 0)
        or not np.array_equal(state_ids, target_state_ids)
        or not np.array_equal(atom_ids, target_atom_ids)
    ):
        raise RuntimeError("TMLR V9B1 prospective feature/target join drift")

    inference_weights_from_features(atom_x, lengths)
    target_state_weight = np.add.reduceat(
        atom_weights.astype(np.float64), np.r_[0, np.cumsum(lengths[:-1])],
    )
    if not np.allclose(target_state_weight, 1.0, rtol=0.0, atol=2e-7):
        raise RuntimeError("TMLR V9B1 target weights do not conserve valid area")
    if (
        not np.array_equal(action_x[:, :, 0], action_x[:, :, 7])
        or not np.all(np.isin(action_x[:, :, 0], (0.0, 1.0)))
        or not np.array_equal(action_x[:, :, 0].astype(bool), feasible)
        or not np.all(feasible[:, 0])
    ):
        raise RuntimeError("TMLR V9B1 plan-feasibility contract drift")
    _assert_target_defined_implies_plan_feasible(state_target, feasible)
    if np.any(np.isfinite(atom_target) & ~np.repeat(feasible, lengths, axis=0)):
        raise RuntimeError("TMLR V9B1 atomic target feasibility drift")
    reconstructed = reconstruct_target_states(
        atom_target=atom_target,
        target_weights=atom_weights,
        state_lengths=lengths,
    )
    defined = np.isfinite(state_target)
    if (
        not np.array_equal(np.isfinite(reconstructed), defined)
        or not np.allclose(
            reconstructed[defined], state_target.astype(np.float64)[defined],
            rtol=0.0, atol=2e-7,
        )
    ):
        raise RuntimeError("TMLR V9B1 target-only reconstruction drift")

    group_hash = _canonical_group_hash(group_ids)
    if require_frozen_population:
        unique, counts = np.unique(group_ids, return_counts=True)
        if (
            n_states != FULLFIT_STATE_COUNT
            or n_atoms != FULLFIT_ATOM_COUNTS[role_id]
            or len(unique) != FULLFIT_GROUP_COUNT
            or not np.all(counts == 20)
            or group_hash != FULLFIT_GROUP_IDS_SHA256
        ):
            raise RuntimeError("TMLR V9B1 frozen S1364 population drift")
    return FullFitBundle(
        role_id=role_id,
        features={key: np.asarray(value) for key, value in features.items()},
        targets={key: np.asarray(value) for key, value in targets.items()},
        group_ids_sha256=group_hash,
    )


def load_fullfit_bundle(
    *, role_id: str, feature_stream: BinaryIO, target_stream: BinaryIO,
) -> FullFitBundle:
    """Load exactly two already-authenticated NPZ streams and nothing else."""

    return validate_fullfit_bundle_arrays(
        role_id=role_id,
        features=_load_exact_npz(feature_stream, FEATURE_ARRAY_NAMES),
        targets=_load_exact_npz(target_stream, TARGET_ARRAY_NAMES),
        require_frozen_population=True,
    )


def scale_fullfit_features(bundle: FullFitBundle, *, atomic: bool) -> ProspectiveFeatures:
    """Fit frozen scaler semantics on every row of the S1364 FIT population."""

    state = np.asarray(bundle.features["state_x"])
    action = np.asarray(bundle.features["action_x16"])
    if (
        state.ndim != 2 or state.shape[1] != STATE_DIM
        or action.shape != (len(state), len(ACTIONS), ACTION_DIM)
        or not np.isfinite(state).all() or not np.isfinite(action).all()
    ):
        raise ValueError("TMLR V9B1 full-fit scaler input geometry drift")
    state_scaler = fit_scaler(state)
    action_scaler = fit_scaler(action.reshape(-1, ACTION_DIM))
    scaled_state = state_scaler.transform(state)
    scaled_action = action_scaler.transform(
        action.reshape(-1, ACTION_DIM),
    ).reshape(action.shape)
    atom_scaler = None
    scaled_atom = None
    if atomic:
        atom = np.asarray(bundle.features["atom_x"])
        if atom.ndim != 2 or atom.shape[1] != ATOM_DIM or not np.isfinite(atom).all():
            raise ValueError("TMLR V9B1 full-fit atom scaler geometry drift")
        atom_scaler = fit_scaler(atom)
        scaled_atom = atom_scaler.transform(atom)
    outputs = (scaled_state, scaled_action) + (() if scaled_atom is None else (scaled_atom,))
    if any(not np.isfinite(value).all() for value in outputs):
        raise RuntimeError("TMLR V9B1 scaled feature contains a non-finite value")
    return ProspectiveFeatures(
        state=scaled_state,
        action=scaled_action,
        atom=scaled_atom,
        scalers=ProspectiveScalers(
            state=state_scaler, action=action_scaler, atom=atom_scaler,
        ),
    )


def build_fullfit_model(family: str, protocol: Mapping[str, Any]) -> nn.Module:
    if family not in FAMILIES:
        raise ValueError("TMLR V9B1 model family is unregistered")
    model = (
        build_phase2_model("R3_P")
        if family == "R3_P"
        else build_registered_model(family, protocol)
    )
    observed = parameter_count(model)
    observed_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if (
        observed != EXPECTED_PARAMETER_COUNTS[family]
        or observed_class != EXPECTED_MODEL_CLASSES[family]
    ):
        raise RuntimeError(
            f"TMLR V9B1 model identity drift: "
            f"{family}=({observed_class},{observed})"
        )
    return model


def _as_tensor(
    array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        result = result.to(dtype)
    return result.to(device)


def _atomic_forward(
    model: nn.Module, scaled: ProspectiveFeatures,
    bundle: FullFitBundle, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaled.atom is None:
        raise RuntimeError("TMLR V9B1 atomic family is missing atom features")
    lengths = _as_tensor(bundle.features["state_lengths"], device, torch.long)
    weights = _as_tensor(
        inference_weights_from_features(
            bundle.features["atom_x"], bundle.features["state_lengths"],
        ), device,
    )
    arguments = (
        _as_tensor(scaled.atom, device),
        _as_tensor(scaled.state, device),
        _as_tensor(scaled.action, device),
        lengths,
        weights,
    )
    if len(scaled.atom) >= MEMORY_BOUNDED_ATOM_THRESHOLD:
        return memory_bounded_shared_atomic_forward(model, *arguments)
    return model(*arguments)


def _loss_closure(
    *, family: str, model: nn.Module, scaled: ProspectiveFeatures,
    bundle: FullFitBundle, device: torch.device,
) -> Callable[[], torch.Tensor]:
    state_target = _as_tensor(bundle.targets["state_target"], device)
    state_mask = torch.ones(len(state_target), dtype=torch.bool, device=device)
    if family in GLOBAL_FAMILIES:
        global_x = torch.cat((
            _as_tensor(scaled.state, device)[:, None, :].expand(-1, len(ACTIONS), -1),
            _as_tensor(scaled.action, device),
        ), dim=2)

        def global_loss() -> torch.Tensor:
            return _state_huber(model(global_x), state_target, state_mask)

        return global_loss

    atom_target = _as_tensor(bundle.targets["atom_target"], device)
    atom_weights = _as_tensor(bundle.targets["atom_weights"], device)
    lengths = _as_tensor(bundle.features["state_lengths"], device, torch.long)

    if family == "R3_P":
        # AtomicObjectiveInputs has a historical cost field, but the frozen R3
        # objective does not inspect it.  An internal zero placeholder closes
        # the type boundary without loading, deriving, or persisting any cost.
        zero_cost_placeholder = torch.zeros(len(ACTIONS), device=device)

        def r3_loss() -> torch.Tensor:
            atom_prediction, state_prediction = _atomic_forward(
                model, scaled, bundle, device,
            )
            return r3_objective(AtomicObjectiveInputs(
                atom_prediction=atom_prediction,
                state_prediction=state_prediction,
                atom_target=atom_target,
                state_target=state_target,
                atom_weights=atom_weights,
                state_lengths=lengths,
                state_mask=state_mask,
                normalized_cost=zero_cost_placeholder,
            ))["total"]

        return r3_loss

    def atomic_state_loss() -> torch.Tensor:
        _, state_prediction = _atomic_forward(model, scaled, bundle, device)
        return _state_huber(state_prediction, state_target, state_mask)

    return atomic_state_loss


def run_fixed_epoch_loop(
    *, optimizer: torch.optim.Optimizer, loss_closure: Callable[[], torch.Tensor],
    epochs: int,
) -> tuple[int, bool]:
    """Run exactly ``epochs`` steps; no stopping, scheduler, or best-state path."""

    if type(epochs) is not int or epochs <= 0:
        raise ValueError("TMLR V9B1 fixed epoch count must be a positive integer")
    finite = True
    completed = 0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("TMLR V9B1 training loss is non-finite")
        loss.backward()
        optimizer.step()
        completed += 1
    if completed != epochs:
        raise RuntimeError("TMLR V9B1 fixed epoch loop terminated early")
    return completed, finite


def train_fullfit_job(
    *, job: FullFitJob, bundle: FullFitBundle,
    protocol: Mapping[str, Any], device: torch.device,
) -> FullFitTrainingResult:
    """Train one exact job from scratch on all FIT rows and return final state."""

    validate_job(job)
    if bundle.role_id != job.bundle_role:
        raise RuntimeError("TMLR V9B1 family/bundle registration drift")
    _seed(job.seed)
    scaled = scale_fullfit_features(bundle, atomic=job.family in ATOMIC_FAMILIES)
    model = build_fullfit_model(job.family, protocol).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(OPTIMIZER_RECIPE["learning_rate"]),
        weight_decay=float(OPTIMIZER_RECIPE["weight_decay"]),
    )
    model.train()
    completed, finite = run_fixed_epoch_loop(
        optimizer=optimizer,
        loss_closure=_loss_closure(
            family=job.family, model=model, scaled=scaled,
            bundle=bundle, device=device,
        ),
        epochs=job.epochs,
    )
    state = _snapshot(model)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("TMLR V9B1 final model state is non-finite")
    objective_id = (
        "R3_FROZEN_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK_NO_COST"
        if job.family == "R3_P"
        else "STATE_HUBER_DELTA_0_05"
    )
    return FullFitTrainingResult(
        job=job,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        parameter_count=parameter_count(model),
        state_dict=state,
        scalers=scaler_payload(scaled.scalers),
        objective_id=objective_id,
        epochs_completed=completed,
        training_loss_finite=finite,
    )


def checkpoint_payload(
    result: FullFitTrainingResult, *, epoch_lock_sha256: str,
    bundle_identities: Mapping[str, Mapping[str, Any]],
    implementation_commit: str, implementation_sha256: Mapping[str, str],
    config_sha256: str,
) -> dict[str, Any]:
    """Build the strict no-optimizer/no-result checkpoint payload."""

    validate_job(result.job)
    if set(bundle_identities) != {
        "feature_manifest", "feature_array", "target_manifest", "target_array",
    }:
        raise RuntimeError("TMLR V9B1 checkpoint bundle identity set drift")
    expected_scalers = {
        "state_mean", "state_scale", "action_mean", "action_scale",
    }
    if result.job.family in ATOMIC_FAMILIES:
        expected_scalers |= {"atom_mean", "atom_scale"}
    scaler_shapes = {
        "state_mean": (STATE_DIM,),
        "state_scale": (STATE_DIM,),
        "action_mean": (ACTION_DIM,),
        "action_scale": (ACTION_DIM,),
        "atom_mean": (ATOM_DIM,),
        "atom_scale": (ATOM_DIM,),
    }
    expected_objective = (
        "R3_FROZEN_ATOM_ABSOLUTE_RELATIVE_SIGN_RANK_NO_COST"
        if result.job.family == "R3_P"
        else "STATE_HUBER_DELTA_0_05"
    )
    if (
        result.model_class != EXPECTED_MODEL_CLASSES[result.job.family]
        or result.parameter_count != EXPECTED_PARAMETER_COUNTS[result.job.family]
        or result.epochs_completed != result.job.epochs
        or result.training_loss_finite is not True
        or result.objective_id != expected_objective
        or set(result.scalers) != expected_scalers
        or any(
            np.asarray(result.scalers[name]).shape != scaler_shapes[name]
            or np.asarray(result.scalers[name]).dtype != np.dtype(np.float32)
            or not np.isfinite(result.scalers[name]).all()
            or (
                name.endswith("_scale")
                and np.any(np.asarray(result.scalers[name]) <= 0)
            )
            for name in expected_scalers
        )
        or not result.state_dict
        or any(
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            for value in result.state_dict.values()
        )
    ):
        raise RuntimeError("TMLR V9B1 checkpoint training-result contract drift")
    hex_digits = set("0123456789abcdef")
    hashes = [epoch_lock_sha256, config_sha256, *implementation_sha256.values()]
    if (
        len(implementation_commit) != 40
        or any(char not in hex_digits for char in implementation_commit)
        or not implementation_sha256
        or any(len(value) != 64 or any(char not in hex_digits for char in value) for value in hashes)
    ):
        raise RuntimeError("TMLR V9B1 checkpoint authority identity drift")
    return {
        "schema_version": "rail3.tmlr-v9b1-fullfit-checkpoint.v1",
        "family": result.job.family,
        "seed": result.job.seed,
        "fullfit_epoch": result.job.epochs,
        "epochs_completed": result.epochs_completed,
        "epoch_rule_id": EPOCH_RULE_ID,
        "epoch_lock_sha256": epoch_lock_sha256,
        "config_sha256": config_sha256,
        "bundle_role": result.job.bundle_role,
        "bundle_identities": dict(bundle_identities),
        "prospective_schema_id": PROSPECTIVE_SCHEMA_ID,
        "feature_dimensions": {
            "atom": ATOM_DIM,
            "state": STATE_DIM,
            "action": ACTION_DIM,
            "global_per_action": GLOBAL_DIM,
        },
        "model_class": result.model_class,
        "parameter_count": result.parameter_count,
        "objective_id": result.objective_id,
        "optimizer_recipe": dict(OPTIMIZER_RECIPE),
        "prediction_semantics": PREDICTION_SEMANTICS,
        "ensemble_id": ENSEMBLE_ID,
        "implementation_commit": implementation_commit,
        "implementation_sha256": dict(implementation_sha256),
        "model_state_dict": dict(result.state_dict),
        # Tensor-only scientific payload keeps the authenticated checkpoint
        # compatible with torch.load(..., weights_only=True) at inference.
        "scalers": {
            name: torch.from_numpy(np.asarray(result.scalers[name]).copy())
            for name in sorted(result.scalers)
        },
    }


__all__ = [
    "ATOMIC_FAMILIES",
    "ENSEMBLE_ID",
    "EPOCH_RULE_ID",
    "FAMILIES",
    "FAMILY_BUNDLE",
    "FULLFIT_EPOCHS",
    "FullFitBundle",
    "FullFitJob",
    "FullFitTrainingResult",
    "SEEDS",
    "build_fullfit_model",
    "checkpoint_payload",
    "expected_jobs",
    "load_fullfit_bundle",
    "run_fixed_epoch_loop",
    "scale_fullfit_features",
    "train_fullfit_job",
    "validate_fullfit_bundle_arrays",
    "validate_job",
]
