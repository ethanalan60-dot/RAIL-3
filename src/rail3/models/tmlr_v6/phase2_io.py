"""Registered FIT-only F0/F1 views for TMLR V6 phase 2.

This module is the only F0/F1 data boundary.  It authenticates the frozen
prospective bundles, restricts them to one registered scale, and constructs a
non-destructive remaining-action view in physical action order.  In
particular, A3 is masked on *both* filtrations before scaling or supervision.

The module never opens a prediction, first-batch metric, post-action
diagnostic, or protected split.  Target arrays remain physically separate
from predictor arrays even though both are returned by the validated view.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.models.tmlr_v6.cost_profiles import assert_fit_only_path
from rail3.models.tmlr_v6.schema import ACTION_DIM, ATOM_DIM, STATE_DIM
from rail3.models.tmlr_v6.training import (
    ACTIONS,
    FEATURE_ARRAY_NAMES,
    PROVENANCE_BUNDLE_REGISTRY,
    PROTOCOL_PATH,
    TARGET_ARRAY_NAMES,
    ZERO_EXTERNAL_ACCESS,
    ProspectiveBundle,
    _assert_target_defined_implies_plan_feasible,
    _load_exact_npz,
    _repository_path,
    _strings,
    _validate_part_manifest,
    _validated_bundle_identity_registry,
    _verify_registered_bundle_provenance,
    inference_weights_from_features,
    load_protocol,
    reconstruct_target_states,
    sha256_file,
)


PHASE2_ADDENDUM_PATH = Path(
    "configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json"
)
PHASE2_VIEW_REGISTRY_PATH = Path(
    "artifacts/audits/tmlr_v6/phase2_f0_f1_view_registry.json"
)
PHASE2_F0_F1_OUTPUT_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/phase2-f0-f1-crossfit"
)
VIEW_REGISTRY_SCHEMA = "rail3.tmlr-v6.phase2-f0-f1-view-registry.v1"
VIEW_REGISTRY_STATUS = "TMLR_V6_PHASE2_F0_F1_VIEWS_PASS"
SCALES = ("S250", "S500", "S1000", "S1364")
SCALE_GROUP_COUNTS = {"S250": 250, "S500": 500, "S1000": 1000, "S1364": 1364}
FILTRATIONS = ("F0_P", "F1_P")
FOLDS = tuple(f"fold_{index}" for index in range(5))
SEEDS = (13, 37, 71)
PHYSICAL_ACTION_ORDER = ("STOP", "A3", "A4", "A5", "A6")
REMAINING_PHYSICAL_INDICES = (0, 2, 3, 4)
REMAINING_ACTIONS = ("STOP", "A4", "A5", "A6")
A3_PHYSICAL_INDEX = 1
FEASIBLE_FEATURE_INDEX = 0
PRECONDITION_FEATURE_INDEX = 7
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Phase2View:
    """One scale/filtration view with A3 irreversibly masked in memory."""

    view_id: str
    filtration_id: str
    scale: str
    bundle: ProspectiveBundle
    fold_manifest: Mapping[str, Any]
    fold_manifest_sha256: str
    global_state_indices: np.ndarray
    global_atom_indices: np.ndarray
    features: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    complete_case: np.ndarray
    formal_decision_population: np.ndarray


def view_id(*, filtration_id: str, scale: str) -> str:
    if filtration_id not in FILTRATIONS or scale not in SCALES:
        raise ValueError("unregistered TMLR V6 F0/F1 view")
    return f"{filtration_id}_{scale}_REMAINING_A3_MASKED"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_logical_path_not_symlinked(
    repository: Path, logical_path: Path, *, description: str,
) -> None:
    """Reject a logical file and in-repository ancestors before resolution."""

    root = repository.resolve()
    requested = logical_path if logical_path.is_absolute() else root / logical_path
    if requested.is_symlink() or any(
        parent.is_symlink() for parent in requested.parents
        if parent == root or root in parent.parents
    ):
        raise RuntimeError(f"TMLR V6 F0/F1 {description} is symlinked")


def _assert_single_link_regular(path: Path, *, description: str) -> None:
    """Reject hardlinks and special files before any fixed input is opened."""

    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"TMLR V6 F0/F1 {description} is absent") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(
            f"TMLR V6 F0/F1 {description} must be one regular single-link file"
        )


def _exact_identity(path: Path, *, logical_path: Path | None = None) -> dict[str, Any]:
    _assert_single_link_regular(path, description="identity input")
    return {
        "path": (path if logical_path is None else logical_path).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_bundle_provenance_sources_single_link(
    repository: Path, identities: Mapping[str, Any],
) -> None:
    """Preflight every file the shared provenance verifier may open."""

    logical_paths: list[Path] = []
    for key in ("build_report", "fit_only_input_inventory"):
        value = identities.get(key, {})
        if isinstance(value, Mapping) and value.get("path"):
            logical_paths.append(Path(str(value["path"])))
    builder = identities.get("committed_builder", {})
    if isinstance(builder, Mapping) and builder.get("path"):
        logical_paths.append(Path(str(builder["path"])))
    implementation = identities.get("builder_implementation", {})
    sources = implementation.get("source_sha256", {}) if isinstance(
        implementation, Mapping,
    ) else {}
    if isinstance(sources, Mapping):
        logical_paths.extend(Path(str(path)) for path in sources)
    roles = identities.get("roles", {})
    if isinstance(roles, Mapping):
        for role in roles.values():
            partitions = role.get("partitions", {}) if isinstance(
                role, Mapping,
            ) else {}
            if not isinstance(partitions, Mapping):
                continue
            for partition in partitions.values():
                if not isinstance(partition, Mapping):
                    continue
                for key in ("manifest", "array"):
                    value = partition.get(key, {})
                    if isinstance(value, Mapping) and value.get("path"):
                        logical_paths.append(Path(str(value["path"])))
    if not logical_paths or any(path.as_posix() in {"", "."} for path in logical_paths):
        raise RuntimeError("TMLR V6 F0/F1 bundle provenance paths are incomplete")
    for logical_path in logical_paths:
        _assert_logical_path_not_symlinked(
            repository, logical_path, description="bundle provenance source",
        )
        _assert_single_link_regular(
            _repository_path(repository, logical_path),
            description="bundle provenance source",
        )


def load_phase2_addendum(repository: Path) -> dict[str, Any]:
    """Load and validate only the pre-metric frozen semantic addendum.

    The phase-2 launch-lock itself is validated by ``phase2_contract``.  This
    pre-lock reader exists for view/cost registration and unit-testable
    semantic checks; it does not authorize training.
    """

    repository = repository.resolve()
    _assert_logical_path_not_symlinked(
        repository, PHASE2_ADDENDUM_PATH, description="semantic addendum",
    )
    _assert_single_link_regular(
        _repository_path(repository, PHASE2_ADDENDUM_PATH),
        description="semantic addendum",
    )
    # The shared contract is the trust root.  This module only layers the
    # F0/F1-specific invariants below; it does not invent a second lock schema.
    from rail3.models.tmlr_v6.phase2_contract import (
        load_phase2_addendum as load_common_phase2_addendum,
    )
    payload = load_common_phase2_addendum(repository)
    expected_header = {
        "schema_version": "rail3.tmlr-v6-p0-phase2-semantic-addendum.v1",
        "addendum_id": "tmlr-v6-p0-phase2-semantic-addendum-v1",
        "status": (
            "TMLR_V6_PHASE2_SEMANTICS_FROZEN_BEFORE_FIRST_BATCH_METRIC_INSPECTION"
        ),
    }
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError("TMLR V6 phase-2 semantic addendum is not frozen")
    freeze = payload.get("freeze_declaration", {})
    if (
        freeze.get("first_batch_metrics_inspected_before_freeze") is not False
        or freeze.get("data_or_artifact_content_read_to_construct_addendum") is not False
        or freeze.get("protected_input_read") is not False
        or freeze.get("semantic_changes_after_first_batch_metric_inspection_forbidden")
        is not True
    ):
        raise RuntimeError("TMLR V6 phase-2 freeze declaration drift")
    common = payload.get("common_semantics", {})
    f0_f1 = payload.get("f0_f1", {})
    costs = payload.get("cost_profiles", {})
    jobs = payload.get("job_matrix", {}).get("f0_f1", {})
    semantic_drift = (
        tuple(common.get("physical_action_order", ())) != PHYSICAL_ACTION_ORDER
        or tuple(f0_f1.get("scales", ())) != SCALES
        or tuple(f0_f1.get("physical_slots", ())) != PHYSICAL_ACTION_ORDER
        or tuple(f0_f1.get("remaining_physical_indices", ()))
        != REMAINING_PHYSICAL_INDICES
        or tuple(f0_f1.get("remaining_actions", ())) != REMAINING_ACTIONS
        or tuple(f0_f1.get("action_scaler", {}).get("fit_rows", ()))
        != REMAINING_PHYSICAL_INDICES
        or tuple(f0_f1.get("action_scaler", {}).get("transform_rows", ()))
        != REMAINING_PHYSICAL_INDICES
        or f0_f1.get("action_scaler", {}).get("A3_row_excluded_from_fit") is not True
        or f0_f1.get("atom_auxiliary_loss") is not False
        or f0_f1.get("future_A4_A6_outcome_or_runtime_input") is not False
        or tuple(costs.get("f0_f1_required_scales", ())) != SCALES
        or int(costs.get("required_future_registry", {}).get("required_rows", -1))
        != 100
        or int(jobs.get("total_jobs", -1)) != 120
    )
    if semantic_drift:
        raise RuntimeError("TMLR V6 phase-2 F0/F1 semantic contract drift")
    parent = payload.get("parent_protocol", {})
    parent_path = _repository_path(repository, Path(str(parent.get("path", ""))))
    _assert_single_link_regular(parent_path, description="parent protocol")
    if (
        parent_path.is_symlink()
        or not parent_path.is_file()
        or parent_path.stat().st_size != int(parent.get("bytes", -1))
        or sha256_file(parent_path) != str(parent.get("sha256", ""))
    ):
        raise RuntimeError("TMLR V6 phase-2 parent protocol identity drift")
    return payload


def _validate_full_bundle_arrays(bundle: ProspectiveBundle) -> None:
    """Apply the first-batch physical-array checks to the registered F1 role."""

    features = bundle.features
    targets = bundle.targets
    atom_x = np.asarray(features["atom_x"])
    state_x = np.asarray(features["state_x"])
    action_x = np.asarray(features["action_x16"])
    lengths = np.asarray(features["state_lengths"], dtype=np.int64)
    state_ids = _strings(features["state_ids"], name="feature state_ids")
    group_ids = _strings(features["image_group_ids"], name="feature image_group_ids")
    atom_ids = _strings(features["atom_ids"], name="feature atom_ids")
    target_state_ids = _strings(targets["state_ids"], name="target state_ids")
    target_atom_ids = _strings(targets["atom_ids"], name="target atom ids")
    n_states = len(state_ids)
    n_atoms = len(atom_ids)
    if (
        atom_x.shape != (n_atoms, ATOM_DIM)
        or state_x.shape != (n_states, STATE_DIM)
        or action_x.shape != (n_states, len(ACTIONS), ACTION_DIM)
        or atom_x.dtype != np.dtype(np.float32)
        or state_x.dtype != np.dtype(np.float32)
        or action_x.dtype != np.dtype(np.float32)
        or lengths.shape != (n_states,)
        or group_ids.shape != (n_states,)
        or np.any(lengths <= 0)
        or int(lengths.sum()) != n_atoms
        or not all(np.isfinite(value).all() for value in (atom_x, state_x, action_x))
        or len(set(state_ids.tolist())) != n_states
        or len(set(atom_ids.tolist())) != n_atoms
    ):
        raise RuntimeError("TMLR V6 registered F1 feature geometry drift")
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
        raise RuntimeError("TMLR V6 registered F1 feature/target join drift")
    inference_weights_from_features(atom_x, lengths)
    target_state_weight = np.add.reduceat(
        atom_weights.astype(np.float64), np.r_[0, np.cumsum(lengths[:-1])]
    )
    repeated_feasible = np.repeat(feasible, lengths, axis=0)
    if (
        not np.allclose(target_state_weight, 1.0, rtol=0.0, atol=2e-7)
        or not np.array_equal(action_x[:, :, 0], action_x[:, :, 7])
        or not np.all(np.isin(action_x[:, :, 0], (0.0, 1.0)))
        or not np.array_equal(action_x[:, :, 0].astype(bool), feasible)
        or np.any(np.isfinite(atom_target) & ~repeated_feasible)
        or not np.all(feasible[:, 0])
    ):
        raise RuntimeError("TMLR V6 registered F1 feasibility/weight drift")
    _assert_target_defined_implies_plan_feasible(state_target, feasible)
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
        raise RuntimeError("TMLR V6 registered F1 target reconstruction drift")
    missing = feasible & ~defined
    incomplete = np.any(missing, axis=1)
    missingness = {
        "plan_feasible_target_undefined_actions": int(missing.sum()),
        "incomplete_target_states": int(incomplete.sum()),
        "incomplete_target_state_ids_sha256": _canonical_sha256(
            sorted(state_ids[incomplete].tolist())
        ),
    }
    if bundle.target_manifest.get("target_missingness") != missingness:
        raise RuntimeError("TMLR V6 registered F1 target missingness drift")
    counts = {"states": n_states, "atoms": n_atoms, "state_actions": n_states * 5}
    for manifest in (bundle.feature_manifest, bundle.target_manifest):
        if any(
            int(manifest.get("counts", {}).get(key, -1)) != value
            for key, value in counts.items()
        ):
            raise RuntimeError("TMLR V6 registered F1 manifest count drift")


def load_registered_f0_f1_bundle(
    repository: Path, *, filtration_id: str,
) -> ProspectiveBundle:
    """Load either protocol-bound F0 or F1 without widening first-batch roles."""

    if filtration_id not in FILTRATIONS:
        raise ValueError("unregistered TMLR V6 filtration")
    repository = repository.resolve()
    _assert_logical_path_not_symlinked(
        repository, PROTOCOL_PATH, description="parent protocol",
    )
    _assert_single_link_regular(
        _repository_path(repository, PROTOCOL_PATH),
        description="parent protocol",
    )
    protocol = load_protocol(repository)
    identities = _validated_bundle_identity_registry(protocol)
    _assert_bundle_provenance_sources_single_link(repository, identities)
    _verify_registered_bundle_provenance(repository, identities)
    registration = PROVENANCE_BUNDLE_REGISTRY[filtration_id]
    for description, logical_path in (
        ("feature manifest", registration.feature_manifest),
        ("feature array", registration.feature_array),
        ("target manifest", registration.target_manifest),
        ("target array", registration.target_array),
    ):
        _assert_logical_path_not_symlinked(
            repository, logical_path, description=description,
        )
        _assert_single_link_regular(
            _repository_path(repository, logical_path), description=description,
        )
    feature_manifest, feature_manifest_sha, feature_array_sha = _validate_part_manifest(
        repository=repository,
        registration=registration,
        manifest_path=registration.feature_manifest,
        array_path=registration.feature_array,
        partition="prospective_features",
        bundle_identities=identities,
    )
    target_manifest, target_manifest_sha, target_array_sha = _validate_part_manifest(
        repository=repository,
        registration=registration,
        manifest_path=registration.target_manifest,
        array_path=registration.target_array,
        partition="targets",
        bundle_identities=identities,
    )
    features = _load_exact_npz(
        _repository_path(repository, registration.feature_array), FEATURE_ARRAY_NAMES,
    )
    targets = _load_exact_npz(
        _repository_path(repository, registration.target_array), TARGET_ARRAY_NAMES,
    )
    bundle = ProspectiveBundle(
        registration=registration,
        feature_manifest=feature_manifest,
        target_manifest=target_manifest,
        features=features,
        targets=targets,
        feature_manifest_sha256=feature_manifest_sha,
        target_manifest_sha256=target_manifest_sha,
        feature_array_sha256=feature_array_sha,
        target_array_sha256=target_array_sha,
    )
    _validate_full_bundle_arrays(bundle)
    if len(bundle.features["state_ids"]) != 27_280:
        raise RuntimeError("TMLR V6 F0/F1 full registered population drift")
    return bundle


def load_registered_scale_fold_manifest(
    repository: Path, *, scale: str, addendum: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Authenticate a scale fold manifest before parsing it."""

    if scale not in SCALES:
        raise ValueError("unregistered TMLR V6 F0/F1 scale")
    repository = repository.resolve()
    frozen = dict(addendum or load_phase2_addendum(repository))
    identity = frozen.get("bound_parent_registries", {}).get("fold_registry", {}).get(scale)
    if not isinstance(identity, Mapping):
        raise RuntimeError("TMLR V6 phase-2 fold identity is absent")
    logical = Path(str(identity.get("path", "")))
    path = _repository_path(repository, logical)
    assert_fit_only_path(path)
    _assert_single_link_regular(path, description=f"{scale} fold manifest")
    digest = str(identity.get("sha256", ""))
    if (
        _SHA256.fullmatch(digest) is None
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != digest
        or ("bytes" in identity and path.stat().st_size != int(identity["bytes"]))
    ):
        raise RuntimeError(f"TMLR V6 {scale} fold identity drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("TMLR V6 scale fold manifest is not an object")
    records = list(payload.get("groups", ()))
    group_ids = [str(item.get("image_group_id", "")) for item in records]
    fold_ids = {str(item.get("fold_id", "")) for item in records}
    expected_scale = "S_MAX" if scale == "S1364" else scale
    if (
        payload.get("scale") != expected_scale
        or int(payload.get("group_count", -1)) != SCALE_GROUP_COUNTS[scale]
        or int(payload.get("fold_count", -1)) != 5
        or len(group_ids) != SCALE_GROUP_COUNTS[scale]
        or len(set(group_ids)) != len(group_ids)
        or "" in group_ids
        or fold_ids != set(FOLDS)
        or (payload.get("checks") is not None and not all(
            bool(value) for value in payload["checks"].values()
        ))
    ):
        raise RuntimeError(f"TMLR V6 {scale} fold contract drift")
    return payload, digest


def complete_case_remaining(
    state_target: np.ndarray, feasible: np.ndarray,
) -> np.ndarray:
    """Return C[h] over STOP/A4/A5/A6 after the bilateral A3 mask."""

    target = np.asarray(state_target)
    plan = np.asarray(feasible)
    if target.ndim != 2 or target.shape[1] != 5 or plan.shape != target.shape:
        raise ValueError("TMLR V6 remaining complete-case arrays must be [state,5]")
    if plan.dtype.kind != "b" or np.isinf(target).any():
        raise ValueError("TMLR V6 remaining complete-case inputs are invalid")
    if np.any(plan[:, A3_PHYSICAL_INDEX]) or np.any(
        np.isfinite(target[:, A3_PHYSICAL_INDEX])
    ):
        raise RuntimeError("TMLR V6 remaining complete-case received unmasked A3")
    remaining_target = target[:, REMAINING_PHYSICAL_INDICES]
    remaining_plan = plan[:, REMAINING_PHYSICAL_INDICES]
    return ~np.any(remaining_plan & ~np.isfinite(remaining_target), axis=1)


def _slice_states_and_atoms(
    bundle: ProspectiveBundle, state_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    lengths = np.asarray(bundle.features["state_lengths"], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    atom_indices = np.concatenate([
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index in state_indices
    ])
    state_feature_names = {"state_x", "action_x16", "state_lengths", "state_ids", "image_group_ids"}
    atom_feature_names = {"atom_x", "atom_ids"}
    state_target_names = {"state_target", "feasible", "state_ids"}
    atom_target_names = {"atom_target", "atom_weights", "atom_ids"}
    features = {
        name: np.asarray(value)[state_indices].copy()
        if name in state_feature_names
        else np.asarray(value)[atom_indices].copy()
        for name, value in bundle.features.items()
        if name in state_feature_names | atom_feature_names
    }
    targets = {
        name: np.asarray(value)[state_indices].copy()
        if name in state_target_names
        else np.asarray(value)[atom_indices].copy()
        for name, value in bundle.targets.items()
        if name in state_target_names | atom_target_names
    }
    if set(features) != FEATURE_ARRAY_NAMES or set(targets) != TARGET_ARRAY_NAMES:
        raise RuntimeError("TMLR V6 F0/F1 view lost a physical array")
    return atom_indices, features, targets


def make_registered_f0_f1_view(
    *, bundle: ProspectiveBundle, fold_manifest: Mapping[str, Any],
    fold_manifest_sha256: str, filtration_id: str, scale: str,
) -> Phase2View:
    """Construct one deterministic in-memory registered view.

    This pure constructor is intentionally exposed for synthetic contract
    tests.  Production callers should use :func:`load_registered_f0_f1_view`.
    """

    if filtration_id not in FILTRATIONS or scale not in SCALES:
        raise ValueError("unregistered TMLR V6 F0/F1 view")
    if bundle.registration.role_id != filtration_id:
        raise RuntimeError("TMLR V6 F0/F1 filtration/bundle role drift")
    groups = {str(item["image_group_id"]) for item in fold_manifest.get("groups", ())}
    if len(groups) != SCALE_GROUP_COUNTS[scale]:
        raise RuntimeError("TMLR V6 F0/F1 scale group count drift")
    full_group_ids = _strings(
        bundle.features["image_group_ids"], name="F0/F1 image_group_ids",
    )
    state_indices = np.flatnonzero(np.isin(full_group_ids, tuple(groups)))
    if (
        len(state_indices) != 20 * SCALE_GROUP_COUNTS[scale]
        or set(full_group_ids[state_indices].tolist()) != groups
    ):
        raise RuntimeError("TMLR V6 F0/F1 view population drift")
    atom_indices, features, targets = _slice_states_and_atoms(bundle, state_indices)
    action_x = np.asarray(features["action_x16"])
    state_target = np.asarray(targets["state_target"])
    atom_target = np.asarray(targets["atom_target"])
    feasible = np.asarray(targets["feasible"])
    if filtration_id == "F1_P" and (
        np.any(feasible[:, A3_PHYSICAL_INDEX])
        or np.any(np.isfinite(state_target[:, A3_PHYSICAL_INDEX]))
        or np.any(np.isfinite(atom_target[:, A3_PHYSICAL_INDEX]))
        or np.any(action_x[:, A3_PHYSICAL_INDEX, FEASIBLE_FEATURE_INDEX] != 0.0)
        or np.any(action_x[:, A3_PHYSICAL_INDEX, PRECONDITION_FEATURE_INDEX] != 0.0)
    ):
        raise RuntimeError("TMLR V6 registered F1 source is not already A3-masked")
    # F0 receives a derived mask; F1 is re-written identically after asserting
    # its source contract so downstream code cannot distinguish the mechanism.
    feasible[:, A3_PHYSICAL_INDEX] = False
    state_target[:, A3_PHYSICAL_INDEX] = np.nan
    atom_target[:, A3_PHYSICAL_INDEX] = np.nan
    action_x[:, A3_PHYSICAL_INDEX, FEASIBLE_FEATURE_INDEX] = 0.0
    action_x[:, A3_PHYSICAL_INDEX, PRECONDITION_FEATURE_INDEX] = 0.0
    if (
        np.any(feasible[:, A3_PHYSICAL_INDEX])
        or np.any(np.isfinite(state_target[:, A3_PHYSICAL_INDEX]))
        or np.any(np.isfinite(atom_target[:, A3_PHYSICAL_INDEX]))
        or np.any(action_x[:, A3_PHYSICAL_INDEX, [0, 7]] != 0.0)
    ):
        raise RuntimeError("TMLR V6 bilateral A3 mask failed")
    complete = complete_case_remaining(state_target, feasible)
    eligible = feasible & np.isfinite(state_target)
    formal = complete & eligible[:, 0] & np.any(
        eligible[:, REMAINING_PHYSICAL_INDICES[1:]], axis=1,
    )
    return Phase2View(
        view_id=view_id(filtration_id=filtration_id, scale=scale),
        filtration_id=filtration_id,
        scale=scale,
        bundle=bundle,
        fold_manifest=dict(fold_manifest),
        fold_manifest_sha256=fold_manifest_sha256,
        global_state_indices=state_indices.astype(np.int32),
        global_atom_indices=atom_indices.astype(np.int32),
        features=features,
        targets=targets,
        complete_case=complete,
        formal_decision_population=formal,
    )


def load_registered_f0_f1_view(
    repository: Path, *, filtration_id: str, scale: str,
) -> Phase2View:
    """Authenticate bundle/folds, select the scale, and apply the A3 mask."""

    addendum = load_phase2_addendum(repository)
    bundle = load_registered_f0_f1_bundle(repository, filtration_id=filtration_id)
    folds, fold_sha = load_registered_scale_fold_manifest(
        repository, scale=scale, addendum=addendum,
    )
    return make_registered_f0_f1_view(
        bundle=bundle,
        fold_manifest=folds,
        fold_manifest_sha256=fold_sha,
        filtration_id=filtration_id,
        scale=scale,
    )


def assert_aligned_f0_f1_views(f0: Phase2View, f1: Phase2View) -> None:
    """Require identical remaining targets/feasibility on aligned states."""

    if f0.filtration_id != "F0_P" or f1.filtration_id != "F1_P" or f0.scale != f1.scale:
        raise RuntimeError("TMLR V6 F0/F1 aligned-view identity drift")
    if (
        not np.array_equal(f0.features["state_ids"], f1.features["state_ids"])
        or not np.array_equal(f0.global_state_indices, f1.global_state_indices)
        or not np.array_equal(
            f0.targets["feasible"][:, REMAINING_PHYSICAL_INDICES],
            f1.targets["feasible"][:, REMAINING_PHYSICAL_INDICES],
        )
    ):
        raise RuntimeError("TMLR V6 F0/F1 remaining feasibility differs")
    left = f0.targets["state_target"][:, REMAINING_PHYSICAL_INDICES]
    right = f1.targets["state_target"][:, REMAINING_PHYSICAL_INDICES]
    if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.array_equal(
        np.nan_to_num(left), np.nan_to_num(right), equal_nan=True,
    ):
        raise RuntimeError("TMLR V6 F0/F1 remaining state targets differ")
    if not np.array_equal(f0.complete_case, f1.complete_case):
        raise RuntimeError("TMLR V6 F0/F1 complete-case populations differ")


def load_aligned_f0_f1_views(
    repository: Path, *, scale: str,
) -> tuple[Phase2View, Phase2View]:
    f0 = load_registered_f0_f1_view(repository, filtration_id="F0_P", scale=scale)
    f1 = load_registered_f0_f1_view(repository, filtration_id="F1_P", scale=scale)
    assert_aligned_f0_f1_views(f0, f1)
    return f0, f1


def view_registry_row(view: Phase2View) -> dict[str, Any]:
    """Return the target-free identity/count record for one registered view."""

    state_ids = _strings(view.features["state_ids"], name="view state_ids")
    group_ids = _strings(view.features["image_group_ids"], name="view group ids")
    return {
        "view_id": view.view_id,
        "filtration_id": view.filtration_id,
        "scale": view.scale,
        "source_bundle_role_id": view.bundle.registration.role_id,
        "feature_manifest_sha256": view.bundle.feature_manifest_sha256,
        "feature_array_sha256": view.bundle.feature_array_sha256,
        "target_manifest_sha256": view.bundle.target_manifest_sha256,
        "target_array_sha256": view.bundle.target_array_sha256,
        "fold_manifest_sha256": view.fold_manifest_sha256,
        "states": len(state_ids),
        "atoms": len(view.features["atom_ids"]),
        "groups": len(set(group_ids.tolist())),
        "state_ids_sha256": _canonical_sha256(state_ids.tolist()),
        "group_ids_sha256": _canonical_sha256(sorted(set(group_ids.tolist()))),
        "global_state_indices_sha256": hashlib.sha256(
            view.global_state_indices.astype("<i4", copy=False).tobytes()
        ).hexdigest(),
        "global_atom_indices_sha256": hashlib.sha256(
            view.global_atom_indices.astype("<i4", copy=False).tobytes()
        ).hexdigest(),
        "physical_action_order": list(PHYSICAL_ACTION_ORDER),
        "remaining_physical_indices": list(REMAINING_PHYSICAL_INDICES),
        "remaining_actions": list(REMAINING_ACTIONS),
        "a3_mask": {
            "feasible": False,
            "state_target": "NaN",
            "atom_target": "NaN",
            "raw_action_feature_0": 0.0,
            "raw_action_feature_7": 0.0,
        },
        "complete_case_definition": (
            "not any(plan_feasible and target_undefined) over physical rows [0,2,3,4]"
        ),
    }


def build_f0_f1_view_registry(
    views: Sequence[Phase2View], *, addendum_identity: Mapping[str, Any],
    implementation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact eight-view registry without reading any metrics."""

    rows = sorted(
        (view_registry_row(item) for item in views),
        key=lambda row: (SCALES.index(row["scale"]), FILTRATIONS.index(row["filtration_id"])),
    )
    expected = {(scale, filtration) for scale in SCALES for filtration in FILTRATIONS}
    observed = {(str(row["scale"]), str(row["filtration_id"])) for row in rows}
    if len(rows) != 8 or observed != expected:
        raise RuntimeError("TMLR V6 F0/F1 view registry must contain eight exact views")
    if (
        set(implementation_identity) != {
            "commit", "combined_source_sha256", "source_sha256",
        }
        or re.fullmatch(r"[0-9a-f]{40}", str(implementation_identity.get("commit", "")))
        is None
        or _SHA256.fullmatch(str(implementation_identity.get(
            "combined_source_sha256", "",
        ))) is None
        or not isinstance(implementation_identity.get("source_sha256"), Mapping)
    ):
        raise RuntimeError("TMLR V6 F0/F1 view implementation identity drift")
    for scale in SCALES:
        pair = [item for item in views if item.scale == scale]
        assert_aligned_f0_f1_views(
            next(item for item in pair if item.filtration_id == "F0_P"),
            next(item for item in pair if item.filtration_id == "F1_P"),
        )
    return {
        "schema_version": VIEW_REGISTRY_SCHEMA,
        "status": VIEW_REGISTRY_STATUS,
        "addendum": dict(addendum_identity),
        "implementation": dict(implementation_identity),
        "physical_action_order": list(PHYSICAL_ACTION_ORDER),
        "remaining_physical_indices": list(REMAINING_PHYSICAL_INDICES),
        "remaining_actions": list(REMAINING_ACTIONS),
        "views": rows,
        "checks": {
            "exact_eight_views": True,
            "f0_a3_masked": True,
            "f1_a3_masked": True,
            "remaining_targets_equal": True,
            "remaining_feasibility_equal": True,
            "complete_case_recomputed_after_mask": True,
            "prediction_inputs_target_free": True,
            "protected_access_zero": True,
        },
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
    }


def load_f0_f1_view_registry(
    repository: Path, launch_lock: Mapping[str, Any], *,
    verified_views: Sequence[Phase2View] | None = None,
) -> dict[str, Any]:
    """Load and deterministically authenticate the derived view registry.

    The semantic addendum intentionally did not add this derived file to the
    launch-lock's exact-key set.  Its trust root is instead the lock-bound
    addendum plus bundle/fold/implementation identities.  With
    ``verified_views=None`` this function reconstructs all eight rows and
    compares the canonical file byte-for-byte.  A runner may pass its already
    authenticated current view to avoid reloading all bundles; the full audit
    performs the all-eight comparison once.
    """

    _assert_logical_path_not_symlinked(
        repository,
        PHASE2_VIEW_REGISTRY_PATH,
        description="view registry logical path",
    )
    repository = repository.resolve()
    if hasattr(launch_lock, "payload"):
        launch_lock = launch_lock.payload  # type: ignore[assignment,union-attr]
    if not isinstance(launch_lock, Mapping):
        raise RuntimeError("TMLR V6 F0/F1 view-registry launch-lock type drift")
    path = _repository_path(repository, PHASE2_VIEW_REGISTRY_PATH)
    _assert_single_link_regular(path, description="view registry")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != VIEW_REGISTRY_SCHEMA
        or payload.get("status") != VIEW_REGISTRY_STATUS
        or len(payload.get("views", ())) != 8
        or payload.get("heldout_access") != ZERO_EXTERNAL_ACCESS
        or not all(bool(value) for value in payload.get("checks", {}).values())
    ):
        raise RuntimeError("TMLR V6 F0/F1 view registry contract drift")
    registered_ids = [str(row.get("view_id", "")) for row in payload["views"]]
    expected_ids = {
        view_id(filtration_id=filtration, scale=scale)
        for scale in SCALES for filtration in FILTRATIONS
    }
    if len(set(registered_ids)) != 8 or set(registered_ids) != expected_ids:
        raise RuntimeError("TMLR V6 F0/F1 view registry key product drift")
    if path.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise RuntimeError("TMLR V6 F0/F1 view registry is not canonical bytes")
    addendum_path = _repository_path(repository, PHASE2_ADDENDUM_PATH)
    addendum_identity = _exact_identity(
        addendum_path, logical_path=PHASE2_ADDENDUM_PATH,
    )
    if (
        payload.get("addendum") != addendum_identity
        or launch_lock.get("phase2_semantic_addendum", {}).get("path")
        != addendum_identity["path"]
        or int(launch_lock.get("phase2_semantic_addendum", {}).get("bytes", -1))
        != addendum_identity["bytes"]
        or launch_lock.get("phase2_semantic_addendum", {}).get("sha256")
        != addendum_identity["sha256"]
    ):
        raise RuntimeError("TMLR V6 F0/F1 view registry addendum identity drift")
    if verified_views is None:
        views: list[Phase2View] = []
        for scale in SCALES:
            views.extend(load_aligned_f0_f1_views(repository, scale=scale))
        expected = build_f0_f1_view_registry(
            views,
            addendum_identity=addendum_identity,
            implementation_identity=launch_lock.get("phase2_implementation", {}),
        )
        if path.read_bytes() != canonical_json_bytes(expected) + b"\n":
            raise RuntimeError("TMLR V6 F0/F1 view registry reconstructed-byte drift")
    else:
        if payload.get("implementation") != launch_lock.get("phase2_implementation"):
            raise RuntimeError("TMLR V6 F0/F1 view registry implementation drift")
        rows = {str(row.get("view_id")): row for row in payload["views"]}
        for view in verified_views:
            if rows.get(view.view_id) != view_registry_row(view):
                raise RuntimeError("TMLR V6 F0/F1 current registered view drift")
    return payload


def split_groups(
    fold_manifest: Mapping[str, Any], *, outer_fold: str, seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Reproduce the frozen inner split at every registered scale."""

    if outer_fold not in FOLDS or seed not in SEEDS:
        raise ValueError("unregistered TMLR V6 phase-2 split")
    records = list(fold_manifest.get("groups", ()))
    heldout = {
        str(item["image_group_id"])
        for item in records if str(item["fold_id"]) == outer_fold
    }
    candidates = sorted(
        str(item["image_group_id"])
        for item in records if str(item["fold_id"]) != outer_fold
    )
    stop_count = len(candidates) // 5

    def key(group_id: str) -> tuple[str, str]:
        payload = canonical_json_bytes({
            "namespace": f"m06e-inner-{fold_manifest['scale']}-v1",
            "seed": seed,
            "outer_fold": outer_fold,
            "image_group_id": group_id,
        })
        return hashlib.sha256(payload).hexdigest(), group_id

    stop = set(sorted(candidates, key=key)[:stop_count])
    train = set(candidates) - stop
    all_groups = {str(item["image_group_id"]) for item in records}
    if (
        not train or not stop or not heldout
        or train & stop or train & heldout or stop & heldout
        or train | stop | heldout != all_groups
    ):
        raise RuntimeError("TMLR V6 phase-2 image-group split isolation failed")
    return train, stop, heldout


__all__ = [
    "A3_PHYSICAL_INDEX",
    "FILTRATIONS",
    "FOLDS",
    "PHASE2_ADDENDUM_PATH",
    "PHASE2_F0_F1_OUTPUT_ROOT",
    "PHASE2_VIEW_REGISTRY_PATH",
    "PHYSICAL_ACTION_ORDER",
    "Phase2View",
    "REMAINING_ACTIONS",
    "REMAINING_PHYSICAL_INDICES",
    "SCALES",
    "SEEDS",
    "VIEW_REGISTRY_SCHEMA",
    "VIEW_REGISTRY_STATUS",
    "assert_aligned_f0_f1_views",
    "build_f0_f1_view_registry",
    "complete_case_remaining",
    "load_aligned_f0_f1_views",
    "load_f0_f1_view_registry",
    "load_phase2_addendum",
    "load_registered_f0_f1_bundle",
    "load_registered_f0_f1_view",
    "load_registered_scale_fold_manifest",
    "make_registered_f0_f1_view",
    "split_groups",
    "view_id",
    "view_registry_row",
]
