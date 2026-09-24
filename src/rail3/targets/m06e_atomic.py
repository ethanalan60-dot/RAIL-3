"""M06-E deployment-safe atomic features and target-only residual decomposition."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import numpy as np

from rail3.contracts import stable_id
from rail3.regions.m06e_causal import (
    M06E_CONFIG_SHA256,
    assert_m06e_causal_label_index_binding,
)


FEATURE_SCHEMA = "rail3.m06e.atomic-residual-feature.v1"
TARGET_SCHEMA = "rail3.m06e.atomic-residual-target.v1"
ADAPTIVE_ACTION_CODES = ("STOP", "A3", "A4", "A5", "A6")
_FORBIDDEN_FEATURE_TOKENS = (
    "gt", "label", "iou", "gain", "oracle", "target", "residual",
    "future", "prediction_mask", "candidate_score", "model_score", "outcome",
)


def _contains_forbidden_feature_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
                return str(key)
            found = _contains_forbidden_feature_key(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _contains_forbidden_feature_key(child)
            if found is not None:
                return found
    return None


def build_atomic_residual_features(
    *,
    partition: Mapping[str, Any],
    actions: Iterable[Mapping[str, Any]],
    canonical_candidate_ids: Iterable[str],
    fold_id: str | None,
) -> list[dict[str, Any]]:
    """Create only F_t-measurable atom/action feature rows."""

    assert_m06e_causal_label_index_binding(partition)
    atoms = list(partition["atoms"])
    if not atoms:
        raise ValueError("M06-E feature construction requires causal atoms")
    action_items = tuple(dict(item) for item in actions)
    if tuple(item.get("action_code") for item in action_items) != ADAPTIVE_ACTION_CODES:
        raise ValueError("M06-E feature actions must be ordered STOP,A3,A4,A5,A6")
    forbidden = _contains_forbidden_feature_key(action_items)
    if forbidden is not None:
        raise ValueError(f"post-action/target field is forbidden in features: {forbidden}")
    required = {
        "action_code", "action_id", "action_family", "feasible", "source_candidate_id",
        "prompt_lineage", "incremental_cost_seconds",
    }
    if any(not required <= set(item) for item in action_items):
        raise ValueError("M06-E action feature descriptor is incomplete")
    if float(action_items[0]["incremental_cost_seconds"]) != 0.0:
        raise ValueError("STOP incremental cost must be zero")

    full_area = int(partition["width"]) * int(partition["height"])
    areas = np.asarray([int(atom["area"]) for atom in atoms], dtype=np.float64)
    fractions = areas / full_area
    area_statistics = [
        float(fractions.min()), float(np.median(fractions)), float(fractions.mean()),
        float(fractions.max()), float(fractions.std()),
    ]
    candidate_count = len(partition["base_candidate_order"])
    disagreement_area = sum(
        int(atom["area"]) for atom in atoms
        if candidate_count >= 2 and 0 < sum(atom["membership_signature"]) < candidate_count
    )
    conflict_area = sum(
        int(atom["area"]) for atom in atoms if len(atom["base_action_incidence"]) >= 2
    )
    state_disagreement = disagreement_area / full_area
    state_conflict = conflict_area / full_area
    base_statuses = [str(item.get("outcome", "unknown")) for item in partition["base_action_lineage"]]
    base_cost = sum(float(item.get("cost_seconds", 0.0)) for item in partition["base_action_lineage"])
    canonical_ids = frozenset(str(value) for value in canonical_candidate_ids)
    width = int(partition["width"])
    height = int(partition["height"])
    result = []
    for atom in atoms:
        x0, y0, x1, y1 = (int(value) for value in atom["bbox"])
        cx, cy = (float(value) for value in atom["centroid"])
        signature = list(atom["membership_signature"])
        incidence = list(atom["base_candidate_incidence"])
        canonical_bit = int(
            bool(canonical_ids.intersection(incidence))
        )
        for action in action_items:
            code = str(action["action_code"])
            incremental = float(action["incremental_cost_seconds"])
            if incremental < 0:
                raise ValueError("M06-E incremental cost cannot be negative")
            action_id = str(action["action_id"])
            feature_id = stable_id("m06e_feature", {
                "schema_version": FEATURE_SCHEMA,
                "config_sha256": partition["config_sha256"],
                "causal_atom_id": atom["causal_atom_id"],
                "state_id": partition["state_id"],
                "action_code": code,
                "action_id": action_id,
                "fold_id": fold_id,
            })
            result.append({
                "schema_version": FEATURE_SCHEMA,
                "config_sha256": partition["config_sha256"],
                "feature_id": feature_id,
                "causal_atom_id": atom["causal_atom_id"],
                "state_id": partition["state_id"],
                "asset_id": partition["asset_id"],
                "image_group_id": partition["image_group_id"],
                "class_id": int(partition["class_id"]),
                "action_code": code,
                "action_id": action_id,
                "fold_id": fold_id,
                "membership_signature": signature,
                "area_fraction": float(atom["area"]) / full_area,
                "bbox_normalized": [x0 / width, y0 / height, x1 / width, y1 / height],
                "centroid_normalized": [cx / width, cy / height],
                "component_index": int(atom["component_id"]),
                "adjacency_degree": len(atom["adjacency"]),
                "shared_boundary_fraction": (
                    sum(int(value) for value in atom["shared_boundary_lengths"].values())
                    / max(1.0, 2.0 * ((x1 - x0) + (y1 - y0)))
                ),
                "base_candidate_incidence": incidence,
                "canonical_prediction_bit": canonical_bit,
                "base_disagreement": bool(
                    candidate_count >= 2 and 0 < sum(signature) < candidate_count
                ),
                "base_consensus": bool(candidate_count and sum(signature) == candidate_count),
                "atom_count": len(atoms),
                "atom_area_statistics": area_statistics,
                "base_result_statuses": base_statuses,
                "state_disagreement_fraction": state_disagreement,
                "state_conflict_fraction": state_conflict,
                "base_cost_seconds": base_cost,
                "action_family": str(action["action_family"]),
                "feasible": bool(action["feasible"]),
                "source_candidate_id": action["source_candidate_id"],
                "prompt_lineage": dict(action["prompt_lineage"]),
                "incremental_cost_seconds": incremental,
                "total_cost_seconds": base_cost + incremental,
                "availability_flags": {
                    **dict(atom["availability_flags"]),
                    "action_precondition_available": bool(action["feasible"]),
                },
            })
    return result


def build_atomic_residual_targets(
    *,
    partition: Mapping[str, Any],
    labels: np.ndarray,
    action_code: str,
    action_id: str,
    target_status: str,
    prediction_mask: np.ndarray | None,
    target_reason: str | None,
    target_only_adaptive_outcome_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Overlay fit GT and one action mask only in the target namespace."""

    if action_code not in ADAPTIVE_ACTION_CODES:
        raise ValueError("M06-E residual target action is invalid")
    if target_status not in {"defined", "failed", "infeasible"}:
        raise ValueError("M06-E residual target status is invalid")
    if action_code == "STOP" and target_only_adaptive_outcome_sha256 is not None:
        raise ValueError("STOP target cannot claim an adaptive outcome identity")
    if target_only_adaptive_outcome_sha256 is not None and len(target_only_adaptive_outcome_sha256) != 64:
        raise ValueError("adaptive outcome identity must be SHA-256")
    label_array = np.asarray(labels)
    shape = (int(partition["height"]), int(partition["width"]))
    if label_array.shape != shape:
        raise ValueError("M06-E label geometry differs from causal partition")
    valid = label_array != 255
    state_valid = int(valid.sum())
    if state_valid <= 0:
        raise ValueError("M06-E state has no supervised valid pixels")
    label_map = assert_m06e_causal_label_index_binding(partition)
    atoms = list(partition["atoms"])
    valid_counts = np.bincount(label_map[valid], minlength=len(atoms)).astype(np.int64)

    defined = target_status == "defined"
    if defined:
        if prediction_mask is None:
            raise ValueError("defined M06-E target requires a prediction mask")
        prediction = np.asarray(prediction_mask, dtype=bool)
        if prediction.shape != shape:
            raise ValueError("M06-E prediction geometry differs from causal partition")
        truth = label_array == int(partition["class_id"])
        mismatch = (prediction != truth) & valid
        mismatch_counts = np.bincount(
            label_map[mismatch], minlength=len(atoms)
        ).astype(np.int64)
        state_mismatch: int | None = int(mismatch.sum())
        state_residual: float | None = state_mismatch / state_valid
    else:
        if prediction_mask is not None:
            raise ValueError("failed/infeasible target must not carry a fabricated mask")
        mismatch_counts = None
        state_mismatch = None
        state_residual = None

    rows = []
    for index, atom in enumerate(atoms):
        atom_valid = int(valid_counts[index])
        row_status = "zero_valid_atom" if atom_valid == 0 else target_status
        atom_mismatch = (
            None if mismatch_counts is None or atom_valid == 0
            else int(mismatch_counts[index])
        )
        atom_residual = None if atom_mismatch is None else atom_mismatch / atom_valid
        causal_atom_id = str(atom["causal_atom_id"])
        target_id = stable_id("m06e_target", {
            "schema_version": TARGET_SCHEMA,
            "config_sha256": partition["config_sha256"],
            "causal_atom_id": causal_atom_id,
            "state_id": partition["state_id"],
            "action_code": action_code,
            "action_id": action_id,
            "target_status": row_status,
            "target_only_adaptive_outcome_sha256": target_only_adaptive_outcome_sha256,
        })
        rows.append({
            "schema_version": TARGET_SCHEMA,
            "config_sha256": partition["config_sha256"],
            "target_id": target_id,
            "causal_atom_id": causal_atom_id,
            "state_id": partition["state_id"],
            "asset_id": partition["asset_id"],
            "image_group_id": partition["image_group_id"],
            "class_id": int(partition["class_id"]),
            "action_code": action_code,
            "action_id": action_id,
            "valid_pixel_count": atom_valid,
            "mismatch_pixel_count": atom_mismatch,
            "atom_residual": atom_residual,
            "state_valid_pixel_count": state_valid,
            "state_mismatch_pixel_count": state_mismatch,
            "state_residual": state_residual,
            "target_status": row_status,
            "target_reason": "ZERO_VALID_ATOM" if atom_valid == 0 else target_reason,
            "target_only_adaptive_outcome_sha256": target_only_adaptive_outcome_sha256,
        })
    reconstructed_valid = sum(row["valid_pixel_count"] for row in rows)
    reconstructed_mismatch = (
        None if not defined else sum(
            int(row["mismatch_pixel_count"] or 0) for row in rows
        )
    )
    audit = {
        "state_id": partition["state_id"],
        "action_code": action_code,
        "target_status": target_status,
        "state_valid_pixel_count": state_valid,
        "direct_state_mismatch_pixel_count": state_mismatch,
        "reconstructed_valid_pixel_count": reconstructed_valid,
        "reconstructed_mismatch_pixel_count": reconstructed_mismatch,
        "valid_reconstruction_exact": reconstructed_valid == state_valid,
        "mismatch_reconstruction_exact": (
            reconstructed_mismatch == state_mismatch if defined else True
        ),
        "void_pixels": int((~valid).sum()),
        "void_excluded": True,
    }
    if not audit["valid_reconstruction_exact"] or not audit["mismatch_reconstruction_exact"]:
        raise RuntimeError("STOP-BLOCKED_M06E_RESIDUAL_RECONSTRUCTION")
    return rows, audit


def assert_feature_json_is_filtration_safe(record: Mapping[str, Any]) -> None:
    """Fail closed on forbidden key names in an emitted feature record."""

    found = _contains_forbidden_feature_key(record)
    if found is not None:
        raise ValueError(f"M06-E emitted feature is not filtration-safe: {found}")
    # Values are inspected too for the explicit diagnostic namespace marker.
    encoded = json.dumps(record, sort_keys=True)
    if "OFFLINE_NONDEPLOYABLE_DIAGNOSTIC" in encoded:
        raise ValueError("retrospective diagnostic entered deployment features")
