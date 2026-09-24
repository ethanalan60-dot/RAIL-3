"""Dimension-preserving R1 feature encoding for M07-A filtrations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from rail3.models.m06e.features import action_vector, atom_vector, state_vector


ACTION_CODES = ("STOP", "A3", "A4", "A5", "A6")
CONTEXT_PREFIX = "M07A_CONTEXT:"


def observation_context(outcome: str | None) -> str:
    if outcome is None:
        return f"{CONTEXT_PREFIX}A3_NOT_OBSERVED"
    normalized = str(outcome).upper()
    if normalized not in {"RESULT", "NO_RESULT", "FAILURE"}:
        raise ValueError("M07-A A3 observation status is invalid")
    return f"{CONTEXT_PREFIX}A3_{normalized}"


def build_feature_matrices(
    *, partition: Mapping[str, Any], actions: Iterable[Mapping[str, Any]],
    canonical_candidate_ids: Iterable[str], a3_outcome: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Encode one F1 state with the exact R1 vector dimensions."""

    action_items = tuple(dict(item) for item in actions)
    if tuple(item.get("action_code") for item in action_items) != ACTION_CODES:
        raise ValueError("M07-A physical action order drift")
    atoms = list(partition["atoms"])
    if not atoms or tuple(partition["base_action_codes"]) != ("A0", "A1", "A2", "A3"):
        raise ValueError("M07-A F1 feature partition drift")
    statuses = [str(item.get("outcome", "other")) for item in partition["base_action_lineage"]]
    if len(statuses) != 4 or statuses[3] != a3_outcome:
        raise ValueError("M07-A A3 status/partition drift")
    base_cost = sum(
        float(item.get("cost_seconds", 0.0))
        for item in partition["base_action_lineage"]
    )
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
        if candidate_count >= 2
        and 0 < sum(atom["membership_signature"]) < candidate_count
    )
    conflict_area = sum(
        int(atom["area"]) for atom in atoms
        if len(atom["base_action_incidence"]) >= 2
    )
    canonical_ids = frozenset(str(value) for value in canonical_candidate_ids)
    width, height = int(partition["width"]), int(partition["height"])
    context = observation_context(a3_outcome)
    rows = []
    for atom in atoms:
        x0, y0, x1, y1 = (int(value) for value in atom["bbox"])
        cx, cy = (float(value) for value in atom["centroid"])
        signature = list(atom["membership_signature"])
        incidence = list(atom["base_candidate_incidence"])
        rows.append({
            "class_id": int(partition["class_id"]),
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
            "base_candidate_incidence": incidence + [context],
            "canonical_prediction_bit": int(bool(canonical_ids.intersection(incidence))),
            "base_disagreement": bool(
                candidate_count >= 2 and 0 < sum(signature) < candidate_count
            ),
            "base_consensus": bool(candidate_count and sum(signature) == candidate_count),
            "atom_count": len(atoms),
            "atom_area_statistics": area_statistics,
            "base_result_statuses": statuses[:3],
            "state_disagreement_fraction": disagreement_area / full_area,
            "state_conflict_fraction": conflict_area / full_area,
            "base_cost_seconds": base_cost,
            "availability_flags": dict(atom["availability_flags"]),
        })
    atom_x = np.stack([atom_vector(row) for row in rows]).astype(np.float32)
    state_x = state_vector(rows[0]).astype(np.float32)
    action_rows = []
    for item in action_items:
        row = dict(item)
        row["availability_flags"] = {
            "action_precondition_available": bool(row["feasible"]),
        }
        action_rows.append(row)
    action_x = np.stack([action_vector(row) for row in action_rows]).astype(np.float32)
    if atom_x.shape[1] != 27 or state_x.shape != (41,) or action_x.shape != (5, 17):
        raise RuntimeError("M07-A R1 feature dimensions drifted")
    return atom_x, state_x, action_x, base_cost


def contextualize_f0_atom_features(atom_x: np.ndarray) -> np.ndarray:
    """Add the pre-registered not-observed token to existing F0 hash channels."""

    values = np.asarray(atom_x, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 27:
        raise ValueError("M07-A F0 atom feature shape drift")
    # Index 18 is log1p(candidate incidence count); indices 19:27 are the
    # normalized eight-bin signed incidence hash from the exact R1 encoder.
    counts = np.rint(np.expm1(values[:, 18])).astype(np.int64)
    if np.any(counts < 0):
        raise ValueError("M07-A F0 incidence count is invalid")
    token_row = {
        "area_fraction": 0.0,
        "bbox_normalized": [0.0] * 4,
        "centroid_normalized": [0.0] * 2,
        "component_index": 0,
        "adjacency_degree": 0,
        "shared_boundary_fraction": 0.0,
        "canonical_prediction_bit": 0,
        "base_disagreement": False,
        "base_consensus": False,
        "availability_flags": {},
        "membership_signature": [],
        "base_candidate_incidence": [observation_context(None)],
    }
    token = atom_vector(token_row)[19:27]
    old_sums = values[:, 19:27] * np.maximum(1.0, np.sqrt(counts))[:, None]
    new_counts = counts + 1
    values[:, 19:27] = (
        old_sums + token[None, :]
    ) / np.sqrt(new_counts)[:, None]
    values[:, 18] = np.log1p(new_counts).astype(np.float32)
    return values
