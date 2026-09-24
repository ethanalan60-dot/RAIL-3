"""V2A exact atomic partitions for one VOC image-class trajectory state."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.evaluation.voc_v2a import V2A_PROTOCOL_SHA256
from rail3.regions.atomic import CandidateMaskInput, build_atomic_partition


VOC_STATE_ATOM_SCHEMA = "rail3.voc-state-atomic-region.v1"


def _encode_label_map(label_map: np.ndarray) -> list[list[int]]:
    flat = np.asarray(label_map, dtype=np.int64).reshape(-1)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    return [
        [int(flat[start]), int(end - start)]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _boundary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    padded = np.pad(array, 1, constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return array & ~eroded


def _dilate(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    padded = np.pad(array, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _centroid(runs: list[list[int]], area: int) -> list[float]:
    x_sum = 0.0
    y_sum = 0.0
    for y, x0, x1 in runs:
        length = x1 - x0
        x_sum += (x0 + x1 - 1) * length / 2
        y_sum += y * length
    return [x_sum / area, y_sum / area]


def _shared_boundaries(
    label_map: np.ndarray, atom_ids: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    horizontal = np.stack(
        (label_map[:, :-1].reshape(-1), label_map[:, 1:].reshape(-1)), axis=1
    )
    vertical = np.stack(
        (label_map[:-1, :].reshape(-1), label_map[1:, :].reshape(-1)), axis=1
    )
    pairs = np.concatenate((horizontal, vertical), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not pairs.size:
        return [], {atom_id: {} for atom_id in atom_ids}
    pairs.sort(axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    edges = []
    per_atom: dict[str, dict[str, int]] = {atom_id: {} for atom_id in atom_ids}
    for (left_index, right_index), count in zip(unique.tolist(), counts.tolist()):
        left = atom_ids[int(left_index)]
        right = atom_ids[int(right_index)]
        length = int(count)
        edges.append({
            "left_atom_id": left,
            "right_atom_id": right,
            "shared_boundary_length": length,
        })
        per_atom[left][right] = length
        per_atom[right][left] = length
    return sorted(edges, key=lambda item: (
        item["left_atom_id"], item["right_atom_id"]
    )), per_atom


def _candidate_identity(candidate: CandidateMaskInput) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "mask_sha256": candidate.mask_sha256,
        "action_id": candidate.action_id,
        "action_code": candidate.action_code,
        "prompt_source": candidate.prompt_source,
    }


def build_voc_state_atomic_partition(
    *,
    asset_id: str,
    image_sha256: str,
    state_id: str,
    class_id: int,
    width: int,
    height: int,
    candidates: Iterable[CandidateMaskInput],
    action_lineage: Iterable[Mapping[str, Any]],
    config_hash: str = V2A_PROTOCOL_SHA256,
) -> dict[str, Any]:
    """Build a full-raster, GT-free, deterministic four-connected partition."""

    if not asset_id or not state_id or class_id not in range(1, 21):
        raise ValueError("VOC state atom identity is invalid")
    if width <= 0 or height <= 0:
        raise ValueError("VOC state atom geometry is invalid")
    if len(image_sha256) != 64 or len(config_hash) != 64:
        raise ValueError("VOC state atom hashes must be SHA-256")
    lineage = tuple(dict(item) for item in action_lineage)
    if tuple(item.get("action_code") for item in lineage) != (
        "A0", "A1", "A2", "A3", "A4", "A5", "A6"
    ):
        raise ValueError("VOC state action lineage must contain ordered A0-A6")
    supplied = tuple(candidates)
    if any(candidate.class_id != class_id for candidate in supplied):
        raise ValueError("VOC state atom candidate class drift")
    if len({candidate.candidate_id for candidate in supplied}) != len(supplied):
        raise ValueError("VOC state atom candidate IDs must be unique")

    if supplied:
        base = build_atomic_partition(
            asset_id=asset_id,
            image_sha256=image_sha256,
            candidates=supplied,
        )
        if (base.width, base.height) != (width, height):
            raise ValueError("VOC state atom candidate geometry differs from image")
        ordered = base.candidate_order
        old_label_map = base.label_map
        base_atoms = list(base.atoms)
    else:
        ordered = ()
        old_label_map = np.zeros((height, width), dtype=np.int32)
        base_atoms = [{
            "member_candidate_ids": [],
            "member_action_ids": [],
            "member_action_codes": [],
            "area": width * height,
            "bbox_xyxy": [0, 0, width, height],
            "runs": [[y, 0, width] for y in range(height)],
        }]
    candidate_index = {
        candidate.candidate_id: index for index, candidate in enumerate(ordered)
    }
    ordered_identity = [_candidate_identity(candidate) for candidate in ordered]

    signatures: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for old_index, atom in enumerate(base_atoms):
        member = set(atom["member_candidate_ids"])
        signature = tuple(
            int(candidate.candidate_id in member) for candidate in ordered
        )
        signatures[signature].append(old_index)
    component_index = {}
    for signature, indices in signatures.items():
        for index, old_index in enumerate(sorted(
            indices,
            key=lambda item: tuple(tuple(run) for run in base_atoms[item]["runs"]),
        )):
            component_index[old_index] = index

    enriched = []
    for old_index, atom in enumerate(base_atoms):
        member_ids = list(atom["member_candidate_ids"])
        member_indices = [candidate_index[value] for value in member_ids]
        signature = [
            int(candidate.candidate_id in set(member_ids)) for candidate in ordered
        ]
        actions = sorted(
            {ordered[index].action_code for index in member_indices},
            key=("A0", "A1", "A2", "A3", "A4", "A5", "A6").index,
        )
        if not member_ids:
            conflict_type = "all_zero_background"
        elif len(member_ids) == len(ordered):
            conflict_type = "all_candidate_consensus"
        elif len(actions) >= 2:
            conflict_type = "cross_action_disagreement"
        elif len(member_ids) >= 2:
            conflict_type = "within_action_candidate_structure"
        else:
            conflict_type = "single_candidate_support"
        runs = [list(run) for run in atom["runs"]]
        area = int(atom["area"])
        atom_id = stable_id("voc_state_atom", {
            "schema_version": VOC_STATE_ATOM_SCHEMA,
            "config_hash": config_hash,
            "asset_id": asset_id,
            "state_id": state_id,
            "class_id": class_id,
            "image_sha256": image_sha256,
            "ordered_candidates": ordered_identity,
            "membership_signature": signature,
            "runs": runs,
        })
        enriched.append({
            "old_index": old_index,
            "atom_id": atom_id,
            "image_id": asset_id,
            "state_id": state_id,
            "class_id": class_id,
            "signature": signature,
            "area": area,
            "bbox": list(atom["bbox_xyxy"]),
            "centroid": _centroid(runs, area),
            "connected_component_index": component_index[old_index],
            "candidate_incidence": member_ids,
            "action_incidence": actions,
            "prompt_lineage": [
                _candidate_identity(ordered[index]) for index in member_indices
            ],
            "overlap_conflict_type": conflict_type,
            "class_conflict": False,
            "exact_merged_status": "exact_unmerged",
            "schema_version": VOC_STATE_ATOM_SCHEMA,
            "config_hash": config_hash,
            "runs": runs,
        })
    enriched.sort(key=lambda atom: atom["atom_id"])
    new_index_by_old = {
        int(atom["old_index"]): index for index, atom in enumerate(enriched)
    }
    # Vectorized relabeling is semantically identical to the historical flat
    # Python loop and matters for M06-E's many full-raster zero-candidate states.
    relabel = np.empty(len(enriched), dtype=np.int32)
    for old_index, new_index in new_index_by_old.items():
        relabel[old_index] = new_index
    label_map = relabel[old_label_map]
    atom_ids = [str(atom["atom_id"]) for atom in enriched]
    if len(enriched) == 1:
        edges, boundaries = [], {atom_ids[0]: {}}
    else:
        edges, boundaries = _shared_boundaries(label_map, atom_ids)
    for atom in enriched:
        atom_id = str(atom["atom_id"])
        atom["adjacency"] = sorted(boundaries[atom_id])
        atom["shared_boundary_length"] = dict(sorted(boundaries[atom_id].items()))
        atom.pop("old_index")

    candidate_to_atoms = {
        candidate.candidate_id: sorted(
            atom["atom_id"] for atom in enriched
            if candidate.candidate_id in atom["candidate_incidence"]
        )
        for candidate in ordered
    }
    reconstruction_errors = {}
    for candidate_index_value, candidate in enumerate(ordered):
        atom_indices = [
            index for index, atom in enumerate(enriched)
            if atom["signature"][candidate_index_value]
        ]
        reconstructed = np.isin(label_map, atom_indices)
        reconstruction_errors[candidate.candidate_id] = int(np.count_nonzero(
            reconstructed != np.asarray(candidate.mask, dtype=bool)
        ))
    area_sum = sum(int(atom["area"]) for atom in enriched)
    runs_valid = all(
        0 <= y < height and 0 <= x0 < x1 <= width
        for atom in enriched for y, x0, x1 in atom["runs"]
    )
    ids_valid = len(atom_ids) == len(set(atom_ids)) and all(atom_ids)
    coverage = np.bincount(label_map.reshape(-1), minlength=len(enriched))
    non_overlap = (
        label_map.shape == (height, width)
        and bool(np.all(label_map >= 0))
        and int(coverage.sum()) == width * height
    )

    if ordered:
        membership_count = np.asarray([
            sum(atom["signature"]) for atom in enriched
        ], dtype=np.int32)
        atom_areas = np.asarray([atom["area"] for atom in enriched], dtype=np.int64)
        disagreement_pixels = int(atom_areas[
            (membership_count > 0) & (membership_count < len(ordered))
        ].sum()) if len(ordered) >= 2 else 0
        consensus_pixels = int(atom_areas[
            membership_count == len(ordered)
        ].sum())
        union_dense = membership_count[label_map] > 0
        disagreement_dense = (
            (membership_count[label_map] > 0)
            & (membership_count[label_map] < len(ordered))
        ) if len(ordered) >= 2 else np.zeros_like(label_map, dtype=bool)
        boundary_band = _dilate(_boundary(union_dense))
        boundary_disagreement_pixels = int(np.count_nonzero(
            disagreement_dense & boundary_band
        ))
        interior_consensus_pixels = int(np.count_nonzero(
            (membership_count[label_map] == len(ordered)) & ~boundary_band
        ))
    else:
        disagreement_pixels = 0
        consensus_pixels = 0
        boundary_disagreement_pixels = 0
        interior_consensus_pixels = 0
    full_pixels = width * height
    invariants = {
        "coverage_pixels": int(coverage.sum()),
        "construction_valid_pixels": full_pixels,
        "construction_void_pixels": 0,
        "every_pixel_assigned_exactly_once": bool(non_overlap),
        "non_overlap": bool(non_overlap),
        "area_conserved": area_sum == full_pixels,
        "reconstruction_error_pixels": sum(reconstruction_errors.values()),
        "candidate_reconstruction_errors": reconstruction_errors,
        "candidate_order_canonicalized": True,
        "stable_atom_ids": True,
        "lineage_complete": len(lineage) == 7,
        "runs_in_bounds": runs_valid,
        "atom_ids_valid_and_unique": ids_valid,
        "no_negative_or_zero_area": all(int(atom["area"]) > 0 for atom in enriched),
    }
    if not all(
        value == 0 if key == "reconstruction_error_pixels" else bool(value)
        for key, value in invariants.items()
        if key != "candidate_reconstruction_errors"
        and key != "construction_void_pixels"
    ):
        raise RuntimeError("VOC state atomic-region invariant failure")
    return {
        "schema_version": VOC_STATE_ATOM_SCHEMA,
        "config_hash": config_hash,
        "asset_id": asset_id,
        "image_sha256": image_sha256,
        "state_id": state_id,
        "class_id": class_id,
        "width": width,
        "height": height,
        "connectivity": 4,
        "construction_valid_domain": "full_deployment_observable_image_raster",
        "candidate_order": ordered_identity,
        "action_lineage": list(lineage),
        "no_result_action_count": sum(
            item.get("outcome") == "no_result" for item in lineage
        ),
        "infeasible_action_count": sum(
            item.get("outcome") == "infeasible" for item in lineage
        ),
        "zero_candidate_state": not bool(ordered),
        "atom_label_map_rle": _encode_label_map(label_map),
        "atoms": enriched,
        "adjacency": edges,
        "candidate_to_atoms": candidate_to_atoms,
        "invariants": invariants,
        "statistics": {
            "atom_count": len(enriched),
            "candidate_count": len(ordered),
            "all_zero_background_atom_count": sum(
                atom["overlap_conflict_type"] == "all_zero_background"
                for atom in enriched
            ),
            "candidate_disagreement_pixels": disagreement_pixels,
            "candidate_disagreement_denominator_pixels": (
                full_pixels if len(ordered) >= 2 else 0
            ),
            "boundary_disagreement_pixels": boundary_disagreement_pixels,
            "interior_consensus_pixels": interior_consensus_pixels,
            "all_candidate_consensus_pixels": consensus_pixels,
            "atoms_influenced_by_noncanonical_action": sum(
                any(code != "A0" for code in atom["action_incidence"])
                for atom in enriched
            ),
            "canonical_only_atom_count": sum(
                atom["action_incidence"] == ["A0"] for atom in enriched
            ),
            "class_conflict_atom_count": 0,
            "class_conflict_pixels": 0,
        },
    }


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
