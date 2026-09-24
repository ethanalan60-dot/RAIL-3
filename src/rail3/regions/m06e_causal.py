"""Filtration-safe M06-E causal atoms built only from the frozen A0/A1/A2 base."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from rail3.contracts import stable_id
from rail3.regions.atomic import CandidateMaskInput, decode_label_map
from rail3.regions.voc_state_atomic import build_voc_state_atomic_partition


CAUSAL_ATOM_SCHEMA = "rail3.m06e.causal-atom.v1"
M06E_CONFIG_SHA256 = "186989353bc8fcbf5e79095ebd200a0f17f3c1fc21ce4945e6371bbdf6752817"
M06E_CAUSAL_IMPL_REVISION = "m06e-causal-v2-remapped-label-index"
BASE_ACTION_CODES = ("A0", "A1", "A2")


def _encode_label_map(label_map: np.ndarray) -> list[list[int]]:
    flat = np.asarray(label_map, dtype=np.int64).reshape(-1)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    return [
        [int(flat[start]), int(end - start)]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _runs_are_four_connected(runs: Iterable[Iterable[int]]) -> bool:
    normalized = [tuple(int(value) for value in run) for run in runs]
    if not normalized:
        return False
    parent = list(range(len(normalized)))
    rank = [0] * len(normalized)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left = find(left)
        right = find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    by_row: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for index, (y, x0, x1) in enumerate(normalized):
        by_row[y].append((x0, x1, index))
    for row in by_row.values():
        row.sort()
        for left, right in zip(row, row[1:]):
            if left[1] == right[0]:
                union(left[2], right[2])
    for y, row in by_row.items():
        previous = by_row.get(y - 1, ())
        left = 0
        right = 0
        while left < len(previous) and right < len(row):
            px0, px1, pindex = previous[left]
            x0, x1, index = row[right]
            if px1 <= x0:
                left += 1
            elif x1 <= px0:
                right += 1
            else:
                union(pindex, index)
                if px1 <= x1:
                    left += 1
                else:
                    right += 1
    root = find(0)
    return all(find(index) == root for index in range(len(normalized)))


def assert_m06e_causal_label_index_binding(
    partition: Mapping[str, Any],
    *,
    require_v2_revision: bool = True,
) -> np.ndarray:
    """Verify that every stored label index names the same sorted atom geometry."""

    if (
        require_v2_revision
        and partition.get("causal_atom_impl_revision") != M06E_CAUSAL_IMPL_REVISION
    ):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_IMPLEMENTATION_REVISION")
    width = int(partition["width"])
    height = int(partition["height"])
    atoms = list(partition["atoms"])
    if not atoms:
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    label_map = decode_label_map(partition["atom_label_map_rle"], width, height)
    if (
        label_map.shape != (height, width)
        or int(label_map.min()) < 0
        or int(label_map.max()) >= len(atoms)
    ):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    geometry_map = np.full((height, width), -1, dtype=np.int32)
    for index, atom in enumerate(atoms):
        runs = list(atom["runs"])
        if not _runs_are_four_connected(runs):
            raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_ATOM_CONNECTIVITY")
        calculated_area = 0
        x_min = width
        y_min = height
        x_max = 0
        y_max = 0
        x_sum = 0.0
        y_sum = 0.0
        previous: tuple[int, int, int] | None = None
        for raw_run in runs:
            y, x0, x1 = (int(value) for value in raw_run)
            current = (y, x0, x1)
            if (
                not (0 <= y < height and 0 <= x0 < x1 <= width)
                or (previous is not None and current <= previous)
                or bool(np.any(geometry_map[y, x0:x1] != -1))
            ):
                raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
            previous = current
            geometry_map[y, x0:x1] = index
            length = x1 - x0
            calculated_area += length
            x_min = min(x_min, x0)
            y_min = min(y_min, y)
            x_max = max(x_max, x1)
            y_max = max(y_max, y + 1)
            x_sum += (x0 + x1 - 1) * length / 2
            y_sum += y * length
        if (
            calculated_area != int(atom["area"])
            or [x_min, y_min, x_max, y_max] != list(atom["bbox"])
            or not np.allclose(
                np.asarray(atom["centroid"], dtype=np.float64),
                np.asarray([x_sum / calculated_area, y_sum / calculated_area]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    if bool(np.any(geometry_map < 0)) or not np.array_equal(label_map, geometry_map):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    counts = np.bincount(label_map.reshape(-1), minlength=len(atoms))
    if counts.tolist() != [int(atom["area"]) for atom in atoms]:
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    if int(counts.sum()) != width * height:
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    return label_map


def assert_m06e_causal_partition_integrity(
    partition: Mapping[str, Any],
    *,
    candidates: Iterable[CandidateMaskInput],
    label_map: np.ndarray | None = None,
) -> None:
    """Exhaustive GT-free creation gate for one v2 causal partition."""

    decoded = (
        assert_m06e_causal_label_index_binding(partition)
        if label_map is None else np.asarray(label_map, dtype=np.int32)
    )
    if label_map is not None:
        stored = decode_label_map(
            partition["atom_label_map_rle"],
            int(partition["width"]),
            int(partition["height"]),
        )
        if not np.array_equal(decoded, stored):
            raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
        assert_m06e_causal_label_index_binding(partition)
    atoms = list(partition["atoms"])
    supplied = tuple(sorted(candidates, key=lambda item: (
        item.mask_sha256, item.candidate_id, item.action_id
    )))
    expected_order = [{
        "candidate_id": candidate.candidate_id,
        "mask_sha256": candidate.mask_sha256,
        "action_id": candidate.action_id,
        "action_code": candidate.action_code,
        "prompt_source": candidate.prompt_source,
    } for candidate in supplied]
    if list(partition["base_candidate_order"]) != expected_order:
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_CANDIDATE_ORDER")
    signatures = np.asarray(
        [atom["membership_signature"] for atom in atoms], dtype=np.uint8
    )
    if signatures.shape != (len(atoms), len(supplied)):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_SIGNATURE_BINDING")
    shape = (int(partition["height"]), int(partition["width"]))
    for candidate_index, candidate in enumerate(supplied):
        expected = np.asarray(candidate.mask, dtype=bool)
        if expected.shape != shape:
            raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_CANDIDATE_GEOMETRY")
        from_label_map = signatures[:, candidate_index][decoded].astype(bool)
        from_runs = np.zeros(shape, dtype=bool)
        for atom, member in zip(atoms, signatures[:, candidate_index].tolist()):
            if member:
                for y, x0, x1 in atom["runs"]:
                    from_runs[int(y), int(x0):int(x1)] = True
        if (
            not np.array_equal(from_label_map, expected)
            or not np.array_equal(from_runs, expected)
            or not np.array_equal(from_label_map, from_runs)
        ):
            raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_BASE_MASK_RECONSTRUCTION")


def build_m06e_causal_partition(
    *,
    asset_id: str,
    sample_id: str,
    image_group_id: str,
    image_sha256: str,
    state_id: str,
    class_id: int,
    width: int,
    height: int,
    candidates: Iterable[CandidateMaskInput],
    base_action_lineage: Iterable[Mapping[str, Any]],
    config_sha256: str = M06E_CONFIG_SHA256,
) -> dict[str, Any]:
    """Build full-raster causal atoms without GT or any adaptive-action outcome."""

    if not sample_id or not image_group_id:
        raise ValueError("M06-E causal atom sample/group identity is empty")
    if len(config_sha256) != 64:
        raise ValueError("M06-E causal atom config identity must be SHA-256")
    supplied = tuple(candidates)
    if any(candidate.action_code not in BASE_ACTION_CODES for candidate in supplied):
        raise ValueError("adaptive action candidate is forbidden in M06-E causal atoms")
    lineage = tuple(dict(item) for item in base_action_lineage)
    if tuple(item.get("action_code") for item in lineage) != BASE_ACTION_CODES:
        raise ValueError("M06-E base lineage must be ordered A0,A1,A2")
    forbidden_lineage_fields = {
        "gt", "label", "iou", "gain", "target", "oracle", "adaptive_mask",
    }
    if any(forbidden_lineage_fields & set(item) for item in lineage):
        raise ValueError("label/target field is forbidden in M06-E base lineage")

    # The mature full-raster connected-component implementation is reused with
    # inert padding only to satisfy its historical A0--A6 lineage shape. That
    # lineage does not enter its atom IDs; this function discards it completely.
    padded = list(lineage) + [{
        "action_code": code, "action_id": f"not-observed-{code}",
        "cache_key": None, "feasible": False, "outcome": "not_observed",
        "reason_code": "OUTSIDE_FILTRATION", "candidate_ids": [],
        "prompt_source": code,
    } for code in ("A3", "A4", "A5", "A6")]
    base = build_voc_state_atomic_partition(
        asset_id=asset_id,
        image_sha256=image_sha256,
        state_id=state_id,
        class_id=class_id,
        width=width,
        height=height,
        candidates=supplied,
        action_lineage=padded,
        config_hash=config_sha256,
    )
    ordered_candidates = list(base["candidate_order"])
    availability = {
        code: bool(item.get("outcome") == "result")
        for code, item in zip(BASE_ACTION_CODES, lineage)
    }

    id_map: dict[str, str] = {}
    for atom in base["atoms"]:
        id_map[str(atom["atom_id"])] = stable_id("m06e_causal_atom", {
            "schema_version": CAUSAL_ATOM_SCHEMA,
            "config_sha256": config_sha256,
            "asset_id": asset_id,
            "sample_id": sample_id,
            "image_group_id": image_group_id,
            "state_id": state_id,
            "class_id": class_id,
            "image_sha256": image_sha256,
            "ordered_base_candidates": ordered_candidates,
            "membership_signature": atom["signature"],
            "runs": atom["runs"],
        })

    atoms = []
    for old_index, atom in enumerate(base["atoms"]):
        old_id = str(atom["atom_id"])
        shared = {
            id_map[str(neighbor)]: int(length)
            for neighbor, length in atom["shared_boundary_length"].items()
        }
        atoms.append({
            "_old_index": old_index,
            "schema_version": CAUSAL_ATOM_SCHEMA,
            "config_sha256": config_sha256,
            "causal_atom_id": id_map[old_id],
            "asset_id": asset_id,
            "sample_id": sample_id,
            "image_group_id": image_group_id,
            "state_id": state_id,
            "class_id": class_id,
            "width": width,
            "height": height,
            "connectivity": 4,
            "membership_signature": list(atom["signature"]),
            "area": int(atom["area"]),
            "bbox": list(atom["bbox"]),
            "centroid": [float(value) for value in atom["centroid"]],
            "component_id": int(atom["connected_component_index"]),
            "base_candidate_incidence": list(atom["candidate_incidence"]),
            "base_action_incidence": list(atom["action_incidence"]),
            "adjacency": sorted(shared),
            "shared_boundary_lengths": dict(sorted(shared.items())),
            "lineage": list(atom["prompt_lineage"]),
            "availability_flags": dict(availability),
            "runs": [list(run) for run in atom["runs"]],
        })
    atoms.sort(key=lambda item: item["causal_atom_id"])
    old_to_new = np.empty(len(atoms), dtype=np.int32)
    for new_index, atom in enumerate(atoms):
        old_to_new[int(atom.pop("_old_index"))] = new_index
    if sorted(old_to_new.tolist()) != list(range(len(atoms))):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BIJECTION")
    old_label_map = decode_label_map(base["atom_label_map_rle"], width, height)
    if int(old_label_map.min()) < 0 or int(old_label_map.max()) >= len(atoms):
        raise RuntimeError("STOP-BLOCKED_M06E_CAUSAL_LABEL_INDEX_BINDING")
    label_map = old_to_new[old_label_map]

    candidate_to_atoms = {
        candidate_id: sorted(id_map[old] for old in atom_ids)
        for candidate_id, atom_ids in base["candidate_to_atoms"].items()
    }
    adjacency = [{
        "left_causal_atom_id": id_map[str(edge["left_atom_id"])],
        "right_causal_atom_id": id_map[str(edge["right_atom_id"])],
        "shared_boundary_length": int(edge["shared_boundary_length"]),
    } for edge in base["adjacency"]]
    adjacency.sort(key=lambda item: (
        item["left_causal_atom_id"], item["right_causal_atom_id"]
    ))
    result = {
        "schema_version": "rail3.m06e.causal-partition.v1",
        "causal_atom_impl_revision": M06E_CAUSAL_IMPL_REVISION,
        "config_sha256": config_sha256,
        "asset_id": asset_id,
        "sample_id": sample_id,
        "image_group_id": image_group_id,
        "image_sha256": image_sha256,
        "state_id": state_id,
        "class_id": class_id,
        "width": width,
        "height": height,
        "connectivity": 4,
        "construction_valid_domain": "full_deployment_observable_image_raster",
        "base_action_codes": list(BASE_ACTION_CODES),
        "base_action_lineage": list(lineage),
        "base_candidate_order": ordered_candidates,
        "atom_label_map_rle": _encode_label_map(label_map),
        "atoms": atoms,
        "adjacency": adjacency,
        "candidate_to_atoms": candidate_to_atoms,
        "zero_candidate_state": bool(base["zero_candidate_state"]),
        "invariants": {
            **base["invariants"],
            "adaptive_candidate_count": 0,
            "gt_field_count": 0,
            "base_lineage_complete": True,
            "label_index_binding_exact": True,
            "geometry_binding_exact": True,
            "base_mask_reconstruction_from_runs_exact": True,
            "base_mask_reconstruction_from_label_map_exact": True,
        },
        "statistics": base["statistics"],
    }
    assert_m06e_causal_partition_integrity(
        result, candidates=supplied, label_map=label_map
    )
    return result
