"""Deterministic regional controls for the frozen diagnostic panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rail3.models.m06e.features import atom_vector, state_vector


@dataclass(frozen=True)
class RegionPartition:
    """A valid-pixel label map with deterministic region metadata."""

    label_map: np.ndarray
    rectangles: tuple[tuple[int, int, int, int], ...]
    kind: str

    @property
    def region_count(self) -> int:
        return len(self.rectangles)


def representation_usage(representation_id: str) -> tuple[str, bool]:
    """Return the frozen deployability label and future-information flag."""

    if representation_id in {"G1", "G2", "G3", "G4"}:
        return "FIT_ONLY_REPRESENTATION_CONTROL", False
    if representation_id == "G5":
        return "OFFLINE_NONDEPLOYABLE_DIAGNOSTIC", True
    raise ValueError("unknown diagnostic representation")


def _valid_bbox(valid: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(valid)
    if len(xx) == 0:
        raise ValueError("a representation requires at least one valid pixel")
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def guillotine_partition(valid: np.ndarray, region_count: int) -> RegionPartition:
    """Create exactly K deterministic rectangular supports over valid pixels."""

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or region_count < 1 or region_count > int(valid.sum()):
        raise ValueError("invalid guillotine geometry or region count")
    initial = _valid_bbox(valid)
    integral = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def area(rectangle: tuple[int, int, int, int]) -> int:
        x0, y0, x1, y1 = rectangle
        return int(
            integral[y1, x1] - integral[y0, x1]
            - integral[y1, x0] + integral[y0, x0]
        )

    rectangles = [(initial, area(initial))]
    while len(rectangles) < region_count:
        candidates = [
            (valid_area, rectangle, index)
            for index, (rectangle, valid_area) in enumerate(rectangles)
            if valid_area > 1
        ]
        if not candidates:
            raise RuntimeError("guillotine partition cannot reach requested K")
        _, rectangle, index = max(
            candidates, key=lambda item: (item[0], -item[1][1], -item[1][0], -item[2]),
        )
        x0, y0, x1, y1 = rectangle
        axes = ("x", "y") if (x1 - x0) >= (y1 - y0) else ("y", "x")
        best: tuple[tuple[int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        total = area(rectangle)
        for axis_rank, axis in enumerate(axes):
            limits = range(x0 + 1, x1) if axis == "x" else range(y0 + 1, y1)
            for cut in limits:
                left = (x0, y0, cut, y1) if axis == "x" else (x0, y0, x1, cut)
                right = (cut, y0, x1, y1) if axis == "x" else (x0, cut, x1, y1)
                left_area = area(left)
                right_area = total - left_area
                if left_area == 0 or right_area == 0:
                    continue
                key = (abs(left_area - right_area), axis_rank, cut)
                if best is None or key < best[0]:
                    best = key, left, right
            if best is not None and best[0][1] == axis_rank:
                break
        if best is None:
            raise RuntimeError("positive-area rectangle has no deterministic split")
        rectangles[index:index + 1] = [
            (best[1], area(best[1])), (best[2], area(best[2])),
        ]

    rectangles_only = [item[0] for item in rectangles]
    rectangles_only.sort(key=lambda item: (item[1], item[0], item[3], item[2]))
    labels = np.full(valid.shape, -1, dtype=np.int32)
    for region, (x0, y0, x1, y1) in enumerate(rectangles_only):
        use = valid[y0:y1, x0:x1]
        block = labels[y0:y1, x0:x1]
        block[use] = region
    if np.any(labels[valid] < 0) or np.any(labels[~valid] >= 0):
        raise RuntimeError("guillotine valid-pixel coverage failure")
    if set(np.unique(labels[valid]).tolist()) != set(range(region_count)):
        raise RuntimeError("guillotine canonical region IDs are incomplete")
    return RegionPartition(labels, tuple(rectangles_only), "AREA_MATCHED_RECTANGULAR_PARTITION")


def _runs(binary: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Return per-row half-open foreground runs with stable run IDs."""

    result: list[list[tuple[int, int, int]]] = []
    next_id = 0
    for row in np.asarray(binary, dtype=np.uint8):
        padded = np.r_[np.uint8(0), row, np.uint8(0)]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        current = []
        for start, stop in changes.reshape(-1, 2):
            current.append((int(start), int(stop), next_id))
            next_id += 1
        result.append(current)
    return result


def connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Deterministic 4-connected components using row-run union-find."""

    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("connected components requires a 2-D mask")
    rows = _runs(binary)
    count = sum(len(row) for row in rows)
    if count == 0:
        return np.full(binary.shape, -1, dtype=np.int32), 0
    parent = np.arange(count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while value != root:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for previous, current in zip(rows, rows[1:]):
        left = 0
        for start, stop, run_id in current:
            while left < len(previous) and previous[left][1] <= start:
                left += 1
            cursor = left
            while cursor < len(previous) and previous[cursor][0] < stop:
                union(run_id, previous[cursor][2])
                cursor += 1
    roots = sorted({find(index) for index in range(count)})
    canonical = {root: index for index, root in enumerate(roots)}
    labels = np.full(binary.shape, -1, dtype=np.int32)
    for y, row in enumerate(rows):
        for start, stop, run_id in row:
            labels[y, start:stop] = canonical[find(run_id)]
    return labels, len(roots)


def union_component_partition(observable_union: np.ndarray, valid: np.ndarray) -> RegionPartition:
    """Partition valid pixels into union and complement components."""

    observable_union = np.asarray(observable_union, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if observable_union.shape != valid.shape or valid.ndim != 2:
        raise ValueError("union and valid masks must be aligned")
    inside, inside_count = connected_components(observable_union & valid)
    outside, outside_count = connected_components((~observable_union) & valid)
    labels = np.full(valid.shape, -1, dtype=np.int32)
    labels[inside >= 0] = inside[inside >= 0]
    labels[outside >= 0] = outside[outside >= 0] + inside_count
    region_count = inside_count + outside_count
    yy, xx = np.nonzero(labels >= 0)
    region = labels[yy, xx]
    min_x = np.full(region_count, labels.shape[1], dtype=np.int32)
    min_y = np.full(region_count, labels.shape[0], dtype=np.int32)
    max_x = np.full(region_count, -1, dtype=np.int32)
    max_y = np.full(region_count, -1, dtype=np.int32)
    np.minimum.at(min_x, region, xx); np.minimum.at(min_y, region, yy)
    np.maximum.at(max_x, region, xx); np.maximum.at(max_y, region, yy)
    rectangles = [
        (int(min_x[index]), int(min_y[index]), int(max_x[index]) + 1, int(max_y[index]) + 1)
        for index in range(region_count)
    ]
    if np.any(labels[valid] < 0) or np.any(labels[~valid] >= 0):
        raise RuntimeError("union/complement valid-pixel coverage failure")
    return RegionPartition(labels, tuple(rectangles), "CANDIDATE_UNION_COMPONENTS")


def _region_geometry(label_map: np.ndarray, region_count: int) -> list[dict[str, Any]]:
    height, width = label_map.shape
    yy, xx = np.nonzero(label_map >= 0)
    region = label_map[yy, xx]
    areas = np.bincount(region, minlength=region_count).astype(np.int64)
    if len(areas) != region_count or np.any(areas <= 0):
        raise ValueError("regional label map contains an empty region")
    adjacency: list[dict[int, int]] = [dict() for _ in range(region_count)]
    pair_blocks = []
    for left, right in ((label_map[:, :-1], label_map[:, 1:]), (label_map[:-1], label_map[1:])):
        use = (left >= 0) & (right >= 0) & (left != right)
        if use.any():
            pair_blocks.append(np.sort(np.column_stack((left[use], right[use])), axis=1))
    if pair_blocks:
        pairs, pair_counts = np.unique(np.concatenate(pair_blocks), axis=0, return_counts=True)
        for (a, b), count in zip(pairs.tolist(), pair_counts.tolist()):
            adjacency[a][b] = int(count)
            adjacency[b][a] = int(count)
    min_x = np.full(region_count, width, dtype=np.int32)
    min_y = np.full(region_count, height, dtype=np.int32)
    max_x = np.full(region_count, -1, dtype=np.int32)
    max_y = np.full(region_count, -1, dtype=np.int32)
    np.minimum.at(min_x, region, xx); np.minimum.at(min_y, region, yy)
    np.maximum.at(max_x, region, xx); np.maximum.at(max_y, region, yy)
    sum_x = np.bincount(region, weights=xx, minlength=region_count)
    sum_y = np.bincount(region, weights=yy, minlength=region_count)
    result = []
    for index in range(region_count):
        x0, y0 = int(min_x[index]), int(min_y[index])
        x1, y1 = int(max_x[index]) + 1, int(max_y[index]) + 1
        result.append({
            "area": int(areas[index]),
            "bbox": (x0, y0, x1, y1),
            "centroid": (float(sum_x[index] / areas[index]), float(sum_y[index] / areas[index])),
            "adjacency": adjacency[index],
            "height": height,
            "width": width,
        })
    return result


def regional_feature_matrices(
    *, partition: RegionPartition, class_id: int,
    base_lineage: Sequence[Mapping[str, Any]], base_cost_seconds: float,
    observable_union: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a generic region control in the exact R1 feature dimensions."""

    if tuple(str(item["action_code"]) for item in base_lineage) != ("A0", "A1", "A2"):
        raise ValueError("regional control requires exact F0 lineage")
    geometry = _region_geometry(partition.label_map, partition.region_count)
    full_area = partition.label_map.size
    fractions = np.asarray([item["area"] / full_area for item in geometry])
    statistics = [
        float(fractions.min()), float(np.median(fractions)), float(fractions.mean()),
        float(fractions.max()), float(fractions.std()),
    ]
    availability = {
        str(item["action_code"]): bool(item.get("outcome") == "result")
        for item in base_lineage
    }
    statuses = [str(item.get("outcome", "other")) for item in base_lineage]
    union_fraction = 0.0 if observable_union is None else float(
        np.count_nonzero(np.asarray(observable_union, dtype=bool) & (partition.label_map >= 0))
        / max(1, np.count_nonzero(partition.label_map >= 0))
    )
    rows = []
    for index, item in enumerate(geometry):
        x0, y0, x1, y1 = item["bbox"]
        cx, cy = item["centroid"]
        inside = bool(
            observable_union is not None
            and np.any(np.asarray(observable_union, dtype=bool)[partition.label_map == index])
        )
        incidence = ["OBSERVABLE_UNION"] if inside else []
        rows.append({
            "class_id": class_id,
            "membership_signature": [int(inside)] if observable_union is not None else [],
            "area_fraction": item["area"] / full_area,
            "bbox_normalized": [x0 / item["width"], y0 / item["height"], x1 / item["width"], y1 / item["height"]],
            "centroid_normalized": [cx / item["width"], cy / item["height"]],
            "component_index": index,
            "adjacency_degree": len(item["adjacency"]),
            "shared_boundary_fraction": sum(item["adjacency"].values()) / max(1.0, 2.0 * ((x1 - x0) + (y1 - y0))),
            "base_candidate_incidence": incidence,
            "canonical_prediction_bit": 0,
            "base_disagreement": False,
            "base_consensus": False,
            "atom_count": partition.region_count,
            "atom_area_statistics": statistics,
            "base_result_statuses": statuses,
            "state_disagreement_fraction": union_fraction,
            "state_conflict_fraction": 0.0,
            "base_cost_seconds": base_cost_seconds,
            "availability_flags": availability,
        })
    atom_x = np.stack([atom_vector(row) for row in rows]).astype(np.float32)
    state_x = state_vector(rows[0]).astype(np.float32)
    if atom_x.shape != (partition.region_count, 27) or state_x.shape != (41,):
        raise RuntimeError("regional control feature dimensions drifted")
    return atom_x, state_x


def observed_partition_feature_matrices(
    *, partition: Mapping[str, Any], canonical_candidate_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode an F* partition while preserving the exact R1 dimensions."""

    atoms = list(partition["atoms"])
    lineage = list(partition["base_action_lineage"])
    if tuple(item["action_code"] for item in lineage) != ("A0", "A1", "A2", "A3", "A4", "A5", "A6"):
        raise ValueError("retrospective partition does not contain A0--A6")
    full_area = int(partition["width"]) * int(partition["height"])
    fractions = np.asarray([int(atom["area"]) / full_area for atom in atoms])
    statistics = [
        float(fractions.min()), float(np.median(fractions)), float(fractions.mean()),
        float(fractions.max()), float(fractions.std()),
    ]
    candidate_count = len(partition["base_candidate_order"])
    disagreement = sum(
        int(atom["area"]) for atom in atoms
        if candidate_count >= 2 and 0 < sum(atom["membership_signature"]) < candidate_count
    )
    conflict = sum(int(atom["area"]) for atom in atoms if len(atom["base_action_incidence"]) >= 2)
    canonical = frozenset(str(value) for value in canonical_candidate_ids)
    availability = {
        str(item["action_code"]): bool(item.get("outcome") == "result")
        for item in lineage
    }
    width, height = int(partition["width"]), int(partition["height"])
    rows = []
    for atom in atoms:
        x0, y0, x1, y1 = map(int, atom["bbox"])
        cx, cy = map(float, atom["centroid"])
        incidence = list(atom["base_candidate_incidence"])
        signature = list(atom["membership_signature"])
        rows.append({
            "class_id": int(partition["class_id"]),
            "membership_signature": signature,
            "area_fraction": int(atom["area"]) / full_area,
            "bbox_normalized": [x0 / width, y0 / height, x1 / width, y1 / height],
            "centroid_normalized": [cx / width, cy / height],
            "component_index": int(atom["component_id"]),
            "adjacency_degree": len(atom["adjacency"]),
            "shared_boundary_fraction": sum(map(int, atom["shared_boundary_lengths"].values())) / max(1.0, 2.0 * ((x1 - x0) + (y1 - y0))),
            "base_candidate_incidence": incidence,
            "canonical_prediction_bit": int(bool(canonical.intersection(incidence))),
            "base_disagreement": bool(candidate_count >= 2 and 0 < sum(signature) < candidate_count),
            "base_consensus": bool(candidate_count and sum(signature) == candidate_count),
            "atom_count": len(atoms),
            "atom_area_statistics": statistics,
            "base_result_statuses": [str(item.get("outcome", "other")) for item in lineage[:3]],
            "state_disagreement_fraction": disagreement / full_area,
            "state_conflict_fraction": conflict / full_area,
            "base_cost_seconds": sum(float(item.get("cost_seconds", 0.0)) for item in lineage),
            "availability_flags": availability,
        })
    atom_x = np.stack([atom_vector(row) for row in rows]).astype(np.float32)
    state_x = state_vector(rows[0]).astype(np.float32)
    return atom_x, state_x


def regional_targets(
    *, label_map: np.ndarray, region_count: int, labels: np.ndarray,
    class_id: int, action_predictions: Sequence[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact regional and state residuals with void pixels excluded."""

    labels = np.asarray(labels)
    label_map = np.asarray(label_map)
    if labels.shape != label_map.shape or len(action_predictions) != 5:
        raise ValueError("regional target inputs are not aligned")
    valid = labels != 255
    if np.any(label_map[valid] < 0) or np.any(label_map[~valid] >= 0):
        raise ValueError("regional label map does not exactly cover valid pixels")
    counts = np.bincount(label_map[valid], minlength=region_count).astype(np.int64)
    if len(counts) != region_count or np.any(counts <= 0):
        raise ValueError("regional target contains an empty region")
    weights = (counts / counts.sum()).astype(np.float32)
    atom_target = np.full((region_count, 5), np.nan, dtype=np.float32)
    state_target = np.full(5, np.nan, dtype=np.float32)
    truth = labels == int(class_id)
    for action, prediction in enumerate(action_predictions):
        if prediction is None:
            continue
        prediction = np.asarray(prediction, dtype=bool)
        if prediction.shape != labels.shape:
            raise ValueError("regional action prediction geometry drift")
        mismatch = (prediction != truth) & valid
        errors = np.bincount(label_map[mismatch], minlength=region_count).astype(np.int64)
        atom_target[:, action] = (errors / counts).astype(np.float32)
        state_target[action] = np.float32(errors.sum() / counts.sum())
        reconstructed = float(np.sum(atom_target[:, action] * weights))
        if not np.isclose(reconstructed, state_target[action], rtol=0, atol=2e-7):
            raise RuntimeError("regional residual reconstruction drift")
    return atom_target, state_target, weights
