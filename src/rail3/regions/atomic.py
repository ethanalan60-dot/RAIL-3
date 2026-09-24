"""Exact prompt-induced atomic regions without label access or pixel Python loops."""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rail3.contracts import canonical_json_bytes, stable_id


ATOM_SCHEMA_VERSION = "rail3.prompt-atomic-region.v1"


@dataclass(frozen=True)
class CandidateMaskInput:
    candidate_id: str
    action_id: str
    action_code: str
    class_id: int
    prompt_source: str
    mask_sha256: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.mask)
        if not self.candidate_id or not self.action_id or not self.prompt_source:
            raise ValueError("atomic candidate lineage fields cannot be empty")
        if self.class_id <= 0 or array.ndim != 2 or not bool(array.any()):
            raise ValueError("atomic candidate mask/class is invalid")
        if not np.isin(array, (False, True, 0, 1)).all():
            raise ValueError("atomic candidate mask is not binary")
        actual_hash = hashlib.sha256(canonical_json_bytes({
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "mask": np.packbits(array.astype(bool), bitorder="little").tolist(),
        })).hexdigest()
        if self.mask_sha256 != actual_hash:
            raise ValueError("atomic candidate mask hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        action_id: str,
        action_code: str,
        class_id: int,
        prompt_source: str,
        mask: np.ndarray,
    ) -> "CandidateMaskInput":
        array = np.asarray(mask, dtype=bool)
        digest = hashlib.sha256(canonical_json_bytes({
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "mask": np.packbits(array, bitorder="little").tolist(),
        })).hexdigest()
        return cls(
            candidate_id, action_id, action_code, class_id, prompt_source, digest, array
        )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        value = len(self.parent)
        self.parent.append(value)
        self.rank.append(0)
        return value

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class _Run:
    run_id: int
    y: int
    x0: int
    x1: int
    signature: bytes


def _row_runs(
    packed_signatures: np.ndarray,
    valid: np.ndarray,
    y: int,
    union_find: _UnionFind,
) -> list[_Run]:
    width = valid.shape[1]
    row_valid = valid[y]
    change = np.ones(width, dtype=bool)
    if width > 1:
        same_signature = np.all(
            packed_signatures[:, y, 1:] == packed_signatures[:, y, :-1], axis=0
        )
        change[1:] = (
            row_valid[1:] != row_valid[:-1]
        ) | (row_valid[1:] & row_valid[:-1] & ~same_signature)
    starts = np.flatnonzero(change)
    ends = np.concatenate((starts[1:], np.asarray([width], dtype=np.int64)))
    result = []
    for x0, x1 in zip(starts.tolist(), ends.tolist()):
        if not row_valid[x0]:
            continue
        result.append(_Run(
            union_find.add(),
            y,
            int(x0),
            int(x1),
            packed_signatures[:, y, x0].tobytes(),
        ))
    return result


def _connect_adjacent_runs(previous: list[_Run], current: list[_Run], uf: _UnionFind) -> None:
    left = 0
    right = 0
    while left < len(previous) and right < len(current):
        a = previous[left]
        b = current[right]
        if a.x1 <= b.x0:
            left += 1
            continue
        if b.x1 <= a.x0:
            right += 1
            continue
        if a.signature == b.signature:
            uf.union(a.run_id, b.run_id)
        if a.x1 <= b.x1:
            left += 1
        else:
            right += 1


def _encode_label_map(label_map: np.ndarray) -> list[list[int]]:
    flat = np.asarray(label_map, dtype=np.int64).reshape(-1)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    return [
        [int(flat[start]), int(end - start)]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def decode_label_map(
    encoded: Iterable[Iterable[int]], width: int, height: int
) -> np.ndarray:
    values = []
    counts = []
    for pair in encoded:
        value, count = (int(item) for item in pair)
        if count <= 0:
            raise ValueError("atomic label-map run length must be positive")
        values.append(value)
        counts.append(count)
    flat = np.repeat(np.asarray(values, dtype=np.int32), np.asarray(counts, dtype=np.int64))
    if flat.size != width * height:
        raise ValueError("atomic label-map RLE geometry mismatch")
    return flat.reshape((height, width))


@dataclass
class AtomicPartition:
    asset_id: str
    image_sha256: str
    width: int
    height: int
    candidate_order: tuple[CandidateMaskInput, ...]
    valid_mask: np.ndarray
    label_map: np.ndarray
    atoms: tuple[dict[str, Any], ...]
    adjacency_edges: tuple[tuple[str, str], ...]
    candidate_to_atoms: dict[str, tuple[str, ...]]
    construction_seconds: float
    reconstruction_error_pixels: int
    schema_version: str = ATOM_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "valid_domain": {
                "source": "deployment_observable_mask",
                "valid_pixels": int(self.valid_mask.sum()),
                "void_pixels": int((~self.valid_mask).sum()),
            },
            "candidate_order": [
                {
                    "candidate_id": candidate.candidate_id,
                    "action_id": candidate.action_id,
                    "action_code": candidate.action_code,
                    "class_id": candidate.class_id,
                    "prompt_source": candidate.prompt_source,
                    "mask_sha256": candidate.mask_sha256,
                }
                for candidate in self.candidate_order
            ],
            "atom_label_map_rle": _encode_label_map(self.label_map),
            "atoms": list(self.atoms),
            "adjacency_edges": [list(edge) for edge in self.adjacency_edges],
            "candidate_to_atoms": {
                key: list(value) for key, value in sorted(self.candidate_to_atoms.items())
            },
            "invariants": {
                "coverage_pixels": sum(int(atom["area"]) for atom in self.atoms),
                "valid_pixels": int(self.valid_mask.sum()),
                "non_overlap": True,
                "area_conserved": (
                    sum(int(atom["area"]) for atom in self.atoms)
                    == int(self.valid_mask.sum())
                ),
                "reconstruction_error_pixels": self.reconstruction_error_pixels,
                "stable_ids": True,
                "candidate_order_canonicalized": True,
            },
            "construction_seconds": self.construction_seconds,
        }


def build_atomic_partition(
    *,
    asset_id: str,
    image_sha256: str,
    candidates: Iterable[CandidateMaskInput],
    valid_mask: np.ndarray | None = None,
) -> AtomicPartition:
    started = time.perf_counter()
    ordered = tuple(sorted(candidates, key=lambda item: (
        item.mask_sha256, item.candidate_id, item.action_id
    )))
    if not ordered:
        raise ValueError("atomic partition requires at least one candidate mask")
    shape = np.asarray(ordered[0].mask).shape
    if any(np.asarray(candidate.mask).shape != shape for candidate in ordered):
        raise ValueError("atomic candidate mask geometries differ")
    height, width = (int(value) for value in shape)
    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape or not bool(valid.any()):
        raise ValueError("atomic valid domain is empty or has wrong geometry")
    stack = np.stack([np.asarray(candidate.mask, dtype=bool) & valid for candidate in ordered])
    packed = np.packbits(stack, axis=0, bitorder="little")
    uf = _UnionFind()
    all_runs: list[_Run] = []
    previous: list[_Run] = []
    for y in range(height):
        current = _row_runs(packed, valid, y, uf)
        _connect_adjacent_runs(previous, current, uf)
        all_runs.extend(current)
        previous = current

    grouped: dict[int, list[_Run]] = defaultdict(list)
    for run in all_runs:
        grouped[uf.find(run.run_id)].append(run)
    atom_payloads = []
    for runs in grouped.values():
        runs = sorted(runs, key=lambda run: (run.y, run.x0, run.x1))
        signature = runs[0].signature
        unpacked = np.unpackbits(
            np.frombuffer(signature, dtype=np.uint8), bitorder="little"
        )[:len(ordered)].astype(bool)
        member_indices = np.flatnonzero(unpacked).tolist()
        member_ids = [ordered[index].candidate_id for index in member_indices]
        class_ids = sorted({ordered[index].class_id for index in member_indices})
        run_geometry = [[run.y, run.x0, run.x1] for run in runs]
        area = sum(run.x1 - run.x0 for run in runs)
        bbox = (
            min(run.x0 for run in runs),
            min(run.y for run in runs),
            max(run.x1 for run in runs),
            max(run.y for run in runs) + 1,
        )
        atom_id = stable_id("atom", {
            "schema_version": ATOM_SCHEMA_VERSION,
            "asset_id": asset_id,
            "image_sha256": image_sha256,
            "candidate_order": [candidate.candidate_id for candidate in ordered],
            "member_candidate_ids": member_ids,
            "runs": run_geometry,
        })
        atom_payloads.append({
            "atom_id": atom_id,
            "member_candidate_ids": member_ids,
            "member_action_ids": sorted({ordered[index].action_id for index in member_indices}),
            "member_action_codes": sorted({ordered[index].action_code for index in member_indices}),
            "class_ids": class_ids,
            "class_conflict": len(class_ids) >= 2,
            "all_zero_background_signature": not member_ids,
            "area": area,
            "bbox_xyxy": list(bbox),
            "runs": run_geometry,
            "parent_atom_ids": [],
        })
    atom_payloads.sort(key=lambda atom: atom["atom_id"])
    atom_by_root: dict[int, int] = {}
    root_by_geometry = {
        tuple(tuple(value) for value in atom["runs"]): index
        for index, atom in enumerate(atom_payloads)
    }
    for root, runs in grouped.items():
        geometry = tuple(
            (run.y, run.x0, run.x1)
            for run in sorted(runs, key=lambda run: (run.y, run.x0, run.x1))
        )
        atom_by_root[root] = root_by_geometry[geometry]
    label_map = np.full(shape, -1, dtype=np.int32)
    for run in all_runs:
        label_map[run.y, run.x0:run.x1] = atom_by_root[uf.find(run.run_id)]
    if bool((label_map[valid] < 0).any()) or bool((label_map[~valid] != -1).any()):
        raise RuntimeError("atomic label map violates valid/void domain")

    horizontal = np.stack((label_map[:, :-1].reshape(-1), label_map[:, 1:].reshape(-1)), axis=1)
    vertical = np.stack((label_map[:-1, :].reshape(-1), label_map[1:, :].reshape(-1)), axis=1)
    pairs = np.concatenate((horizontal, vertical), axis=0)
    pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] != pairs[:, 1])]
    if pairs.size:
        pairs.sort(axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        edges = tuple(sorted(
            tuple(sorted((atom_payloads[int(left)]["atom_id"], atom_payloads[int(right)]["atom_id"])))
            for left, right in unique_pairs
        ))
    else:
        edges = ()

    candidate_to_atoms: dict[str, list[str]] = {
        candidate.candidate_id: [] for candidate in ordered
    }
    for atom in atom_payloads:
        for candidate_id in atom["member_candidate_ids"]:
            candidate_to_atoms[candidate_id].append(atom["atom_id"])
    candidate_to_atoms_tuple = {
        key: tuple(sorted(value)) for key, value in candidate_to_atoms.items()
    }
    error_pixels = 0
    for candidate in ordered:
        member_atom_indices = np.asarray([
            index for index, atom in enumerate(atom_payloads)
            if candidate.candidate_id in atom["member_candidate_ids"]
        ], dtype=np.int32)
        reconstructed = np.isin(label_map, member_atom_indices) & valid
        expected = np.asarray(candidate.mask, dtype=bool) & valid
        error_pixels += int(np.count_nonzero(reconstructed != expected))
    if error_pixels:
        raise RuntimeError("atomic candidate reconstruction is not lossless")
    if sum(int(atom["area"]) for atom in atom_payloads) != int(valid.sum()):
        raise RuntimeError("atomic region area is not conserved")
    return AtomicPartition(
        asset_id=asset_id,
        image_sha256=image_sha256,
        width=width,
        height=height,
        candidate_order=ordered,
        valid_mask=valid,
        label_map=label_map,
        atoms=tuple(atom_payloads),
        adjacency_edges=edges,
        candidate_to_atoms=candidate_to_atoms_tuple,
        construction_seconds=time.perf_counter() - started,
        reconstruction_error_pixels=error_pixels,
    )


def attach_lineage(previous: AtomicPartition, current: AtomicPartition) -> dict[str, Any]:
    if (
        previous.asset_id != current.asset_id
        or previous.label_map.shape != current.label_map.shape
        or not np.array_equal(previous.valid_mask, current.valid_mask)
    ):
        raise ValueError("atomic lineage partitions are not comparable")
    pairs = np.stack((previous.label_map.reshape(-1), current.label_map.reshape(-1)), axis=1)
    pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
    unique = np.unique(pairs, axis=0)
    parents_by_child: dict[int, set[int]] = defaultdict(set)
    for parent, child in unique.tolist():
        parents_by_child[int(child)].add(int(parent))
    if any(len(parents) != 1 for parents in parents_by_child.values()):
        raise RuntimeError("adding candidates merged prior atoms instead of only refining them")
    links = []
    for child_index, parents in sorted(parents_by_child.items()):
        parent_index = next(iter(parents))
        links.append({
            "parent_atom_id": previous.atoms[parent_index]["atom_id"],
            "child_atom_id": current.atoms[child_index]["atom_id"],
        })
        current.atoms[child_index]["parent_atom_ids"] = [
            previous.atoms[parent_index]["atom_id"]
        ]
    return {
        "from_candidate_count": len(previous.candidate_order),
        "to_candidate_count": len(current.candidate_order),
        "from_atom_count": len(previous.atoms),
        "to_atom_count": len(current.atoms),
        "links": links,
    }


def deterministic_merge_view(
    partition: AtomicPartition, *, minimum_area: int = 4
) -> dict[str, Any]:
    if minimum_area <= 0:
        raise ValueError("minimum merge-view area must be positive")
    index_by_id = {atom["atom_id"]: index for index, atom in enumerate(partition.atoms)}
    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in partition.adjacency_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    mapping = {atom["atom_id"]: atom["atom_id"] for atom in partition.atoms}
    tiny = sorted(
        (atom for atom in partition.atoms if int(atom["area"]) < minimum_area),
        key=lambda atom: (int(atom["area"]), atom["atom_id"]),
    )
    for atom in tiny:
        candidates = {
            atom_id for atom_id in neighbors.get(atom["atom_id"], set())
            if int(partition.atoms[index_by_id[atom_id]]["area"]) >= minimum_area
        }
        if not candidates:
            continue
        target = sorted(
            candidates,
            key=lambda atom_id: (
                -int(partition.atoms[index_by_id[atom_id]]["area"]), atom_id
            ),
        )[0]
        mapping[atom["atom_id"]] = target
    groups: dict[str, list[str]] = defaultdict(list)
    for original, merged in sorted(mapping.items()):
        groups[merged].append(original)
    return {
        "schema_version": "rail3.deterministic-atom-merge-view.v1",
        "minimum_area": minimum_area,
        "rule": (
            "tiny atom to adjacent non-tiny largest-area atom; stable atom-ID tie-break; "
            "retain tiny atom when no eligible neighbor exists"
        ),
        "original_to_merged": mapping,
        "merged_groups": dict(sorted(groups.items())),
    }


def partition_statistics(partition: AtomicPartition) -> dict[str, Any]:
    areas = np.asarray([int(atom["area"]) for atom in partition.atoms], dtype=np.int64)
    valid_pixels = int(partition.valid_mask.sum())
    conflict_pixels = sum(
        int(atom["area"]) for atom in partition.atoms if atom["class_conflict"]
    )
    background_pixels = sum(
        int(atom["area"])
        for atom in partition.atoms if atom["all_zero_background_signature"]
    )
    memberships = np.asarray([
        len(atom["member_candidate_ids"]) for atom in partition.atoms
    ], dtype=np.int64)
    prompt_memberships = np.asarray([
        len(atom["member_action_ids"]) for atom in partition.atoms
    ], dtype=np.int64)
    return {
        "atom_count": int(areas.size),
        "area_min": int(areas.min()),
        "area_median": float(np.median(areas)),
        "area_p90": float(np.quantile(areas, 0.90)),
        "area_p95": float(np.quantile(areas, 0.95)),
        "area_max": int(areas.max()),
        "singleton_atom_count": int((areas == 1).sum()),
        "tiny_atom_count_lt4": int((areas < 4).sum()),
        "candidate_memberships_per_atom_mean": float(memberships.mean()),
        "prompt_memberships_per_atom_mean": float(prompt_memberships.mean()),
        "class_conflict_pixel_fraction": conflict_pixels / valid_pixels,
        "all_zero_background_pixel_fraction": background_pixels / valid_pixels,
        "construction_seconds": partition.construction_seconds,
        "label_map_bytes": int(partition.label_map.nbytes),
        "candidate_stack_boolean_bytes": int(
            len(partition.candidate_order) * partition.width * partition.height
        ),
    }
