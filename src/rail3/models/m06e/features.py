"""Deterministic numeric encoding of the frozen deployment-safe M06-E fields."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ACTION_CODES = ("STOP", "A3", "A4", "A5", "A6")
STATUS_CODES = ("result", "no_result", "failure", "other")
HASH_DIM = 8


def _hash_vector(values: list[str], *, namespace: str) -> np.ndarray:
    result = np.zeros(HASH_DIM, dtype=np.float32)
    for value in sorted(values):
        digest = hashlib.sha256(f"{namespace}\0{value}".encode()).digest()
        result[int.from_bytes(digest[:2], "big") % HASH_DIM] += 1.0 if digest[2] & 1 else -1.0
    return result / max(1.0, math.sqrt(len(values)))


def atom_vector(row: Mapping[str, Any]) -> np.ndarray:
    incidence = [str(value) for value in row["base_candidate_incidence"]]
    signature = [int(value) for value in row["membership_signature"]]
    flags = dict(row["availability_flags"])
    values = [
        float(row["area_fraction"]), *map(float, row["bbox_normalized"]),
        *map(float, row["centroid_normalized"]), math.log1p(int(row["component_index"])),
        math.log1p(int(row["adjacency_degree"])), float(row["shared_boundary_fraction"]),
        float(bool(row["canonical_prediction_bit"])), float(bool(row["base_disagreement"])),
        float(bool(row["base_consensus"])), float(flags.get("A0", False)),
        float(flags.get("A1", False)), float(flags.get("A2", False)),
        math.log1p(len(signature)), sum(signature) / max(1, len(signature)),
        math.log1p(len(incidence)),
    ]
    return np.concatenate((np.asarray(values, dtype=np.float32), _hash_vector(incidence, namespace="atom-incidence")))


def state_vector(row: Mapping[str, Any]) -> np.ndarray:
    statuses = list(row["base_result_statuses"])
    one_hot = []
    for status in statuses:
        normalized = status if status in STATUS_CODES[:-1] else "other"
        one_hot.extend(float(normalized == value) for value in STATUS_CODES)
    if len(one_hot) != 12:
        raise ValueError("M06-E base status vector must contain A0,A1,A2")
    class_id = int(row["class_id"])
    classes = [float(class_id == value) for value in range(1, 21)]
    return np.asarray([
        math.log1p(int(row["atom_count"])), *map(float, row["atom_area_statistics"]),
        *one_hot, float(row["state_disagreement_fraction"]),
        float(row["state_conflict_fraction"]), math.log1p(float(row["base_cost_seconds"])),
        *classes,
    ], dtype=np.float32)


def action_vector(row: Mapping[str, Any]) -> np.ndarray:
    code = str(row["action_code"])
    lineage = json.dumps(row["prompt_lineage"], sort_keys=True, separators=(",", ":"))
    source = [] if row["source_candidate_id"] is None else [str(row["source_candidate_id"])]
    hashed = _hash_vector(source + [lineage], namespace="action-lineage")
    return np.concatenate((np.asarray([
        float(bool(row["feasible"])), float(row["source_candidate_id"] is not None),
        *(float(code == value) for value in ACTION_CODES),
        math.log1p(float(row["incremental_cost_seconds"])),
        float(dict(row["availability_flags"]).get("action_precondition_available", False)),
    ], dtype=np.float32), hashed))


@dataclass(frozen=True)
class FrozenScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)


def fit_scaler(values: np.ndarray) -> FrozenScaler:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("scaler requires a finite nonempty matrix")
    mean = array.mean(axis=0)
    scale = array.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return FrozenScaler(mean.astype(np.float32), scale.astype(np.float32))
