"""Audited tensorization of causal features and target-only M06-E supervision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rail3.contracts import canonical_json_bytes
from rail3.models.m06e.features import ACTION_CODES, action_vector, atom_vector, state_vector
from rail3.regions.m06e_causal import M06E_CAUSAL_IMPL_REVISION


BUNDLE_SCHEMA = "rail3.m06e.atomic-tensor-bundle.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    encoded = canonical_json_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_tensor_bundle(
    *, feature_report_path: Path, target_report_path: Path, output_root: Path,
) -> dict[str, Any]:
    """Join rows state-locally while preserving the feature/target boundary."""

    if output_root.exists():
        raise FileExistsError(output_root)
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
    if feature_report.get("status") != "M06_E_CAUSAL_FEATURES_PASS":
        raise ValueError("causal feature report is not PASS")
    if target_report.get("status") != "M06_E_ATOMIC_TARGETS_PASS":
        raise ValueError("atomic target report is not PASS")
    if (
        feature_report.get("causal_atom_impl_revision") != M06E_CAUSAL_IMPL_REVISION
        or target_report.get("causal_atom_impl_revision") != M06E_CAUSAL_IMPL_REVISION
    ):
        raise ValueError("M06-E tensor sources are not causal v2")
    if any(feature_report.get("heldout_access", {}).values()):
        raise ValueError("causal feature report records held-out access")
    if any(target_report.get("heldout_access", {}).get(key, 0) for key in (
        "validation40", "calibration30", "pilot_test30", "official_voc_val",
    )):
        raise ValueError("atomic target report records forbidden held-out access")
    feature_path = Path(feature_report["files"]["features"]["path"])
    state_index_path = Path(feature_report["files"]["state_index"]["path"])
    target_path = Path(target_report["file"]["path"])
    state_target_path = Path(target_report["state_file"]["path"])
    identities = (
        (feature_path, feature_report["files"]["features"]),
        (state_index_path, feature_report["files"]["state_index"]),
        (target_path, target_report["file"]),
        (state_target_path, target_report["state_file"]),
    )
    for path, identity in identities:
        if path.stat().st_size != int(identity["bytes"]) or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"M06-E tensor source identity drift: {path}")

    feature_iter, target_iter = _records(feature_path), _records(target_path)
    state_target_iter = _records(state_target_path)
    atom_x: list[np.ndarray] = []
    state_x: list[np.ndarray] = []
    action_x: list[np.ndarray] = []
    atom_target: list[np.ndarray] = []
    atom_weights: list[float] = []
    state_target: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    base_costs: list[float] = []
    a0_costs: list[float] = []
    a0_no_result_flags: list[bool] = []
    feasible: list[np.ndarray] = []
    b4_scores: list[np.ndarray] = []
    state_lengths: list[int] = []
    state_ids: list[str] = []
    group_ids: list[str] = []
    asset_ids: list[str] = []
    class_ids: list[int] = []
    gt_present: list[bool] = []
    origins: list[str] = []
    target_status_counts: dict[str, int] = {}
    for state_record in _records(state_index_path):
        if state_record.get("causal_atom_impl_revision") != M06E_CAUSAL_IMPL_REVISION:
            raise ValueError("M06-E tensor state index is not causal v2")
        state_truth = next(state_target_iter)
        n_atoms = int(state_record["atom_count"])
        if n_atoms <= 0:
            raise ValueError("M06-E state has no causal atom")
        feature_rows = [next(feature_iter) for _ in range(n_atoms * 5)]
        target_rows = [next(target_iter) for _ in range(n_atoms * 5)]
        state_id = str(state_record["state_id"])
        if state_truth["state_id"] != state_id:
            raise ValueError("M06-E state target order drift")
        first_actions = feature_rows[:5]
        if tuple(row["action_code"] for row in first_actions) != ACTION_CODES:
            raise ValueError("M06-E feature action order drift")
        state_x.append(state_vector(first_actions[0]))
        action_x.append(np.stack([action_vector(row) for row in first_actions]))
        costs.append(np.asarray([row["incremental_cost_seconds"] for row in first_actions], dtype=np.float32))
        base_costs.append(float(first_actions[0]["base_cost_seconds"]))
        partition = json.loads(Path(state_record["partition_path"]).read_text(encoding="utf-8"))
        if partition["state_id"] != state_id:
            raise ValueError("M06-E partition/state tensor join drift")
        a0_costs.append(float(partition["base_action_lineage"][0].get("cost_seconds", 0.0)))
        a0_no_result_flags.append(first_actions[0]["base_result_statuses"][0] == "no_result")
        feasible.append(np.asarray([row["feasible"] for row in first_actions], dtype=bool))
        state_target.append(np.asarray([
            np.nan if state_truth["state_residuals"][code] is None
            else float(state_truth["state_residuals"][code]) for code in ACTION_CODES
        ], dtype=np.float32))
        local_atom_target = np.full((n_atoms, 5), np.nan, dtype=np.float32)
        local_weights = np.zeros(n_atoms, dtype=np.float32)
        incident_area = np.zeros(5, dtype=np.float64)
        adjacency_edges = 0.0
        for atom_index in range(n_atoms):
            base_row = feature_rows[atom_index * 5]
            if any(feature_rows[atom_index * 5 + action]["causal_atom_id"] != base_row["causal_atom_id"] for action in range(5)):
                raise ValueError("M06-E atom feature block drift")
            atom_x.append(atom_vector(base_row))
            adjacency_edges += float(base_row["adjacency_degree"]) / 2.0
            for action_index, code in enumerate(ACTION_CODES):
                target = target_rows[action_index * n_atoms + atom_index]
                if target["state_id"] != state_id or target["causal_atom_id"] != base_row["causal_atom_id"] or target["action_code"] != code:
                    raise ValueError("M06-E atomic target join drift")
                value = target["atom_residual"]
                if value is not None:
                    local_atom_target[atom_index, action_index] = float(value)
                target_status_counts[target["target_status"]] = target_status_counts.get(target["target_status"], 0) + 1
            stop_target = target_rows[atom_index]
            local_weights[atom_index] = float(stop_target["valid_pixel_count"]) / float(stop_target["state_valid_pixel_count"])
            incidence = set(str(value) for value in base_row["base_candidate_incidence"])
            for action_index, action_row in enumerate(first_actions):
                source = action_row["source_candidate_id"]
                if source is not None and str(source) in incidence:
                    incident_area[action_index] += local_weights[atom_index]
        if not np.isclose(local_weights.sum(), 1.0, atol=2e-7):
            raise ValueError("M06-E atom valid-area weights do not conserve one")
        atom_target.extend(local_atom_target)
        atom_weights.extend(local_weights.tolist())
        a0_no_result = float(first_actions[0]["base_result_statuses"][0] == "no_result")
        conflict = float(first_actions[0]["state_conflict_fraction"])
        adjacency_per_atom = adjacency_edges / n_atoms
        b4 = np.asarray([
            0.0 if action == 0 else (
                0.35 * a0_no_result + 0.35 * incident_area[action]
                + 0.15 * conflict + 0.15 * min(adjacency_per_atom, 1.0)
                - 0.10 * np.log1p(float(first_actions[action]["incremental_cost_seconds"]))
            ) for action in range(5)
        ], dtype=np.float32)
        b4_scores.append(b4)
        state_lengths.append(n_atoms)
        state_ids.append(state_id)
        group_ids.append(str(state_record["image_group_id"]))
        asset_ids.append(str(state_record["asset_id"]))
        class_ids.append(int(state_record["class_id"]))
        gt_present.append(bool(state_truth["gt_presence"]))
        origins.append(str(state_record["origin"]))
    try:
        next(feature_iter)
        raise ValueError("orphan M06-E feature rows")
    except StopIteration:
        pass
    try:
        next(target_iter)
        raise ValueError("orphan M06-E target rows")
    except StopIteration:
        pass
    try:
        next(state_target_iter)
        raise ValueError("orphan M06-E state targets")
    except StopIteration:
        pass
    arrays = {
        "atom_x": np.stack(atom_x).astype(np.float32),
        "state_x": np.stack(state_x).astype(np.float32),
        "action_x": np.stack(action_x).astype(np.float32),
        "atom_target": np.stack(atom_target).astype(np.float32),
        "atom_weights": np.asarray(atom_weights, dtype=np.float32),
        "state_target": np.stack(state_target).astype(np.float32),
        "cost_seconds": np.stack(costs).astype(np.float32),
        "base_cost_seconds": np.asarray(base_costs, dtype=np.float32),
        "a0_cost_seconds": np.asarray(a0_costs, dtype=np.float32),
        "a0_no_result": np.asarray(a0_no_result_flags, dtype=bool),
        "feasible": np.stack(feasible).astype(bool),
        "b4_score": np.stack(b4_scores).astype(np.float32),
        "state_lengths": np.asarray(state_lengths, dtype=np.int32),
        "class_id": np.asarray(class_ids, dtype=np.int8),
        "gt_present": np.asarray(gt_present, dtype=bool),
    }
    if arrays["atom_x"].shape[0] != int(arrays["state_lengths"].sum()):
        raise RuntimeError("M06-E tensor atom offsets are inconsistent")
    output_root.mkdir(parents=True)
    array_path = output_root / "arrays.npz"
    _atomic_npz(array_path, arrays)
    metadata = {
        "schema_version": BUNDLE_SCHEMA, "status": "M06_E_TENSOR_BUNDLE_PASS",
        "causal_atom_impl_revision": M06E_CAUSAL_IMPL_REVISION,
        "feature_report_sha256": sha256_file(feature_report_path),
        "target_report_sha256": sha256_file(target_report_path),
        "array": {"path": array_path.as_posix(), "bytes": array_path.stat().st_size, "sha256": sha256_file(array_path)},
        "dimensions": {"atom": int(arrays["atom_x"].shape[1]), "state": int(arrays["state_x"].shape[1]), "action_raw": int(arrays["action_x"].shape[2])},
        "counts": {"states": len(state_ids), "atoms": len(atom_x), "state_actions": len(state_ids) * 5, "target_statuses": target_status_counts},
        "state_ids": state_ids, "group_ids": group_ids, "asset_ids": asset_ids, "origins": origins,
        "checks": {"feature_target_join_exact": True, "state_order_exact": True, "atom_weight_conservation": True, "adaptive_mask_feature_count_zero": True},
        "heldout_access": {"validation40": 0, "calibration30": 0, "pilot_test30": 0, "official_voc_val": 0},
    }
    _atomic_json(output_root / "manifest.json", metadata)
    return metadata


class TensorBundle:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        path = Path(self.manifest["array"]["path"])
        if path.stat().st_size != int(self.manifest["array"]["bytes"]) or sha256_file(path) != self.manifest["array"]["sha256"]:
            raise ValueError("M06-E tensor bundle identity drift")
        self.arrays = np.load(path, allow_pickle=False)

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[name])
