#!/usr/bin/env python3
"""Build physically separated, fit-only PROSPECTIVE_V1 tensor bundles.

The feature stage never opens a segmentation label or an adaptive-action
output.  It opens the legacy joined NPZ containers only through an exact member
allowlist and never deserializes their target, feasibility, or realized-action
members.  F0/F1 features are projected from those allowed members, while action
descriptors are reconstructed from registered pre-execution plans.  RECT/UNION
geometry is rebuilt over the full image raster from causal A0/A1/A2 partitions.
The physically separate target stage alone opens the 1,364 fit labels and
cached action outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rail3.cache import CandidateCache
from rail3.cache.atomic_io import atomic_create_bytes
from rail3.contracts import canonical_json_bytes, stable_id
from rail3.data.m06e_sources import (
    TrajectorySource,
    action_observations,
    canonical_observation,
    load_trajectory_sources,
    sha256_file,
    state_by_class,
    union_action,
    union_canonical,
)
from rail3.diagnostic.representations import (
    RegionPartition,
    guillotine_partition,
    regional_feature_matrices,
    union_component_partition,
)
from rail3.models.m06e.data import TensorBundle
from rail3.models.tmlr_v6.descriptors import (
    action_matrix,
    descriptor_from_plan,
    stop_descriptor,
)
from rail3.models.tmlr_v6.schema import (
    ACTION_DIM,
    PROSPECTIVE_BUILDER_IMPLEMENTATION_PATHS,
    PROSPECTIVE_SCHEMA_ID,
)
from rail3.regions.atomic import decode_label_map
from rail3.sam.voc_actions import build_action_plans


ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
ZERO_EXTERNAL_ACCESS = {
    "validation40": 0,
    "calibration30": 0,
    "pilot_test30": 0,
    "official_voc_val": 0,
    "segppd": 0,
    "uav_iap": 0,
}
FIT_MANIFEST_SHA256 = (
    "75dfe57ef4ec84e112240d9154e91d5e40e87891867e0d7fbce5979c4c625acb"
)
FEATURE_SCHEMA = "rail3.tmlr-v6.prospective-feature-bundle.v1"
TARGET_SCHEMA = "rail3.tmlr-v6.prospective-target-bundle.v1"
FEATURE_STATUS = "TMLR_V6_PROSPECTIVE_FEATURE_BUNDLE_PASS"
TARGET_STATUS = "TMLR_V6_PROSPECTIVE_TARGET_BUNDLE_PASS"
TARGET_CONTRACT = "M06E_ABSOLUTE_ADDITIVE_PIXEL_MISMATCH_V1"
TARGET_DEFINITION = "same frozen absolute residual target for all first-batch models"
DEFAULT_INPUT_INVENTORY = Path(
    "artifacts/audits/tmlr_v6/fit_only_input_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/prospective-bundles"
)
FIT_ONLY_MATERIALIZER_PATH = Path(
    "scripts/materialize_tmlr_v6_fit_only_inputs.py"
)
FEATURE_NAMES = frozenset({
    "atom_x", "state_x", "action_x16", "state_lengths", "state_ids",
    "image_group_ids", "atom_ids",
})
TARGET_NAMES = frozenset({
    "atom_target", "atom_weights", "state_target", "feasible",
    "state_ids", "atom_ids",
})
HISTORICAL_FEATURE_INPUTS: Mapping[str, tuple[str, ...]] = {
    "f0": ("atom_x", "state_x", "state_lengths", "base_cost_seconds"),
    "f1": ("atom_x", "state_x", "state_lengths"),
}
HISTORICAL_TARGET_INPUTS: Mapping[str, tuple[str, ...]] = {
    "f0": ("atom_target", "atom_weights", "state_target", "feasible", "state_lengths"),
    "f1": ("atom_target", "atom_weights", "state_target", "feasible", "state_lengths"),
}
FORBIDDEN_PATH_TOKENS = (
    "artifacts/protected", "locked40",
    "validation40", "official_voc_val", "official-voc-val", "calibration30",
    "pilot_test30", "pilot-test30", "segppd", "uav-iap", "uav_iap",
    "segmentation-trainval",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_BUILDER_PATH_KEYS = frozenset({
    "manifest", "input_inventory", "output_root", "dataset_root",
    "baseline_cache", "causal_root",
    "f0_bundle", "f1_bundle", "state_index", "fit100_report",
    "fit100_cache", "m06c_report",
    "m06c_cache", "m06d_report", "m06d_cache", "m06e_report",
    "m06e_cache",
})


ROLE_SPECS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    "f0": ("F0_P", "observable_boolean_cells", ("A0", "A1", "A2")),
    "rect": (
        "RECT_FULL_RASTER_P", "full_raster_area_matched_rectangles",
        ("A0", "A1", "A2"),
    ),
    "union": (
        "UNION_FULL_RASTER_P", "full_raster_candidate_union_components",
        ("A0", "A1", "A2"),
    ),
    "f1": (
        "F1_P", "observable_boolean_cells_after_a3",
        ("A0", "A1", "A2", "A3"),
    ),
}


def _all_action_predictions(
    canonical_result: Any, outcomes: Sequence[Any], shape: tuple[int, int],
) -> list[np.ndarray | None]:
    canonical_defined = (
        canonical_result.candidate is not None
        or canonical_result.no_result is not None
    )
    result: list[np.ndarray | None] = [
        union_canonical(canonical_result, shape) if canonical_defined else None,
    ]
    for outcome in outcomes[2:]:
        defined = outcome.outcome in {"result", "no_result"}
        result.append(union_action(outcome, shape) if defined else None)
    if len(result) != len(ACTIONS):
        raise RuntimeError("cached target action order drift")
    return result


def assert_fit_only_path(path: Path) -> None:
    normalized = path.resolve(strict=False).as_posix().lower()
    matches = [token for token in FORBIDDEN_PATH_TOKENS if token in normalized]
    if matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: " + ", ".join(matches)
        )


def _assert_input_file(path: Path, *, description: str) -> None:
    assert_fit_only_path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or linked fit-only {description}: {path}")


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    _assert_input_file(path, description=description)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return payload


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    _assert_input_file(path, description=description)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class InputInventoryGuard:
    """Verify every consumed fit-only file against the exhaustive inventory."""

    repository: Path
    stage: str
    rows: Mapping[str, Mapping[str, Any]]
    expected_paths: frozenset[str]
    consumed: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls, *, repository: Path, stage: str, payload: Mapping[str, Any],
        expected_registry_count: int = 220_989,
    ) -> "InputInventoryGuard":
        if stage not in {"features", "targets"}:
            raise ValueError("unknown input-inventory stage")
        raw_rows = payload.get("files")
        if (
            not isinstance(raw_rows, list)
            or len(raw_rows) != expected_registry_count
        ):
            raise RuntimeError("fit-only inventory file registry cardinality drift")
        rows: dict[str, dict[str, Any]] = {}
        category_rows: dict[str, list[dict[str, Any]]] = {}
        total_bytes = 0
        for raw in raw_rows:
            if not isinstance(raw, Mapping) or set(raw) != {
                "category", "path", "bytes", "sha256",
            }:
                raise RuntimeError("fit-only inventory file-row schema drift")
            relative = str(raw["path"])
            candidate = PurePosixPath(relative)
            if (
                candidate.is_absolute() or ".." in candidate.parts
                or not candidate.parts or relative in rows
                or any(token in relative.lower() for token in FORBIDDEN_PATH_TOKENS)
            ):
                raise RuntimeError("fit-only inventory file path is unsafe or duplicated")
            row = {
                "category": str(raw["category"]),
                "path": relative,
                "bytes": int(raw["bytes"]),
                "sha256": str(raw["sha256"]),
            }
            if row["bytes"] <= 0 or SHA256_RE.fullmatch(row["sha256"]) is None:
                raise RuntimeError("fit-only inventory file identity is invalid")
            rows[relative] = row
            category_rows.setdefault(row["category"], []).append(row)
            total_bytes += row["bytes"]

        lineage = payload.get("lineage")
        if not isinstance(lineage, Mapping):
            raise RuntimeError("fit-only inventory lineage is absent")
        fit_relative = str(lineage.get("fit_manifest", ""))
        fit_row = {
            "category": "fit_manifest",
            "path": fit_relative,
            "bytes": int(lineage.get("fit_manifest_bytes", -1)),
            "sha256": str(lineage.get("fit_manifest_sha256", "")),
        }
        if (
            fit_relative != "data/manifests/voc2012-m06e-fit1364.json"
            or fit_relative in rows or fit_row["bytes"] <= 0
            or SHA256_RE.fullmatch(fit_row["sha256"]) is None
        ):
            raise RuntimeError("fit-only inventory fit-manifest identity drift")
        rows[fit_relative] = fit_row

        categories = payload.get("categories")
        if not isinstance(categories, Mapping) or set(categories) != set(category_rows):
            raise RuntimeError("fit-only inventory category registry drift")
        for category, observed_rows in category_rows.items():
            registered = categories[category]
            if not isinstance(registered, Mapping) or set(registered) != {
                "files", "bytes", "inventory_sha256",
            }:
                raise RuntimeError("fit-only inventory category schema drift")
            if (
                int(registered["files"]) != len(observed_rows)
                or int(registered["bytes"])
                != sum(int(item["bytes"]) for item in observed_rows)
                or str(registered["inventory_sha256"])
                != hashlib.sha256(canonical_json_bytes(observed_rows)).hexdigest()
            ):
                raise RuntimeError("fit-only inventory category identity drift")
        if int(payload.get("counts", {}).get("bytes", -1)) != total_bytes:
            raise RuntimeError("fit-only inventory total byte count drift")

        core = {
            fit_relative,
            str(payload["builder_path_map"]["state_index"]),
            "artifacts/voc2012/m06e/atomic-bundle-v2/manifest.json",
            "artifacts/voc2012/m06e/atomic-bundle-v2/arrays.npz",
            "artifacts/voc2012/m07a/bundle-f1/manifest.json",
            "artifacts/voc2012/m07a/bundle-f1/arrays.npz",
        }
        stage_categories = {"candidate_cache", "causal_partition"}
        if stage == "targets":
            stage_categories |= {
                "trajectory_report", "trajectory_cache",
                "fit_segmentation_label",
            }
        expected = core | {
            path for path, row in rows.items()
            if row["category"] in stage_categories
        }
        missing = expected.difference(rows)
        expected_count = 54_566 if stage == "features" else 219_614
        if missing or (
            expected_registry_count == 220_989 and len(expected) != expected_count
        ):
            raise RuntimeError("fit-only inventory expected-input registry is incomplete")
        return cls(
            repository=repository.resolve(), stage=stage, rows=rows,
            expected_paths=frozenset(expected),
        )

    def verify(self, path: Path, *, category: str) -> Mapping[str, Any]:
        relative = _repository_relative(self.repository, path)
        row = self.rows.get(relative)
        if row is None or row.get("category") != category:
            raise RuntimeError(
                f"{self.stage} input is absent from exact inventory: {relative}"
            )
        if relative not in self.expected_paths:
            raise RuntimeError(f"{self.stage} opened a non-stage input: {relative}")
        _assert_input_file(path, description=f"inventoried {category}")
        if (
            path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"inventoried input identity drift: {relative}")
        self.consumed[relative] = row
        return row

    def complete(self) -> dict[str, Any]:
        observed = set(self.consumed)
        if observed != set(self.expected_paths):
            missing = sorted(self.expected_paths - observed)
            extra = sorted(observed - self.expected_paths)
            raise RuntimeError(
                "fit-only consumed-input set drift: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        rows = [dict(self.consumed[path]) for path in sorted(self.consumed)]
        return {
            "stage": self.stage,
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "combined_sha256": hashlib.sha256(
                canonical_json_bytes(rows)
            ).hexdigest(),
            "exact_expected_set_consumed": True,
        }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _repository_relative(repository: Path, path: Path) -> str:
    candidate = path if path.is_absolute() else repository / path
    return candidate.resolve().relative_to(repository.resolve()).as_posix()


def _committed_builder_implementation(repository: Path) -> dict[str, Any]:
    paths = tuple(PROSPECTIVE_BUILDER_IMPLEMENTATION_PATHS)
    if not paths or len(set(paths)) != len(paths):
        raise RuntimeError("prospective builder dependency registry is invalid")
    for relative in paths:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("prospective builder dependency path is unsafe")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=repository,
            capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                f"prospective builder dependency is not committed: {relative}"
            )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("prospective builder dependencies must be committed and clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("prospective builder implementation commit is malformed")
    source_sha256: dict[str, str] = {}
    for relative in paths:
        current = repository / relative
        digest = sha256_file(current)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=repository,
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(
                f"prospective builder dependency differs from Git: {relative}"
            )
        source_sha256[relative] = digest
    return {
        "commit": commit,
        "source_sha256": source_sha256,
        "combined_sha256": hashlib.sha256(
            canonical_json_bytes(source_sha256)
        ).hexdigest(),
    }


def _committed_builder_identity(
    repository: Path, implementation: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = _repository_relative(repository, path)
    commit = str(implementation["commit"])
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repository, check=True, capture_output=True,
    ).stdout
    committed_sha = hashlib.sha256(committed).hexdigest()
    working_sha = str(implementation["source_sha256"].get(relative, ""))
    if committed_sha != working_sha:
        raise RuntimeError("prospective bundle builder differs from committed bytes")
    return {"path": relative, "sha256": working_sha, "commit": commit}


def _validate_inventory_payload(payload: Mapping[str, Any]) -> None:
    counts = payload.get("counts", {})
    checks = payload.get("checks", {})
    access = payload.get("heldout_access", {})
    materializer = payload.get("materializer")
    path_map = payload.get("builder_path_map")
    if (
        payload.get("schema_version")
        != "rail3.tmlr-v6-fit-only-input-inventory.v1"
        or payload.get("status") != "TMLR_V6_FIT_ONLY_INPUTS_COMPLETE"
        or not checks
        or not all(bool(value) for value in checks.values())
        or int(counts.get("files", -1)) != 220_989
        or int(counts.get("fit_image_groups", -1)) != 1364
        or int(counts.get("symlinks", -1)) != 0
        or int(counts.get("forbidden_path_token_matches", -1)) != 0
        or set(access) != set(ZERO_EXTERNAL_ACCESS)
        or any(int(access[name]) != 0 for name in ZERO_EXTERNAL_ACCESS)
        or not isinstance(materializer, Mapping)
        or set(materializer) != {"path", "bytes", "sha256", "commit"}
        or materializer.get("path") != FIT_ONLY_MATERIALIZER_PATH.as_posix()
        or int(materializer.get("bytes", -1)) <= 0
        or SHA256_RE.fullmatch(str(materializer.get("sha256", ""))) is None
        or COMMIT_RE.fullmatch(str(materializer.get("commit", ""))) is None
        or str(materializer.get("commit")) != str(payload.get("destination_git_head"))
        or not isinstance(path_map, Mapping)
        or set(map(str, path_map)) != EXPECTED_BUILDER_PATH_KEYS
    ):
        raise RuntimeError("fit-only input inventory boundary or completeness drift")
    for value in path_map.values():
        candidate = PurePosixPath(str(value))
        if (
            candidate.is_absolute() or ".." in candidate.parts
            or not candidate.parts
            or any(token in candidate.as_posix().lower() for token in FORBIDDEN_PATH_TOKENS)
        ):
            raise RuntimeError("fit-only inventory builder path map is unsafe")


def _verify_inventory_materializer(
    repository: Path, payload: Mapping[str, Any],
) -> None:
    identity = payload["materializer"]
    path = repository / str(identity["path"])
    _assert_input_file(path, description="committed fit-only materializer")
    if (
        path.stat().st_size != int(identity["bytes"])
        or sha256_file(path) != str(identity["sha256"])
    ):
        raise RuntimeError("fit-only materializer working identity drift")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(identity["path"])],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("fit-only materializer is not committed and clean")
    commit = str(identity["commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("fit-only materializer commit is not an ancestor")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{identity['path']}"], cwd=repository,
        check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != str(identity["sha256"]):
        raise RuntimeError("fit-only materializer committed identity drift")


def _validate_builder_paths(
    *, repository: Path, args: argparse.Namespace, payload: Mapping[str, Any],
) -> None:
    registered = {str(key): str(value) for key, value in payload["builder_path_map"].items()}
    for key in EXPECTED_BUILDER_PATH_KEYS:
        actual = _repository_relative(repository, Path(getattr(args, key)))
        if actual != registered[key]:
            raise RuntimeError(f"builder argument differs from inventory path map: {key}")
    fit_manifest = str(payload["lineage"].get("fit_manifest", ""))
    if _repository_relative(repository, args.manifest) != fit_manifest:
        raise RuntimeError("builder fit manifest differs from inventory lineage")
    causal_expected = PurePosixPath(registered["state_index"]).parent.as_posix()
    if _repository_relative(repository, args.causal_root) != causal_expected:
        raise RuntimeError("builder causal root differs from inventory state-index root")
    if _repository_relative(repository, args.input_inventory) != DEFAULT_INPUT_INVENTORY.as_posix():
        raise RuntimeError("builder input-inventory path is not canonical")
    if _repository_relative(repository, args.output_root) != DEFAULT_OUTPUT_ROOT.as_posix():
        raise RuntimeError("builder output root is not canonical")


def _fit_only_inventory_identity(
    repository: Path, path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_fit_only_path(path)
    _assert_input_file(path, description="fit-only input inventory")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("fit-only input inventory is not a JSON object")
    _validate_inventory_payload(payload)
    _verify_inventory_materializer(repository, payload)
    return {
        "path": _repository_relative(repository, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": str(payload["status"]),
    }, dict(payload)


def _partition_path(
    causal_root: Path, record: Mapping[str, Any],
    *, input_guard: InputInventoryGuard | None = None,
) -> Path:
    registered = PurePosixPath(str(record.get("partition_path", "")))
    marker = PurePosixPath("artifacts/voc2012/m06e/causal-v2")
    try:
        suffix = registered.relative_to(marker)
    except ValueError as error:
        raise RuntimeError("causal partition path escaped its frozen root") from error
    if (
        len(suffix.parts) != 3
        or suffix.parts[0] != "atoms"
        or not suffix.name.endswith(".json")
        or suffix.stem != str(record.get("state_id"))
    ):
        raise RuntimeError("causal partition path contract drift")
    path = causal_root / suffix
    if input_guard is not None:
        input_guard.verify(path, category="causal_partition")
    _assert_input_file(path, description="causal partition")
    expected_bytes = int(record.get("partition_bytes", -1))
    expected_sha = str(record.get("partition_sha256", ""))
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise RuntimeError("causal partition identity drift")
    return path


def _observable_union(partition: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct A0/A1/A2 union solely from causal atom incidence."""

    height, width = int(partition["height"]), int(partition["width"])
    if (
        partition.get("construction_valid_domain")
        != "full_deployment_observable_image_raster"
        or tuple(partition.get("base_action_codes", ())) != ("A0", "A1", "A2")
    ):
        raise RuntimeError("causal partition is not a full-raster F0 observation")
    result = np.zeros((height, width), dtype=bool)
    area = 0
    for atom in partition.get("atoms", ()):
        use = bool(atom.get("base_candidate_incidence"))
        for run in atom.get("runs", ()):
            if len(run) != 3:
                raise RuntimeError("causal atom run contract drift")
            y, start, stop = map(int, run)
            if not (0 <= y < height and 0 <= start < stop <= width):
                raise RuntimeError("causal atom run escaped full raster")
            if use:
                result[y, start:stop] = True
                area += stop - start
    if int(result.sum()) != area:
        raise RuntimeError("causal observable-union runs overlap")
    return result


def _regional_atom_ids(
    *, role: str, state_id: str, partition: RegionPartition,
) -> np.ndarray:
    values = []
    for index, rectangle in enumerate(partition.rectangles):
        values.append(stable_id(
            f"tmlr_v6_{role}_atom",
            {
                "state_id": state_id,
                "region_index": index,
                "bbox": list(map(int, rectangle)),
                "geometry": ROLE_SPECS[role][1],
            },
        ))
    return np.asarray(values)


def _indexed_atom_ids(
    *, role: str, state_id: str, count: int,
) -> np.ndarray:
    return np.asarray([
        stable_id(
            f"tmlr_v6_{role}_atom",
            {"state_id": state_id, "atom_index": index},
        )
        for index in range(count)
    ])


def full_raster_targets(
    *, partition: RegionPartition, labels: np.ndarray, class_id: int,
    action_predictions: Sequence[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply GT validity after feature geometry, retaining void-only regions."""

    label_map = np.asarray(partition.label_map)
    labels = np.asarray(labels)
    if (
        labels.ndim != 2
        or label_map.shape != labels.shape
        or len(action_predictions) != len(ACTIONS)
        or np.any(label_map < 0)
    ):
        raise ValueError("full-raster target inputs are not aligned")
    region_count = partition.region_count
    valid = labels != 255
    state_valid = int(valid.sum())
    if state_valid <= 0:
        raise RuntimeError("fit label has no target-valid pixel")
    valid_counts = np.bincount(
        label_map[valid], minlength=region_count,
    ).astype(np.int64)
    if len(valid_counts) != region_count:
        raise RuntimeError("full-raster region cardinality drift")
    weights = (valid_counts / state_valid).astype(np.float32)
    atom_target = np.full((region_count, len(ACTIONS)), np.nan, dtype=np.float32)
    state_target = np.full(len(ACTIONS), np.nan, dtype=np.float32)
    truth = labels == int(class_id)
    defined_regions = valid_counts > 0
    for action_index, prediction in enumerate(action_predictions):
        if prediction is None:
            continue
        prediction = np.asarray(prediction, dtype=bool)
        if prediction.shape != labels.shape:
            raise RuntimeError("cached action output geometry drift")
        mismatch = (prediction != truth) & valid
        mismatch_counts = np.bincount(
            label_map[mismatch], minlength=region_count,
        ).astype(np.int64)
        atom_target[defined_regions, action_index] = (
            mismatch_counts[defined_regions] / valid_counts[defined_regions]
        ).astype(np.float32)
        direct = int(mismatch.sum())
        if int(mismatch_counts.sum()) != direct:
            raise RuntimeError("full-raster additive mismatch reconstruction drift")
        state_target[action_index] = np.float32(direct / state_valid)
        reconstructed = float(np.nansum(
            atom_target[:, action_index].astype(np.float64)
            * weights.astype(np.float64)
        ))
        if not np.isclose(reconstructed, state_target[action_index], rtol=0, atol=2e-7):
            raise RuntimeError("full-raster weighted residual reconstruction drift")
    zero_valid = int((valid_counts == 0).sum())
    if (
        not np.isclose(
            weights.astype(np.float64).sum(), 1.0, rtol=0, atol=2e-7,
        )
        or np.any(weights[~defined_regions] != 0)
        or np.any(np.isfinite(atom_target[~defined_regions]))
    ):
        raise RuntimeError("zero-valid full-raster target convention drift")
    return atom_target, state_target, weights, zero_valid


def _trajectory_sources(args: argparse.Namespace) -> list[TrajectorySource]:
    return [
        TrajectorySource("fit100", 100, args.fit100_report, args.fit100_cache),
        TrajectorySource("m06c_additional", 400, args.m06c_report, args.m06c_cache),
        TrajectorySource("m06d_additional", 500, args.m06d_report, args.m06d_cache),
        TrajectorySource("m06e_additional", 364, args.m06e_report, args.m06e_cache),
    ]


def _candidate_object_path(root: Path, cache_key: str) -> Path:
    match = re.fullmatch(r"candidate_cache_([0-9a-f]{64})", cache_key)
    if match is None:
        raise RuntimeError("registered candidate cache key is invalid")
    return root / match.group(1)[:2] / f"{cache_key}.json"


def _trajectory_object_path(root: Path, cache_key: str) -> Path:
    match = re.fullmatch(r"trajectory_cache_([0-9a-f]{64})", cache_key)
    if match is None:
        raise RuntimeError("registered trajectory cache key is invalid")
    return root / match.group(1)[:2] / f"{cache_key}.json"


def _load_core(
    args: argparse.Namespace, *, physical_partition: str,
    input_guard: InputInventoryGuard | None = None,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Mapping[str, np.ndarray]],
    dict[str, Any], dict[str, Any],
]:
    if input_guard is not None:
        input_guard.verify(args.manifest, category="fit_manifest")
    manifest = _read_json(args.manifest, description="S1364 manifest")
    if (
        sha256_file(args.manifest) != FIT_MANIFEST_SHA256
        or int(manifest.get("count", -1)) != 1364
        or len(manifest.get("records", ())) != 1364
        or not manifest.get("checks", {}).get("no_heldout_content", False)
    ):
        raise RuntimeError("S1364 fit manifest identity or boundary drift")
    if input_guard is not None:
        input_guard.verify(args.state_index, category="causal_top_level")
    state_records = _read_jsonl(args.state_index, description="causal state index")
    if len(state_records) != 27_280:
        raise RuntimeError("causal state index cardinality drift")
    for path in (args.f0_bundle, args.f1_bundle):
        assert_fit_only_path(path)
        if input_guard is not None:
            input_guard.verify(path / "manifest.json", category="bundle_leaf")
            input_guard.verify(path / "arrays.npz", category="bundle_leaf")
    f0_bundle, f1_bundle = TensorBundle(args.f0_bundle), TensorBundle(args.f1_bundle)
    if physical_partition == "prospective_features":
        allowlist = HISTORICAL_FEATURE_INPUTS
    elif physical_partition == "targets":
        allowlist = HISTORICAL_TARGET_INPUTS
    else:
        raise ValueError("unknown historical physical input partition")
    bundles = {
        "f0": {name: f0_bundle.array(name) for name in allowlist["f0"]},
        "f1": {name: f1_bundle.array(name) for name in allowlist["f1"]},
    }
    f0_manifest, f1_manifest = f0_bundle.manifest, f1_bundle.manifest
    state_ids = [str(record["state_id"]) for record in state_records]
    if (
        state_ids != list(f0_manifest.get("state_ids", ()))
        or state_ids != list(f1_manifest.get("state_ids", ()))
        or list(f0_manifest.get("group_ids", ())) != list(f1_manifest.get("group_ids", ()))
    ):
        raise RuntimeError("historical feature/target state order drift")
    return manifest, state_records, bundles, f0_manifest, f1_manifest


def _plan_action_x16(
    *, item: Mapping[str, Any], state_record: Mapping[str, Any],
    partition_payload: Mapping[str, Any], baseline_cache: CandidateCache,
    baseline_cache_root: Path | None = None,
    input_guard: InputInventoryGuard | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    class_id = int(state_record["class_id"])
    base_lineage = list(partition_payload.get("base_action_lineage", ()))
    if tuple(item.get("action_code") for item in base_lineage) != ("A0", "A1", "A2"):
        raise RuntimeError("feature-stage causal lineage is not A0/A1/A2")
    state_id = str(state_record["state_id"])
    # canonical_observation needs only the registered canonical key and state
    # identity.  This synthetic shell deliberately has no realized adaptive
    # runtime, outcome, cache key, candidate count, or reason code.
    class_state = {
        "canonical_cache_key": str(base_lineage[0]["cache_key"]),
        "actions": [
            {"action_code": code, "state_id": state_id}
            for code in ("A1", "A2", "A3", "A4", "A5", "A6")
        ],
    }
    if input_guard is not None:
        if baseline_cache_root is None:
            raise RuntimeError("inventory guard requires an explicit candidate-cache root")
        input_guard.verify(
            _candidate_object_path(
                baseline_cache_root, str(class_state["canonical_cache_key"]),
            ),
            category="candidate_cache",
        )
    canonical_result, _, _, canonical_action_id = canonical_observation(
        item=item, class_state=class_state, baseline_cache=baseline_cache,
        class_id=class_id,
    )
    plans = build_action_plans(
        dict(item), class_id=class_id, canonical_result=canonical_result,
        canonical_cache_key=str(class_state["canonical_cache_key"]),
    )
    adaptive = tuple(plans[2:])
    if tuple(plan.action_code for plan in adaptive) != ACTIONS[1:]:
        raise RuntimeError("registered prospective action plan order drift")
    if any(
        plan.action_id != str(lineage["action_id"])
        or plan.feasible != bool(lineage["feasible"])
        or plan.source_candidate_id != lineage.get("source_candidate_id")
        for plan, lineage in zip(plans[:2], base_lineage[1:])
    ):
        raise RuntimeError("registered A1/A2 plans differ from causal lineage")
    matrix = action_matrix(
        stop=stop_descriptor(
            state_id=str(state_record["state_id"]),
            canonical_action_id=canonical_action_id,
        ),
        adaptive=tuple(descriptor_from_plan(plan) for plan in adaptive),
    )
    f1_matrix = matrix.copy()
    f1_matrix[1, 0] = 0.0
    f1_matrix[1, 7] = 0.0
    return matrix, f1_matrix


@lru_cache(maxsize=None)
def _full_raster_rectangle(
    height: int, width: int, region_count: int,
) -> RegionPartition:
    """Cache immutable all-valid rectangle geometry shared by equal shapes/K."""

    result = guillotine_partition(
        np.ones((height, width), dtype=bool), region_count,
    )
    result.label_map.setflags(write=False)
    return result


def _partitions(
    *, partition_payload: Mapping[str, Any], f0_atom_count: int,
) -> tuple[RegionPartition, RegionPartition, np.ndarray]:
    height, width = int(partition_payload["height"]), int(partition_payload["width"])
    observable_union = _observable_union(partition_payload)
    rectangle = _full_raster_rectangle(height, width, f0_atom_count)
    # Empty observable unions dominate this fit population.  For either
    # constant mask, the canonical 4-connected union/complement partition is
    # exactly one full-raster component, so avoid two full image traversals.
    if not observable_union.any() or observable_union.all():
        label_map = np.zeros((height, width), dtype=np.int32)
        label_map.setflags(write=False)
        union = RegionPartition(
            label_map=label_map,
            rectangles=((0, 0, width, height),),
            kind="CANDIDATE_UNION_COMPONENTS",
        )
    else:
        union = union_component_partition(
            observable_union, np.ones((height, width), dtype=bool),
        )
    return rectangle, union, observable_union


def _concatenate(rows: Mapping[str, list[np.ndarray]], names: Iterable[str]) -> dict[str, np.ndarray]:
    result = {}
    for name in names:
        values = rows[name]
        if not values:
            raise RuntimeError(f"no rows accumulated for {name}")
        result[name] = np.concatenate(values)
    return result


def build_features(
    args: argparse.Namespace, *, input_guard: InputInventoryGuard | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    manifest, state_records, bundles, f0_manifest, _ = _load_core(
        args, physical_partition="prospective_features", input_guard=input_guard,
    )
    baseline_cache = CandidateCache(args.baseline_cache)
    manifest_by_asset = {
        str(item["asset_id"]): item for item in manifest["records"]
    }
    f0, f1 = bundles["f0"], bundles["f1"]
    f0_offsets = np.r_[0, np.cumsum(np.asarray(f0["state_lengths"], dtype=np.int64))]
    f1_offsets = np.r_[0, np.cumsum(np.asarray(f1["state_lengths"], dtype=np.int64))]
    rows: dict[str, dict[str, list[np.ndarray]]] = {
        role: {name: [] for name in FEATURE_NAMES} for role in ROLE_SPECS
    }
    for index, state_record in enumerate(state_records):
        asset_id = str(state_record["asset_id"])
        item = manifest_by_asset[asset_id]
        state_id = str(state_record["state_id"])
        partition_payload = _read_json(
            _partition_path(
                args.causal_root, state_record, input_guard=input_guard,
            ),
            description="causal partition",
        )
        if partition_payload.get("state_id") != state_id:
            raise RuntimeError("causal partition state identity drift")
        f0_start, f0_stop = int(f0_offsets[index]), int(f0_offsets[index + 1])
        f1_start, f1_stop = int(f1_offsets[index]), int(f1_offsets[index + 1])
        f0_count, f1_count = f0_stop - f0_start, f1_stop - f1_start
        if f0_count != int(state_record["atom_count"]):
            raise RuntimeError("historical F0 atom length differs from causal state index")
        action_x, f1_action_x = _plan_action_x16(
            item=item, state_record=state_record,
            partition_payload=partition_payload,
            baseline_cache=baseline_cache,
            baseline_cache_root=args.baseline_cache,
            input_guard=input_guard,
        )
        rectangle, union, observable_union = _partitions(
            partition_payload=partition_payload, f0_atom_count=f0_count,
        )
        base_lineage = list(partition_payload["base_action_lineage"])
        base_cost = float(f0["base_cost_seconds"][index])
        rect_x, rect_state_x = regional_feature_matrices(
            partition=rectangle, class_id=int(state_record["class_id"]),
            base_lineage=base_lineage, base_cost_seconds=base_cost,
        )
        union_x, union_state_x = regional_feature_matrices(
            partition=union, class_id=int(state_record["class_id"]),
            base_lineage=base_lineage, base_cost_seconds=base_cost,
            observable_union=observable_union,
        )
        common = {
            "state_x": np.asarray(f0["state_x"][index:index + 1], dtype=np.float32),
            "action_x16": action_x[None, ...].astype(np.float32),
            "state_ids": np.asarray([state_id]),
            "image_group_ids": np.asarray([str(state_record["image_group_id"])]),
        }
        role_values = {
            "f0": {
                "atom_x": np.asarray(f0["atom_x"][f0_start:f0_stop], dtype=np.float32),
                "state_lengths": np.asarray([f0_count], dtype=np.int32),
                "atom_ids": _indexed_atom_ids(role="f0", state_id=state_id, count=f0_count),
                **common,
            },
            "rect": {
                "atom_x": rect_x.astype(np.float32),
                "state_x": rect_state_x[None, :].astype(np.float32),
                "state_lengths": np.asarray([rectangle.region_count], dtype=np.int32),
                "atom_ids": _regional_atom_ids(role="rect", state_id=state_id, partition=rectangle),
                **{name: value for name, value in common.items() if name != "state_x"},
            },
            "union": {
                "atom_x": union_x.astype(np.float32),
                "state_x": union_state_x[None, :].astype(np.float32),
                "state_lengths": np.asarray([union.region_count], dtype=np.int32),
                "atom_ids": _regional_atom_ids(role="union", state_id=state_id, partition=union),
                **{name: value for name, value in common.items() if name != "state_x"},
            },
            "f1": {
                "atom_x": np.asarray(f1["atom_x"][f1_start:f1_stop], dtype=np.float32),
                "state_x": np.asarray(f1["state_x"][index:index + 1], dtype=np.float32),
                "action_x16": f1_action_x[None, ...].astype(np.float32),
                "state_lengths": np.asarray([f1_count], dtype=np.int32),
                "state_ids": np.asarray([state_id]),
                "image_group_ids": np.asarray([str(state_record["image_group_id"])]),
                "atom_ids": _indexed_atom_ids(role="f1", state_id=state_id, count=f1_count),
            },
        }
        for role, values in role_values.items():
            for name, value in values.items():
                rows[role][name].append(value)
        if (index + 1) % 500 == 0 or index + 1 == len(state_records):
            print(json.dumps({
                "stage": "TMLR_V6_PROSPECTIVE_FEATURES",
                "states": index + 1,
                "total": len(state_records),
            }, sort_keys=True), flush=True)
    outputs = {
        role: _concatenate(role_rows, FEATURE_NAMES)
        for role, role_rows in rows.items()
    }
    expected_states = len(state_records)
    for role, arrays in outputs.items():
        lengths = np.asarray(arrays["state_lengths"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths[:-1])]
        inference_weights = np.asarray(arrays["atom_x"][:, 0], dtype=np.float64)
        weight_sums = np.add.reduceat(inference_weights, offsets)
        if (
            arrays["state_x"].shape != (expected_states, 41)
            or arrays["action_x16"].shape != (expected_states, 5, ACTION_DIM)
            or arrays["atom_x"].shape[1] != 27
            or int(arrays["state_lengths"].sum()) != len(arrays["atom_x"])
            or not all(np.isfinite(arrays[name]).all() for name in ("atom_x", "state_x", "action_x16"))
            or len(set(arrays["atom_ids"].tolist())) != len(arrays["atom_ids"])
            or np.any(inference_weights <= 0)
            or not np.allclose(weight_sums, 1.0, rtol=0, atol=2e-6)
            or not np.array_equal(
                arrays["action_x16"][:, :, 0], arrays["action_x16"][:, :, 7],
            )
            or not np.all(np.isin(arrays["action_x16"][:, :, 0], (0.0, 1.0)))
            or not np.all(arrays["action_x16"][:, 0, 0] == 1.0)
        ):
            raise RuntimeError(f"{role} prospective feature geometry drift")
    if list(f0_manifest["group_ids"]) != outputs["f0"]["image_group_ids"].tolist():
        raise RuntimeError("prospective feature group order drift")
    return outputs


def build_targets(
    args: argparse.Namespace, *, input_guard: InputInventoryGuard | None = None,
) -> tuple[
    dict[str, dict[str, np.ndarray]], dict[str, int],
]:
    manifest, state_records, bundles, _, _ = _load_core(
        args, physical_partition="targets", input_guard=input_guard,
    )
    sources = _trajectory_sources(args)
    if input_guard is not None:
        for source in sources:
            input_guard.verify(source.report_path, category="trajectory_report")
    trajectory_index, _ = load_trajectory_sources(
        manifest=manifest, sources=sources,
    )
    baseline_cache = CandidateCache(args.baseline_cache)
    manifest_by_asset = {
        str(item["asset_id"]): item for item in manifest["records"]
    }
    records_by_asset: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(state_records):
        records_by_asset.setdefault(str(record["asset_id"]), []).append((index, record))
    f0, f1 = bundles["f0"], bundles["f1"]
    f0_offsets = np.r_[0, np.cumsum(np.asarray(f0["state_lengths"], dtype=np.int64))]
    f1_offsets = np.r_[0, np.cumsum(np.asarray(f1["state_lengths"], dtype=np.int64))]
    rows: dict[str, dict[str, list[np.ndarray]]] = {
        role: {name: [] for name in TARGET_NAMES} for role in ROLE_SPECS
    }
    zero_valid = Counter()
    processed = 0
    from PIL import Image
    for asset_id in [str(item["asset_id"]) for item in manifest["records"]]:
        item = manifest_by_asset[asset_id]
        image_record, cache, origin = trajectory_index[asset_id]
        if origin != str(item["origin"]):
            raise RuntimeError("trajectory origin drift")
        label_path = (
            args.dataset_root / "VOCdevkit" / "VOC2012" / "SegmentationClass"
            / f"{item['sample_id']}.png"
        )
        if input_guard is not None:
            input_guard.verify(label_path, category="fit_segmentation_label")
        _assert_input_file(label_path, description="S1364 fit label")
        with Image.open(label_path) as image:
            labels = np.asarray(image, dtype=np.uint8)
        for index, state_record in records_by_asset[asset_id]:
            state_id = str(state_record["state_id"])
            class_id = int(state_record["class_id"])
            class_state = state_by_class(image_record)[class_id]
            if input_guard is not None:
                input_guard.verify(
                    _candidate_object_path(
                        args.baseline_cache,
                        str(class_state["canonical_cache_key"]),
                    ),
                    category="candidate_cache",
                )
                for action in class_state["actions"]:
                    input_guard.verify(
                        _trajectory_object_path(
                            Path(cache.root), str(action["cache_key"]),
                        ),
                        category="trajectory_cache",
                    )
            canonical_result, _, _, _ = canonical_observation(
                item=item, class_state=class_state, baseline_cache=baseline_cache,
                class_id=class_id,
            )
            _, outcomes = action_observations(
                class_state=class_state, cache=cache, class_id=class_id,
            )
            plans = build_action_plans(
                dict(item), class_id=class_id,
                canonical_result=canonical_result,
                canonical_cache_key=str(class_state["canonical_cache_key"]),
            )
            if tuple(plan.action_code for plan in plans[2:]) != ACTIONS[1:]:
                raise RuntimeError("target-stage prospective plan order drift")
            f0_plan_feasible = np.asarray(
                [True, *(bool(plan.feasible) for plan in plans[2:])],
                dtype=bool,
            )
            f1_plan_feasible = f0_plan_feasible.copy()
            f1_plan_feasible[1] = False
            predictions = _all_action_predictions(canonical_result, outcomes, labels.shape)
            partition_payload = _read_json(
                _partition_path(
                    args.causal_root, state_record, input_guard=input_guard,
                ),
                description="causal partition",
            )
            f0_start, f0_stop = int(f0_offsets[index]), int(f0_offsets[index + 1])
            f1_start, f1_stop = int(f1_offsets[index]), int(f1_offsets[index + 1])
            f0_count, f1_count = f0_stop - f0_start, f1_stop - f1_start
            rectangle, union, _ = _partitions(
                partition_payload=partition_payload, f0_atom_count=f0_count,
            )
            rect_atom, rect_state, rect_weight, rect_zero = full_raster_targets(
                partition=rectangle, labels=labels, class_id=class_id,
                action_predictions=predictions,
            )
            union_atom, union_state, union_weight, union_zero = full_raster_targets(
                partition=union, labels=labels, class_id=class_id,
                action_predictions=predictions,
            )
            expected_state = np.asarray(f0["state_target"][index], dtype=np.float32)
            if (
                not np.array_equal(rect_state, expected_state, equal_nan=True)
                or not np.array_equal(union_state, expected_state, equal_nan=True)
            ):
                raise RuntimeError("full-raster state target differs from frozen F0")
            frozen_f0_feasible = np.asarray(f0["feasible"][index], dtype=bool)
            frozen_f1_feasible = np.asarray(f1["feasible"][index], dtype=bool)
            if (
                not np.array_equal(frozen_f0_feasible, f0_plan_feasible)
                or not np.array_equal(frozen_f1_feasible, f1_plan_feasible)
            ):
                raise RuntimeError(
                    "historical feasibility differs from registered pre-action plans"
                )
            feasible = f0_plan_feasible[None, :]
            common = {
                "state_target": expected_state[None, :],
                "feasible": feasible,
                "state_ids": np.asarray([state_id]),
            }
            role_values = {
                "f0": {
                    "atom_target": np.asarray(f0["atom_target"][f0_start:f0_stop], dtype=np.float32),
                    "atom_weights": np.asarray(f0["atom_weights"][f0_start:f0_stop], dtype=np.float32),
                    "atom_ids": _indexed_atom_ids(role="f0", state_id=state_id, count=f0_count),
                    **common,
                },
                "rect": {
                    "atom_target": rect_atom,
                    "atom_weights": rect_weight,
                    "atom_ids": _regional_atom_ids(role="rect", state_id=state_id, partition=rectangle),
                    **common,
                },
                "union": {
                    "atom_target": union_atom,
                    "atom_weights": union_weight,
                    "atom_ids": _regional_atom_ids(role="union", state_id=state_id, partition=union),
                    **common,
                },
                "f1": {
                    "atom_target": np.asarray(f1["atom_target"][f1_start:f1_stop], dtype=np.float32),
                    "atom_weights": np.asarray(f1["atom_weights"][f1_start:f1_stop], dtype=np.float32),
                    "state_target": np.asarray(f1["state_target"][index:index + 1], dtype=np.float32),
                    "feasible": f1_plan_feasible[None, :],
                    "state_ids": np.asarray([state_id]),
                    "atom_ids": _indexed_atom_ids(role="f1", state_id=state_id, count=f1_count),
                },
            }
            for role, values in role_values.items():
                for name, value in values.items():
                    rows[role][name].append(value)
            zero_valid["rect"] += rect_zero
            zero_valid["union"] += union_zero
            processed += 1
            if processed % 500 == 0 or processed == len(state_records):
                print(json.dumps({
                    "stage": "TMLR_V6_PROSPECTIVE_TARGETS",
                    "states": processed,
                    "total": len(state_records),
                }, sort_keys=True), flush=True)
    outputs = {
        role: _concatenate(role_rows, TARGET_NAMES)
        for role, role_rows in rows.items()
    }
    for role, arrays in outputs.items():
        defined = np.isfinite(arrays["state_target"])
        plan_feasible = np.asarray(arrays["feasible"], dtype=bool)
        if (
            arrays["state_target"].shape != (27_280, 5)
            or arrays["feasible"].shape != (27_280, 5)
            or arrays["feasible"].dtype.kind != "b"
            or len(arrays["atom_target"]) != len(arrays["atom_weights"])
            or np.any(arrays["atom_weights"] < 0)
            or np.any(defined & ~plan_feasible)
        ):
            raise RuntimeError(f"{role} prospective target geometry drift")
    return outputs, dict(zero_valid)


def _target_missingness(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    plan_feasible = np.asarray(arrays["feasible"], dtype=bool)
    defined = np.isfinite(np.asarray(arrays["state_target"]))
    if plan_feasible.shape != defined.shape or np.any(defined & ~plan_feasible):
        raise RuntimeError("defined target does not imply pre-action feasibility")
    missing = plan_feasible & ~defined
    incomplete = np.any(missing, axis=1)
    state_ids = np.asarray(arrays["state_ids"]).astype(str)
    incomplete_ids = sorted(state_ids[incomplete].tolist())
    return {
        "plan_feasible_target_undefined_actions": int(missing.sum()),
        "incomplete_target_states": int(incomplete.sum()),
        "incomplete_target_state_ids_sha256": hashlib.sha256(
            canonical_json_bytes(incomplete_ids)
        ).hexdigest(),
    }


def _write_role_part(
    *, repository: Path, output_root: Path, role: str, partition: str,
    arrays: Mapping[str, np.ndarray], input_identities: Mapping[str, Any],
    builder_identity: Mapping[str, Any],
    builder_implementation: Mapping[str, Any],
    fit_only_input_inventory: Mapping[str, Any],
    consumed_inputs: Mapping[str, Any],
    zero_valid_regions: int = 0,
    target_missingness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    role_id, geometry, source_actions = ROLE_SPECS[role]
    if partition == "prospective_features":
        directory, filename = output_root / role / "features", "features.npz"
        schema, status, expected_names = FEATURE_SCHEMA, FEATURE_STATUS, FEATURE_NAMES
    elif partition == "targets":
        directory, filename = output_root / role / "target", "targets.npz"
        schema, status, expected_names = TARGET_SCHEMA, TARGET_STATUS, TARGET_NAMES
    else:
        raise ValueError("unknown physical prospective partition")
    if set(arrays) != expected_names:
        raise RuntimeError(f"{role} {partition} array schema drift")
    array_path = directory / filename
    _atomic_npz(array_path, arrays)
    n_states = len(arrays["state_ids"])
    n_atoms = len(arrays["atom_ids"])
    manifest: dict[str, Any] = {
        "schema_version": schema,
        "status": status,
        "physical_partition": partition,
        "role_id": role_id,
        "geometry": geometry,
        "prospective_schema_id": PROSPECTIVE_SCHEMA_ID,
        "action_order": list(ACTIONS),
        "array_names": sorted(expected_names),
        "array": {
            "path": _repository_relative(repository, array_path),
            "bytes": array_path.stat().st_size,
            "sha256": sha256_file(array_path),
        },
        "counts": {
            "states": n_states,
            "atoms": n_atoms,
            "state_actions": n_states * len(ACTIONS),
            "zero_valid_regions": int(zero_valid_regions),
        },
        "input_identities": dict(input_identities),
        "committed_builder": dict(builder_identity),
        "builder_implementation": dict(builder_implementation),
        "fit_only_input_inventory": dict(fit_only_input_inventory),
        "consumed_fit_only_inputs": dict(consumed_inputs),
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
    }
    if partition == "prospective_features":
        manifest.update({
            "runtime_feature_count": 0,
            "future_mask_feature_count": 0,
            "ground_truth_feature_count": 0,
            "action_dimension": ACTION_DIM,
            "post_projection_only": True,
            "source_actions": list(source_actions),
            "inference_weight_source": "feature_area_fraction_full_raster",
            "ground_truth_inference_input": False,
            "checks": {
                "feature_target_physical_separation": True,
                "realized_runtime_removed_before_scaling": True,
                "registered_plan_descriptors_only": True,
                "legacy_joined_container_member_allowlist_enforced": True,
                "committed_action_protocol_dependency": True,
                "no_future_mask_feature": True,
                "no_ground_truth_feature": True,
                "full_raster_geometry_before_gt_validity": True,
                "protected_access_zero": True,
            },
        })
    else:
        if target_missingness is None:
            raise RuntimeError("target missingness audit is required")
        manifest.update({
            "target_contract_id": TARGET_CONTRACT,
            "target_definition": TARGET_DEFINITION,
            "target_weight_source": "valid_pixel_fraction_target_only",
            "target_missingness": dict(target_missingness),
            "checks": {
                "fit_labels_only": True,
                "cached_action_outputs_only": True,
                "same_frozen_state_target": True,
                "zero_valid_region_weight_zero": True,
                "zero_valid_region_target_nan": True,
                "protected_access_zero": True,
                "defined_target_implies_plan_feasible": True,
                "execution_failure_missingness_is_not_feature_feasibility": True,
                "committed_action_protocol_dependency": True,
            },
        })
    atomic_create_bytes(
        directory / "manifest.json", canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def _input_identities(
    repository: Path, args: argparse.Namespace,
) -> dict[str, Any]:
    paths = {
        "fit_manifest": args.manifest,
        "state_index": args.state_index,
        "f0_manifest": args.f0_bundle / "manifest.json",
        "f0_array": args.f0_bundle / "arrays.npz",
        "f1_manifest": args.f1_bundle / "manifest.json",
        "f1_array": args.f1_bundle / "arrays.npz",
    }
    return {
        name: {
            "path": _repository_relative(repository, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _write_build_report(
    *, repository: Path, output_root: Path, stages: Sequence[str],
    manifests: Sequence[Mapping[str, Any]], started: float,
    builder_identity: Mapping[str, Any],
    builder_implementation: Mapping[str, Any],
    fit_only_input_inventory: Mapping[str, Any],
    consumed_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    report_path = output_root / "build-report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {
        "schema_version": "rail3.tmlr-v6.prospective-bundle-build.v1",
        "status": "TMLR_V6_PROSPECTIVE_BUNDLES_COMPLETE",
        "stages": list(stages),
        "roles": {
            f"{manifest['role_id']}:{manifest['physical_partition']}": {
                "physical_partition": manifest["physical_partition"],
                "geometry": manifest["geometry"],
                "states": manifest["counts"]["states"],
                "atoms": manifest["counts"]["atoms"],
                "array": manifest["array"],
            }
            for manifest in manifests
        },
        "fit_labels_opened": 1364 if "targets" in stages else 0,
        "committed_builder": dict(builder_identity),
        "builder_implementation": dict(builder_implementation),
        "fit_only_input_inventory": dict(fit_only_input_inventory),
        "consumed_fit_only_inputs": {
            stage: dict(identity) for stage, identity in consumed_inputs.items()
        },
        "sam_inference_used": False,
        "prompt_or_trajectory_generation_used": False,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_create_bytes(report_path, canonical_json_bytes(report) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mirror = Path("data/v6_fit_only/source_mirror")
    parser.add_argument("--stage", choices=("features", "targets", "all"), default="all")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/voc2012-m06e-fit1364.json"))
    parser.add_argument("--dataset-root", type=Path, default=mirror / "data/raw/voc2012/extracted")
    parser.add_argument("--baseline-cache", type=Path, default=mirror / "artifacts/candidates/voc2012/sam31-text-v1")
    parser.add_argument("--causal-root", type=Path, default=mirror / "artifacts/voc2012/m06e/causal-v2")
    parser.add_argument("--state-index", type=Path, default=mirror / "artifacts/voc2012/m06e/causal-v2/state-index.jsonl")
    parser.add_argument("--f0-bundle", type=Path, default=Path("artifacts/voc2012/m06e/atomic-bundle-v2"))
    parser.add_argument("--f1-bundle", type=Path, default=Path("artifacts/voc2012/m07a/bundle-f1"))
    parser.add_argument("--fit100-report", type=Path, default=mirror / "artifacts/voc2012/fit100-trajectory-merged.json")
    parser.add_argument("--fit100-cache", type=Path, default=mirror / "artifacts/trajectories/voc2012/m04-m05-v1/fit100-merged")
    parser.add_argument("--m06c-report", type=Path, default=mirror / "artifacts/voc2012/m06c/trajectory-new400-merged.json")
    parser.add_argument("--m06c-cache", type=Path, default=mirror / "artifacts/trajectories/voc2012/m06c/new400")
    parser.add_argument("--m06d-report", type=Path, default=mirror / "artifacts/voc2012/m06d/trajectory-new500-merged.json")
    parser.add_argument("--m06d-cache", type=Path, default=mirror / "artifacts/voc2012/m06d/trajectory-cache-merged")
    parser.add_argument("--m06e-report", type=Path, default=mirror / "artifacts/voc2012/m06e/trajectory-new364-merged.json")
    parser.add_argument("--m06e-cache", type=Path, default=mirror / "artifacts/voc2012/m06e/trajectory-cache-merged")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-inventory", type=Path, default=DEFAULT_INPUT_INVENTORY)
    args = parser.parse_args()

    repository = Path.cwd().resolve()
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if Path(root).resolve() != repository:
        raise RuntimeError("prospective bundle builder requires the worktree root")
    for value in vars(args).values():
        if isinstance(value, Path):
            assert_fit_only_path(value)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("prospective bundle construction is CPU-only")
    builder_implementation = _committed_builder_implementation(repository)
    builder_identity = _committed_builder_identity(
        repository, builder_implementation,
    )
    inventory_identity, inventory_payload = _fit_only_inventory_identity(
        repository, args.input_inventory,
    )
    _validate_builder_paths(
        repository=repository, args=args, payload=inventory_payload,
    )
    started = time.perf_counter()
    stages = ("features", "targets") if args.stage == "all" else (args.stage,)
    identities = _input_identities(repository, args)
    manifests: list[Mapping[str, Any]] = []
    consumption: dict[str, Mapping[str, Any]] = {}
    zero_valid: Mapping[str, int] = {}
    if "features" in stages:
        feature_guard = InputInventoryGuard.from_payload(
            repository=repository, stage="features", payload=inventory_payload,
        )
        feature_arrays = build_features(args, input_guard=feature_guard)
        consumption["features"] = feature_guard.complete()
        for role in ROLE_SPECS:
            manifests.append(_write_role_part(
                repository=repository, output_root=args.output_root, role=role,
                partition="prospective_features", arrays=feature_arrays[role],
                input_identities=identities,
                builder_identity=builder_identity,
                builder_implementation=builder_implementation,
                fit_only_input_inventory=inventory_identity,
                consumed_inputs=consumption["features"],
            ))
    if "targets" in stages:
        target_guard = InputInventoryGuard.from_payload(
            repository=repository, stage="targets", payload=inventory_payload,
        )
        target_arrays, zero_valid = build_targets(
            args, input_guard=target_guard,
        )
        if "features" in stages:
            for role in ROLE_SPECS:
                plan_feasible = np.asarray(
                    feature_arrays[role]["action_x16"][:, :, 0], dtype=bool,
                )
                if not np.array_equal(
                    plan_feasible, np.asarray(target_arrays[role]["feasible"]),
                ):
                    raise RuntimeError(
                        f"{role} feature-plan/target feasibility alignment drift"
                    )
            first_batch_feasible = np.asarray(
                target_arrays["f0"]["feasible"], dtype=bool,
            )
            for role in ("rect", "union"):
                if not np.array_equal(
                    first_batch_feasible,
                    np.asarray(target_arrays[role]["feasible"], dtype=bool),
                ):
                    raise RuntimeError("first-batch plan feasibility differs by role")
        consumption["targets"] = target_guard.complete()
        for role in ROLE_SPECS:
            missingness = _target_missingness(target_arrays[role])
            manifests.append(_write_role_part(
                repository=repository, output_root=args.output_root, role=role,
                partition="targets", arrays=target_arrays[role],
                input_identities=identities,
                builder_identity=builder_identity,
                builder_implementation=builder_implementation,
                fit_only_input_inventory=inventory_identity,
                consumed_inputs=consumption["targets"],
                zero_valid_regions=int(zero_valid.get(role, 0)),
                target_missingness=missingness,
            ))
    if args.stage == "all":
        _write_build_report(
            repository=repository, output_root=args.output_root, stages=stages,
            manifests=manifests, started=started,
            builder_identity=builder_identity,
            builder_implementation=builder_implementation,
            fit_only_input_inventory=inventory_identity,
            consumed_inputs=consumption,
        )
    print(json.dumps({
        "status": "TMLR_V6_PROSPECTIVE_BUNDLE_STAGE_COMPLETE",
        "stages": list(stages),
        "roles": list(ROLE_SPECS),
        "heldout_access": ZERO_EXTERNAL_ACCESS,
        "sam_inference_used": False,
        "wall_seconds": time.perf_counter() - started,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
