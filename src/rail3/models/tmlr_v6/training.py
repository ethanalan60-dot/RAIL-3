"""Unified first-batch cross-fitting for the frozen TMLR V6 P0 protocol.

The model boundary is physically prospective.  A registered role supplies a
feature-only NPZ and a separate target-only NPZ.  The trainer cannot open the
historical joined atomic bundle, a post-action diagnostic artifact, or an
action-cost table.  Only the 16-column action projection may enter a model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.m06e.models import (
    GlobalResidual,
    SharedAtomicResidual,
    parameter_count,
)
from rail3.models.tmlr_v6.models import (
    CapacityMatchedGlobalResidual,
    ProspectiveSharedAtomicResidual,
    solve_capacity_matched_width,
)
from rail3.models.tmlr_v6.schema import (
    ACTION_DIM,
    ATOM_DIM,
    GLOBAL_DIM,
    PROSPECTIVE_BUILDER_IMPLEMENTATION_PATHS,
    PROSPECTIVE_SCHEMA_ID,
    STATE_DIM,
    scale_predictor_features,
    scaler_payload,
    validate_schema_contract,
)


PROTOCOL_PATH = Path("configs/experiments/tmlr_v6_p0_prospective_capacity.json")
FOLD_MANIFEST_PATH = Path("data/manifests/voc2012-m06e-s_max-folds.json")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/first-batch-crossfit"
)
GPU_LOCK_ROOT = Path("artifacts/voc2012/tmlr-v6-p0/gpu-locks")
FIRST_BATCH_MODEL_IDS = (
    "R0_SMALL_P",
    "R0_CM_P",
    "R1_P",
    "RECT_P",
    "UNION_P",
)
FOLDS = tuple(f"fold_{index}" for index in range(5))
SEEDS = (13, 37, 71)
GPU_ASSIGNMENT: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "0": ("fold_0", "fold_2", "fold_4"),
    "1": ("fold_1", "fold_3"),
})
ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
LEGACY_EXTERNAL_SPLITS = (
    "validation40",
    "calibration30",
    "pilot_test30",
    "official_voc_val",
)
EXTERNAL_SPLITS = (
    *LEGACY_EXTERNAL_SPLITS,
    "segppd",
    "uav_iap",
)
LEGACY_ZERO_EXTERNAL_ACCESS = {name: 0 for name in LEGACY_EXTERNAL_SPLITS}
ZERO_EXTERNAL_ACCESS = {name: 0 for name in EXTERNAL_SPLITS}
FEATURE_MANIFEST_SCHEMA = "rail3.tmlr-v6.prospective-feature-bundle.v1"
TARGET_MANIFEST_SCHEMA = "rail3.tmlr-v6.prospective-target-bundle.v1"
FEATURE_MANIFEST_STATUS = "TMLR_V6_PROSPECTIVE_FEATURE_BUNDLE_PASS"
TARGET_MANIFEST_STATUS = "TMLR_V6_PROSPECTIVE_TARGET_BUNDLE_PASS"
BUNDLE_IDENTITY_SCHEMA = "rail3.tmlr-v6.prospective-bundle-identities.v1"
BUNDLE_IDENTITY_STATUS = "TMLR_V6_PROSPECTIVE_BUNDLE_IDENTITIES_FROZEN"
FIT_ONLY_INPUT_INVENTORY_PATH = Path(
    "artifacts/audits/tmlr_v6/fit_only_input_inventory.json"
)
PROSPECTIVE_BUNDLE_BUILD_REPORT_PATH = Path(
    "artifacts/voc2012/tmlr-v6-p0/prospective-bundles/build-report.json"
)
PROSPECTIVE_BUNDLE_BUILDER_PATH = Path(
    "scripts/build_tmlr_v6_prospective_bundles.py"
)
FIT_ONLY_MATERIALIZER_PATH = Path(
    "scripts/materialize_tmlr_v6_fit_only_inputs.py"
)
RUNTIME_FEATURE_AUDIT_PATH = Path(
    "artifacts/audits/tmlr_v6/runtime_feature_audit.json"
)
RUNTIME_FEATURE_AUDIT_STATUS = (
    "RETRO_V4_FUTURE_RUNTIME_CONFIRMED__"
    "PROSPECTIVE_V1_SCHEMA_AND_PLAN_GUARD_DEFINED"
)
RUNTIME_FEATURE_AUDIT_SOURCE_PATHS = (
    PROTOCOL_PATH,
    Path("scripts/audit_tmlr_v6_runtime_features.py"),
    PROSPECTIVE_BUNDLE_BUILDER_PATH,
    Path("scripts/build_rail_diagnostic_representation_controls.py"),
    Path("src/rail3/data/m06e_sources.py"),
    Path("src/rail3/diagnostic/baselines.py"),
    Path("src/rail3/diagnostic/representation_training.py"),
    Path("src/rail3/models/m06e/features.py"),
    Path("src/rail3/models/m06e/losses.py"),
    Path("src/rail3/models/m06e/training.py"),
    Path("src/rail3/models/m06g/training.py"),
    Path("src/rail3/models/m07a/features.py"),
    Path("src/rail3/models/m07a/training.py"),
    Path("src/rail3/models/tmlr_v6/cost_profiles.py"),
    Path("src/rail3/models/tmlr_v6/descriptors.py"),
    Path("src/rail3/models/tmlr_v6/schema.py"),
    Path("src/rail3/models/tmlr_v6/training.py"),
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_PATH_FRAGMENTS = (
    "artifacts/protected",
    "protected/",
    "validation40",
    "locked40",
    "calibration30",
    "pilot-test30",
    "pilot_test30",
    "official-voc-val",
    "official_voc_val",
    "official-val",
    "official_val",
    "segmentation-trainval",
    "trainval-v1",
    "segppd",
    "uav",
    "uav-iap",
    "uav_iap",
)
FEATURE_ARRAY_NAMES = frozenset({
    "atom_x",
    "state_x",
    "action_x16",
    "state_lengths",
    "state_ids",
    "image_group_ids",
    "atom_ids",
})
TARGET_ARRAY_NAMES = frozenset({
    "atom_target",
    "atom_weights",
    "state_target",
    "feasible",
    "state_ids",
    "atom_ids",
})
MEMORY_BOUNDED_ATOM_THRESHOLD = 1_000_000
MEMORY_BOUNDED_ATOM_CHUNK = 65_536
VRAM_LIMIT = 40 * 1024**3
RUN_SECONDS_LIMIT = 1800.0
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
IMPLEMENTATION_PATHS = (
    Path("src/rail3/__init__.py"),
    Path("src/rail3/contracts/__init__.py"),
    Path("src/rail3/contracts/records.py"),
    Path("src/rail3/contracts/serialization.py"),
    Path("src/rail3/models/__init__.py"),
    Path("src/rail3/models/m06e/__init__.py"),
    Path("src/rail3/models/m06e/features.py"),
    Path("src/rail3/models/m06e/models.py"),
    Path("src/rail3/models/tmlr_v6/__init__.py"),
    Path("src/rail3/models/tmlr_v6/schema.py"),
    Path("src/rail3/models/tmlr_v6/models.py"),
    Path("src/rail3/models/tmlr_v6/training.py"),
    Path("scripts/run_tmlr_v6_first_batch_crossfit.py"),
    PROTOCOL_PATH,
)


@dataclass(frozen=True)
class BundleRegistration:
    """An immutable prospective feature/target role."""

    role_id: str
    root: Path
    geometry: str

    @property
    def feature_manifest(self) -> Path:
        return self.root / "features" / "manifest.json"

    @property
    def feature_array(self) -> Path:
        return self.root / "features" / "features.npz"

    @property
    def target_manifest(self) -> Path:
        return self.root / "target" / "manifest.json"

    @property
    def target_array(self) -> Path:
        return self.root / "target" / "targets.npz"


PROSPECTIVE_BUNDLE_ROOT = Path(
    "artifacts/voc2012/tmlr-v6-p0/prospective-bundles"
)
BUNDLE_REGISTRY: dict[str, BundleRegistration] = {
    "F0_P": BundleRegistration(
        role_id="F0_P",
        root=PROSPECTIVE_BUNDLE_ROOT / "f0",
        geometry="observable_boolean_cells",
    ),
    "RECT_FULL_RASTER_P": BundleRegistration(
        role_id="RECT_FULL_RASTER_P",
        root=PROSPECTIVE_BUNDLE_ROOT / "rect",
        geometry="full_raster_area_matched_rectangles",
    ),
    "UNION_FULL_RASTER_P": BundleRegistration(
        role_id="UNION_FULL_RASTER_P",
        root=PROSPECTIVE_BUNDLE_ROOT / "union",
        geometry="full_raster_candidate_union_components",
    ),
}

# F1 is built and provenance-registered with the same transaction, but is not
# a first-batch training role.  Keeping it outside BUNDLE_REGISTRY prevents an
# accidental first-batch use while retaining an exact future extension point.
PROVENANCE_BUNDLE_REGISTRY: dict[str, BundleRegistration] = {
    **BUNDLE_REGISTRY,
    "F1_P": BundleRegistration(
        role_id="F1_P",
        root=PROSPECTIVE_BUNDLE_ROOT / "f1",
        geometry="observable_boolean_cells_after_a3",
    ),
}


@dataclass(frozen=True)
class ProspectiveBundle:
    """A validated, ID-aligned pair of physically separate arrays."""

    registration: BundleRegistration
    feature_manifest: Mapping[str, Any]
    target_manifest: Mapping[str, Any]
    features: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    feature_manifest_sha256: str
    target_manifest_sha256: str
    feature_array_sha256: str
    target_array_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_path_fragments(path: Path | str) -> tuple[str, ...]:
    """Return protected-split markers from untrusted lexical path text."""

    normalized = str(path).replace("\\", "/").lower()
    matches = [
        fragment for fragment in FORBIDDEN_PATH_FRAGMENTS
        if fragment in normalized
    ]
    if "protected" in normalized.split("/") and "protected" not in matches:
        matches.append("protected")
    return tuple(matches)


def _repository_path(repository: Path, path: Path) -> Path:
    """Validate one lexical worktree path before following any symlink.

    Protocol paths are untrusted until this boundary returns.  In particular,
    resolving first would erase evidence that a registered logical leaf or one
    of its worktree-relative ancestors is a symlink.  The order here is part of
    the fit-only security contract: lexical checks, complete symlink-chain
    checks, then canonical containment and protected-path checks.
    """

    lexical_repository = repository.absolute()
    raw_matches = _forbidden_path_fragments(path)
    if raw_matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: "
            + ", ".join(raw_matches)
        )
    if ".." in path.parts:
        raise RuntimeError("TMLR V6 path escapes the repository")
    lexical = path if path.is_absolute() else lexical_repository / path
    try:
        relative = lexical.relative_to(lexical_repository)
    except ValueError as error:
        raise RuntimeError("TMLR V6 path escapes the repository") from error

    relative_matches = _forbidden_path_fragments(relative)
    if relative_matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: "
            + ", ".join(relative_matches)
        )

    current = lexical_repository
    if current.is_symlink():
        raise RuntimeError("TMLR V6 repository root is symlinked")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(
                f"TMLR V6 repository path contains a symlink: {current}"
            )

    resolved_repository = lexical_repository.resolve()
    if resolved_repository != lexical_repository:
        raise RuntimeError(
            "TMLR V6 repository root or its ancestor is symlinked"
        )
    resolved = lexical.resolve(strict=False)
    try:
        resolved_relative = resolved.relative_to(resolved_repository)
    except ValueError as error:
        raise RuntimeError("TMLR V6 path escapes the repository") from error
    resolved_matches = _forbidden_path_fragments(resolved_relative)
    if resolved_matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: "
            + ", ".join(resolved_matches)
        )
    return resolved


def _logical_path(repository: Path, path: Path) -> str:
    return _repository_path(repository, path).relative_to(repository.resolve()).as_posix()


def _verify_file(path: Path, identity: Mapping[str, Any], *, description: str) -> None:
    if (
        path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
        or path.stat().st_size != int(identity.get("bytes", -1))
        or sha256_file(path) != str(identity.get("sha256"))
    ):
        raise RuntimeError(f"TMLR V6 {description} identity drift")


def _exact_file_identity(
    value: Any, *, expected_path: Path, description: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"TMLR V6 {description} identity schema drift")
    identity = {
        "path": str(value["path"]),
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
    }
    if (
        identity["path"] != expected_path.as_posix()
        or identity["bytes"] <= 0
        or _SHA256.fullmatch(identity["sha256"]) is None
    ):
        raise RuntimeError(f"TMLR V6 {description} identity is invalid")
    return identity


def _exact_builder_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "commit", "sha256"}:
        raise RuntimeError("TMLR V6 committed builder identity schema drift")
    result = {
        "path": str(value["path"]),
        "commit": str(value["commit"]),
        "sha256": str(value["sha256"]),
    }
    if (
        result["path"] != PROSPECTIVE_BUNDLE_BUILDER_PATH.as_posix()
        or _COMMIT.fullmatch(result["commit"]) is None
        or _SHA256.fullmatch(result["sha256"]) is None
    ):
        raise RuntimeError("TMLR V6 committed builder identity is invalid")
    return result


def _exact_builder_implementation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "commit", "source_sha256", "combined_sha256",
    }:
        raise RuntimeError("TMLR V6 builder implementation identity schema drift")
    commit = str(value["commit"])
    sources = value["source_sha256"]
    expected_paths = set(PROSPECTIVE_BUILDER_IMPLEMENTATION_PATHS)
    if (
        _COMMIT.fullmatch(commit) is None
        or not isinstance(sources, Mapping)
        or set(map(str, sources)) != expected_paths
    ):
        raise RuntimeError("TMLR V6 builder implementation source registry drift")
    normalized_sources = {str(path): str(sources[path]) for path in sorted(sources)}
    if any(_SHA256.fullmatch(digest) is None for digest in normalized_sources.values()):
        raise RuntimeError("TMLR V6 builder implementation source SHA is invalid")
    combined = str(value["combined_sha256"])
    if (
        _SHA256.fullmatch(combined) is None
        or hashlib.sha256(canonical_json_bytes(normalized_sources)).hexdigest()
        != combined
    ):
        raise RuntimeError("TMLR V6 builder implementation aggregate drift")
    return {
        "commit": commit,
        "source_sha256": normalized_sources,
        "combined_sha256": combined,
    }


def _exact_inventory_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "bytes", "sha256", "status",
    }:
        raise RuntimeError("TMLR V6 fit-only inventory identity schema drift")
    result = {
        "path": str(value["path"]),
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
        "status": str(value["status"]),
    }
    if (
        result["path"] != FIT_ONLY_INPUT_INVENTORY_PATH.as_posix()
        or result["bytes"] <= 0
        or _SHA256.fullmatch(result["sha256"]) is None
        or result["status"] != "TMLR_V6_FIT_ONLY_INPUTS_COMPLETE"
    ):
        raise RuntimeError("TMLR V6 fit-only inventory identity is invalid")
    return result


def _validated_bundle_identity_registry(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    value = protocol.get("prospective_bundle_identities")
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "status", "build_report", "committed_builder",
        "builder_implementation", "fit_only_input_inventory", "roles",
    }:
        raise RuntimeError("TMLR V6 prospective bundle identity registry is absent or incomplete")
    if (
        value.get("schema_version") != BUNDLE_IDENTITY_SCHEMA
        or value.get("status") != BUNDLE_IDENTITY_STATUS
    ):
        raise RuntimeError("TMLR V6 prospective bundle identity registry is not frozen")
    build_report = _exact_file_identity(
        value.get("build_report"),
        expected_path=PROSPECTIVE_BUNDLE_BUILD_REPORT_PATH,
        description="prospective bundle build report",
    )
    builder = _exact_builder_identity(value.get("committed_builder"))
    builder_implementation = _exact_builder_implementation(
        value.get("builder_implementation")
    )
    if builder["commit"] != builder_implementation["commit"]:
        raise RuntimeError("TMLR V6 builder script/implementation commit drift")
    if (
        builder["sha256"]
        != builder_implementation["source_sha256"][
            PROSPECTIVE_BUNDLE_BUILDER_PATH.as_posix()
        ]
    ):
        raise RuntimeError("TMLR V6 builder script/source SHA drift")
    inventory = _exact_inventory_identity(value.get("fit_only_input_inventory"))
    roles_value = value.get("roles")
    if not isinstance(roles_value, Mapping):
        raise RuntimeError("TMLR V6 prospective bundle role registry is invalid")
    observed_roles = set(map(str, roles_value))
    required_roles = set(PROVENANCE_BUNDLE_REGISTRY)
    if observed_roles != required_roles:
        raise RuntimeError("TMLR V6 prospective bundle role registry drift")
    roles: dict[str, Any] = {}
    for role_id in sorted(observed_roles):
        registration = PROVENANCE_BUNDLE_REGISTRY[role_id]
        role = roles_value[role_id]
        if not isinstance(role, Mapping) or set(role) != {
            "root", "geometry", "partitions",
        }:
            raise RuntimeError(f"TMLR V6 {role_id} provenance schema drift")
        if (
            str(role.get("root")) != registration.root.as_posix()
            or str(role.get("geometry")) != registration.geometry
        ):
            raise RuntimeError(f"TMLR V6 {role_id} registered geometry/path drift")
        partitions = role.get("partitions")
        if not isinstance(partitions, Mapping) or set(partitions) != {
            "prospective_features", "targets",
        }:
            raise RuntimeError(f"TMLR V6 {role_id} partition registry drift")
        normalized_parts = {}
        for partition, manifest_path, array_path in (
            (
                "prospective_features", registration.feature_manifest,
                registration.feature_array,
            ),
            ("targets", registration.target_manifest, registration.target_array),
        ):
            part = partitions[partition]
            if not isinstance(part, Mapping) or set(part) != {"manifest", "array"}:
                raise RuntimeError(f"TMLR V6 {role_id}:{partition} identity schema drift")
            normalized_parts[partition] = {
                "manifest": _exact_file_identity(
                    part.get("manifest"), expected_path=manifest_path,
                    description=f"{role_id}:{partition} manifest",
                ),
                "array": _exact_file_identity(
                    part.get("array"), expected_path=array_path,
                    description=f"{role_id}:{partition} array",
                ),
            }
        roles[role_id] = {
            "root": registration.root.as_posix(),
            "geometry": registration.geometry,
            "partitions": normalized_parts,
        }
    return {
        "schema_version": BUNDLE_IDENTITY_SCHEMA,
        "status": BUNDLE_IDENTITY_STATUS,
        "build_report": build_report,
        "committed_builder": builder,
        "builder_implementation": builder_implementation,
        "fit_only_input_inventory": inventory,
        "roles": roles,
    }


def _verify_committed_builder(
    repository: Path, identity: Mapping[str, Any],
) -> None:
    """Bind the builder to both its working bytes and its registered commit."""

    repository = repository.resolve()
    normalized = _exact_builder_identity(identity)
    path = _repository_path(repository, Path(normalized["path"]))
    if (
        path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
        or sha256_file(path) != normalized["sha256"]
    ):
        raise RuntimeError("TMLR V6 committed builder working identity drift")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", normalized["path"]],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 committed builder is not clean")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", normalized["commit"], "HEAD"],
        cwd=repository, capture_output=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("TMLR V6 committed builder commit is not an ancestor")
    committed = subprocess.run(
        ["git", "show", f"{normalized['commit']}:{normalized['path']}"],
        cwd=repository, check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != normalized["sha256"]:
        raise RuntimeError("TMLR V6 committed builder Git identity drift")


def _verify_builder_implementation(
    repository: Path, identity: Mapping[str, Any],
) -> None:
    repository = repository.resolve()
    normalized = _exact_builder_implementation(identity)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", normalized["commit"], "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("TMLR V6 builder implementation commit is not an ancestor")
    paths = tuple(sorted(normalized["source_sha256"]))
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 builder implementation dependencies are not clean")
    for relative in paths:
        expected = normalized["source_sha256"][relative]
        actual = _repository_path(repository, Path(relative))
        if (
            actual.is_symlink()
            or not actual.is_file()
            or sha256_file(actual) != expected
        ):
            raise RuntimeError(
                f"TMLR V6 builder implementation dependency drift: {relative}"
            )
        committed = subprocess.run(
            ["git", "show", f"{normalized['commit']}:{relative}"],
            cwd=repository, check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(
                f"TMLR V6 builder committed dependency drift: {relative}"
            )


def _verify_fit_only_input_inventory(
    repository: Path, identity: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _exact_inventory_identity(identity)
    path = _repository_path(repository, Path(normalized["path"]))
    _verify_file(path, normalized, description="fit-only input inventory")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("TMLR V6 fit-only input inventory is not an object")
    counts = payload.get("counts", {})
    checks = payload.get("checks", {})
    materializer = payload.get("materializer")
    if (
        payload.get("schema_version")
        != "rail3.tmlr-v6-fit-only-input-inventory.v1"
        or payload.get("status") != normalized["status"]
        or not isinstance(counts, Mapping)
        or int(counts.get("files", -1)) != 220_989
        or int(counts.get("fit_image_groups", -1)) != 1_364
        or int(counts.get("symlinks", -1)) != 0
        or int(counts.get("forbidden_path_token_matches", -1)) != 0
        or not isinstance(checks, Mapping)
        or not checks
        or not all(bool(value) for value in checks.values())
        or not isinstance(materializer, Mapping)
        or set(materializer) != {"path", "bytes", "sha256", "commit"}
        or materializer.get("path") != FIT_ONLY_MATERIALIZER_PATH.as_posix()
        or int(materializer.get("bytes", -1)) <= 0
        or _SHA256.fullmatch(str(materializer.get("sha256", ""))) is None
        or _COMMIT.fullmatch(str(materializer.get("commit", ""))) is None
        or str(materializer.get("commit")) != str(payload.get("destination_git_head"))
    ):
        raise RuntimeError("TMLR V6 fit-only input inventory boundary drift")
    assert_zero_external_access(payload.get("heldout_access", {}))
    materializer_path = _repository_path(
        repository, Path(str(materializer["path"])),
    )
    _verify_file(
        materializer_path, materializer,
        description="committed fit-only materializer",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(materializer["path"])],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 fit-only materializer is not clean")
    commit = str(materializer["commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("TMLR V6 fit-only materializer commit is not an ancestor")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{materializer['path']}"],
        cwd=repository, check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != str(materializer["sha256"]):
        raise RuntimeError("TMLR V6 fit-only materializer Git identity drift")
    return dict(payload)


def _validated_consumed_inputs(value: Any, *, stage: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "stage", "files", "bytes", "combined_sha256",
        "exact_expected_set_consumed",
    }:
        raise RuntimeError("TMLR V6 consumed-input identity schema drift")
    result = {
        "stage": str(value["stage"]),
        "files": int(value["files"]),
        "bytes": int(value["bytes"]),
        "combined_sha256": str(value["combined_sha256"]),
        "exact_expected_set_consumed": value["exact_expected_set_consumed"],
    }
    if (
        result["stage"] != stage or result["files"] <= 0
        or result["bytes"] <= 0
        or _SHA256.fullmatch(result["combined_sha256"]) is None
        or result["exact_expected_set_consumed"] is not True
    ):
        raise RuntimeError("TMLR V6 consumed-input identity drift")
    return result


def _verify_bundle_build_report(
    repository: Path, registry: Mapping[str, Any],
) -> dict[str, Any]:
    identity = registry["build_report"]
    path = _repository_path(repository, Path(str(identity["path"])))
    _verify_file(path, identity, description="prospective bundle build report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping) or set(report) != {
        "schema_version", "status", "stages", "roles", "fit_labels_opened",
        "committed_builder", "builder_implementation",
        "fit_only_input_inventory", "sam_inference_used",
        "prompt_or_trajectory_generation_used", "heldout_access",
        "git_head", "wall_seconds",
        "consumed_fit_only_inputs",
    }:
        raise RuntimeError("TMLR V6 prospective bundle build report schema drift")
    if (
        report.get("schema_version")
        != "rail3.tmlr-v6.prospective-bundle-build.v1"
        or report.get("status") != "TMLR_V6_PROSPECTIVE_BUNDLES_COMPLETE"
        or report.get("stages") != ["features", "targets"]
        or int(report.get("fit_labels_opened", -1)) != 1_364
        or report.get("committed_builder") != registry["committed_builder"]
        or report.get("builder_implementation")
        != registry["builder_implementation"]
        or report.get("fit_only_input_inventory")
        != registry["fit_only_input_inventory"]
        or report.get("sam_inference_used") is not False
        or report.get("prompt_or_trajectory_generation_used") is not False
        or report.get("git_head") != registry["builder_implementation"]["commit"]
        or not math.isfinite(float(report.get("wall_seconds", math.nan)))
        or float(report.get("wall_seconds", -1.0)) <= 0.0
    ):
        raise RuntimeError("TMLR V6 prospective bundle build report drift")
    assert_zero_external_access(report.get("heldout_access", {}))
    consumed = report.get("consumed_fit_only_inputs")
    if not isinstance(consumed, Mapping) or set(consumed) != {"features", "targets"}:
        raise RuntimeError("TMLR V6 build-report consumed-input registry drift")
    for stage in ("features", "targets"):
        _validated_consumed_inputs(consumed[stage], stage=stage)
    observed = report.get("roles")
    if not isinstance(observed, Mapping):
        raise RuntimeError("TMLR V6 prospective build-report role registry drift")
    expected_keys = {
        f"{role_id}:{partition}"
        for role_id in PROVENANCE_BUNDLE_REGISTRY
        for partition in ("prospective_features", "targets")
    }
    if set(map(str, observed)) != expected_keys:
        raise RuntimeError("TMLR V6 prospective build-report role set drift")
    role_counts: dict[str, tuple[int, int]] = {}
    for role_id, role in registry["roles"].items():
        for partition in ("prospective_features", "targets"):
            key = f"{role_id}:{partition}"
            item = observed[key]
            if not isinstance(item, Mapping) or set(item) != {
                "physical_partition", "geometry", "states", "atoms", "array",
            }:
                raise RuntimeError(f"TMLR V6 build-report entry schema drift: {key}")
            states = int(item.get("states", -1))
            atoms = int(item.get("atoms", -1))
            if (
                item.get("physical_partition") != partition
                or item.get("geometry") != role["geometry"]
                or states != 27_280
                or atoms <= 0
                or item.get("array") != role["partitions"][partition]["array"]
            ):
                raise RuntimeError(f"TMLR V6 build-report entry drift: {key}")
            prior = role_counts.setdefault(role_id, (states, atoms))
            if prior != (states, atoms):
                raise RuntimeError(f"TMLR V6 feature/target count drift: {role_id}")
    return dict(report)


def _verify_registered_bundle_provenance(
    repository: Path, registry: Mapping[str, Any],
) -> None:
    _verify_builder_implementation(repository, registry["builder_implementation"])
    _verify_committed_builder(repository, registry["committed_builder"])
    _verify_fit_only_input_inventory(
        repository, registry["fit_only_input_inventory"],
    )
    _verify_bundle_build_report(repository, registry)


def verify_runtime_feature_audit(
    repository: Path, protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a source-derived runtime audit without trusting its own claims."""

    repository = repository.resolve()
    path = _repository_path(repository, RUNTIME_FEATURE_AUDIT_PATH)
    if (
        path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
    ):
        raise RuntimeError("TMLR V6 runtime feature audit is absent or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("TMLR V6 runtime feature audit is not an object")
    expected_checks = {
        "historical_runtime_indices_exact": True,
        "historical_training_heldout_path_traced": True,
        "all_requested_historical_models_accounted": True,
        "prospective_action_dimension_16": True,
        "prospective_plan_exact_type_guard_present": True,
        "prospective_cost_api_physically_separate": True,
        "prospective_full_raster_guard_present": True,
        "protected_or_raw_label_read": False,
        "training_or_evaluation_run": False,
    }
    expected_scope = {
        "historical_models": [
            "R0", "R1", "R2", "R3", "Q2", "F0", "F1", "LL4TTA",
            "L2D", "SPO_PLUS", "RECTANGLE", "CANDIDATE_UNION",
        ],
        "read_source_code_only": True,
        "data_or_manifest_read": False,
        "training_run": False,
    }
    prospective = payload.get("prospective_feature_schema", {})
    if (
        payload.get("schema_version")
        != "rail3.tmlr-v6.runtime-feature-audit.v1"
        or payload.get("status") != RUNTIME_FEATURE_AUDIT_STATUS
        or payload.get("registered_starting_head")
        != protocol.get("starting_git_commit")
        or payload.get("scope") != expected_scope
        or payload.get("checks") != expected_checks
        or not isinstance(prospective, Mapping)
        or prospective.get("id") != PROSPECTIVE_SCHEMA_ID
        or int(prospective.get("action_dimension", -1)) != ACTION_DIM
        or prospective.get("predictor_cost_input") is not False
        or prospective.get("post_action_mask_input") is not False
        or prospective.get("future_action_outcome_input") is not False
    ):
        raise RuntimeError("TMLR V6 runtime feature audit contract drift")
    assert_zero_external_access(payload.get("protected_access", {}))
    audited_head = str(payload.get("audited_repository_head", ""))
    if _COMMIT.fullmatch(audited_head) is None:
        raise RuntimeError("TMLR V6 runtime feature audit commit is malformed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited_head, "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("TMLR V6 runtime feature audit commit is not an ancestor")
    sources = payload.get("source_files")
    expected_paths = {item.as_posix() for item in RUNTIME_FEATURE_AUDIT_SOURCE_PATHS}
    if not isinstance(sources, Mapping) or set(map(str, sources)) != expected_paths:
        raise RuntimeError("TMLR V6 runtime feature audit source registry drift")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted(expected_paths)],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 runtime audit source files are not clean")
    for relative in sorted(expected_paths):
        identity = sources[relative]
        if not isinstance(identity, Mapping) or set(identity) != {"bytes", "sha256"}:
            raise RuntimeError("TMLR V6 runtime audit source identity schema drift")
        actual = _repository_path(repository, Path(relative))
        _verify_file(actual, identity, description=f"runtime audit source {relative}")
        committed = subprocess.run(
            ["git", "show", f"{audited_head}:{relative}"],
            cwd=repository, check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != str(identity["sha256"]):
            raise RuntimeError("TMLR V6 runtime audit committed-source drift")
    return {
        "path": RUNTIME_FEATURE_AUDIT_PATH.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assert_zero_external_access(access: Mapping[str, Any]) -> None:
    if set(access) != set(EXTERNAL_SPLITS):
        raise RuntimeError("TMLR V6 external-access schema drift")
    if any(
        type(access[name]) is not int or access[name] != 0
        for name in EXTERNAL_SPLITS
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")


def assert_legacy_zero_external_access(access: Mapping[str, Any]) -> None:
    """Validate immutable RETRO artifacts without weakening new V6 outputs."""

    if set(access) != set(LEGACY_EXTERNAL_SPLITS):
        raise RuntimeError("TMLR V6 legacy external-access schema drift")
    if any(
        type(access[name]) is not int or access[name] != 0
        for name in LEGACY_EXTERNAL_SPLITS
    ):
        raise RuntimeError("STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS")


def assert_no_legacy_representation_path(path: Path) -> None:
    """Reject the retrospective G1/G2 bundle family by path, fail closed."""

    normalized = path.as_posix().lower().replace("_", "-")
    forbidden = (
        "diagnostic-core/representations",
        "bundle-g1",
        "bundle-g2",
        "/g1/",
        "/g2/",
    )
    if any(token in normalized for token in forbidden):
        raise RuntimeError("legacy G1/G2 representation bundles are not prospective")


def _model_registry(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    first_batch = protocol.get("models", {}).get("first_batch", {})
    if tuple(first_batch) != FIRST_BATCH_MODEL_IDS:
        raise RuntimeError("TMLR V6 first-batch model registry drift")
    expected = {
        "R0_SMALL_P": ("F0_P", "global_small", 7809),
        "R0_CM_P": ("F0_P", "global_capacity_matched", 103369),
        "R1_P": ("F0_P", "shared_atomic", 103521),
        "RECT_P": ("RECT_FULL_RASTER_P", "shared_atomic", 103521),
        "UNION_P": ("UNION_FULL_RASTER_P", "shared_atomic", 103521),
    }
    for model_id, (bundle, kind, parameters) in expected.items():
        spec = first_batch.get(model_id, {})
        if (
            spec.get("bundle") != bundle
            or spec.get("kind") != kind
            or int(spec.get("parameters", -1)) != parameters
        ):
            raise RuntimeError(f"TMLR V6 model contract drift: {model_id}")
    return first_batch


def _validate_protocol_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen protocol, including its external bundle anchors."""

    if (
        protocol.get("schema_version")
        != "rail3.tmlr-v6-p0-prospective-capacity.v1"
        or protocol.get("protocol_id") != "tmlr-v6-p0-prospective-capacity"
        or protocol.get("status")
        != "TMLR_V6_P0_PROTOCOL_FROZEN_BEFORE_TRAINING"
    ):
        raise RuntimeError("TMLR V6 P0 protocol is not frozen")
    validate_schema_contract(protocol)
    _model_registry(protocol)
    training = protocol.get("training", {})
    expected_training = {
        "scale": "S1364",
        "outer_folds": list(FOLDS),
        "seeds": list(SEEDS),
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "max_epochs": 120,
        "patience": 15,
        "minimum_inner_stop_improvement": 1e-8,
        "same_target": True,
        "primary_training_loss": (
            "state-level Huber(delta=0.05) over the same defined state-action residual targets"
        ),
        "inner_stopping_loss": (
            "the same state-level Huber loss on the seed-specific inner-stop groups"
        ),
        "atom_target_auxiliary_loss": False,
        "same_action_set": list(ACTIONS),
        "first_batch_job_count": 75,
        "gpu_assignment": {
            device: list(folds) for device, folds in GPU_ASSIGNMENT.items()
        },
        "ddp": False,
        "concurrent_jobs_per_gpu_max": 1,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    }
    drift = {
        key: (training.get(key), value)
        for key, value in expected_training.items()
        if training.get(key) != value
    }
    if drift:
        raise RuntimeError(f"TMLR V6 first-batch training contract drift: {drift}")
    representation = protocol.get("prospective_representation_build", {})
    expected_representation = {
        "base_information": (
            "nonempty individual candidates from observed A0/A1/A2 only"
        ),
        "rectangle_domain": "full image raster",
        "union_component_domain": "full image raster",
        "ground_truth_void_affects_feature_geometry": False,
        "ground_truth_void_used_only_for_target_validity": True,
        "inference_aggregation": (
            "sum atom predictions using label-free full-raster region-area fractions"
        ),
        "target_reconstruction": (
            "sum atom targets using target-only ground-truth-valid pixel fractions"
        ),
        "G5_future_action_atoms_forbidden": True,
    }
    representation_drift = {
        key: (representation.get(key), value)
        for key, value in expected_representation.items()
        if representation.get(key) != value
    }
    if representation_drift:
        raise RuntimeError(
            f"TMLR V6 representation/weight contract drift: {representation_drift}"
        )
    capacity = protocol.get("capacity_match", {})
    domain = capacity.get("search_domain", [])
    if len(domain) != 2:
        raise RuntimeError("TMLR V6 capacity search domain drift")
    solution = solve_capacity_matched_width(
        input_dim=int(capacity.get("global_input_dimension", -1)),
        target_parameter_count=int(capacity.get("target_parameter_count", -1)),
        search_min=int(domain[0]),
        search_max=int(domain[1]),
    )
    if (
        solution.width != int(capacity.get("selected_width", -1))
        or solution.parameter_count
        != int(capacity.get("selected_parameter_count", -1))
        or abs(solution.difference_percent)
        > float(capacity.get("allowed_absolute_percent", -1.0))
    ):
        raise RuntimeError("TMLR V6 deterministic capacity solution drift")
    _validated_bundle_identity_registry(protocol)
    return dict(protocol)


def load_protocol(repository: Path) -> dict[str, Any]:
    path = _repository_path(repository, PROTOCOL_PATH)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, Mapping):
        raise RuntimeError("TMLR V6 P0 protocol is not a JSON object")
    return _validate_protocol_payload(protocol)


def validate_fold_manifest(
    repository: Path, protocol: Mapping[str, Any], path: Path = FOLD_MANIFEST_PATH,
) -> dict[str, Any]:
    registered = protocol["frozen_fit_inputs"]["folds"]["S1364"]
    actual_path = _repository_path(repository, path)
    expected_path = _repository_path(repository, Path(str(registered["path"])))
    if actual_path != expected_path:
        raise RuntimeError("TMLR V6 fold path is not registered")
    if sha256_file(actual_path) != str(registered["sha256"]):
        raise RuntimeError("TMLR V6 frozen S1364 fold identity drift")
    manifest = json.loads(actual_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups", [])
    group_ids = [str(item.get("image_group_id")) for item in groups]
    fold_ids = {str(item.get("fold_id")) for item in groups}
    if (
        manifest.get("scale") != "S_MAX"
        or int(manifest.get("group_count", -1)) != 1364
        or int(manifest.get("fold_count", -1)) != 5
        or len(group_ids) != 1364
        or len(set(group_ids)) != 1364
        or fold_ids != set(FOLDS)
        or not all(bool(value) for value in manifest.get("checks", {}).values())
    ):
        raise RuntimeError("TMLR V6 frozen S1364 fold contract drift")
    return manifest


def _validate_part_manifest(
    *, repository: Path, registration: BundleRegistration,
    manifest_path: Path, array_path: Path, partition: str,
    bundle_identities: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    assert_no_legacy_representation_path(manifest_path)
    assert_no_legacy_representation_path(array_path)
    actual_manifest = _repository_path(repository, manifest_path)
    actual_array = _repository_path(repository, array_path)
    try:
        frozen_part = bundle_identities["roles"][registration.role_id][
            "partitions"
        ][partition]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"TMLR V6 {registration.role_id}:{partition} is not protocol-registered"
        ) from error
    registered_manifest = frozen_part["manifest"]
    registered_array = frozen_part["array"]
    if (
        _repository_path(repository, Path(str(registered_manifest["path"])))
        != actual_manifest
        or _repository_path(repository, Path(str(registered_array["path"])))
        != actual_array
    ):
        raise RuntimeError("TMLR V6 registered prospective part path drift")
    # The protocol-authenticated manifest must be verified before its content
    # is parsed.  Its array declaration is never accepted as its own trust root.
    _verify_file(
        actual_manifest, registered_manifest,
        description=f"{registration.role_id}:{partition} manifest",
    )
    manifest = json.loads(actual_manifest.read_text(encoding="utf-8"))
    if partition == "prospective_features":
        expected_schema = FEATURE_MANIFEST_SCHEMA
        expected_status = FEATURE_MANIFEST_STATUS
        expected_names = FEATURE_ARRAY_NAMES
    elif partition == "targets":
        expected_schema = TARGET_MANIFEST_SCHEMA
        expected_status = TARGET_MANIFEST_STATUS
        expected_names = TARGET_ARRAY_NAMES
    else:
        raise ValueError("unregistered physical partition")
    expected = {
        "schema_version": expected_schema,
        "status": expected_status,
        "physical_partition": partition,
        "role_id": registration.role_id,
        "geometry": registration.geometry,
        "prospective_schema_id": PROSPECTIVE_SCHEMA_ID,
        "action_order": list(ACTIONS),
        "array_names": sorted(expected_names),
    }
    drift = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if drift:
        raise RuntimeError(
            f"TMLR V6 {registration.role_id} {partition} manifest drift: {drift}"
        )
    assert_zero_external_access(manifest.get("heldout_access", {}))
    _validated_consumed_inputs(
        manifest.get("consumed_fit_only_inputs"),
        stage="features" if partition == "prospective_features" else "targets",
    )
    if not manifest.get("checks") or not all(
        bool(value) for value in manifest["checks"].values()
    ):
        raise RuntimeError("TMLR V6 prospective bundle checks are incomplete")
    if partition == "prospective_features":
        feature_boundary = {
            "runtime_feature_count": 0,
            "future_mask_feature_count": 0,
            "ground_truth_feature_count": 0,
            "action_dimension": ACTION_DIM,
            "post_projection_only": True,
            "source_actions": (
                ["A0", "A1", "A2", "A3"]
                if registration.role_id == "F1_P"
                else ["A0", "A1", "A2"]
            ),
            "inference_weight_source": "feature_area_fraction_full_raster",
            "ground_truth_inference_input": False,
        }
        if any(manifest.get(key) != value for key, value in feature_boundary.items()):
            raise RuntimeError("TMLR V6 prospective feature boundary drift")
        if not bool(
            manifest["checks"].get(
                "legacy_joined_container_member_allowlist_enforced"
            )
        ):
            raise RuntimeError("TMLR V6 legacy container allowlist audit is absent")
    else:
        missingness = manifest.get("target_missingness")
        if (
            manifest.get("target_contract_id")
            != "M06E_ABSOLUTE_ADDITIVE_PIXEL_MISMATCH_V1"
            or manifest.get("target_definition")
            != "same frozen absolute residual target for all first-batch models"
            or manifest.get("target_weight_source")
            != "valid_pixel_fraction_target_only"
            or not isinstance(missingness, Mapping)
            or set(missingness) != {
                "plan_feasible_target_undefined_actions",
                "incomplete_target_states",
                "incomplete_target_state_ids_sha256",
            }
            or int(missingness.get(
                "plan_feasible_target_undefined_actions", -1,
            )) < 0
            or int(missingness.get("incomplete_target_states", -1)) < 0
            or _SHA256.fullmatch(str(missingness.get(
                "incomplete_target_state_ids_sha256", "",
            ))) is None
        ):
            raise RuntimeError("TMLR V6 first-batch target-definition drift")
    if (
        manifest.get("committed_builder")
        != bundle_identities["committed_builder"]
        or manifest.get("builder_implementation")
        != bundle_identities["builder_implementation"]
        or manifest.get("fit_only_input_inventory")
        != bundle_identities["fit_only_input_inventory"]
    ):
        raise RuntimeError("TMLR V6 prospective manifest provenance drift")
    array_identity = manifest.get("array", {})
    if array_identity != registered_array:
        raise RuntimeError("TMLR V6 prospective array identity is not protocol-frozen")
    if _repository_path(
        repository, Path(str(array_identity.get("path", "")))
    ) != actual_array:
        raise RuntimeError("TMLR V6 prospective array path drift")
    _verify_file(
        actual_array, registered_array,
        description=f"{registration.role_id}:{partition} array",
    )
    return (
        manifest,
        str(registered_manifest["sha256"]),
        str(registered_array["sha256"]),
    )


def _strings(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise RuntimeError(f"TMLR V6 {name} must be a one-dimensional string array")
    result = value.astype(str)
    if np.any(result == ""):
        raise RuntimeError(f"TMLR V6 {name} contains an empty identifier")
    return result


def _load_exact_npz(path: Path, expected: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise RuntimeError(
                f"TMLR V6 physical array schema drift: "
                f"expected={sorted(expected)}, observed={sorted(payload.files)}"
            )
        return {name: np.asarray(payload[name]) for name in sorted(expected)}


def _assert_target_defined_implies_plan_feasible(
    state_target: np.ndarray, feasible: np.ndarray,
) -> None:
    if np.any(
        np.isfinite(np.asarray(state_target)) & ~np.asarray(feasible, dtype=bool)
    ):
        raise RuntimeError(
            "TMLR V6 defined target must imply pre-action plan feasibility"
        )


def load_registered_bundle(
    repository: Path, role_id: str,
) -> ProspectiveBundle:
    if role_id not in BUNDLE_REGISTRY:
        raise ValueError("unregistered TMLR V6 prospective bundle role")
    registration = BUNDLE_REGISTRY[role_id]
    protocol = load_protocol(repository)
    bundle_identities = _validated_bundle_identity_registry(protocol)
    _verify_registered_bundle_provenance(repository, bundle_identities)
    feature_manifest, feature_manifest_sha, feature_array_sha = (
        _validate_part_manifest(
            repository=repository,
            registration=registration,
            manifest_path=registration.feature_manifest,
            array_path=registration.feature_array,
            partition="prospective_features",
            bundle_identities=bundle_identities,
        )
    )
    target_manifest, target_manifest_sha, target_array_sha = (
        _validate_part_manifest(
            repository=repository,
            registration=registration,
            manifest_path=registration.target_manifest,
            array_path=registration.target_array,
            partition="targets",
            bundle_identities=bundle_identities,
        )
    )
    features = _load_exact_npz(
        _repository_path(repository, registration.feature_array), FEATURE_ARRAY_NAMES,
    )
    targets = _load_exact_npz(
        _repository_path(repository, registration.target_array), TARGET_ARRAY_NAMES,
    )
    atom_x = np.asarray(features["atom_x"])
    state_x = np.asarray(features["state_x"])
    action_x = np.asarray(features["action_x16"])
    lengths = np.asarray(features["state_lengths"], dtype=np.int64)
    state_ids = _strings(features["state_ids"], name="feature state_ids")
    group_ids = _strings(
        features["image_group_ids"], name="feature image_group_ids",
    )
    atom_ids = _strings(features["atom_ids"], name="feature atom_ids")
    target_state_ids = _strings(targets["state_ids"], name="target state_ids")
    target_atom_ids = _strings(targets["atom_ids"], name="target atom_ids")
    n_states = len(state_ids)
    n_atoms = len(atom_ids)
    if (
        atom_x.shape != (n_atoms, ATOM_DIM)
        or state_x.shape != (n_states, STATE_DIM)
        or action_x.shape != (n_states, len(ACTIONS), ACTION_DIM)
        or atom_x.dtype != np.dtype(np.float32)
        or state_x.dtype != np.dtype(np.float32)
        or action_x.dtype != np.dtype(np.float32)
        or np.asarray(features["state_lengths"]).dtype.kind not in {"i", "u"}
        or lengths.shape != (n_states,)
        or group_ids.shape != (n_states,)
        or np.any(lengths <= 0)
        or int(lengths.sum()) != n_atoms
        or not all(np.isfinite(value).all() for value in (atom_x, state_x, action_x))
        or len(set(state_ids.tolist())) != n_states
        or len(set(atom_ids.tolist())) != n_atoms
    ):
        raise RuntimeError("TMLR V6 prospective feature geometry drift")
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
        raise RuntimeError("TMLR V6 prospective feature/target join drift")
    inference_weights_from_features(atom_x, lengths)
    target_state_weight = np.add.reduceat(
        atom_weights.astype(np.float64),
        np.r_[0, np.cumsum(lengths[:-1])],
    )
    if not np.allclose(target_state_weight, 1.0, rtol=0.0, atol=2e-7):
        raise RuntimeError("TMLR V6 target weights do not conserve valid area")
    repeated_feasible = np.repeat(feasible, lengths, axis=0)
    if (
        not np.array_equal(action_x[:, :, 0], action_x[:, :, 7])
        or not np.all(np.isin(action_x[:, :, 0], (0.0, 1.0)))
        or not np.array_equal(
            np.asarray(action_x[:, :, 0], dtype=bool), feasible,
        )
    ):
        raise RuntimeError("TMLR V6 target feasibility is not plan-derived")
    _assert_target_defined_implies_plan_feasible(state_target, feasible)
    if (
        np.any(np.isfinite(atom_target) & ~repeated_feasible)
        or not np.all(feasible[:, 0])
    ):
        raise RuntimeError("TMLR V6 target feasibility contract drift")
    reconstructed_target = reconstruct_target_states(
        atom_target=atom_target,
        target_weights=atom_weights,
        state_lengths=lengths,
    )
    defined_state = np.isfinite(state_target)
    missing = feasible & ~defined_state
    incomplete = np.any(missing, axis=1)
    missingness = {
        "plan_feasible_target_undefined_actions": int(missing.sum()),
        "incomplete_target_states": int(incomplete.sum()),
        "incomplete_target_state_ids_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(state_ids[incomplete].tolist()))
        ).hexdigest(),
    }
    if target_manifest.get("target_missingness") != missingness:
        raise RuntimeError("TMLR V6 target missingness manifest drift")
    if (
        not np.array_equal(np.isfinite(reconstructed_target), defined_state)
        or not np.allclose(
            reconstructed_target[defined_state],
            state_target.astype(np.float64)[defined_state],
            rtol=0.0,
            atol=2e-7,
        )
    ):
        raise RuntimeError("TMLR V6 target-only additive reconstruction drift")
    manifest_counts = {
        "states": n_states,
        "atoms": n_atoms,
        "state_actions": n_states * len(ACTIONS),
    }
    for manifest in (feature_manifest, target_manifest):
        if any(
            int(manifest.get("counts", {}).get(key, -1)) != value
            for key, value in manifest_counts.items()
        ):
            raise RuntimeError("TMLR V6 prospective manifest count drift")
    return ProspectiveBundle(
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


def _validate_bundle_population(
    bundle: ProspectiveBundle, fold_manifest: Mapping[str, Any],
) -> None:
    state_ids = _strings(bundle.features["state_ids"], name="state_ids")
    group_ids = _strings(bundle.features["image_group_ids"], name="image_group_ids")
    fold_groups = {str(item["image_group_id"]) for item in fold_manifest["groups"]}
    unique, counts = np.unique(group_ids, return_counts=True)
    if (
        len(state_ids) != 27_280
        or set(unique.tolist()) != fold_groups
        or not np.all(counts == 20)
    ):
        raise RuntimeError("TMLR V6 S1364 bundle population drift")


def inference_weights_from_features(
    atom_x: np.ndarray, state_lengths: np.ndarray,
) -> np.ndarray:
    """Derive label-free prediction weights from full-raster area fraction."""

    features = np.asarray(atom_x)
    lengths = np.asarray(state_lengths, dtype=np.int64)
    if (
        features.ndim != 2
        or features.shape[1] != ATOM_DIM
        or lengths.ndim != 1
        or np.any(lengths <= 0)
        or int(lengths.sum()) != len(features)
    ):
        raise ValueError("TMLR V6 inference-weight feature geometry drift")
    weights = features[:, 0].astype(np.float32, copy=True)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise RuntimeError("TMLR V6 label-free inference weights are invalid")
    state_weight = np.add.reduceat(
        weights.astype(np.float64),
        np.r_[0, np.cumsum(lengths[:-1])],
    )
    if not np.allclose(state_weight, 1.0, rtol=0.0, atol=2e-6):
        raise RuntimeError(
            "TMLR V6 label-free full-raster weights do not conserve one"
        )
    return weights


def reconstruct_target_states(
    *, atom_target: np.ndarray, target_weights: np.ndarray,
    state_lengths: np.ndarray,
) -> np.ndarray:
    """Audit target reconstruction without exposing weights to prediction."""

    target = np.asarray(atom_target, dtype=np.float64)
    weights = np.asarray(target_weights, dtype=np.float64)
    lengths = np.asarray(state_lengths, dtype=np.int64)
    if (
        target.ndim != 2
        or target.shape[1] != len(ACTIONS)
        or weights.shape != (len(target),)
        or lengths.ndim != 1
        or np.any(lengths <= 0)
        or int(lengths.sum()) != len(target)
    ):
        raise ValueError("TMLR V6 target reconstruction inputs are invalid")
    offsets = np.r_[0, np.cumsum(lengths)]
    result = np.full((len(lengths), len(ACTIONS)), np.nan, dtype=np.float64)
    for state_index in range(len(lengths)):
        start, stop = offsets[state_index], offsets[state_index + 1]
        local_target = target[start:stop]
        local_weight = weights[start:stop, None]
        defined_weight = np.sum(local_weight * np.isfinite(local_target), axis=0)
        defined = defined_weight > 0
        if not np.allclose(
            defined_weight[defined], 1.0, rtol=0.0, atol=2e-7,
        ):
            raise RuntimeError(
                "TMLR V6 partially defined target does not conserve valid area"
            )
        reconstructed = np.sum(
            local_weight * np.nan_to_num(local_target), axis=0,
        )
        result[state_index, defined] = reconstructed[defined]
    return result


def _inner_split(
    fold_manifest: Mapping[str, Any], outer_fold: str, seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Exact frozen image-group split, copied without semantic change."""

    if outer_fold not in FOLDS:
        raise ValueError("unknown TMLR V6 outer fold")
    heldout = {
        str(item["image_group_id"])
        for item in fold_manifest["groups"]
        if item["fold_id"] == outer_fold
    }
    candidates = sorted(
        str(item["image_group_id"])
        for item in fold_manifest["groups"]
        if item["fold_id"] != outer_fold
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
    if (
        train & stop
        or train & heldout
        or stop & heldout
        or len(train | stop | heldout) != len(fold_manifest["groups"])
    ):
        raise RuntimeError("TMLR V6 image-group split isolation failed")
    return train, stop, heldout


def build_registered_model(
    model_id: str, protocol: Mapping[str, Any],
) -> nn.Module:
    if model_id == "R0_SMALL_P":
        model: nn.Module = GlobalResidual(GLOBAL_DIM)
    elif model_id == "R0_CM_P":
        capacity = protocol["capacity_match"]
        domain = capacity["search_domain"]
        solution = solve_capacity_matched_width(
            input_dim=int(capacity["global_input_dimension"]),
            target_parameter_count=int(capacity["target_parameter_count"]),
            search_min=int(domain[0]),
            search_max=int(domain[1]),
        )
        model = CapacityMatchedGlobalResidual(GLOBAL_DIM, solution.width)
    elif model_id in {"R1_P", "RECT_P", "UNION_P"}:
        model = ProspectiveSharedAtomicResidual(ATOM_DIM, STATE_DIM, ACTION_DIM)
    else:
        raise ValueError("unregistered TMLR V6 first-batch model")
    expected = int(_model_registry(protocol)[model_id]["parameters"])
    if parameter_count(model) != expected:
        raise RuntimeError(f"TMLR V6 parameter-count drift: {model_id}")
    return model


def memory_bounded_shared_atomic_forward(
    model: SharedAtomicResidual,
    atom_features: torch.Tensor,
    state_features: torch.Tensor,
    action_features: torch.Tensor,
    state_lengths: torch.Tensor,
    inference_weights: torch.Tensor,
    *,
    chunk_size: int = MEMORY_BOUNDED_ATOM_CHUNK,
    checkpoint_activations: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure execution-only port of the frozen shared-atomic graph.

    Chunking and activation checkpointing change neither modules, parameters,
    targets, losses, optimizer, nor additive aggregation.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    atom = model.atom_encoder(atom_features)
    mean = torch.segment_reduce(atom, "mean", lengths=state_lengths)
    maximum = torch.segment_reduce(atom, "max", lengths=state_lengths)
    context = model.state_context(torch.cat((mean, maximum, state_features), dim=1))
    action_ids = torch.arange(5, device=action_features.device).expand(
        action_features.shape[0], 5,
    )
    action = model.action_encoder(torch.cat((
        model.action_id_embedding(action_ids), action_features,
    ), dim=2))
    atom_states = torch.repeat_interleave(
        torch.arange(state_features.shape[0], device=atom.device), state_lengths,
    )

    def predict_chunk(
        atom_chunk: torch.Tensor,
        context_chunk: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat((
            atom_chunk[:, None, :].expand(-1, 5, -1),
            context_chunk[:, None, :].expand(-1, 5, -1),
            action_chunk,
        ), dim=2)
        return model.residual_predictor(fused).squeeze(-1)

    chunks: list[torch.Tensor] = []
    use_checkpoint = checkpoint_activations and torch.is_grad_enabled()
    for start in range(0, atom.shape[0], chunk_size):
        stop = min(start + chunk_size, atom.shape[0])
        state_index = atom_states[start:stop]
        arguments = (
            atom[start:stop],
            context[state_index],
            action[state_index],
        )
        if use_checkpoint:
            prediction = checkpoint(
                predict_chunk,
                *arguments,
                use_reentrant=True,
                preserve_rng_state=True,
            )
        else:
            prediction = predict_chunk(*arguments)
        chunks.append(prediction)
    atomic = torch.cat(chunks, dim=0)
    state = torch.segment_reduce(
        atomic * inference_weights[:, None], "sum", lengths=state_lengths,
    )
    return atomic, state


def _state_action_mean(
    loss: torch.Tensor, defined: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    use = defined & state_mask[:, None]
    per_state = (loss * use).sum(dim=1) / use.sum(dim=1).clamp_min(1)
    selected = state_mask & use.any(dim=1)
    if not selected.any():
        raise ValueError("TMLR V6 loss selection contains no supervised states")
    return per_state[selected].mean()


def _state_huber(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    defined = torch.isfinite(target)
    raw = F.huber_loss(
        prediction, torch.nan_to_num(target), reduction="none", delta=0.05,
    )
    return _state_action_mean(raw, defined, mask)


def _atomic_state_losses(
    atom_prediction: torch.Tensor,
    state_prediction: torch.Tensor,
    atom_target: torch.Tensor,
    state_target: torch.Tensor,
    atom_weights: torch.Tensor,
    state_lengths: torch.Tensor,
    state_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        atom_prediction.shape != atom_target.shape
        or state_prediction.shape != state_target.shape
    ):
        raise ValueError("TMLR V6 residual prediction/target tensors are not aligned")
    atom_defined = torch.isfinite(atom_target)
    raw_atom = F.huber_loss(
        atom_prediction, torch.nan_to_num(atom_target), reduction="none", delta=0.10,
    )
    weighted = raw_atom * atom_weights[:, None] * atom_defined
    atom_sa = torch.segment_reduce(weighted, "sum", lengths=state_lengths)
    valid_weight = torch.segment_reduce(
        atom_weights[:, None].expand(-1, 5) * atom_defined,
        "sum",
        lengths=state_lengths,
    )
    atom_sa = atom_sa / valid_weight.clamp_min(1e-12)
    state_defined = torch.isfinite(state_target)
    atom_loss = _state_action_mean(
        atom_sa, state_defined & (valid_weight > 0), state_mask,
    )
    raw_state = F.huber_loss(
        state_prediction,
        torch.nan_to_num(state_target),
        reduction="none",
        delta=0.05,
    )
    state_loss = _state_action_mean(raw_state, state_defined, state_mask)
    return atom_loss, state_loss


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cuda_determinism_environment_contract() -> dict[str, str]:
    """Return the exact process environment required by deterministic CUDA."""

    return {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG}


def assert_cuda_determinism_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail before CUDA work unless cuBLAS determinism is explicitly enabled."""

    environment = os.environ if environ is None else environ
    observed = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "TMLR V6 deterministic CUDA environment requires "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}; "
            f"observed={observed!r}"
        )


def _registered_gpu_for_fold(outer_fold: str) -> str:
    assigned = [
        device for device, folds in GPU_ASSIGNMENT.items()
        if outer_fold in folds
    ]
    if len(assigned) != 1:
        raise RuntimeError("TMLR V6 frozen fold-to-GPU assignment is ambiguous")
    return assigned[0]


def _validated_visible_device(
    protocol: Mapping[str, Any], outer_fold: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Bind one registered fold to its single frozen physical GPU."""

    expected_assignment = {
        device: list(folds) for device, folds in GPU_ASSIGNMENT.items()
    }
    observed_assignment = protocol.get("training", {}).get("gpu_assignment")
    if observed_assignment != expected_assignment:
        raise RuntimeError("TMLR V6 physical-GPU assignment contract drift")
    environment = os.environ if environ is None else environ
    visible_device = environment.get("CUDA_VISIBLE_DEVICES")
    if visible_device != _registered_gpu_for_fold(outer_fold):
        raise RuntimeError(
            "TMLR V6 worker violates the frozen physical-GPU assignment"
        )
    return str(visible_device)


_GPU_LEASE_GUARD = object()


@dataclass(frozen=True)
class _GpuLease:
    visible_device: str
    descriptor: int
    lock_path: Path
    kernel_socket: socket.socket
    kernel_address: bytes
    guard: object


def _gpu_kernel_lease_address(
    repository: Path, visible_device: str,
) -> bytes:
    """Return a Linux abstract-socket name bound to one repo and GPU."""

    try:
        repository_identity = os.stat(repository, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("TMLR V6 repository identity is unavailable") from error
    digest = hashlib.sha256(
        (
            f"{repository}\0{repository_identity.st_dev}:"
            f"{repository_identity.st_ino}"
        ).encode("utf-8")
    ).hexdigest()[:48]
    # A leading NUL selects Linux's kernel-owned abstract namespace.  Unlike
    # a filesystem socket path, this name cannot be renamed or unlinked while
    # the owning socket remains open.
    return (
        b"\0rail3.tmlr-v6."
        + digest.encode("ascii")
        + b".gpu."
        + visible_device.encode("ascii")
    )


def _acquire_gpu_kernel_lease(
    repository: Path, visible_device: str,
) -> tuple[socket.socket, bytes]:
    """Acquire the non-replaceable kernel anchor for one physical GPU."""

    address = _gpu_kernel_lease_address(repository, visible_device)
    try:
        kernel_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    except OSError as error:
        raise RuntimeError("TMLR V6 kernel GPU lease is unavailable") from error
    try:
        kernel_socket.bind(address)
    except OSError as error:
        kernel_socket.close()
        if error.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"TMLR V6 physical GPU {visible_device} already has a worker"
            ) from error
        raise RuntimeError("TMLR V6 kernel GPU lease acquisition failed") from error
    return kernel_socket, address


def _validate_gpu_lock_descriptor(descriptor: int, lock_path: Path) -> None:
    """Reject aliases and special files before modifying a durable lock."""

    try:
        opened = os.fstat(descriptor)
        named = os.stat(lock_path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("TMLR V6 GPU lock identity is unavailable") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise RuntimeError(
            "TMLR V6 GPU lock must be one regular single-link file"
        )


def _assert_active_gpu_lease(lease: _GpuLease, visible_device: str) -> None:
    if (
        not isinstance(lease, _GpuLease)
        or lease.guard is not _GPU_LEASE_GUARD
        or lease.visible_device != visible_device
    ):
        raise RuntimeError("TMLR V6 training requires an active GPU lease")
    try:
        if lease.kernel_socket.fileno() < 0:
            raise OSError("closed kernel lease socket")
        bound_address = lease.kernel_socket.getsockname()
    except OSError as error:
        raise RuntimeError("TMLR V6 GPU lease is no longer active") from error
    if bound_address != lease.kernel_address:
        raise RuntimeError("TMLR V6 kernel GPU lease identity drift")
    try:
        _validate_gpu_lock_descriptor(lease.descriptor, lease.lock_path)
    except RuntimeError as error:
        raise RuntimeError("TMLR V6 GPU lease identity drift") from error


@contextmanager
def _exclusive_gpu_worker_lock(
    repository: Path, visible_device: str,
) -> Iterator[_GpuLease]:
    """Hold the only training lease for one frozen physical GPU."""

    if visible_device not in GPU_ASSIGNMENT:
        raise RuntimeError("TMLR V6 physical GPU identifier is invalid")
    repository = _repository_path(repository.absolute(), Path("."))
    kernel_socket, kernel_address = _acquire_gpu_kernel_lease(
        repository, visible_device,
    )
    try:
        lock_root = _repository_path(repository, GPU_LOCK_ROOT)
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_root = _repository_path(repository, GPU_LOCK_ROOT)
        if not lock_root.is_dir():
            raise RuntimeError("TMLR V6 GPU lock root is not a directory")
        lock_path = _repository_path(
            repository, GPU_LOCK_ROOT / f"gpu_{visible_device}.lock",
        )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            _validate_gpu_lock_descriptor(descriptor, lock_path)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"TMLR V6 physical GPU {visible_device} already has a worker"
                ) from error
            # Re-check immediately before the first mutation.  In particular,
            # a pre-existing hardlink can never be truncated or overwritten.
            _validate_gpu_lock_descriptor(descriptor, lock_path)
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            lease = _GpuLease(
                visible_device=visible_device,
                descriptor=descriptor,
                lock_path=lock_path,
                kernel_socket=kernel_socket,
                kernel_address=kernel_address,
                guard=_GPU_LEASE_GUARD,
            )
            _assert_active_gpu_lease(lease, visible_device)
            yield lease
        finally:
            # Retain the durable path.  Kernel ownership, rather than deleting
            # a file or interpreting a stale PID, defines lock lifetime.
            os.close(descriptor)
    finally:
        kernel_socket.close()


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TMLR V6 worker requires exactly one visible CUDA device")
    return torch.device("cuda:0")


def _initialize_cuda_telemetry(device: torch.device) -> int:
    current = torch.cuda.current_device()
    if device.index is None or current != device.index:
        raise RuntimeError("TMLR V6 CUDA telemetry device drift")
    torch.cuda.reset_peak_memory_stats(current)
    return current


def _tensor(
    array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        result = result.to(dtype)
    return result.to(device)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _implementation_identity(
    repository: Path,
) -> tuple[str, dict[str, str], str]:
    relative = [path.as_posix() for path in IMPLEMENTATION_PATHS]
    for path in relative:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=repository,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"TMLR V6 implementation is not committed: {path}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("TMLR V6 implementation paths are not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_hashes = {
        path: sha256_file(_repository_path(repository, Path(path)))
        for path in relative
    }
    combined = hashlib.sha256(canonical_json_bytes(source_hashes)).hexdigest()
    return commit, source_hashes, combined


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_artifact_identity(
    repository: Path, identity: Mapping[str, Any], *, kind: str,
) -> None:
    _verify_file(
        _repository_path(repository, Path(str(identity["path"]))),
        identity,
        description=f"completed {kind}",
    )


def _validate_local_run_artifact(
    repository: Path, run_dir: Path, value: Any, *, kind: str, filename: str,
) -> None:
    expected_path = run_dir / filename
    expected_logical = _logical_path(repository, expected_path)
    if not isinstance(value, Mapping) or set(value) != {
        "path", "bytes", "sha256",
    }:
        raise RuntimeError(f"completed TMLR V6 {kind} identity schema drift")
    if str(value.get("path")) != expected_logical:
        raise RuntimeError("completed TMLR V6 artifact escapes its run directory")
    identity = _exact_file_identity(
        value,
        expected_path=Path(expected_logical),
        description=f"completed {kind}",
    )
    _verify_file(expected_path, identity, description=f"completed {kind}")


def _run_staging_path(run_dir: Path) -> Path:
    return run_dir.with_name(f"{run_dir.name}.incomplete")


def _assert_no_stale_run_staging(staging_dir: Path) -> None:
    if staging_dir.exists() or staging_dir.is_symlink():
        raise RuntimeError(
            f"stale incomplete TMLR V6 run requires human resolution: {staging_dir}"
        )


def _prepare_run_staging(run_dir: Path) -> Path:
    """Create the deterministic sibling transaction directory, fail closed."""

    staging_dir = _run_staging_path(run_dir)
    _assert_no_stale_run_staging(staging_dir)
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(f"TMLR V6 final run already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    return staging_dir


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_run_staging(staging_dir: Path, run_dir: Path) -> None:
    """Atomically expose a complete run; never remove or overwrite state."""

    if (
        staging_dir != _run_staging_path(run_dir)
        or staging_dir.is_symlink()
        or not staging_dir.is_dir()
        or not (staging_dir / "run-manifest.json").is_file()
    ):
        raise RuntimeError("TMLR V6 run staging transaction is incomplete")
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(f"TMLR V6 final run appeared during publish: {run_dir}")
    _fsync_directory(staging_dir)
    staging_dir.rename(run_dir)
    _fsync_directory(run_dir.parent)


def _verify_existing(
    repository: Path, run_dir: Path, expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not run_dir.exists() and not run_dir.is_symlink():
        return None
    manifest_path = run_dir / "run-manifest.json"
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError(f"incomplete TMLR V6 run exists: {run_dir}")
    entries = list(run_dir.iterdir())
    if {item.name for item in entries} != {
        "run-manifest.json", "predictions.npz", "checkpoint.pt",
    }:
        raise RuntimeError("completed TMLR V6 run file set drift")
    for item in entries:
        try:
            metadata = item.stat(follow_symlinks=False)
        except OSError as error:
            raise RuntimeError("completed TMLR V6 run file set drift") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("completed TMLR V6 run file set drift")
    # No manifest bytes are opened until the exact three-entry, regular,
    # non-symlink, single-link transaction tree has passed.
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"completed TMLR V6 run identity drift: {drift}")
    assert_zero_external_access(report.get("heldout_access", {}))
    _validate_local_run_artifact(
        repository, run_dir, report.get("prediction"),
        kind="prediction", filename="predictions.npz",
    )
    _validate_local_run_artifact(
        repository, run_dir, report.get("checkpoint"),
        kind="checkpoint", filename="checkpoint.pt",
    )
    if (
        not math.isfinite(float(report.get("gpu_wall_seconds", math.nan)))
        or float(report.get("gpu_wall_seconds", math.inf)) > RUN_SECONDS_LIMIT
        or not math.isfinite(float(report.get("wall_seconds", math.nan)))
        or float(report.get("wall_seconds", math.inf)) > RUN_SECONDS_LIMIT
        or int(report.get("peak_vram_bytes", VRAM_LIMIT)) >= VRAM_LIMIT
        or int(report.get("peak_vram_bytes", -1)) < 0
        or not math.isfinite(float(report.get("inner_stop_loss", math.nan)))
        or int(report.get("best_epoch", 0)) <= 0
        or int(report.get("epochs_completed", 0)) <= 0
    ):
        raise RuntimeError("completed TMLR V6 run exceeds resource limits")
    return report


def run_crossfit(
    *, model_id: str, outer_fold: str, seed: int,
    repository: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
    fold_manifest_path: Path = FOLD_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate and run one job while holding its mandatory physical-GPU lease."""

    assert_cuda_determinism_environment()
    repository = _repository_path(repository.absolute(), Path("."))
    if (
        model_id not in FIRST_BATCH_MODEL_IDS
        or outer_fold not in FOLDS
        or seed not in SEEDS
    ):
        raise ValueError("unregistered TMLR V6 first-batch job")
    protocol = load_protocol(repository)
    visible_device = _validated_visible_device(protocol, outer_fold)
    registered_fold_path = Path(str(
        protocol["frozen_fit_inputs"]["folds"]["S1364"]["path"]
    ))
    if _repository_path(repository, fold_manifest_path) != _repository_path(
        repository, registered_fold_path,
    ):
        raise RuntimeError("TMLR V6 fold path is not registered")
    if _repository_path(repository, output_root) != _repository_path(
        repository, DEFAULT_OUTPUT_ROOT,
    ):
        raise RuntimeError("TMLR V6 output root is not the registered path")
    # All launch arguments and the frozen fold-to-device mapping are checked
    # before the lock directory or durable lock file can be created.
    with _exclusive_gpu_worker_lock(
        repository, visible_device,
    ) as gpu_lease:
        return _run_crossfit_with_gpu_lease(
            model_id=model_id,
            outer_fold=outer_fold,
            seed=seed,
            repository=repository,
            output_root=output_root,
            fold_manifest_path=fold_manifest_path,
            protocol=protocol,
            visible_device=visible_device,
            gpu_lease=gpu_lease,
        )


def _run_crossfit_with_gpu_lease(
    *, model_id: str, outer_fold: str, seed: int,
    repository: Path, output_root: Path,
    fold_manifest_path: Path, protocol: Mapping[str, Any],
    visible_device: str, gpu_lease: _GpuLease,
) -> dict[str, Any]:
    """Train or immutably resume after the public launch gate holds a lease."""

    _assert_active_gpu_lease(gpu_lease, visible_device)
    if _validated_visible_device(protocol, outer_fold) != visible_device:
        raise RuntimeError("TMLR V6 GPU lease assignment drift")
    fold_manifest = validate_fold_manifest(
        repository, protocol, fold_manifest_path,
    )
    if _repository_path(repository, output_root) != _repository_path(
        repository, DEFAULT_OUTPUT_ROOT,
    ):
        raise RuntimeError("TMLR V6 output root is not the registered path")
    run_dir = _repository_path(
        repository,
        output_root / "S1364" / model_id / outer_fold / f"seed_{seed}",
    )
    staging_dir = _run_staging_path(run_dir)
    _assert_no_stale_run_staging(staging_dir)
    model_spec = _model_registry(protocol)[model_id]
    role_id = str(model_spec["bundle"])
    bundle = load_registered_bundle(repository, role_id)
    _validate_bundle_population(bundle, fold_manifest)
    runtime_audit_identity = verify_runtime_feature_audit(repository, protocol)
    implementation_commit, source_hashes, implementation_sha = (
        _implementation_identity(repository)
    )
    protocol_sha = sha256_file(_repository_path(repository, PROTOCOL_PATH))
    fold_sha = sha256_file(_repository_path(repository, fold_manifest_path))
    train_groups, stop_groups, heldout_groups = _inner_split(
        fold_manifest, outer_fold, seed,
    )
    group_ids = _strings(bundle.features["image_group_ids"], name="image_group_ids")
    train_np = np.isin(group_ids, tuple(train_groups))
    stop_np = np.isin(group_ids, tuple(stop_groups))
    heldout_np = np.isin(group_ids, tuple(heldout_groups))
    outer_train_np = train_np | stop_np
    if (
        not train_np.any()
        or not stop_np.any()
        or not heldout_np.any()
        or np.any(train_np & stop_np)
        or np.any(train_np & heldout_np)
        or np.any(stop_np & heldout_np)
        or not np.all(train_np | stop_np | heldout_np)
    ):
        raise RuntimeError("TMLR V6 fit partition is not isolated")
    counts = {
        "inner_train_states": int(train_np.sum()),
        "inner_stop_states": int(stop_np.sum()),
        "outer_heldout_fit_states": int(heldout_np.sum()),
        "bundle_states": len(group_ids),
        "bundle_atoms": len(bundle.features["atom_ids"]),
    }
    run_id = stable_id("tmlr_v6_first_batch_run", {
        "protocol_sha256": protocol_sha,
        "implementation_sha256": implementation_sha,
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "runtime_feature_audit_sha256": runtime_audit_identity["sha256"],
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "model_id": model_id,
        "outer_fold": outer_fold,
        "seed": seed,
    })
    expected = {
        "schema_version": "rail3.tmlr-v6.first-batch-run.v1",
        "status": "PASS",
        "phase": "fit_s1364_first_batch_outer_crossfit",
        "run_id": run_id,
        "model_id": model_id,
        "bundle_role_id": role_id,
        "geometry": bundle.registration.geometry,
        "outer_fold": outer_fold,
        "seed": seed,
        "scale": "S1364",
        "protocol_sha256": protocol_sha,
        "implementation_commit": implementation_commit,
        "implementation_sha256": implementation_sha,
        "implementation_source_sha256": source_hashes,
        "feature_manifest_sha256": bundle.feature_manifest_sha256,
        "feature_array_sha256": bundle.feature_array_sha256,
        "target_manifest_sha256": bundle.target_manifest_sha256,
        "target_array_sha256": bundle.target_array_sha256,
        "fold_manifest_sha256": fold_sha,
        "runtime_feature_audit": runtime_audit_identity,
        "cuda_determinism_environment": cuda_determinism_environment_contract(),
        "visible_device": visible_device,
        "feature_schema_id": PROSPECTIVE_SCHEMA_ID,
        "input_dimensions": {
            "atom": ATOM_DIM,
            "state": STATE_DIM,
            "action_projected": ACTION_DIM,
            "global": GLOBAL_DIM,
        },
        "physical_input_partitions": ["prospective_features", "targets"],
        "post_action_diagnostics_opened": False,
        "predictor_cost_input": False,
        "cost_objective_input": False,
        "inference_weight_source": "feature_area_fraction_full_raster",
        "target_weight_source": "valid_pixel_fraction_target_only",
        "target_weight_in_prediction_graph": False,
        "primary_training_loss": protocol["training"]["primary_training_loss"],
        "inner_stopping_loss": protocol["training"]["inner_stopping_loss"],
        "atom_target_auxiliary_loss": False,
        "parameter_count": int(model_spec["parameters"]),
        "train_group_ids": sorted(train_groups),
        "inner_stop_group_ids": sorted(stop_groups),
        "outer_heldout_fit_group_ids": sorted(heldout_groups),
        "counts": counts,
        "heldout_access": dict(ZERO_EXTERNAL_ACCESS),
        "fit_only": True,
        "sam_inference": False,
        "prompt_or_trajectory_generation": False,
        "protected_reader_invoked": False,
        "ddp": False,
    }
    existing = _verify_existing(repository, run_dir, expected)
    if existing is not None:
        return existing

    is_atomic = model_spec["kind"] == "shared_atomic"
    scaled = scale_predictor_features(
        atom_x=bundle.features["atom_x"] if is_atomic else None,
        state_lengths=bundle.features["state_lengths"] if is_atomic else None,
        state_x=bundle.features["state_x"],
        action_x16=bundle.features["action_x16"],
        outer_train_mask=outer_train_np,
    )
    started = time.perf_counter()
    started_at = _utc_now()
    _seed(seed)
    device = _device()
    telemetry_index = _initialize_cuda_telemetry(device)
    staging_dir = _prepare_run_staging(run_dir)
    state_x = _tensor(scaled.state, device, torch.float32)
    action_x = _tensor(scaled.action, device, torch.float32)
    state_target = _tensor(bundle.targets["state_target"], device, torch.float32)
    train_mask = _tensor(train_np, device, torch.bool)
    stop_mask = _tensor(stop_np, device, torch.bool)
    global_x = torch.cat((
        state_x[:, None, :].expand(-1, len(ACTIONS), -1), action_x,
    ), dim=2)
    if global_x.shape[2] != GLOBAL_DIM:
        raise RuntimeError("TMLR V6 projected global dimension drift")
    model = build_registered_model(model_id, protocol).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    atom_x: torch.Tensor | None = None
    inference_weights: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    memory_bounded = False
    if is_atomic:
        if scaled.atom is None:
            raise RuntimeError("TMLR V6 atomic model has no scaled atomic features")
        atom_x = _tensor(scaled.atom, device, torch.float32)
        inference_weights = _tensor(
            inference_weights_from_features(
                bundle.features["atom_x"], bundle.features["state_lengths"],
            ),
            device,
            torch.float32,
        )
        lengths = _tensor(bundle.features["state_lengths"], device, torch.long)
        memory_bounded = atom_x.shape[0] >= MEMORY_BOUNDED_ATOM_THRESHOLD

    def objective(
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not is_atomic:
            prediction = model(global_x.reshape(-1, GLOBAL_DIM)).reshape(-1, 5)
            return _state_huber(prediction, state_target, mask), prediction, None
        assert (
            isinstance(model, SharedAtomicResidual)
            and atom_x is not None
            and inference_weights is not None
            and lengths is not None
        )
        if memory_bounded:
            atomic_prediction, state_prediction = (
                memory_bounded_shared_atomic_forward(
                    model,
                    atom_x,
                    state_x,
                    action_x,
                    lengths,
                    inference_weights,
                    checkpoint_activations=model.training,
                )
            )
        else:
            atomic_prediction, state_prediction = model(
                atom_x, state_x, action_x, lengths, inference_weights,
            )
        state_loss = _state_huber(state_prediction, state_target, mask)
        return state_loss, state_prediction, atomic_prediction

    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    completed = 0
    max_epochs = int(protocol["training"]["max_epochs"])
    patience = int(protocol["training"]["patience"])
    minimum_improvement = float(
        protocol["training"]["minimum_inner_stop_improvement"]
    )
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(train_mask)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite TMLR V6 first-batch training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            inner_value = float(objective(stop_mask)[0].item())
        completed = epoch
        if inner_value < best_value - minimum_improvement:
            best_value = inner_value
            best_epoch = epoch
            stale = 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= patience:
                break
        if time.perf_counter() - started > RUN_SECONDS_LIMIT:
            raise RuntimeError("TMLR V6 first-batch single-run wall limit exceeded")
    if best_state is None:
        raise RuntimeError("TMLR V6 early stopping produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        _, state_prediction, atom_prediction = objective(stop_mask)
    if not torch.isfinite(state_prediction).all():
        raise RuntimeError("non-finite TMLR V6 state prediction")
    state_indices = np.flatnonzero(heldout_np)
    state_ids = _strings(bundle.features["state_ids"], name="state_ids")
    output: dict[str, np.ndarray] = {
        "state_index": state_indices.astype(np.int32),
        "state_id": state_ids[state_indices],
        "state_prediction": (
            state_prediction[state_indices].detach().cpu().numpy().astype(np.float32)
        ),
    }
    if atom_prediction is not None:
        if not torch.isfinite(atom_prediction).all():
            raise RuntimeError("non-finite TMLR V6 atom prediction")
        lengths_np = np.asarray(bundle.features["state_lengths"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths_np)]
        atom_indices = np.concatenate([
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index in state_indices
        ])
        atom_ids = _strings(bundle.features["atom_ids"], name="atom_ids")
        output.update({
            "atom_index": atom_indices.astype(np.int32),
            "atom_id": atom_ids[atom_indices],
            "atom_prediction": (
                atom_prediction[atom_indices]
                .detach().cpu().numpy().astype(np.float32)
            ),
        })
    prediction_path = staging_dir / "predictions.npz"
    checkpoint_path = staging_dir / "checkpoint.pt"
    final_prediction_path = run_dir / prediction_path.name
    final_checkpoint_path = run_dir / checkpoint_path.name
    _atomic_npz(prediction_path, **output)
    _atomic_checkpoint(checkpoint_path, {
        "model": best_state,
        "model_id": model_id,
        "bundle_role_id": role_id,
        "feature_schema_id": PROSPECTIVE_SCHEMA_ID,
        "input_dimensions": expected["input_dimensions"],
        "scalers": scaler_payload(scaled.scalers),
    })
    wall_seconds = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_reserved(telemetry_index))
    if wall_seconds > RUN_SECONDS_LIMIT or peak_vram >= VRAM_LIMIT:
        raise RuntimeError("TMLR V6 first-batch per-run resource limit exceeded")
    report = {
        **expected,
        "architecture_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "optimizer": "AdamW",
        "learning_rate": float(protocol["training"]["learning_rate"]),
        "weight_decay": float(protocol["training"]["weight_decay"]),
        "max_epochs": max_epochs,
        "patience": patience,
        "minimum_inner_stop_improvement": minimum_improvement,
        "best_epoch": best_epoch,
        "epochs_completed": completed,
        "inner_stop_loss": best_value,
        "execution_memory_mode": (
            f"ACTIVATION_CHECKPOINTED_ATOM_CHUNKS_{MEMORY_BOUNDED_ATOM_CHUNK}"
            if memory_bounded else "FULL_BATCH"
        ),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "wall_seconds": wall_seconds,
        "gpu_wall_seconds": wall_seconds,
        "peak_vram_bytes": peak_vram,
        "prediction": {
            "path": _logical_path(repository, final_prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        },
        "checkpoint": {
            "path": _logical_path(repository, final_checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
    }
    _atomic_json(staging_dir / "run-manifest.json", report)
    _publish_run_staging(staging_dir, run_dir)
    return report


__all__ = [
    "BUNDLE_REGISTRY",
    "CUBLAS_WORKSPACE_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "EXTERNAL_SPLITS",
    "FIRST_BATCH_MODEL_IDS",
    "FOLDS",
    "LEGACY_EXTERNAL_SPLITS",
    "LEGACY_ZERO_EXTERNAL_ACCESS",
    "SEEDS",
    "ZERO_EXTERNAL_ACCESS",
    "assert_legacy_zero_external_access",
    "assert_cuda_determinism_environment",
    "assert_zero_external_access",
    "build_registered_model",
    "cuda_determinism_environment_contract",
    "load_protocol",
    "load_registered_bundle",
    "inference_weights_from_features",
    "memory_bounded_shared_atomic_forward",
    "run_crossfit",
    "verify_runtime_feature_audit",
]
