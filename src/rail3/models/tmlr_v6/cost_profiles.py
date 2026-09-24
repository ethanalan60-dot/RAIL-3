"""Fit-only, action-type cost profiles for TMLR V6 decisions.

This module consumes label-free trajectory *reports*.  It never loads a
feature bundle, target table, prediction cache, trajectory cache, or protected
split.  Costs are decision inputs kept outside every residual predictor.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROFILE_ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
ADAPTIVE_ACTIONS = PROFILE_ACTIONS[1:]
TRAJECTORY_REPORT_KEYS = ("fit100", "new400", "new500", "new364")
COST_SOURCE_IDENTITY_SCHEMA = "rail3.tmlr-v6.cost-profile-sources.v1"
COST_PROFILE_PROTOCOL_PATH = Path(
    "configs/experiments/tmlr_v6_p0_prospective_capacity.json"
)
COST_PROFILE_IMPLEMENTATION_PATHS = (
    Path("scripts/build_tmlr_v6_action_cost_profiles.py"),
    Path("src/rail3/models/tmlr_v6/cost_profiles.py"),
)
_IDENTITY_FIELDS = frozenset({"path", "bytes", "sha256"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PROTECTED_PATH_TOKENS = (
    "artifacts/protected",
    "validation40",
    "locked40",
    "official_voc_val",
    "official-voc-val",
    "calibration30",
    "pilot_test30",
    "pilot-test30",
    "segmentation-trainval",
    "segppd",
    "uav-iap",
    "uav_iap",
)
CSV_FIELDS = (
    "action",
    "fold",
    "seed",
    "n",
    "n_total",
    "n_infeasible",
    "median_seconds",
    "mean_seconds",
    "q25_seconds",
    "q75_seconds",
    "pooled_lower_median_normalizer_seconds",
    "normalized_median",
    "outer_training_group_count",
    "source_manifest_sha256",
    "profile_sha256",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_identity(value: Any, *, name: str) -> dict[str, object]:
    """Return one exact protocol identity without opening its file."""

    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise RuntimeError(
            f"TMLR_V6_COST_SOURCE_REGISTRATION_REQUIRED: {name} must have "
            "exact path/bytes/sha256"
        )
    logical = str(value["path"])
    path = Path(logical)
    if not logical or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"TMLR V6 {name} path is not repository-relative")
    assert_fit_only_path(path)
    try:
        byte_count = int(value["bytes"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"TMLR V6 {name} byte identity is invalid") from error
    digest = str(value["sha256"])
    if byte_count <= 0 or _SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError(f"TMLR V6 {name} identity is invalid")
    return {"path": path.as_posix(), "bytes": byte_count, "sha256": digest}


def protocol_cost_source_registrations(
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    """Extract the exact source registrations needed by the cost builder.

    Keeping this registration subset separate from generated cost-artifact
    identities avoids a circular protocol hash while still binding the
    derivation to the frozen protocol.
    """

    frozen = protocol.get("frozen_fit_inputs")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("TMLR_V6_COST_SOURCE_REGISTRATION_REQUIRED")
    folds = frozen.get("folds")
    reports = frozen.get("trajectory_reports")
    if not isinstance(folds, Mapping) or not isinstance(reports, Mapping):
        raise RuntimeError("TMLR_V6_COST_SOURCE_REGISTRATION_REQUIRED")
    if set(reports) != set(TRAJECTORY_REPORT_KEYS):
        raise RuntimeError(
            "TMLR_V6_COST_SOURCE_REGISTRATION_REQUIRED: trajectory report "
            "keys must be fit100/new400/new500/new364 exactly"
        )
    report_identities = {
        key: _validated_identity(
            reports[key], name=f"{key} trajectory report",
        )
        for key in TRAJECTORY_REPORT_KEYS
    }
    result: dict[str, object] = {
        "fit_manifest": _validated_identity(
            frozen.get("s1364_manifest"), name="S1364 manifest",
        ),
        "fold_manifest": _validated_identity(
            folds.get("S1364"), name="S1364 fold manifest",
        ),
        "trajectory_reports": report_identities,
    }
    legacy_hashes = frozen.get("trajectory_report_sha256")
    if legacy_hashes is not None:
        expected_hashes = {
            key: report_identities[key]["sha256"] for key in TRAJECTORY_REPORT_KEYS
        }
        if not isinstance(legacy_hashes, Mapping) or dict(legacy_hashes) != expected_hashes:
            raise RuntimeError("TMLR V6 legacy trajectory SHA registration drift")
    return result


def _origin_protocol_identity(
    protocol: Mapping[str, Any], registrations: Mapping[str, Any],
) -> dict[str, object]:
    schema = str(protocol.get("schema_version", ""))
    protocol_id = str(protocol.get("protocol_id", ""))
    if not schema or not protocol_id:
        raise RuntimeError("TMLR V6 cost source protocol identity is incomplete")
    return {
        "path": COST_PROFILE_PROTOCOL_PATH.as_posix(),
        "schema_version": schema,
        "protocol_id": protocol_id,
        "registered_source_set_sha256": canonical_sha256(registrations),
    }


def _repository_source_path(
    repository: Path, requested: Path, *, expected_logical: str, name: str,
) -> Path:
    """Resolve one source only after its requested logical path is registered."""

    lexical_repository = repository.absolute()
    if (
        lexical_repository.is_symlink()
        or lexical_repository.resolve() != lexical_repository
    ):
        raise RuntimeError("TMLR V6 repository root is symlinked")
    if requested.is_absolute():
        if ".." in requested.parts:
            raise RuntimeError(f"TMLR V6 {name} path escapes the worktree")
        try:
            logical = requested.relative_to(lexical_repository).as_posix()
        except ValueError as error:
            raise RuntimeError(f"TMLR V6 {name} path escapes the worktree") from error
    else:
        if ".." in requested.parts:
            raise RuntimeError(f"TMLR V6 {name} path escapes the worktree")
        logical = requested.as_posix()
    if logical != expected_logical:
        raise RuntimeError(f"TMLR V6 {name} requested path/registration drift")
    candidate = lexical_repository / expected_logical
    assert_fit_only_path(candidate)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(lexical_repository)
    except ValueError as error:
        raise RuntimeError(f"TMLR V6 {name} path escapes the worktree") from error
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise RuntimeError(f"TMLR V6 {name} is missing or aliased") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(
            f"TMLR V6 {name} must be one regular single-link file"
        )
    return candidate


def verify_registered_cost_sources(
    *, repository: Path, protocol: Mapping[str, Any],
    fit_manifest_path: Path, fold_manifest_path: Path,
    trajectory_report_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Verify every CLI path before reading any registered source bytes."""

    registrations = protocol_cost_source_registrations(protocol)
    report_registrations = registrations["trajectory_reports"]
    if set(trajectory_report_paths) != set(TRAJECTORY_REPORT_KEYS):
        raise RuntimeError("TMLR V6 trajectory report request set drift")
    requested = {
        "fit_manifest": _repository_source_path(
            repository, fit_manifest_path,
            expected_logical=str(registrations["fit_manifest"]["path"]),
            name="S1364 manifest",
        ),
        "fold_manifest": _repository_source_path(
            repository, fold_manifest_path,
            expected_logical=str(registrations["fold_manifest"]["path"]),
            name="S1364 fold manifest",
        ),
        "trajectory_reports": {
            key: _repository_source_path(
                repository, trajectory_report_paths[key],
                expected_logical=str(report_registrations[key]["path"]),
                name=f"{key} trajectory report",
            )
            for key in TRAJECTORY_REPORT_KEYS
        },
    }
    # No source file bytes are read until all requested paths pass above.
    flattened = [
        ("fit_manifest", requested["fit_manifest"], registrations["fit_manifest"]),
        ("fold_manifest", requested["fold_manifest"], registrations["fold_manifest"]),
        *[
            (
                f"trajectory_reports.{key}",
                requested["trajectory_reports"][key],
                report_registrations[key],
            )
            for key in TRAJECTORY_REPORT_KEYS
        ],
    ]
    for name, path, registration in flattened:
        if (
            path.stat().st_size != int(registration["bytes"])
            or sha256_file(path) != str(registration["sha256"])
        ):
            raise RuntimeError(f"TMLR V6 {name} registered identity drift")
    return {
        "schema_version": COST_SOURCE_IDENTITY_SCHEMA,
        "origin_protocol": _origin_protocol_identity(protocol, registrations),
        "fit_manifest": dict(registrations["fit_manifest"]),
        "fold_manifest": dict(registrations["fold_manifest"]),
        "trajectory_reports": {
            key: dict(report_registrations[key]) for key in TRAJECTORY_REPORT_KEYS
        },
    }


def validate_cost_profile_source_identities(
    *, repository: Path, protocol: Mapping[str, Any], identities: Any,
) -> dict[str, object]:
    """Require an exact artifact-to-protocol source set and reverify its files."""

    if not isinstance(identities, Mapping):
        raise RuntimeError("TMLR V6 cost profile source identities are missing")
    expected_keys = {
        "schema_version", "origin_protocol", "fit_manifest", "fold_manifest",
        "trajectory_reports",
    }
    if set(identities) != expected_keys:
        raise RuntimeError("TMLR V6 cost profile source identity set drift")
    reports = identities.get("trajectory_reports")
    if not isinstance(reports, Mapping) or set(reports) != set(TRAJECTORY_REPORT_KEYS):
        raise RuntimeError("TMLR V6 cost profile trajectory source set drift")
    normalized_fit = _validated_identity(
        identities.get("fit_manifest"), name="recorded S1364 manifest",
    )
    normalized_fold = _validated_identity(
        identities.get("fold_manifest"), name="recorded S1364 fold manifest",
    )
    normalized_reports = {
        key: _validated_identity(
            reports[key], name=f"recorded {key} trajectory report",
        )
        for key in TRAJECTORY_REPORT_KEYS
    }
    verified = verify_registered_cost_sources(
        repository=repository,
        protocol=protocol,
        fit_manifest_path=Path(str(normalized_fit["path"])),
        fold_manifest_path=Path(str(normalized_fold["path"])),
        trajectory_report_paths={
            key: Path(str(normalized_reports[key]["path"]))
            for key in TRAJECTORY_REPORT_KEYS
        },
    )
    if canonical_json_bytes(dict(identities)) != canonical_json_bytes(verified):
        raise RuntimeError("TMLR V6 cost profile source identity drift")
    return verified


def committed_cost_profile_implementation_identity(
    repository: Path,
) -> dict[str, object]:
    """Bind profile generation to committed, clean builder/source bytes."""

    repository = repository.resolve()
    logical = tuple(path.as_posix() for path in COST_PROFILE_IMPLEMENTATION_PATHS)
    for value in logical:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", value], cwd=repository,
            capture_output=True, text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"cost-profile implementation is not committed: {value}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *logical], cwd=repository,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("cost-profile implementation paths are not clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    sources: dict[str, str] = {}
    for value in logical:
        current = repository / value
        digest = sha256_file(current)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{value}"], cwd=repository,
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != digest:
            raise RuntimeError(f"cost-profile implementation commit drift: {value}")
        sources[value] = digest
    return {
        "commit": commit,
        "source_sha256": sources,
        "combined_sha256": canonical_sha256(sources),
    }


def validate_cost_profile_implementation_identity(
    repository: Path, identity: Any,
) -> dict[str, object]:
    """Authenticate the recorded generator commit without trusting its report."""

    if not isinstance(identity, Mapping) or set(identity) != {
        "commit", "source_sha256", "combined_sha256",
    }:
        raise RuntimeError("TMLR V6 cost-profile implementation identity is missing")
    commit = str(identity["commit"])
    sources = identity["source_sha256"]
    expected_paths = {
        path.as_posix() for path in COST_PROFILE_IMPLEMENTATION_PATHS
    }
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(sources, Mapping)
        or set(sources) != expected_paths
        or any(_SHA256_PATTERN.fullmatch(str(value)) is None for value in sources.values())
        or str(identity["combined_sha256"]) != canonical_sha256(dict(sources))
    ):
        raise RuntimeError("TMLR V6 cost-profile implementation identity drift")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("TMLR V6 cost-profile implementation commit is not an ancestor")
    for path in sorted(expected_paths):
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True,
        )
        if (
            result.returncode != 0
            or hashlib.sha256(result.stdout).hexdigest() != str(sources[path])
        ):
            raise RuntimeError(
                f"TMLR V6 cost-profile committed implementation drift: {path}"
            )
    return {
        "commit": commit,
        "source_sha256": dict(sources),
        "combined_sha256": str(identity["combined_sha256"]),
    }


def assert_fit_only_path(path: Path) -> None:
    """Reject a path before opening it when it names a protected source."""

    raw = str(path).replace("\\", "/").lower()
    matches = [token for token in PROTECTED_PATH_TOKENS if token in raw]
    if matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: " + ", ".join(matches)
        )
    if ".." in path.parts:
        raise RuntimeError("TMLR V6 fit-only path contains a parent traversal")
    lexical = path.absolute()
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("TMLR V6 fit-only path contains a symlink")
    resolved = lexical.resolve(strict=False)
    resolved_text = resolved.as_posix().lower()
    resolved_matches = [
        token for token in PROTECTED_PATH_TOKENS if token in resolved_text
    ]
    if resolved_matches:
        raise RuntimeError(
            "STOP-BLOCKED_TMLR_V6_PROTECTED_ACCESS: "
            + ", ".join(resolved_matches)
        )


def load_fit_only_json(path: Path) -> dict[str, Any]:
    assert_fit_only_path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError("TMLR V6 fit-only JSON is absent") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(
            "TMLR V6 fit-only JSON must be one regular single-link file"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cost-profile source must be a JSON object")
    return payload


def lower_empirical_median(values: Iterable[float]) -> float:
    """Match torch.median's lower-middle convention for an even population."""

    array = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("lower empirical median requires finite positive values")
    if np.any(array <= 0):
        raise ValueError("lower empirical median requires finite positive values")
    return float(array[(len(array) - 1) // 2])


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quantile requires a finite nonempty vector")
    return float(np.quantile(array, probability, method="linear"))


def _fit_index(manifest: Mapping[str, Any]) -> tuple[dict[str, str], set[str]]:
    records = list(manifest.get("records", ()))
    if not records:
        raise ValueError("fit manifest contains no records")
    asset_to_group: dict[str, str] = {}
    for item in records:
        asset_id = str(item["asset_id"])
        group_id = str(item["group_id"])
        if asset_id in asset_to_group:
            raise ValueError("fit manifest repeats an asset")
        asset_to_group[asset_id] = group_id
    groups = set(asset_to_group.values())
    if len(groups) != len(asset_to_group):
        raise ValueError("fit manifest must contain one image per group")
    declared = manifest.get("count")
    if declared is not None and int(declared) != len(records):
        raise ValueError("fit manifest count disagrees with records")
    return asset_to_group, groups


def _fold_index(
    fold_manifest: Mapping[str, Any], *, expected_groups: set[str],
) -> dict[str, set[str]]:
    assignments = list(fold_manifest.get("groups", ()))
    if not assignments:
        raise ValueError("fold manifest contains no groups")
    by_fold: dict[str, set[str]] = {}
    observed: set[str] = set()
    for item in assignments:
        group_id = str(item["image_group_id"])
        fold_id = str(item["fold_id"])
        if group_id in observed:
            raise ValueError("fold manifest repeats an image group")
        observed.add(group_id)
        by_fold.setdefault(fold_id, set()).add(group_id)
    if observed != expected_groups:
        raise ValueError("fold manifest and fit manifest group sets differ")
    if len(by_fold) < 2 or any(not groups for groups in by_fold.values()):
        raise ValueError("cost profiles require at least two nonempty folds")
    declared = fold_manifest.get("group_count")
    if declared is not None and int(declared) != len(observed):
        raise ValueError("fold-manifest group count disagrees with assignments")
    return by_fold


def _runtime_index(
    *, trajectory_reports: Sequence[Mapping[str, Any]],
    asset_to_group: Mapping[str, str],
) -> dict[str, dict[str, list[tuple[bool, float]]]]:
    result = {
        group_id: {action: [] for action in ADAPTIVE_ACTIONS}
        for group_id in asset_to_group.values()
    }
    seen_assets: set[str] = set()
    for report in trajectory_reports:
        if report.get("status") not in (None, "PASS"):
            raise ValueError("trajectory report does not have PASS status")
        if report.get("run_status") not in (None, "COMPLETE"):
            raise ValueError("trajectory report is not complete")
        report_records = list(report.get("records", ()))
        declared_images = report.get("completed_image_count")
        if declared_images is not None and int(declared_images) != len(report_records):
            raise ValueError("trajectory report image count disagrees with records")
        for image in report_records:
            asset_id = str(image["asset_id"])
            if asset_id not in asset_to_group:
                raise ValueError("trajectory report contains an out-of-fit asset")
            if asset_id in seen_assets:
                raise ValueError("trajectory reports repeat an asset")
            seen_assets.add(asset_id)
            group_id = asset_to_group[asset_id]
            states = list(image.get("class_states", ()))
            if not states:
                raise ValueError("trajectory report image contains no class states")
            state_ids: set[str] = set()
            for state in states:
                actions = list(state.get("actions", ()))
                selected = [
                    item for item in actions
                    if str(item.get("action_code")) in ADAPTIVE_ACTIONS
                ]
                codes = tuple(str(item.get("action_code")) for item in selected)
                if codes != ADAPTIVE_ACTIONS:
                    raise ValueError(
                        "trajectory class state must contain ordered A3--A6 exactly once"
                    )
                state_id = str(selected[0].get("state_id", ""))
                if not state_id or state_id in state_ids:
                    raise ValueError("trajectory image repeats or omits a state identity")
                if any(str(item.get("state_id", "")) != state_id for item in selected):
                    raise ValueError("trajectory action records cross state identities")
                state_ids.add(state_id)
                for item in selected:
                    action = str(item["action_code"])
                    feasible = bool(item["feasible"])
                    cost = float(item["action_cost_seconds"])
                    if not math.isfinite(cost):
                        raise ValueError("trajectory action runtime is not finite")
                    if feasible and cost <= 0:
                        raise ValueError(
                            "feasible trajectory action must have a positive runtime"
                        )
                    if not feasible and cost != 0:
                        raise ValueError(
                            "infeasible trajectory action must have exactly zero runtime"
                        )
                    result[group_id][action].append((feasible, cost))
    if seen_assets != set(asset_to_group):
        raise ValueError("trajectory reports do not cover the fit manifest exactly")
    state_counts = {
        len(rows)
        for by_action in result.values()
        for rows in by_action.values()
    }
    if len(state_counts) != 1 or next(iter(state_counts), 0) <= 0:
        raise ValueError("trajectory action coverage differs across groups/actions")
    return result


def _profile_rows(
    *, profile_id: str, training_groups: set[str],
    runtime_index: Mapping[str, Mapping[str, Sequence[tuple[bool, float]]]],
    source_manifest_sha256: str,
) -> list[dict[str, Any]]:
    adaptive_values: dict[str, list[float]] = {}
    adaptive_totals: dict[str, int] = {}
    adaptive_infeasible: dict[str, int] = {}
    for action in ADAPTIVE_ACTIONS:
        observations = [
            observation
            for group_id in sorted(training_groups)
            for observation in runtime_index[group_id][action]
        ]
        values = [cost for feasible, cost in observations if feasible]
        if not values:
            raise ValueError(f"{profile_id} has no feasible positive {action} runtime")
        adaptive_values[action] = values
        adaptive_totals[action] = len(observations)
        adaptive_infeasible[action] = sum(not feasible for feasible, _ in observations)
    pooled = [value for action in ADAPTIVE_ACTIONS for value in adaptive_values[action]]
    normalizer = lower_empirical_median(pooled)
    action_statistics = []
    for action in ADAPTIVE_ACTIONS:
        values = adaptive_values[action]
        action_statistics.append({
            "action": action,
            "n": len(values),
            "n_total": adaptive_totals[action],
            "n_infeasible": adaptive_infeasible[action],
            "median_seconds": _linear_quantile(values, 0.50),
            "mean_seconds": float(np.mean(np.asarray(values, dtype=np.float64))),
            "q25_seconds": _linear_quantile(values, 0.25),
            "q75_seconds": _linear_quantile(values, 0.75),
        })
    profile_sha256 = canonical_sha256({
        "profile": profile_id,
        "training_groups": sorted(training_groups),
        "pooled_lower_median_normalizer_seconds": normalizer,
        "actions": action_statistics,
    })
    common = {
        "fold": profile_id,
        "seed": "ALL",
        "pooled_lower_median_normalizer_seconds": normalizer,
        "outer_training_group_count": len(training_groups),
        "source_manifest_sha256": source_manifest_sha256,
        "profile_sha256": profile_sha256,
    }
    rows: list[dict[str, Any]] = [{
        "action": "STOP",
        "n": 0,
        "n_total": 0,
        "n_infeasible": 0,
        "median_seconds": 0.0,
        "mean_seconds": 0.0,
        "q25_seconds": 0.0,
        "q75_seconds": 0.0,
        "normalized_median": 0.0,
        **common,
    }]
    for statistics in action_statistics:
        rows.append({
            **statistics,
            "normalized_median": statistics["median_seconds"] / normalizer,
            **common,
        })
    return rows


def build_action_type_cost_profiles(
    *, fit_manifest: Mapping[str, Any], fold_manifest: Mapping[str, Any],
    trajectory_reports: Sequence[Mapping[str, Any]],
    source_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build fold-specific outer-training profiles plus one S-fit full profile."""

    asset_to_group, fit_groups = _fit_index(fit_manifest)
    groups_by_fold = _fold_index(fold_manifest, expected_groups=fit_groups)
    runtime_index = _runtime_index(
        trajectory_reports=trajectory_reports, asset_to_group=asset_to_group,
    )
    source_manifest_sha256 = (
        str(source_identities["fit_manifest"]["sha256"])
        if source_identities and "fit_manifest" in source_identities
        else canonical_sha256(fit_manifest)
    )
    rows: list[dict[str, Any]] = []
    for fold in sorted(groups_by_fold):
        outer_training = fit_groups - groups_by_fold[fold]
        if not outer_training:
            raise ValueError("outer-training group set is empty")
        rows.extend(_profile_rows(
            profile_id=fold,
            training_groups=outer_training,
            runtime_index=runtime_index,
            source_manifest_sha256=source_manifest_sha256,
        ))
    rows.extend(_profile_rows(
        profile_id="FULLFIT",
        training_groups=fit_groups,
        runtime_index=runtime_index,
        source_manifest_sha256=source_manifest_sha256,
    ))
    return {
        "schema_version": "rail3.tmlr-v6.action-type-cost-profiles.v1",
        "status": "TMLR_V6_FIT_ONLY_COST_PROFILES_PASS",
        "actions": list(PROFILE_ACTIONS),
        "per_action_estimator": "numpy_linear_quantile_0.5",
        "normalizer_estimator": "lower_empirical_median_pooled_feasible_positive",
        "profile_seed_dependency": False,
        "fit_group_count": len(fit_groups),
        "class_states_per_group": len(
            next(iter(runtime_index.values()))[ADAPTIVE_ACTIONS[0]]
        ),
        "source_identities": dict(source_identities or {}),
        "rows": rows,
        "checks": {
            "label_free_trajectory_reports_only": True,
            "feasible_runtime_strictly_positive": True,
            "infeasible_runtime_exactly_zero": True,
            "outer_heldout_excluded_per_fold": True,
            "stop_exactly_zero": True,
            "profile_seed_independent": True,
            "predictor_input_used": False,
            "protected_read_count": 0,
        },
        "protected_access": {
            "validation40": 0,
            "official_voc_val": 0,
            "calibration30": 0,
            "pilot_test30": 0,
            "segppd": 0,
            "uav_iap": 0,
        },
    }


def csv_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize profile rows deterministically with a fixed column order."""

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in payload["rows"]:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    return handle.getvalue().encode("utf-8")
