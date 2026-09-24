"""Result-blind execution disposition for semantically unresolved E4 states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


E4_SEMANTIC_CLASSIFICATION = "E4_INSUFFICIENT_EVIDENCE"
E4_EXECUTION_DISPOSITION = "QUARANTINED_TECHNICAL_MISSINGNESS"
E4_DISPOSITION_VERSION = "V9B1_E4_MISSINGNESS_DISPOSITION_V1"
TECHNICAL_MISSINGNESS_STATE_DENOMINATOR = 20_000
TECHNICAL_MISSINGNESS_STATE_MAXIMUM = 100
TECHNICAL_MISSINGNESS_STATE_MAXIMUM_FRACTION = 0.005
TECHNICAL_MISSINGNESS_IMAGE_DENOMINATOR = 1_000
TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM = 10
TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM_FRACTION = 0.01
INFERENCE_COMPLETENESS_FAILED = "INFERENCE_COMPLETENESS_FAILED"
E4_EXECUTION_REGISTRY_SCHEMA = "rail3.tmlr-v9b1-e4-execution-registry.v1"
E4_EXECUTION_REGISTRY_VERSION = "V9B1_E4_EXECUTION_REGISTRY_V1"


@dataclass(frozen=True)
class E4TechnicalMissingnessRecord:
    semantic_state_id: str
    image_group_id: str
    class_id: int
    retry_count: int = 1
    maximum_retry: int = 1
    semantic_classification: str = E4_SEMANTIC_CLASSIFICATION
    execution_disposition: str = E4_EXECUTION_DISPOSITION
    semantic_outcome_defined: bool = False
    trajectory_defined: bool = False
    scientific_outcome_imputed: bool = False
    historical_state_reconstructed: bool = False
    canonical_a0_created: bool = False
    additional_retry: bool = False

    def __post_init__(self) -> None:
        if not self.semantic_state_id or not self.image_group_id or self.class_id <= 0:
            raise ValueError("E4 technical-missingness identity is invalid")
        if (
            self.semantic_classification != E4_SEMANTIC_CLASSIFICATION
            or self.execution_disposition != E4_EXECUTION_DISPOSITION
        ):
            raise ValueError("E4 semantic classification or execution disposition drift")
        if self.retry_count != 1 or self.maximum_retry != 1 or self.additional_retry:
            raise ValueError("E4 retry contract drift")
        if any((
            self.semantic_outcome_defined,
            self.trajectory_defined,
            self.scientific_outcome_imputed,
            self.historical_state_reconstructed,
            self.canonical_a0_created,
        )):
            raise ValueError("E4 execution disposition cannot impute a scientific outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_state_id": self.semantic_state_id,
            "image_group_id": self.image_group_id,
            "class_id": self.class_id,
            "semantic_classification": self.semantic_classification,
            "execution_disposition": self.execution_disposition,
            "semantic_outcome_defined": self.semantic_outcome_defined,
            "trajectory_defined": self.trajectory_defined,
            "scientific_outcome_imputed": self.scientific_outcome_imputed,
            "historical_state_reconstructed": self.historical_state_reconstructed,
            "canonical_A0_created": self.canonical_a0_created,
            "retry_count": self.retry_count,
            "maximum_retry": self.maximum_retry,
            "additional_retry": self.additional_retry,
        }


@dataclass(frozen=True)
class E4ExecutionEvidence:
    """Frozen evidence proving that an E4 state has no scientific trajectory."""

    record: E4TechnicalMissingnessRecord
    semantic_resolution: str
    cache_key: str
    cache_path: str
    cache_sha256: str
    retry_attempt_path: str
    retry_attempt_sha256: str
    retry_terminal_path: str
    retry_terminal_sha256: str

    def __post_init__(self) -> None:
        if self.semantic_resolution != "UNRESOLVED_E4":
            raise ValueError("E4 semantic resolution was improperly closed")
        for value in (self.cache_sha256, self.retry_attempt_sha256, self.retry_terminal_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("E4 execution evidence SHA-256 is invalid")
        for value in (self.cache_path, self.retry_attempt_path, self.retry_terminal_path):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("E4 execution evidence path is not repository-relative")

    def disposition_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "semantic_resolution": self.semantic_resolution,
            "execution_evidence_cache_key": self.cache_key,
        }


def _unique_records(
    records: Sequence[E4TechnicalMissingnessRecord],
) -> tuple[E4TechnicalMissingnessRecord, ...]:
    by_state: dict[str, E4TechnicalMissingnessRecord] = {}
    for record in records:
        existing = by_state.get(record.semantic_state_id)
        if existing is not None and existing != record:
            raise ValueError("conflicting E4 dispositions share a semantic state ID")
        by_state[record.semantic_state_id] = record
    return tuple(by_state[key] for key in sorted(by_state))


def evaluate_e4_completeness(
    records: Sequence[E4TechnicalMissingnessRecord],
) -> dict[str, Any]:
    """Apply the frozen, non-tunable completeness gate to unique E4 states."""

    unique = _unique_records(records)
    state_count = len(unique)
    image_group_count = len({record.image_group_id for record in unique})
    state_fraction = state_count / TECHNICAL_MISSINGNESS_STATE_DENOMINATOR
    image_group_fraction = image_group_count / TECHNICAL_MISSINGNESS_IMAGE_DENOMINATOR
    passed = (
        state_count <= TECHNICAL_MISSINGNESS_STATE_MAXIMUM
        and state_fraction <= TECHNICAL_MISSINGNESS_STATE_MAXIMUM_FRACTION
        and image_group_count <= TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM
        and image_group_fraction <= TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM_FRACTION
    )
    return {
        "disposition_version": E4_DISPOSITION_VERSION,
        "technical_missingness_count": state_count,
        "technical_missingness_denominator": TECHNICAL_MISSINGNESS_STATE_DENOMINATOR,
        "technical_missingness_fraction": state_fraction,
        "maximum_technical_missingness_count": TECHNICAL_MISSINGNESS_STATE_MAXIMUM,
        "maximum_technical_missingness_fraction": (
            TECHNICAL_MISSINGNESS_STATE_MAXIMUM_FRACTION
        ),
        "affected_image_group_count": image_group_count,
        "image_group_denominator": TECHNICAL_MISSINGNESS_IMAGE_DENOMINATOR,
        "affected_image_group_fraction": image_group_fraction,
        "maximum_affected_image_group_count": TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM,
        "maximum_affected_image_group_fraction": (
            TECHNICAL_MISSINGNESS_IMAGE_MAXIMUM_FRACTION
        ),
        "completeness_gate": "PASS" if passed else "FAIL",
        "terminal": None if passed else INFERENCE_COMPLETENESS_FAILED,
        "resume_allowed_under_frozen_missingness_policy": passed,
    }


def trajectory_denominator_partition(
    panel_states: Sequence[tuple[str, str]],
    records: Sequence[E4TechnicalMissingnessRecord],
) -> dict[str, Any]:
    """Exclude only undefined E4 trajectories while preserving panel and groups."""

    state_to_group: dict[str, str] = {}
    panel_state_ids: list[str] = []
    panel_image_groups: list[str] = []
    seen_groups: set[str] = set()
    for state_id, image_group_id in panel_states:
        if not state_id or not image_group_id or state_id in state_to_group:
            raise ValueError("panel state identity is empty or duplicated")
        state_to_group[state_id] = image_group_id
        panel_state_ids.append(state_id)
        if image_group_id not in seen_groups:
            panel_image_groups.append(image_group_id)
            seen_groups.add(image_group_id)
    unavailable = _unique_records(records)
    unavailable_ids = {record.semantic_state_id for record in unavailable}
    for record in unavailable:
        if state_to_group.get(record.semantic_state_id) != record.image_group_id:
            raise ValueError("E4 disposition is absent from or conflicts with the panel")
    numerical = [state_id for state_id in panel_state_ids if state_id not in unavailable_ids]
    return {
        "panel_state_ids": panel_state_ids,
        "trajectory_dependent_numerical_state_ids": numerical,
        "technical_unavailable_state_ids": sorted(unavailable_ids),
        "panel_image_group_ids": panel_image_groups,
        "grouped_evaluation_image_group_ids": list(panel_image_groups),
        "panel_membership_preserved": True,
        "image_groups_preserved": True,
        "scientific_outcome_imputed": False,
        "excluded_only_where_trajectory_mathematically_required": True,
    }


def _stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"E4 authority is not a single-link regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise ValueError(f"E4 authority changed while being read: {path}")
    named = path.lstat()
    if named.st_dev != after.st_dev or named.st_ino != after.st_ino or path.is_symlink():
        raise ValueError(f"E4 authority path identity drifted: {path}")
    return b"".join(chunks)


def _read_bound_json(repo_root: Path, relative_path: str, expected_sha256: str) -> dict[str, Any]:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("E4 authority path must be repository-relative")
    encoded = _stable_regular_bytes(repo_root / relative)
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError(f"E4 authority SHA-256 drift: {relative_path}")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"E4 authority JSON is invalid: {relative_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"E4 authority must be a JSON object: {relative_path}")
    return payload


def load_e4_execution_registry(
    repo_root: Path,
    identity: Mapping[str, Any],
) -> tuple[E4ExecutionEvidence, ...]:
    """Load and cross-check the result-blind E4 execution registry."""

    if set(identity) != {"path", "sha256"}:
        raise ValueError("E4 execution registry identity schema is invalid")
    payload = _read_bound_json(repo_root, str(identity["path"]), str(identity["sha256"]))
    if not (
        payload.get("schema_version") == E4_EXECUTION_REGISTRY_SCHEMA
        and payload.get("registry_version") == E4_EXECUTION_REGISTRY_VERSION
        and payload.get("status") == "FROZEN_RESULT_BLIND"
        and payload.get("scope") == {
            "global_disposition_rule": True,
            "image_or_class_special_cases": False,
            "record_identity_is_evidence_not_a_code_branch": True,
        }
        and payload.get("result_blind_accounting") == {
            "COCO_GT_CONTENT_READ_COUNT": 0,
            "PERFORMANCE_RESULT_COUNT": 0,
            "REAL_SAM_CALL_COUNT": 0,
        }
    ):
        raise ValueError("E4 execution registry contract drift")

    authorities = payload.get("authorities")
    if not isinstance(authorities, dict) or set(authorities) != {
        "r2d_disposition_lock", "inference_outcome_policy_lock"
    }:
        raise ValueError("E4 execution registry authorities are invalid")
    r2d = _read_bound_json(repo_root, **{
        "relative_path": authorities["r2d_disposition_lock"]["path"],
        "expected_sha256": authorities["r2d_disposition_lock"]["sha256"],
    })
    _read_bound_json(repo_root, **{
        "relative_path": authorities["inference_outcome_policy_lock"]["path"],
        "expected_sha256": authorities["inference_outcome_policy_lock"]["sha256"],
    })
    historical = r2d.get("historical_state", {})

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("E4 execution registry records are invalid")
    records: list[E4ExecutionEvidence] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("E4 execution registry record is invalid")
        evidence = raw.get("execution_evidence", {})
        retry = raw.get("retry_evidence", {})
        record = E4TechnicalMissingnessRecord(
            semantic_state_id=str(raw.get("semantic_state_id", "")),
            image_group_id=str(raw.get("image_group_id", "")),
            class_id=int(raw.get("class_id", 0)),
            retry_count=int(raw.get("retry_count", -1)),
            maximum_retry=int(raw.get("maximum_retry", -1)),
            semantic_classification=str(raw.get("semantic_classification", "")),
            execution_disposition=str(raw.get("execution_disposition", "")),
            semantic_outcome_defined=bool(raw.get("semantic_outcome_defined", True)),
            trajectory_defined=bool(raw.get("trajectory_defined", True)),
            scientific_outcome_imputed=bool(raw.get("scientific_outcome_imputed", True)),
            historical_state_reconstructed=bool(raw.get("historical_state_reconstructed", True)),
            canonical_a0_created=bool(raw.get("canonical_A0_created", True)),
            additional_retry=bool(raw.get("additional_retry", True)),
        )
        if not (
            raw.get("semantic_resolution") == "UNRESOLVED_E4"
            and evidence.get("required_payload_kind")
            == "FAILURE_EVIDENCE_NOT_SCIENTIFIC_A0"
        ):
            raise ValueError("E4 execution registry semantics drift")
        frozen = E4ExecutionEvidence(
            record=record,
            semantic_resolution=str(raw["semantic_resolution"]),
            cache_key=str(evidence.get("cache_key", "")),
            cache_path=str(evidence.get("path", "")),
            cache_sha256=str(evidence.get("sha256", "")),
            retry_attempt_path=str(retry.get("attempt_path", "")),
            retry_attempt_sha256=str(retry.get("attempt_sha256", "")),
            retry_terminal_path=str(retry.get("terminal_path", "")),
            retry_terminal_sha256=str(retry.get("terminal_sha256", "")),
        )
        comparable = {
            "semantic_state_id": record.semantic_state_id,
            "image_id": record.image_group_id,
            "class_id": record.class_id,
            "semantic_classification": record.semantic_classification,
            "execution_disposition": record.execution_disposition,
            "semantic_outcome_defined": record.semantic_outcome_defined,
            "trajectory_defined": record.trajectory_defined,
            "scientific_outcome_imputed": record.scientific_outcome_imputed,
            "historical_state_reconstructed": record.historical_state_reconstructed,
            "canonical_A0_created": record.canonical_a0_created,
            "retry_count": record.retry_count,
            "maximum_retry": record.maximum_retry,
            "additional_retry": record.additional_retry,
        }
        if any(historical.get(key) != value for key, value in comparable.items()):
            raise ValueError("E4 registry record conflicts with frozen R2D lock")
        cache_payload = _read_bound_json(repo_root, frozen.cache_path, frozen.cache_sha256)
        cache_body = cache_payload.get("payload", {})
        if not (
            cache_payload.get("cache_key") == frozen.cache_key
            and cache_body.get("candidate") is None
            and cache_body.get("no_result") is None
            and isinstance(cache_body.get("failure"), dict)
        ):
            raise ValueError("E4 execution evidence was converted into a scientific A0")
        _read_bound_json(repo_root, frozen.retry_attempt_path, frozen.retry_attempt_sha256)
        _read_bound_json(repo_root, frozen.retry_terminal_path, frozen.retry_terminal_sha256)
        records.append(frozen)

    by_state = {entry.record.semantic_state_id: entry for entry in records}
    if len(by_state) != len(records):
        raise ValueError("E4 execution registry duplicates a semantic state")
    gate = evaluate_e4_completeness(tuple(entry.record for entry in records))
    if gate["completeness_gate"] != "PASS":
        raise ValueError(INFERENCE_COMPLETENESS_FAILED)
    return tuple(by_state[key] for key in sorted(by_state))


def partition_executable_states(
    states: Sequence[Any],
    records: Sequence[E4ExecutionEvidence],
) -> tuple[tuple[Any, ...], tuple[E4ExecutionEvidence, ...]]:
    """Remove only registered E4 trajectories while retaining every other state."""

    state_by_id = {state.semantic_state_id: state for state in states}
    if len(state_by_id) != len(states):
        raise ValueError("semantic state sequence contains duplicate identities")
    unavailable = {entry.record.semantic_state_id: entry for entry in records}
    for state_id, entry in unavailable.items():
        state = state_by_id.get(state_id)
        if state is None or not (
            state.image.canonical_image_id == entry.record.image_group_id
            and state.class_id == entry.record.class_id
        ):
            raise ValueError("E4 execution record is absent from or conflicts with state grid")
    executable = tuple(state for state in states if state.semantic_state_id not in unavailable)
    return executable, tuple(unavailable[key] for key in sorted(unavailable))


def dry_run_e4_cursor_transition(
    states: Sequence[Any],
    records: Sequence[E4ExecutionEvidence],
    semantic_state_id: str,
) -> dict[str, Any]:
    """Prove a cursor can cross one E4 state without inference or imputation."""

    executable, unavailable = partition_executable_states(states, records)
    unavailable_by_id = {entry.record.semantic_state_id: entry for entry in unavailable}
    if semantic_state_id not in unavailable_by_id:
        raise ValueError("dry-run cursor is not a frozen E4 state")
    ordered_ids = [state.semantic_state_id for state in states]
    position = ordered_ids.index(semantic_state_id)
    next_state = next(
        (state for state in states[position + 1:] if state.semantic_state_id not in unavailable_by_id),
        None,
    )
    record = unavailable_by_id[semantic_state_id].record
    same_image = [
        state for state in executable
        if state.image.canonical_image_id == record.image_group_id
    ]
    gate = evaluate_e4_completeness(tuple(entry.record for entry in unavailable))
    return {
        "cursor_before": semantic_state_id,
        "cursor_at_disposition": E4_EXECUTION_DISPOSITION,
        "cursor_after": None if next_state is None else next_state.semantic_state_id,
        "sam_calls_for_state": 0,
        "trajectory_created_for_state": False,
        "a0_created_for_state": False,
        "action_outputs_created_for_state": 0,
        "technical_missingness_count": gate["technical_missingness_count"],
        "completeness_gate": gate["completeness_gate"],
        "same_image_other_state_count": len(same_image),
        "same_image_other_state_ids": [state.semantic_state_id for state in same_image],
        "scientific_outcome_imputed": False,
    }
