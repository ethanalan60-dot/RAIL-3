"""Label-free joins over frozen M06-E canonical and trajectory caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rail3.cache import CandidateCache
from rail3.cache.candidates import CandidateGenerationResult, decode_binary_mask_array
from rail3.contracts import stable_id
from rail3.regions.atomic import CandidateMaskInput
from rail3.sam.sam31_backend import official_sam31_model_spec
from rail3.sam.voc_actions import (
    ACTION_CODES, PROTOCOL_SHA256, ActionOutcome, TrajectoryCache,
)
from rail3.sam.voc_protocol import canonical_voc_prompts, voc_cache_identity


@dataclass(frozen=True)
class TrajectorySource:
    origin: str
    expected_images: int
    report_path: Path
    cache_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory_sources(
    *, manifest: Mapping[str, Any], sources: Sequence[TrajectorySource],
) -> tuple[dict[str, tuple[dict[str, Any], TrajectoryCache, str]], dict[str, Any]]:
    """Bind every manifest asset to exactly one complete label-free source."""

    expected_by_origin: dict[str, set[str]] = {}
    for record in manifest["records"]:
        expected_by_origin.setdefault(str(record["origin"]), set()).add(str(record["asset_id"]))
    if set(expected_by_origin) != {source.origin for source in sources}:
        raise ValueError("M06-E trajectory source origins differ from the S_MAX manifest")
    index: dict[str, tuple[dict[str, Any], TrajectoryCache, str]] = {}
    identities = {}
    for source in sources:
        report = json.loads(source.report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "PASS"
            or report.get("run_status") != "COMPLETE"
            or report.get("protocol_sha256") != PROTOCOL_SHA256
            or int(report.get("completed_image_count", -1)) != source.expected_images
        ):
            raise ValueError(f"M06-E trajectory source is not complete: {source.origin}")
        records = list(report.get("records", ()))
        asset_ids = {str(item["asset_id"]) for item in records}
        if len(records) != source.expected_images or asset_ids != expected_by_origin[source.origin]:
            raise ValueError(f"M06-E trajectory asset set mismatch: {source.origin}")
        cache = TrajectoryCache(source.cache_root)
        report_sha = sha256_file(source.report_path)
        identities[source.origin] = {
            "report_path": source.report_path.as_posix(),
            "report_sha256": report_sha,
            "cache_root": source.cache_root.as_posix(),
            "image_count": len(records),
            "action_record_count": int(report["accounted_action_records"]),
        }
        for record in records:
            asset_id = str(record["asset_id"])
            if asset_id in index:
                raise ValueError("M06-E trajectory sources repeat an asset")
            index[asset_id] = (dict(record), cache, source.origin)
    if len(index) != int(manifest["count"]):
        raise ValueError("M06-E trajectory source join is incomplete")
    return index, identities


def state_by_class(image_record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    states = {int(item["class_id"]): dict(item) for item in image_record["class_states"]}
    if set(states) != set(range(1, 21)):
        raise ValueError("M06-E trajectory image must contain classes 1--20")
    return states


def _candidate_status(result: CandidateGenerationResult) -> tuple[str, str | None]:
    if result.candidate is not None:
        return "result", None
    if result.no_result is not None:
        return "no_result", result.no_result.reason_code
    if result.failure is not None:
        return "failure", result.failure.reason_code
    raise ValueError("canonical candidate result has no outcome")


def _canonical_runtime(result: CandidateGenerationResult) -> float:
    if result.candidate is not None:
        value = result.candidate.runtime_seconds
    elif result.no_result is not None and result.no_result.telemetry is not None:
        value = result.no_result.telemetry.runtime_seconds
    elif result.failure is not None and result.failure.telemetry is not None:
        value = result.failure.telemetry.runtime_seconds
    else:
        value = 0.0
    return max(0.0, float(value or 0.0))


def canonical_observation(
    *, item: Mapping[str, Any], class_state: Mapping[str, Any],
    baseline_cache: CandidateCache, class_id: int,
) -> tuple[CandidateGenerationResult, list[CandidateMaskInput], dict[str, Any], str]:
    prompt, _ = next(
        pair for pair in canonical_voc_prompts() if pair[1].class_id == class_id
    )
    identity = voc_cache_identity(
        dict(item), prompt, model_spec_id=official_sam31_model_spec().model_spec_id
    )
    if class_state.get("canonical_cache_key") != identity.cache_key:
        raise ValueError("M06-E canonical cache identity differs from trajectory state")
    result = baseline_cache.load(identity)
    actions = list(class_state["actions"])
    if tuple(action.get("action_code") for action in actions) != ACTION_CODES:
        raise ValueError("M06-E class state must contain ordered A1--A6")
    state_id = str(actions[0]["state_id"])
    action_id = stable_id("canonical_action", {
        "protocol_sha256": PROTOCOL_SHA256,
        "state_id": state_id,
        "canonical_cache_key": identity.cache_key,
    })
    candidates = [CandidateMaskInput.create(
        candidate_id=candidate.candidate_id,
        action_id=action_id,
        action_code="A0",
        class_id=class_id,
        prompt_source="CANONICAL_TEXT",
        mask=decode_binary_mask_array(candidate.mask_rle, candidate.width, candidate.height),
    ) for candidate in result.all_candidates]
    status, reason = _candidate_status(result)
    lineage = {
        "action_code": "A0", "action_id": action_id, "cache_key": identity.cache_key,
        "feasible": True, "outcome": status, "reason_code": reason,
        "candidate_ids": [candidate.candidate_id for candidate in result.all_candidates],
        "prompt_source": "CANONICAL_TEXT",
        "prompt": {"kind": prompt.kind.value, "text": prompt.text, "prompt_id": prompt.prompt_id},
        "cost_seconds": _canonical_runtime(result),
    }
    return result, candidates, lineage, action_id


def action_observations(
    *, class_state: Mapping[str, Any], cache: TrajectoryCache, class_id: int,
) -> tuple[list[dict[str, Any]], list[ActionOutcome]]:
    references = [dict(item) for item in class_state["actions"]]
    if tuple(item.get("action_code") for item in references) != ACTION_CODES:
        raise ValueError("M06-E action references are not ordered A1--A6")
    outcomes = [cache.load_key(str(item["cache_key"])) for item in references]
    for reference, outcome in zip(references, outcomes):
        if (
            outcome.action_id != reference.get("action_id")
            or outcome.state_id != reference.get("state_id")
            or outcome.action_code != reference.get("action_code")
            or outcome.class_id != class_id
        ):
            raise ValueError("M06-E trajectory reference/cache mismatch")
    return references, outcomes


def causal_base_inputs(
    *, canonical_candidates: Sequence[CandidateMaskInput],
    references: Sequence[Mapping[str, Any]], outcomes: Sequence[ActionOutcome],
    canonical_lineage: Mapping[str, Any], class_id: int,
) -> tuple[list[CandidateMaskInput], list[dict[str, Any]]]:
    candidates = list(canonical_candidates)
    lineage = [dict(canonical_lineage)]
    for reference, outcome in zip(references[:2], outcomes[:2]):
        for candidate in outcome.candidates:
            candidates.append(CandidateMaskInput.create(
                candidate_id=candidate.candidate_id,
                action_id=outcome.action_id,
                action_code=outcome.action_code,
                class_id=class_id,
                prompt_source=outcome.action_code,
                mask=decode_binary_mask_array(
                    candidate.mask_rle, candidate.width, candidate.height
                ),
            ))
        lineage.append({
            "action_code": outcome.action_code,
            "action_id": outcome.action_id,
            "cache_key": reference["cache_key"],
            "feasible": outcome.feasible,
            "outcome": outcome.outcome,
            "reason_code": outcome.reason_code,
            "candidate_ids": [candidate.candidate_id for candidate in outcome.candidates],
            "source_candidate_id": outcome.source_candidate_id,
            "prompt_source": outcome.action_code,
            "prompt": outcome.prompt,
            "transform": outcome.transform,
            "cost_seconds": max(0.0, float(reference.get("action_cost_seconds") or 0.0)),
        })
    if tuple(item["action_code"] for item in lineage) != ("A0", "A1", "A2"):
        raise AssertionError("M06-E causal base assembly drifted")
    return candidates, lineage


def adaptive_feature_descriptors(
    *, canonical_action_id: str, references: Sequence[Mapping[str, Any]],
    outcomes: Sequence[ActionOutcome], action_families: Mapping[str, str],
) -> list[dict[str, Any]]:
    result = [{
        "action_code": "STOP", "action_id": canonical_action_id,
        "action_family": "STOP_A0_CANONICAL", "feasible": True,
        "source_candidate_id": None,
        "prompt_lineage": {"kind": "stop", "base_action_code": "A0"},
        "incremental_cost_seconds": 0.0,
    }]
    for reference, outcome in zip(references[2:], outcomes[2:]):
        result.append({
            "action_code": outcome.action_code,
            "action_id": outcome.action_id,
            "action_family": str(action_families[outcome.action_code]),
            "feasible": outcome.feasible,
            "source_candidate_id": outcome.source_candidate_id,
            "prompt_lineage": {
                "prompt": outcome.prompt,
                "transform": outcome.transform,
                "source_candidate_id": outcome.source_candidate_id,
            },
            "incremental_cost_seconds": max(
                0.0, float(reference.get("action_cost_seconds") or 0.0)
            ),
        })
    if tuple(item["action_code"] for item in result) != ("STOP", "A3", "A4", "A5", "A6"):
        raise AssertionError("M06-E adaptive descriptor assembly drifted")
    return result


def union_canonical(result: CandidateGenerationResult, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in result.all_candidates:
        decoded = decode_binary_mask_array(candidate.mask_rle, candidate.width, candidate.height)
        if decoded.shape != shape:
            raise ValueError("M06-E canonical output geometry drift")
        mask |= decoded
    return mask


def union_action(outcome: ActionOutcome, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in outcome.candidates:
        decoded = decode_binary_mask_array(candidate.mask_rle, candidate.width, candidate.height)
        if decoded.shape != shape:
            raise ValueError("M06-E adaptive output geometry drift")
        mask |= decoded
    return mask
