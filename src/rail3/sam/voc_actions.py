"""Frozen, label-free VOC prompt actions and content-addressed trajectory cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rail3.cache.atomic_io import atomic_create_bytes
from rail3.cache.candidates import (
    CandidateGenerationResult,
    CandidateRecord,
    decode_binary_mask_array,
    encode_binary_mask_array,
)
from rail3.contracts import canonical_json_bytes, stable_id
from rail3.data.voc_taxonomy import VOC_FOREGROUND_CLASSES
from rail3.sam.protocols import PromptKind, PromptRequest
from rail3.sam.sam31_backend import PromptMetadata


PROTOCOL_PATH = Path("configs/experiments/voc_m04_m05_action_protocol_v1.json")
PROTOCOL_SHA256 = "c558595b6be151307f04ae189ff0b8c74f1de2adbf7e0c046db57048fd12f985"
ACTION_CACHE_SCHEMA = "rail3.voc-trajectory-action-cache.v1"
ACTION_CODES = ("A1", "A2", "A3", "A4", "A5", "A6")
MACHINE_TELEMETRY_FIELDS = frozenset({
    "runtime_seconds", "wall_seconds", "action_cost_seconds",
    "serialization_seconds", "peak_allocated_vram_bytes",
    "peak_reserved_vram_bytes", "physical_gpu_index", "logical_device",
    "model_build_count", "checkpoint_load_count", "prompt_call_count",
    "image_session_initialization_count",
})


class TrajectoryCacheError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def load_action_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("VOC action protocol content hash drift")
    payload = json.loads(encoded)
    if payload.get("schema_version") != "rail3.voc-action-protocol.v1":
        raise ValueError("VOC action protocol schema drift")
    if [item.get("action_code") for item in payload.get("actions", ())] != [
        "A0", *ACTION_CODES
    ]:
        raise ValueError("VOC action protocol action ordering drift")
    return payload


def normalized_voc_texts(protocol: dict[str, Any] | None = None) -> dict[int, str]:
    protocol = load_action_protocol() if protocol is None else protocol
    mapping = protocol["normalization_mapping"]
    result: dict[int, str] = {}
    for item in VOC_FOREGROUND_CLASSES:
        normalized_source = " ".join(item.source_name.strip().lower().split())
        if normalized_source not in mapping:
            raise ValueError(f"normalization mapping omits {item.source_name}")
        value = " ".join(str(mapping[normalized_source]).strip().lower().split())
        if not value:
            raise ValueError("normalized prompt cannot be empty")
        result[item.class_id] = value
    if len(mapping) != len(VOC_FOREGROUND_CLASSES):
        raise ValueError("normalization mapping contains unexpected entries")
    return result


def select_source_candidate(result: CandidateGenerationResult) -> CandidateRecord | None:
    candidates = result.all_candidates
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item.model_score, item.candidate_id))[0]


def expanded_crop_box(
    bbox: tuple[int, int, int, int], width: int, height: int, ratio: float = 1.25
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or ratio < 1:
        raise ValueError("source bbox or crop expansion ratio is invalid")
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    expanded_width = (x1 - x0) * ratio
    expanded_height = (y1 - y0) * ratio
    crop_x0 = max(0, math.floor(center_x - expanded_width / 2))
    crop_y0 = max(0, math.floor(center_y - expanded_height / 2))
    crop_x1 = min(width, math.ceil(center_x + expanded_width / 2))
    crop_y1 = min(height, math.ceil(center_y + expanded_height / 2))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError("expanded crop is empty after clipping")
    return crop_x0, crop_y0, crop_x1, crop_y1


def deepest_interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Return x,y at maximum four-connected erosion depth, row-major tie-break."""

    current = np.asarray(mask, dtype=bool)
    if current.ndim != 2 or not bool(current.any()):
        raise ValueError("point action requires a nonempty two-dimensional mask")
    while True:
        padded = np.pad(current, 1, constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        if not bool(eroded.any()):
            y, x = (int(value) for value in np.argwhere(current)[0])
            return x, y
        current = eroded


def horizontal_flip_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("horizontal flip requires a two-dimensional mask")
    return np.ascontiguousarray(array[:, ::-1])


def restore_crop_mask(
    crop_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    array = np.asarray(crop_mask, dtype=bool)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop restoration window is outside original geometry")
    if array.shape != (y1 - y0, x1 - x0):
        raise ValueError("crop output geometry differs from restoration window")
    restored = np.zeros((height, width), dtype=bool)
    restored[y0:y1, x0:x1] = array
    return restored


@dataclass(frozen=True)
class ActionPlan:
    state_id: str
    action_id: str
    action_code: str
    class_id: int
    canonical_text: str
    prompt: PromptRequest | None
    feasible: bool
    reason_code: str | None
    source_candidate_id: str | None
    source_upstream_object_id: int | None
    transform: dict[str, Any]

    def __post_init__(self) -> None:
        if self.action_code not in ACTION_CODES:
            raise ValueError("unknown action code")
        if self.feasible != (self.reason_code is None):
            raise ValueError("action feasibility and reason code disagree")
        if self.feasible != (self.prompt is not None):
            raise ValueError("feasible action must have exactly one prompt")
        expected = stable_id("voc_action", self.identity_payload())
        if self.action_id != expected:
            raise ValueError("action ID differs from deployment-visible identity")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "protocol_sha256": PROTOCOL_SHA256,
            "state_id": self.state_id,
            "action_code": self.action_code,
            "class_id": self.class_id,
            "canonical_text": self.canonical_text,
            "prompt": None if self.prompt is None else prompt_payload(self.prompt),
            "feasible": self.feasible,
            "reason_code": self.reason_code,
            "source_candidate_id": self.source_candidate_id,
            "source_upstream_object_id": self.source_upstream_object_id,
            "transform": self.transform,
        }


def prompt_payload(prompt: PromptRequest) -> dict[str, Any]:
    return {
        "prompt_id": prompt.prompt_id,
        "kind": prompt.kind.value,
        "text": prompt.text,
        "points": [list(point) for point in prompt.points],
        "box": None if prompt.box is None else list(prompt.box),
        "object_id": prompt.object_id,
        "seed": prompt.seed,
        "schema_version": prompt.schema_version,
    }


def _make_plan(
    *,
    state_id: str,
    action_code: str,
    class_id: int,
    canonical_text: str,
    prompt: PromptRequest | None,
    feasible: bool,
    reason_code: str | None,
    source_candidate: CandidateRecord | None,
    transform: dict[str, Any],
) -> ActionPlan:
    values = {
        "protocol_sha256": PROTOCOL_SHA256,
        "state_id": state_id,
        "action_code": action_code,
        "class_id": class_id,
        "canonical_text": canonical_text,
        "prompt": None if prompt is None else prompt_payload(prompt),
        "feasible": feasible,
        "reason_code": reason_code,
        "source_candidate_id": None if source_candidate is None else source_candidate.candidate_id,
        "source_upstream_object_id": None if source_candidate is None else source_candidate.upstream_object_id,
        "transform": transform,
    }
    return ActionPlan(
        state_id=state_id,
        action_id=stable_id("voc_action", values),
        action_code=action_code,
        class_id=class_id,
        canonical_text=canonical_text,
        prompt=prompt,
        feasible=feasible,
        reason_code=reason_code,
        source_candidate_id=values["source_candidate_id"],
        source_upstream_object_id=values["source_upstream_object_id"],
        transform=transform,
    )


def build_action_plans(
    item: dict[str, Any],
    *,
    class_id: int,
    canonical_result: CandidateGenerationResult,
    canonical_cache_key: str,
) -> tuple[ActionPlan, ...]:
    protocol = load_action_protocol()
    class_item = next(item for item in VOC_FOREGROUND_CLASSES if item.class_id == class_id)
    normalized = normalized_voc_texts(protocol)[class_id]
    source = select_source_candidate(canonical_result)
    source_ids = sorted(candidate.candidate_id for candidate in canonical_result.all_candidates)
    state_id = stable_id("voc_trajectory_state", {
        "protocol_sha256": PROTOCOL_SHA256,
        "asset_id": item["asset_id"],
        "image_sha256": item["image_sha256"],
        "width": int(item["width"]),
        "height": int(item["height"]),
        "class_id": class_id,
        "canonical_cache_key": canonical_cache_key,
        "canonical_outcome": (
            "result" if canonical_result.candidate is not None
            else "no_result" if canonical_result.no_result is not None else "failure"
        ),
        "canonical_candidate_ids": source_ids,
    })

    def text_prompt(code: str, text: str) -> PromptRequest:
        return PromptRequest.create(
            kind=PromptKind.TEXT,
            text=text,
            object_id=f"{code.lower()}-{state_id}",
            seed=int(protocol["seed"]),
        )

    plans = [
        _make_plan(
            state_id=state_id,
            action_code="A1",
            class_id=class_id,
            canonical_text=class_item.canonical_prompt,
            prompt=text_prompt("A1", normalized),
            feasible=True,
            reason_code=None,
            source_candidate=None,
            transform={"kind": "identity", "text": normalized},
        ),
        _make_plan(
            state_id=state_id,
            action_code="A2",
            class_id=class_id,
            canonical_text=class_item.canonical_prompt,
            prompt=text_prompt("A2", f"a photo of a {normalized}"),
            feasible=True,
            reason_code=None,
            source_candidate=None,
            transform={"kind": "identity", "text": f"a photo of a {normalized}"},
        ),
        _make_plan(
            state_id=state_id,
            action_code="A3",
            class_id=class_id,
            canonical_text=class_item.canonical_prompt,
            prompt=text_prompt("A3", class_item.canonical_prompt),
            feasible=True,
            reason_code=None,
            source_candidate=None,
            transform={"kind": "horizontal_flip", "restore": "reverse_columns"},
        ),
    ]
    if source is None:
        for code in ("A4", "A5", "A6"):
            plans.append(_make_plan(
                state_id=state_id,
                action_code=code,
                class_id=class_id,
                canonical_text=class_item.canonical_prompt,
                prompt=None,
                feasible=False,
                reason_code="NO_CANONICAL_CANDIDATE",
                source_candidate=None,
                transform={"kind": "unavailable"},
            ))
        return tuple(plans)

    crop_box = expanded_crop_box(
        source.bbox_xyxy, int(item["width"]), int(item["height"]), ratio=1.25
    )
    plans.append(_make_plan(
        state_id=state_id,
        action_code="A4",
        class_id=class_id,
        canonical_text=class_item.canonical_prompt,
        prompt=text_prompt("A4", class_item.canonical_prompt),
        feasible=True,
        reason_code=None,
        source_candidate=source,
        transform={"kind": "crop", "crop_box_xyxy": list(crop_box), "expansion_ratio": 1.25},
    ))
    box_prompt = PromptRequest.create(
        kind=PromptKind.BOX,
        box=source.bbox_xyxy,
        object_id=f"a5-{state_id}-{source.candidate_id}",
        seed=int(protocol["seed"]),
    )
    plans.append(_make_plan(
        state_id=state_id,
        action_code="A5",
        class_id=class_id,
        canonical_text=class_item.canonical_prompt,
        prompt=box_prompt,
        feasible=True,
        reason_code=None,
        source_candidate=source,
        transform={"kind": "candidate_box", "box_xyxy": list(source.bbox_xyxy)},
    ))
    if source.upstream_object_id is None:
        plans.append(_make_plan(
            state_id=state_id,
            action_code="A6",
            class_id=class_id,
            canonical_text=class_item.canonical_prompt,
            prompt=None,
            feasible=False,
            reason_code="NO_UPSTREAM_OBJECT_ID",
            source_candidate=source,
            transform={"kind": "unavailable"},
        ))
    else:
        point = deepest_interior_point(
            decode_binary_mask_array(source.mask_rle, source.width, source.height)
        )
        point_prompt = PromptRequest.create(
            kind=PromptKind.POINTS,
            points=((point[0], point[1], 1),),
            object_id=f"a6-{state_id}-{source.candidate_id}",
            seed=int(protocol["seed"]),
        )
        plans.append(_make_plan(
            state_id=state_id,
            action_code="A6",
            class_id=class_id,
            canonical_text=class_item.canonical_prompt,
            prompt=point_prompt,
            feasible=True,
            reason_code=None,
            source_candidate=source,
            transform={
                "kind": "candidate_point",
                "point_xy": list(point),
                "point_rule": "maximum_four_connected_erosion_depth_row_major",
            },
        ))
    return tuple(plans)


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    action_id: str
    candidate_index: int
    width: int
    height: int
    mask_rle: tuple[int, ...]
    mask_sha256: str
    bbox_xyxy: tuple[int, int, int, int]
    model_score: float
    upstream_object_id: int | None
    upstream_bbox_xywh: tuple[float, float, float, float] | None

    @classmethod
    def from_candidate_record(
        cls,
        candidate: CandidateRecord,
        *,
        plan: ActionPlan,
        original_width: int,
        original_height: int,
    ) -> "ActionCandidate":
        mask = decode_binary_mask_array(candidate.mask_rle, candidate.width, candidate.height)
        kind = plan.transform["kind"]
        if kind == "horizontal_flip":
            mask = horizontal_flip_mask(mask)
        elif kind == "crop":
            mask = restore_crop_mask(
                mask,
                tuple(int(value) for value in plan.transform["crop_box_xyxy"]),
                original_width,
                original_height,
            )
        elif kind not in {"identity", "candidate_box", "candidate_point"}:
            raise ValueError("action candidate has unsupported coordinate transform")
        if mask.shape != (original_height, original_width) or not bool(mask.any()):
            raise ValueError("restored action candidate is empty or has wrong geometry")
        rle = encode_binary_mask_array(mask)
        mask_sha256 = hashlib.sha256(canonical_json_bytes({
            "width": original_width,
            "height": original_height,
            "rle": rle,
        })).hexdigest()
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        candidate_id = stable_id("action_candidate", {
            "action_id": plan.action_id,
            "candidate_index": candidate.candidate_index,
            "mask_sha256": mask_sha256,
            "model_score": candidate.model_score,
            "upstream_object_id": candidate.upstream_object_id,
        })
        return cls(
            candidate_id,
            plan.action_id,
            candidate.candidate_index,
            original_width,
            original_height,
            rle,
            mask_sha256,
            bbox,
            candidate.model_score,
            candidate.upstream_object_id,
            candidate.upstream_bbox_xywh,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
            "candidate_index": self.candidate_index,
            "width": self.width,
            "height": self.height,
            "mask_rle": list(self.mask_rle),
            "mask_sha256": self.mask_sha256,
            "bbox_xyxy": list(self.bbox_xyxy),
            "model_score": self.model_score,
            "upstream_object_id": self.upstream_object_id,
            "upstream_bbox_xywh": (
                None if self.upstream_bbox_xywh is None else list(self.upstream_bbox_xywh)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionCandidate":
        candidate = cls(
            candidate_id=payload["candidate_id"],
            action_id=payload["action_id"],
            candidate_index=int(payload["candidate_index"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            mask_rle=tuple(int(value) for value in payload["mask_rle"]),
            mask_sha256=payload["mask_sha256"],
            bbox_xyxy=tuple(int(value) for value in payload["bbox_xyxy"]),
            model_score=float(payload["model_score"]),
            upstream_object_id=(
                None if payload.get("upstream_object_id") is None
                else int(payload["upstream_object_id"])
            ),
            upstream_bbox_xywh=(
                None if payload.get("upstream_bbox_xywh") is None
                else tuple(float(value) for value in payload["upstream_bbox_xywh"])
            ),
        )
        mask = decode_binary_mask_array(candidate.mask_rle, candidate.width, candidate.height)
        expected_hash = hashlib.sha256(canonical_json_bytes({
            "width": candidate.width,
            "height": candidate.height,
            "rle": candidate.mask_rle,
        })).hexdigest()
        if expected_hash != candidate.mask_sha256:
            raise ValueError("action candidate mask hash mismatch")
        ys, xs = np.nonzero(mask)
        expected_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if candidate.bbox_xyxy != expected_bbox:
            raise ValueError("action candidate bbox mismatch")
        expected_id = stable_id("action_candidate", {
            "action_id": candidate.action_id,
            "candidate_index": candidate.candidate_index,
            "mask_sha256": candidate.mask_sha256,
            "model_score": candidate.model_score,
            "upstream_object_id": candidate.upstream_object_id,
        })
        if expected_id != candidate.candidate_id:
            raise ValueError("action candidate ID mismatch")
        return candidate


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    state_id: str
    action_code: str
    asset_id: str
    image_sha256: str
    class_id: int
    feasible: bool
    reason_code: str | None
    outcome: str
    source_candidate_id: str | None
    prompt: dict[str, Any] | None
    transform: dict[str, Any]
    candidates: tuple[ActionCandidate, ...]
    telemetry: dict[str, Any]
    schema_version: str = "rail3.voc-action-outcome.v1"

    def __post_init__(self) -> None:
        if self.action_code not in ACTION_CODES or self.outcome not in {
            "result", "no_result", "failure", "infeasible"
        }:
            raise ValueError("action outcome code/status is invalid")
        if self.feasible == (self.outcome == "infeasible"):
            raise ValueError("action outcome feasibility/status disagree")
        if (self.outcome == "result") != bool(self.candidates):
            raise ValueError("result status must contain candidates and only result may contain them")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "state_id": self.state_id,
            "action_code": self.action_code,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "class_id": self.class_id,
            "feasible": self.feasible,
            "reason_code": self.reason_code,
            "outcome": self.outcome,
            "source_candidate_id": self.source_candidate_id,
            "prompt": self.prompt,
            "transform": self.transform,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "telemetry": self.telemetry,
        }

    def semantic_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["telemetry"] = {
            key: value for key, value in self.telemetry.items()
            if key not in MACHINE_TELEMETRY_FIELDS
        }
        return payload

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.semantic_dict())).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionOutcome":
        if payload.get("schema_version") != "rail3.voc-action-outcome.v1":
            raise ValueError("action outcome schema mismatch")
        return cls(
            action_id=payload["action_id"],
            state_id=payload["state_id"],
            action_code=payload["action_code"],
            asset_id=payload["asset_id"],
            image_sha256=payload["image_sha256"],
            class_id=int(payload["class_id"]),
            feasible=bool(payload["feasible"]),
            reason_code=payload.get("reason_code"),
            outcome=payload["outcome"],
            source_candidate_id=payload.get("source_candidate_id"),
            prompt=payload.get("prompt"),
            transform=dict(payload["transform"]),
            candidates=tuple(ActionCandidate.from_dict(item) for item in payload["candidates"]),
            telemetry=dict(payload["telemetry"]),
            schema_version=payload["schema_version"],
        )


def outcome_from_generation(
    *,
    plan: ActionPlan,
    item: dict[str, Any],
    result: CandidateGenerationResult | None,
    telemetry: dict[str, Any],
) -> ActionOutcome:
    if not plan.feasible:
        return ActionOutcome(
            action_id=plan.action_id,
            state_id=plan.state_id,
            action_code=plan.action_code,
            asset_id=item["asset_id"],
            image_sha256=item["image_sha256"],
            class_id=plan.class_id,
            feasible=False,
            reason_code=plan.reason_code,
            outcome="infeasible",
            source_candidate_id=plan.source_candidate_id,
            prompt=None,
            transform=plan.transform,
            candidates=(),
            telemetry=telemetry,
        )
    if result is None:
        raise ValueError("feasible action outcome requires generation result")
    if result.failure is not None:
        status = "failure"
        reason = result.failure.reason_code
    elif result.no_result is not None:
        status = "no_result"
        reason = result.no_result.reason_code
    else:
        status = "result"
        reason = None
    candidates = tuple(
        ActionCandidate.from_candidate_record(
            candidate,
            plan=plan,
            original_width=int(item["width"]),
            original_height=int(item["height"]),
        )
        for candidate in result.all_candidates
    )
    return ActionOutcome(
        action_id=plan.action_id,
        state_id=plan.state_id,
        action_code=plan.action_code,
        asset_id=item["asset_id"],
        image_sha256=item["image_sha256"],
        class_id=plan.class_id,
        feasible=True,
        reason_code=reason,
        outcome=status,
        source_candidate_id=plan.source_candidate_id,
        prompt=prompt_payload(plan.prompt) if plan.prompt is not None else None,
        transform=plan.transform,
        candidates=candidates,
        telemetry=telemetry,
    )


@dataclass(frozen=True)
class ActionCacheIdentity:
    cache_key: str
    model_spec_id: str
    asset_id: str
    image_sha256: str
    state_id: str
    action_id: str

    @classmethod
    def create(
        cls, *, model_spec_id: str, item: dict[str, Any], plan: ActionPlan
    ) -> "ActionCacheIdentity":
        payload = {
            "schema_version": ACTION_CACHE_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_spec_id": model_spec_id,
            "asset_id": item["asset_id"],
            "image_sha256": item["image_sha256"],
            "state_id": plan.state_id,
            "action_id": plan.action_id,
        }
        return cls(
            stable_id("trajectory_cache", payload),
            model_spec_id,
            item["asset_id"],
            item["image_sha256"],
            plan.state_id,
            plan.action_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_spec_id": self.model_spec_id,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "state_id": self.state_id,
            "action_id": self.action_id,
        }


class TrajectoryCache:
    def __init__(self, root: Path) -> None:
        workspace = Path.cwd().resolve()
        resolved = root.resolve()
        try:
            self.root = resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("trajectory cache root escapes repository workspace") from exc

    def path_for(self, identity: ActionCacheIdentity) -> Path:
        digest = identity.cache_key.removeprefix("trajectory_cache_")
        return self.root / digest[:2] / f"{identity.cache_key}.json"

    def exists(self, identity: ActionCacheIdentity) -> bool:
        return self.path_for(identity).is_file()

    def store(self, identity: ActionCacheIdentity, outcome: ActionOutcome) -> Path:
        self._validate_identity(identity, outcome)
        payload = outcome.to_dict()
        envelope = {
            "schema_version": ACTION_CACHE_SCHEMA,
            "cache_key": identity.cache_key,
            "identity": identity.to_dict(),
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            "semantic_sha256": outcome.semantic_sha256,
        }
        encoded = canonical_json_bytes(envelope) + b"\n"
        destination = self.path_for(identity)
        try:
            atomic_create_bytes(destination, encoded)
        except FileExistsError:
            existing = self.load_envelope(identity)
            if existing != envelope:
                raise TrajectoryCacheError(
                    "IMMUTABLE_TRAJECTORY_CACHE_CONFLICT",
                    "existing trajectory cache object differs from new payload",
                )
        return destination

    def load_envelope(self, identity: ActionCacheIdentity) -> dict[str, Any]:
        try:
            envelope = json.loads(self.path_for(identity).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrajectoryCacheError(
                "TRAJECTORY_CACHE_JSON_INVALID", "trajectory cache object is unreadable"
            ) from exc
        if envelope.get("schema_version") != ACTION_CACHE_SCHEMA:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_SCHEMA_MISMATCH", "cache schema drift")
        if envelope.get("cache_key") != identity.cache_key:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_KEY_MISMATCH", "cache key drift")
        if envelope.get("identity") != identity.to_dict():
            raise TrajectoryCacheError("TRAJECTORY_CACHE_IDENTITY_MISMATCH", "identity drift")
        payload = envelope.get("payload")
        actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if envelope.get("payload_sha256") != actual:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_PAYLOAD_HASH_MISMATCH", "hash mismatch")
        return envelope

    def load(self, identity: ActionCacheIdentity) -> ActionOutcome:
        envelope = self.load_envelope(identity)
        try:
            outcome = ActionOutcome.from_dict(envelope["payload"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrajectoryCacheError(
                "TRAJECTORY_CACHE_PAYLOAD_INVALID", "action outcome violates schema"
            ) from exc
        self._validate_identity(identity, outcome)
        if envelope.get("semantic_sha256") != outcome.semantic_sha256:
            raise TrajectoryCacheError(
                "TRAJECTORY_CACHE_SEMANTIC_HASH_MISMATCH", "semantic payload hash drift"
            )
        return outcome

    def load_key(self, cache_key: str) -> ActionOutcome:
        if re.fullmatch(r"trajectory_cache_[0-9a-f]{64}", cache_key) is None:
            raise TrajectoryCacheError("TRAJECTORY_CACHE_KEY_INVALID", "cache key format is invalid")
        digest = cache_key.removeprefix("trajectory_cache_")
        path = self.root / digest[:2] / f"{cache_key}.json"
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            identity_payload = envelope["identity"]
            identity = ActionCacheIdentity(
                cache_key=cache_key,
                model_spec_id=identity_payload["model_spec_id"],
                asset_id=identity_payload["asset_id"],
                image_sha256=identity_payload["image_sha256"],
                state_id=identity_payload["state_id"],
                action_id=identity_payload["action_id"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise TrajectoryCacheError(
                "TRAJECTORY_CACHE_JSON_INVALID", "trajectory cache object is unreadable"
            ) from exc
        return self.load(identity)

    @staticmethod
    def _validate_identity(identity: ActionCacheIdentity, outcome: ActionOutcome) -> None:
        if (
            outcome.action_id != identity.action_id
            or outcome.state_id != identity.state_id
            or outcome.asset_id != identity.asset_id
            or outcome.image_sha256 != identity.image_sha256
        ):
            raise TrajectoryCacheError(
                "TRAJECTORY_OUTCOME_IDENTITY_MISMATCH", "outcome and cache identity differ"
            )


def shard_index(asset_id: str, shard_count: int = 2) -> int:
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    digest = hashlib.sha256(asset_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def merge_trajectory_caches(source_roots: Iterable[Path], destination_root: Path) -> dict[str, int]:
    destination = destination_root.resolve()
    workspace = Path.cwd().resolve()
    try:
        destination.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("merge destination escapes repository workspace") from exc
    merged = 0
    duplicates = 0
    for source_root in source_roots:
        source = source_root.resolve()
        try:
            source.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("merge source escapes repository workspace") from exc
        for path in sorted(source.rglob("trajectory_cache_*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            key = envelope.get("cache_key")
            if not isinstance(key, str) or re.fullmatch(r"trajectory_cache_[0-9a-f]{64}", key) is None:
                raise TrajectoryCacheError("MERGE_CACHE_KEY_INVALID", "source cache key is invalid")
            digest = key.removeprefix("trajectory_cache_")
            target = destination / digest[:2] / f"{key}.json"
            encoded = canonical_json_bytes(envelope) + b"\n"
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if existing.get("semantic_sha256") != envelope.get("semantic_sha256"):
                    raise TrajectoryCacheError(
                        "MERGE_SEMANTIC_CONFLICT", "duplicate cache key has different semantics"
                    )
                duplicates += 1
                continue
            atomic_create_bytes(target, encoded)
            merged += 1
    return {"merged": merged, "duplicates": duplicates}


def action_metadata(plan: ActionPlan, sequence_index: int) -> PromptMetadata:
    if plan.prompt is None:
        raise ValueError("infeasible plan has no prompt metadata")
    actual_text = plan.prompt.text if plan.prompt.text is not None else plan.canonical_text
    return PromptMetadata(
        class_id=plan.class_id,
        canonical_text=actual_text,
        family=f"voc_m04_m05_{plan.action_code.lower()}_v1",
        sequence_index=sequence_index,
    )
