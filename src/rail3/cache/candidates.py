"""Candidate records, cache keys, integrity envelopes, and resume policies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rail3.cache.atomic_io import atomic_create_bytes
from rail3.contracts import canonical_json_bytes, stable_id


_SHA256 = re.compile(r"[0-9a-f]{64}")


class CacheCorruptionError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class CandidateOutcomeTelemetry:
    dataset_id: str
    sample_id: str
    asset_id: str
    image_sha256: str
    model_source_commit: str
    checkpoint_sha256: str
    prompt_id: str
    prompt_class_id: int
    canonical_prompt_text: str
    prompt_family: str
    prompt_sequence_index: int
    image_session_initialization_count: int
    model_build_count: int
    prompt_call_count: int
    runtime_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    physical_gpu_index: int
    logical_device: str
    config_hash: str
    cache_key: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.dataset_id, "dataset_id"), (self.sample_id, "sample_id"),
            (self.asset_id, "asset_id"), (self.prompt_id, "prompt_id"),
            (self.canonical_prompt_text, "canonical_prompt_text"),
            (self.prompt_family, "prompt_family"), (self.logical_device, "logical_device"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for value, name in (
            (self.image_sha256, "image_sha256"),
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.config_hash, "config_hash"),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        if re.fullmatch(r"[0-9a-f]{40}", self.model_source_commit) is None:
            raise ValueError("model_source_commit must be a full lowercase Git commit")
        integers = (
            self.prompt_class_id, self.prompt_sequence_index,
            self.image_session_initialization_count, self.model_build_count,
            self.prompt_call_count, self.peak_allocated_vram_bytes,
            self.peak_reserved_vram_bytes, self.physical_gpu_index,
        )
        if any(value < 0 for value in integers):
            raise ValueError("candidate outcome telemetry integers cannot be negative")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("candidate outcome runtime must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateOutcomeTelemetry":
        return cls(
            dataset_id=data["dataset_id"], sample_id=data["sample_id"], asset_id=data["asset_id"],
            image_sha256=data["image_sha256"], model_source_commit=data["model_source_commit"],
            checkpoint_sha256=data["checkpoint_sha256"], prompt_id=data["prompt_id"],
            prompt_class_id=int(data["prompt_class_id"]), canonical_prompt_text=data["canonical_prompt_text"],
            prompt_family=data["prompt_family"], prompt_sequence_index=int(data["prompt_sequence_index"]),
            image_session_initialization_count=int(data["image_session_initialization_count"]),
            model_build_count=int(data["model_build_count"]), prompt_call_count=int(data["prompt_call_count"]),
            runtime_seconds=float(data["runtime_seconds"]),
            peak_allocated_vram_bytes=int(data["peak_allocated_vram_bytes"]),
            peak_reserved_vram_bytes=int(data["peak_reserved_vram_bytes"]),
            physical_gpu_index=int(data["physical_gpu_index"]), logical_device=data["logical_device"],
            config_hash=data["config_hash"], cache_key=data.get("cache_key"),
        )


@dataclass(frozen=True)
class CandidateFailure:
    reason_code: str
    message: str
    retryable: bool
    telemetry: CandidateOutcomeTelemetry | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", self.reason_code) is None:
            raise ValueError("candidate failure reason_code must be uppercase snake case")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "retryable": self.retryable,
            "telemetry": None if self.telemetry is None else self.telemetry.to_dict(),
        }


@dataclass(frozen=True)
class CandidateNoResult:
    """A successful prompt call for which upstream returned no masks."""

    reason_code: str
    message: str
    telemetry: CandidateOutcomeTelemetry | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", self.reason_code) is None:
            raise ValueError("candidate no-result reason_code must be uppercase snake case")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "telemetry": None if self.telemetry is None else self.telemetry.to_dict(),
        }


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    trajectory_id: str
    asset_id: str
    model_spec_id: str
    prompt_id: str
    step_index: int
    candidate_index: int
    width: int
    height: int
    mask_rle: tuple[int, ...]
    mask_sha256: str
    bbox_xyxy: tuple[int, int, int, int]
    model_score: float
    state_id: str
    dataset_id: str | None = None
    sample_id: str | None = None
    image_sha256: str | None = None
    model_source_commit: str | None = None
    checkpoint_sha256: str | None = None
    prompt_class_id: int | None = None
    canonical_prompt_text: str | None = None
    prompt_family: str | None = None
    prompt_sequence_index: int | None = None
    upstream_score_field: str | None = None
    upstream_object_id: int | None = None
    upstream_bbox_xywh: tuple[float, float, float, float] | None = None
    image_session_initialization_count: int | None = None
    model_build_count: int | None = None
    prompt_call_count: int | None = None
    runtime_seconds: float | None = None
    peak_allocated_vram_bytes: int | None = None
    peak_reserved_vram_bytes: int | None = None
    physical_gpu_index: int | None = None
    logical_device: str | None = None
    error_reason_code: str | None = None
    config_hash: str | None = None
    cache_key: str | None = None
    schema_version: str = "rail3.candidate.v2"

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.candidate_index < 0:
            raise ValueError("candidate indices must be non-negative")
        if self.width <= 0 or self.height <= 0 or sum(self.mask_rle) != self.width * self.height:
            raise ValueError("RLE area must exactly match candidate dimensions")
        if any(value < 0 for value in self.mask_rle):
            raise ValueError("RLE counts cannot be negative")
        if _SHA256.fullmatch(self.mask_sha256) is None:
            raise ValueError("mask_sha256 must be lowercase SHA-256")
        x0, y0, x1, y1 = self.bbox_xyxy
        if not (0 <= x0 <= x1 <= self.width and 0 <= y0 <= y1 <= self.height):
            raise ValueError("candidate bounding box is outside the image")
        if not math.isfinite(self.model_score) or not 0.0 <= self.model_score <= 1.0:
            raise ValueError("model score must be finite and within [0, 1]")
        for value, name in (
            (self.image_sha256, "image_sha256"),
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.config_hash, "config_hash"),
        ):
            if value is not None and _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        if self.model_source_commit is not None and re.fullmatch(r"[0-9a-f]{40}", self.model_source_commit) is None:
            raise ValueError("model_source_commit must be a full lowercase Git commit")
        if self.sample_id is not None and not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        for value, name in (
            (self.prompt_class_id, "prompt_class_id"),
            (self.prompt_sequence_index, "prompt_sequence_index"),
            (self.upstream_object_id, "upstream_object_id"),
            (self.image_session_initialization_count, "image_session_initialization_count"),
            (self.model_build_count, "model_build_count"),
            (self.prompt_call_count, "prompt_call_count"),
            (self.peak_allocated_vram_bytes, "peak_allocated_vram_bytes"),
            (self.peak_reserved_vram_bytes, "peak_reserved_vram_bytes"),
            (self.physical_gpu_index, "physical_gpu_index"),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.runtime_seconds is not None and (
            not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0
        ):
            raise ValueError("runtime_seconds must be finite and non-negative")
        if self.upstream_bbox_xywh is not None and (
            len(self.upstream_bbox_xywh) != 4
            or any(not math.isfinite(value) for value in self.upstream_bbox_xywh)
        ):
            raise ValueError("upstream_bbox_xywh must contain four finite values")
        decoded = decode_binary_mask_array(self.mask_rle, self.width, self.height)
        canonical_rle = encode_binary_mask_array(decoded)
        if canonical_rle != self.mask_rle:
            raise ValueError("candidate RLE must use canonical alternating runs")
        expected_mask_sha256 = hashlib.sha256(canonical_json_bytes({
            "width": self.width,
            "height": self.height,
            "rle": self.mask_rle,
        })).hexdigest()
        if self.mask_sha256 != expected_mask_sha256:
            raise ValueError("mask_sha256 does not match candidate RLE")
        import numpy as np

        ys, xs = np.nonzero(decoded)
        expected_bbox = (0, 0, 0, 0) if xs.size == 0 else (
            int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
        )
        if self.bbox_xyxy != expected_bbox:
            raise ValueError("candidate bounding box does not match decoded mask")
        expected_candidate_id = stable_id("candidate", {
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "candidate_index": self.candidate_index,
            "mask_sha256": self.mask_sha256,
        })
        if self.candidate_id != expected_candidate_id:
            raise ValueError("candidate_id does not match canonical candidate identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trajectory_id": self.trajectory_id,
            "asset_id": self.asset_id,
            "model_spec_id": self.model_spec_id,
            "prompt_id": self.prompt_id,
            "step_index": self.step_index,
            "candidate_index": self.candidate_index,
            "width": self.width,
            "height": self.height,
            "mask_rle": list(self.mask_rle),
            "mask_sha256": self.mask_sha256,
            "bbox_xyxy": list(self.bbox_xyxy),
            "model_score": self.model_score,
            "state_id": self.state_id,
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "image_sha256": self.image_sha256,
            "model_source_commit": self.model_source_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_class_id": self.prompt_class_id,
            "canonical_prompt_text": self.canonical_prompt_text,
            "prompt_family": self.prompt_family,
            "prompt_sequence_index": self.prompt_sequence_index,
            "upstream_score_field": self.upstream_score_field,
            "upstream_object_id": self.upstream_object_id,
            "upstream_bbox_xywh": None if self.upstream_bbox_xywh is None else list(self.upstream_bbox_xywh),
            "image_session_initialization_count": self.image_session_initialization_count,
            "model_build_count": self.model_build_count,
            "prompt_call_count": self.prompt_call_count,
            "runtime_seconds": self.runtime_seconds,
            "peak_allocated_vram_bytes": self.peak_allocated_vram_bytes,
            "peak_reserved_vram_bytes": self.peak_reserved_vram_bytes,
            "physical_gpu_index": self.physical_gpu_index,
            "logical_device": self.logical_device,
            "error_reason_code": self.error_reason_code,
            "config_hash": self.config_hash,
            "cache_key": self.cache_key,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        return cls(
            candidate_id=data["candidate_id"], trajectory_id=data["trajectory_id"], asset_id=data["asset_id"],
            model_spec_id=data["model_spec_id"], prompt_id=data["prompt_id"], step_index=int(data["step_index"]),
            candidate_index=int(data["candidate_index"]), width=int(data["width"]), height=int(data["height"]),
            mask_rle=tuple(int(value) for value in data["mask_rle"]), mask_sha256=data["mask_sha256"],
            bbox_xyxy=tuple(int(value) for value in data["bbox_xyxy"]), model_score=float(data["model_score"]),
            state_id=data["state_id"], dataset_id=data.get("dataset_id"), sample_id=data.get("sample_id"),
            image_sha256=data.get("image_sha256"),
            model_source_commit=data.get("model_source_commit"), checkpoint_sha256=data.get("checkpoint_sha256"),
            prompt_class_id=None if data.get("prompt_class_id") is None else int(data["prompt_class_id"]),
            canonical_prompt_text=data.get("canonical_prompt_text"), prompt_family=data.get("prompt_family"),
            prompt_sequence_index=None if data.get("prompt_sequence_index") is None else int(data["prompt_sequence_index"]),
            upstream_score_field=data.get("upstream_score_field"),
            upstream_object_id=None if data.get("upstream_object_id") is None else int(data["upstream_object_id"]),
            upstream_bbox_xywh=None if data.get("upstream_bbox_xywh") is None else tuple(float(value) for value in data["upstream_bbox_xywh"]),
            image_session_initialization_count=None if data.get("image_session_initialization_count") is None else int(data["image_session_initialization_count"]),
            model_build_count=None if data.get("model_build_count") is None else int(data["model_build_count"]),
            prompt_call_count=None if data.get("prompt_call_count") is None else int(data["prompt_call_count"]),
            runtime_seconds=None if data.get("runtime_seconds") is None else float(data["runtime_seconds"]),
            peak_allocated_vram_bytes=None if data.get("peak_allocated_vram_bytes") is None else int(data["peak_allocated_vram_bytes"]),
            peak_reserved_vram_bytes=None if data.get("peak_reserved_vram_bytes") is None else int(data["peak_reserved_vram_bytes"]),
            physical_gpu_index=None if data.get("physical_gpu_index") is None else int(data["physical_gpu_index"]),
            logical_device=data.get("logical_device"), error_reason_code=data.get("error_reason_code"),
            config_hash=data.get("config_hash"), cache_key=data.get("cache_key"),
            schema_version=data.get("schema_version", "rail3.candidate.v1"),
        )


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidate: CandidateRecord | None = None
    failure: CandidateFailure | None = None
    no_result: CandidateNoResult | None = None
    additional_candidates: tuple[CandidateRecord, ...] = ()

    def __post_init__(self) -> None:
        if sum(value is not None for value in (self.candidate, self.failure, self.no_result)) != 1:
            raise ValueError("result must contain exactly one candidate, failure, or no-result")
        if self.candidate is None and self.additional_candidates:
            raise ValueError("additional candidates require a primary candidate")

    @property
    def all_candidates(self) -> tuple[CandidateRecord, ...]:
        if self.candidate is None:
            return ()
        return (self.candidate, *self.additional_candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
            "no_result": None if self.no_result is None else self.no_result.to_dict(),
            "additional_candidates": [candidate.to_dict() for candidate in self.additional_candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateGenerationResult":
        candidate = None if data.get("candidate") is None else CandidateRecord.from_dict(data["candidate"])
        failure_data = data.get("failure")
        failure = None if failure_data is None else CandidateFailure(
            failure_data["reason_code"], failure_data["message"], bool(failure_data["retryable"]),
            None if failure_data.get("telemetry") is None else CandidateOutcomeTelemetry.from_dict(failure_data["telemetry"]),
        )
        no_result_data = data.get("no_result")
        no_result = None if no_result_data is None else CandidateNoResult(
            no_result_data["reason_code"], no_result_data["message"],
            None if no_result_data.get("telemetry") is None else CandidateOutcomeTelemetry.from_dict(no_result_data["telemetry"]),
        )
        additional = tuple(CandidateRecord.from_dict(item) for item in data.get("additional_candidates", ()))
        return cls(candidate, failure, no_result, additional)


@dataclass(frozen=True)
class CacheIdentity:
    model_spec_id: str
    asset_id: str
    image_sha256: str
    prompt_id: str
    state_id: str
    adapter_schema: str = "rail3.candidate-cache.v1"

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.image_sha256) is None:
            raise ValueError("cache image_sha256 must be lowercase SHA-256")

    @property
    def cache_key(self) -> str:
        return stable_id("candidate_cache", {
            "adapter_schema": self.adapter_schema,
            "model_spec_id": self.model_spec_id,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "prompt_id": self.prompt_id,
            "state_id": self.state_id,
        })


@dataclass(frozen=True)
class CacheExecution:
    status: str
    cache_key: str
    result: CandidateGenerationResult | None


class CandidateCache:
    def __init__(self, root: Path) -> None:
        workspace = Path.cwd().resolve()
        resolved = root.resolve()
        try:
            self.root = resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("candidate cache root escapes the repository workspace") from exc

    def path_for(self, identity: CacheIdentity) -> Path:
        digest = identity.cache_key.removeprefix("candidate_cache_")
        return self.root / digest[:2] / f"{identity.cache_key}.json"

    def exists(self, identity: CacheIdentity) -> bool:
        return self.path_for(identity).is_file()

    def store(self, identity: CacheIdentity, result: CandidateGenerationResult) -> Path:
        self._validate_result_identity(identity, result)
        payload = result.to_dict()
        envelope = {
            "schema_version": "rail3.candidate-cache.v1",
            "cache_key": identity.cache_key,
            "identity": {
                "model_spec_id": identity.model_spec_id,
                "asset_id": identity.asset_id,
                "image_sha256": identity.image_sha256,
                "prompt_id": identity.prompt_id,
                "state_id": identity.state_id,
                "adapter_schema": identity.adapter_schema,
            },
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        }
        encoded = canonical_json_bytes(envelope) + b"\n"
        destination = self.path_for(identity)
        try:
            atomic_create_bytes(destination, encoded)
        except FileExistsError:
            existing = destination.read_bytes()
            if existing != encoded:
                raise CacheCorruptionError("IMMUTABLE_CACHE_CONFLICT", "existing cache object differs from new payload")
        return destination

    def load(self, identity: CacheIdentity) -> CandidateGenerationResult:
        path = self.path_for(identity)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheCorruptionError("CACHE_JSON_INVALID", "cache object is unreadable or invalid JSON") from exc
        if envelope.get("schema_version") != "rail3.candidate-cache.v1":
            raise CacheCorruptionError("CACHE_SCHEMA_MISMATCH", "cache schema is not supported")
        if envelope.get("cache_key") != identity.cache_key:
            raise CacheCorruptionError("CACHE_KEY_MISMATCH", "cache key does not match the requested identity")
        expected_identity = {
            "model_spec_id": identity.model_spec_id,
            "asset_id": identity.asset_id,
            "image_sha256": identity.image_sha256,
            "prompt_id": identity.prompt_id,
            "state_id": identity.state_id,
            "adapter_schema": identity.adapter_schema,
        }
        if envelope.get("identity") != expected_identity:
            raise CacheCorruptionError("CACHE_IDENTITY_MISMATCH", "stored identity does not match the requested identity")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise CacheCorruptionError("CACHE_PAYLOAD_INVALID", "cache payload must be an object")
        actual_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if envelope.get("payload_sha256") != actual_hash:
            raise CacheCorruptionError("CACHE_PAYLOAD_HASH_MISMATCH", "cache payload failed its integrity digest")
        try:
            result = CandidateGenerationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorruptionError("CACHE_PAYLOAD_INVALID", "cache payload violates the candidate contract") from exc
        self._validate_result_identity(identity, result)
        return result

    def execute(
        self,
        identity: CacheIdentity,
        generator: Callable[[], CandidateGenerationResult],
        *,
        resume: bool = False,
        skip_existing: bool = False,
        dry_run: bool = False,
    ) -> CacheExecution:
        existing = self.exists(identity)
        if dry_run:
            result = self.load(identity) if existing else None
            return CacheExecution("DRY_RUN_CACHE_HIT" if existing else "DRY_RUN_WOULD_GENERATE", identity.cache_key, result)
        if existing and (resume or skip_existing):
            return CacheExecution("RESUMED_EXISTING" if resume else "SKIPPED_EXISTING", identity.cache_key, self.load(identity))
        if existing:
            raise FileExistsError(self.path_for(identity))
        result = generator()
        self.store(identity, result)
        return CacheExecution("GENERATED", identity.cache_key, result)

    @staticmethod
    def _validate_result_identity(identity: CacheIdentity, result: CandidateGenerationResult) -> None:
        for candidate in result.all_candidates:
            if (
                candidate.model_spec_id != identity.model_spec_id
                or candidate.asset_id != identity.asset_id
                or candidate.prompt_id != identity.prompt_id
                or candidate.state_id != identity.state_id
                or candidate.image_sha256 not in {None, identity.image_sha256}
                or candidate.cache_key not in {None, identity.cache_key}
            ):
                raise CacheCorruptionError("CANDIDATE_IDENTITY_MISMATCH", "candidate fields do not match the cache identity")
        telemetry = None
        if result.failure is not None:
            telemetry = result.failure.telemetry
        elif result.no_result is not None:
            telemetry = result.no_result.telemetry
        if telemetry is not None and (
            telemetry.asset_id != identity.asset_id
            or telemetry.image_sha256 != identity.image_sha256
            or telemetry.prompt_id != identity.prompt_id
            or telemetry.cache_key not in {None, identity.cache_key}
        ):
            raise CacheCorruptionError("CANDIDATE_IDENTITY_MISMATCH", "outcome telemetry does not match the cache identity")


def encode_binary_mask(mask: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    if not mask or not mask[0] or any(len(row) != len(mask[0]) for row in mask):
        raise ValueError("mask must be a non-empty rectangle")
    flat = [value for row in mask for value in row]
    if any(value not in {0, 1} for value in flat):
        raise ValueError("mask values must be binary")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    return tuple(counts)


def encode_binary_mask_array(mask: Any) -> tuple[int, ...]:
    """Encode a two-dimensional NumPy-compatible mask without materializing tuples."""

    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError("mask values must be binary")
    flat = array.astype(bool, copy=False).reshape(-1)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if bool(flat[0]):
        counts.insert(0, 0)
    return tuple(counts)


def decode_binary_mask(counts: tuple[int, ...], width: int, height: int) -> tuple[tuple[int, ...], ...]:
    values: list[int] = []
    current = 0
    for count in counts:
        if count < 0:
            raise ValueError("RLE counts cannot be negative")
        values.extend([current] * count)
        current = 1 - current
    if len(values) != width * height:
        raise ValueError("RLE area does not match dimensions")
    return tuple(tuple(values[row * width:(row + 1) * width]) for row in range(height))


def decode_binary_mask_array(counts: tuple[int, ...], width: int, height: int) -> Any:
    """Decode canonical alternating-run RLE directly to a boolean NumPy array."""

    import numpy as np

    if any(count < 0 for count in counts) or sum(counts) != width * height:
        raise ValueError("RLE area does not match dimensions")
    values = np.arange(len(counts), dtype=np.uint8) % 2
    flat = np.repeat(values, np.asarray(counts, dtype=np.int64)).astype(bool, copy=False)
    return flat.reshape((height, width))
