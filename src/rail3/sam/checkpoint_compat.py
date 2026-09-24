"""Exact-contract SAM 3.1 deterministic RoPE-buffer completion.

The compatibility surface is intentionally bound to one upstream commit and one
checkpoint digest.  It never mutates or writes a checkpoint and never performs a
non-strict real-model load.  The only generated entries are the 64 explicitly
specified persistent, non-learned RoPE real/imaginary buffers.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PINNED_SAM31_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
PINNED_SAM31_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
EXPECTED_CHECKPOINT_KEY_COUNT = 1623
EXPECTED_FINAL_MODEL_KEY_COUNT = 1687
GLOBAL_ATTENTION_BLOCKS = frozenset({7, 15, 23, 31})
ROPE_BUFFER_CLASSIFICATION = "deterministic_nonlearned_buffer"
LOCAL_REAL_SHA256 = "a3fac8f57d90c524e02a0e2ef5d541c5ac9c793a43a0fe94f06a3afd4e478296"
LOCAL_IMAG_SHA256 = "497a8ee9f818ac11f52fbc4501a5c32e7f6b64ab2b7e1c2ffaa62aec73ec3369"
GLOBAL_REAL_SHA256 = "8f869d845513b182a956f1922be3bf496bb4bd187cd44b79469ff9ee2b9cd325"
GLOBAL_IMAG_SHA256 = "2fe08034a01b79157f9f7108259b0b913b46fa460771f508ce02320d7289fb99"


class UnsupportedSam31CheckpointContractError(RuntimeError):
    """Base structured failure for a pinned checkpoint-contract violation."""

    default_reason_code = "unsupported_sam31_checkpoint_contract"

    def __init__(
        self,
        message: str,
        *,
        expected: Any,
        actual: Any,
        source_commit: str,
        checkpoint_hash: str,
        reason_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.default_reason_code
        self.expected = expected
        self.actual = actual
        self.source_commit = source_commit
        self.checkpoint_hash = checkpoint_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "reason_code": self.reason_code,
            "message": str(self),
            "expected": self.expected,
            "actual": self.actual,
            "source_commit": self.source_commit,
            "checkpoint_hash": self.checkpoint_hash,
        }


class UnexpectedMissingKeyError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "unexpected_missing_key"


class UnexpectedCheckpointKeyError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "unexpected_checkpoint_key"


class CheckpointShapeMismatchError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "checkpoint_shape_mismatch"


class CheckpointDtypeMismatchError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "checkpoint_dtype_mismatch"


class GeneratedBufferDeterminismError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "generated_buffer_determinism_failure"


class PinnedSourceMismatchError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "pinned_source_mismatch"


class CheckpointHashMismatchError(UnsupportedSam31CheckpointContractError):
    default_reason_code = "checkpoint_hash_mismatch"


@dataclass(frozen=True)
class DeterministicBufferSpec:
    key: str
    module_path: str
    source_complex_key: str
    component: str
    shape: tuple[int, ...]
    dtype: str
    classification: str
    expected_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "module_path": self.module_path,
            "source_complex_key": self.source_complex_key,
            "component": self.component,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "classification": self.classification,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True)
class Sam31CheckpointContract:
    upstream_commit: str
    checkpoint_sha256: str
    expected_checkpoint_key_count: int
    expected_final_model_key_count: int
    buffers: tuple[DeterministicBufferSpec, ...]

    def validate_structure(self) -> None:
        keys = [spec.key for spec in self.buffers]
        if len(keys) != len(set(keys)):
            raise ValueError("deterministic buffer contract contains duplicate keys")
        if any(spec.classification != ROPE_BUFFER_CLASSIFICATION for spec in self.buffers):
            raise ValueError("deterministic buffer contract classification drift")


@dataclass(frozen=True)
class CheckpointContractAnalysis:
    shared_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[dict[str, Any], ...]
    dtype_mismatches: tuple[dict[str, Any], ...]
    parameter_keys: tuple[str, ...]
    buffer_keys: tuple[str, ...]
    generated_buffer_keys: tuple[str, ...]
    learned_parameter_coverage: float
    checkpoint_key_usage: float
    generated_buffer_hashes: tuple[tuple[str, str], ...]
    generated_buffer_aggregate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shared_key_count": len(self.shared_keys),
            "shared_keys": list(self.shared_keys),
            "missing_key_count": len(self.missing_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_key_count": len(self.unexpected_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "shape_mismatch_count": len(self.shape_mismatches),
            "shape_mismatches": list(self.shape_mismatches),
            "dtype_mismatch_count": len(self.dtype_mismatches),
            "dtype_mismatches": list(self.dtype_mismatches),
            "parameter_key_count": len(self.parameter_keys),
            "buffer_key_count": len(self.buffer_keys),
            "generated_buffer_count": len(self.generated_buffer_keys),
            "generated_buffer_keys": list(self.generated_buffer_keys),
            "learned_parameter_coverage": self.learned_parameter_coverage,
            "checkpoint_key_usage": self.checkpoint_key_usage,
            "generated_buffer_hashes": dict(self.generated_buffer_hashes),
            "generated_buffer_aggregate_sha256": self.generated_buffer_aggregate_sha256,
        }


@dataclass(frozen=True)
class StrictLoadResult:
    status: str
    strict: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "strict": self.strict,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


def _rope_key(block: int, component: str) -> str:
    return (
        "detector.backbone.vision_backbone.trunk.blocks."
        f"{block}.attn.freqs_cis_{component}"
    )


def _rope_module_path(block: int) -> str:
    return f"detector.backbone.vision_backbone.trunk.blocks.{block}.attn"


def _rope_source_key(block: int) -> str:
    return f"{_rope_module_path(block)}.freqs_cis"


def official_rope_buffer_specs() -> tuple[DeterministicBufferSpec, ...]:
    specs: list[DeterministicBufferSpec] = []
    for block in range(32):
        is_global = block in GLOBAL_ATTENTION_BLOCKS
        shape = (5184, 32) if is_global else (576, 32)
        hashes = {
            "real": GLOBAL_REAL_SHA256 if is_global else LOCAL_REAL_SHA256,
            "imag": GLOBAL_IMAG_SHA256 if is_global else LOCAL_IMAG_SHA256,
        }
        for component in ("real", "imag"):
            specs.append(
                DeterministicBufferSpec(
                    key=_rope_key(block, component),
                    module_path=_rope_module_path(block),
                    source_complex_key=_rope_source_key(block),
                    component=component,
                    shape=shape,
                    dtype="torch.float32",
                    classification=ROPE_BUFFER_CLASSIFICATION,
                    expected_sha256=hashes[component],
                )
            )
    result = tuple(specs)
    if len(result) != 64 or len({spec.key for spec in result}) != 64:
        raise AssertionError("official RoPE buffer allowlist must contain exactly 64 keys")
    return result


OFFICIAL_SAM31_CHECKPOINT_CONTRACT = Sam31CheckpointContract(
    upstream_commit=PINNED_SAM31_COMMIT,
    checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
    expected_checkpoint_key_count=EXPECTED_CHECKPOINT_KEY_COUNT,
    expected_final_model_key_count=EXPECTED_FINAL_MODEL_KEY_COUNT,
    buffers=official_rope_buffer_specs(),
)


def tensor_sha256(tensor: Any) -> str:
    """Hash dtype, shape, and canonical CPU bytes without logging tensor data."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(list(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def aggregate_buffer_sha256(buffer_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(buffer_hashes):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(buffer_hashes[key].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in value.shape)


def _dtype(value: Any) -> str:
    return str(value.dtype)


def _error_context(contract: Sam31CheckpointContract) -> dict[str, str]:
    return {
        "source_commit": contract.upstream_commit,
        "checkpoint_hash": contract.checkpoint_sha256,
    }


def validate_pinned_identity(
    *,
    source_commit: str,
    checkpoint_sha256: str,
    contract: Sam31CheckpointContract = OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
) -> None:
    contract.validate_structure()
    if source_commit != contract.upstream_commit:
        raise PinnedSourceMismatchError(
            "SAM 3.1 source commit differs from the exact compatibility contract",
            expected=contract.upstream_commit,
            actual=source_commit,
            source_commit=source_commit,
            checkpoint_hash=checkpoint_sha256,
        )
    if checkpoint_sha256 != contract.checkpoint_sha256:
        raise CheckpointHashMismatchError(
            "SAM 3.1 checkpoint SHA-256 differs from the exact compatibility contract",
            expected=contract.checkpoint_sha256,
            actual=checkpoint_sha256,
            source_commit=source_commit,
            checkpoint_hash=checkpoint_sha256,
        )


def _derived_component(source: Any, component: str) -> Any:
    if component == "real":
        return source.real
    if component == "imag":
        return source.imag
    raise ValueError(f"unknown complex component: {component}")


def analyze_checkpoint_contract(
    model: Any,
    checkpoint_state: Mapping[str, Any],
    *,
    source_commit: str,
    checkpoint_sha256: str,
    contract: Sam31CheckpointContract = OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
) -> CheckpointContractAnalysis:
    """Validate the exact original-checkpoint/final-model contract fail closed."""

    validate_pinned_identity(
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        contract=contract,
    )
    context = _error_context(contract)
    model_state = model.state_dict()
    parameter_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    parameter_keys = set(parameter_map)
    buffer_keys = set(buffer_map)
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    allowlist = {spec.key for spec in contract.buffers}

    if len(checkpoint_keys) != contract.expected_checkpoint_key_count:
        raise UnexpectedCheckpointKeyError(
            "checkpoint key count differs from the exact contract",
            expected=contract.expected_checkpoint_key_count,
            actual=len(checkpoint_keys),
            **context,
        )
    if len(model_keys) != contract.expected_final_model_key_count:
        raise UnexpectedMissingKeyError(
            "final model key count differs from the exact contract",
            expected=contract.expected_final_model_key_count,
            actual=len(model_keys),
            **context,
        )

    collisions = sorted(allowlist & parameter_keys)
    if collisions:
        raise GeneratedBufferDeterminismError(
            "deterministic buffer allowlist collides with learned parameters",
            expected=[],
            actual=collisions,
            reason_code="allowlist_parameter_collision",
            **context,
        )
    non_buffers = sorted(allowlist - buffer_keys)
    if non_buffers:
        raise GeneratedBufferDeterminismError(
            "allowlisted completion key is not a named buffer",
            expected="all allowlisted keys in named_buffers",
            actual=non_buffers,
            reason_code="allowlist_non_buffer_key",
            **context,
        )

    shared = sorted(model_keys & checkpoint_keys)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if unexpected:
        raise UnexpectedCheckpointKeyError(
            "checkpoint contains keys absent from the final model",
            expected=[],
            actual=unexpected,
            **context,
        )
    if set(missing) != allowlist:
        raise UnexpectedMissingKeyError(
            "final model missing-key set differs from the exact 64-buffer allowlist",
            expected=sorted(allowlist),
            actual=missing,
            **context,
        )

    missing_parameters = sorted(parameter_keys - checkpoint_keys)
    if missing_parameters:
        raise UnexpectedMissingKeyError(
            "one or more learned parameters are absent from the original checkpoint",
            expected=[],
            actual=missing_parameters,
            reason_code="learned_parameter_missing",
            **context,
        )

    shape_mismatches = tuple(
        {
            "key": key,
            "expected": list(_shape(model_state[key])),
            "actual": list(_shape(checkpoint_state[key])),
        }
        for key in shared
        if _shape(model_state[key]) != _shape(checkpoint_state[key])
    )
    if shape_mismatches:
        raise CheckpointShapeMismatchError(
            "checkpoint tensor shape differs from the final model",
            expected=[{"key": item["key"], "shape": item["expected"]} for item in shape_mismatches],
            actual=[{"key": item["key"], "shape": item["actual"]} for item in shape_mismatches],
            **context,
        )

    dtype_mismatches = tuple(
        {
            "key": key,
            "expected": _dtype(model_state[key]),
            "actual": _dtype(checkpoint_state[key]),
        }
        for key in shared
        if _dtype(model_state[key]) != _dtype(checkpoint_state[key])
    )
    if dtype_mismatches:
        raise CheckpointDtypeMismatchError(
            "checkpoint tensor dtype differs from the final model",
            expected=[{"key": item["key"], "dtype": item["expected"]} for item in dtype_mismatches],
            actual=[{"key": item["key"], "dtype": item["actual"]} for item in dtype_mismatches],
            **context,
        )

    generated_hashes: dict[str, str] = {}
    for spec in contract.buffers:
        value = buffer_map[spec.key]
        if _shape(value) != spec.shape:
            raise GeneratedBufferDeterminismError(
                "generated buffer shape differs from its exact allowlist entry",
                expected=list(spec.shape),
                actual=list(_shape(value)),
                reason_code="generated_buffer_shape_mismatch",
                **context,
            )
        if _dtype(value) != spec.dtype:
            raise GeneratedBufferDeterminismError(
                "generated buffer dtype differs from its exact allowlist entry",
                expected=spec.dtype,
                actual=_dtype(value),
                reason_code="generated_buffer_dtype_mismatch",
                **context,
            )
        if bool(value.requires_grad):
            raise GeneratedBufferDeterminismError(
                "generated buffer unexpectedly requires gradients",
                expected=False,
                actual=True,
                reason_code="generated_buffer_requires_grad",
                **context,
            )
        observed_hash = tensor_sha256(value)
        if observed_hash != spec.expected_sha256:
            raise GeneratedBufferDeterminismError(
                "generated buffer differs from the pinned deterministic constructor hash",
                expected=spec.expected_sha256,
                actual=observed_hash,
                reason_code="generated_buffer_hash_mismatch",
                **context,
            )
        source = checkpoint_state.get(spec.source_complex_key)
        if source is None:
            raise UnexpectedMissingKeyError(
                "checkpoint lacks the complex RoPE source buffer",
                expected=spec.source_complex_key,
                actual=None,
                reason_code="rope_source_buffer_missing",
                **context,
            )
        derived_hash = tensor_sha256(_derived_component(source, spec.component))
        if observed_hash != derived_hash:
            raise GeneratedBufferDeterminismError(
                "generated buffer differs from the checkpoint complex-buffer component",
                expected=derived_hash,
                actual=observed_hash,
                reason_code="generated_buffer_checkpoint_derivation_mismatch",
                **context,
            )
        generated_hashes[spec.key] = observed_hash

    learned_coverage = (
        1.0 if not parameter_keys else len(parameter_keys & checkpoint_keys) / len(parameter_keys)
    )
    checkpoint_usage = (
        1.0 if not checkpoint_keys else len(shared) / len(checkpoint_keys)
    )
    if learned_coverage != 1.0:
        raise UnexpectedMissingKeyError(
            "learned parameter coverage is not 100 percent",
            expected=1.0,
            actual=learned_coverage,
            reason_code="learned_parameter_coverage_incomplete",
            **context,
        )
    if checkpoint_usage != 1.0:
        raise UnexpectedCheckpointKeyError(
            "not every original checkpoint key is consumed by the final model",
            expected=1.0,
            actual=checkpoint_usage,
            reason_code="checkpoint_key_usage_incomplete",
            **context,
        )

    return CheckpointContractAnalysis(
        shared_keys=tuple(shared),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        shape_mismatches=shape_mismatches,
        dtype_mismatches=dtype_mismatches,
        parameter_keys=tuple(sorted(parameter_keys)),
        buffer_keys=tuple(sorted(buffer_keys)),
        generated_buffer_keys=tuple(sorted(generated_hashes)),
        learned_parameter_coverage=learned_coverage,
        checkpoint_key_usage=checkpoint_usage,
        generated_buffer_hashes=tuple(sorted(generated_hashes.items())),
        generated_buffer_aggregate_sha256=aggregate_buffer_sha256(generated_hashes),
    )


def complete_checkpoint_state_in_memory(
    model: Any,
    checkpoint_state: Mapping[str, Any],
    *,
    source_commit: str,
    checkpoint_sha256: str,
    contract: Sam31CheckpointContract = OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
) -> tuple[dict[str, Any], CheckpointContractAnalysis]:
    """Return a new in-memory state mapping after exact fail-closed validation."""

    analysis = analyze_checkpoint_contract(
        model,
        checkpoint_state,
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        contract=contract,
    )
    model_buffers = dict(model.named_buffers())
    completed_state = dict(checkpoint_state)
    for spec in contract.buffers:
        completed_state[spec.key] = model_buffers[spec.key]
    if len(completed_state) != contract.expected_final_model_key_count:
        raise UnexpectedMissingKeyError(
            "completed in-memory state has an unexpected key count",
            expected=contract.expected_final_model_key_count,
            actual=len(completed_state),
            **_error_context(contract),
        )
    return completed_state, analysis


def load_completed_state_dict_strict(
    model: Any,
    completed_state: Mapping[str, Any],
    *,
    source_commit: str,
    checkpoint_sha256: str,
    contract: Sam31CheckpointContract = OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
) -> StrictLoadResult:
    """Perform the one final production load, always with ``strict=True``."""

    validate_pinned_identity(
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        contract=contract,
    )
    try:
        incompatible = model.load_state_dict(completed_state, strict=True)
    except Exception as exc:
        raise UnsupportedSam31CheckpointContractError(
            "completed state failed the final strict model load",
            expected={"missing_keys": [], "unexpected_keys": []},
            actual={"error_type": type(exc).__name__, "message": str(exc)},
            reason_code="completed_strict_load_failed",
            **_error_context(contract),
        ) from exc
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise UnsupportedSam31CheckpointContractError(
            "final strict load returned incompatible keys",
            expected={"missing_keys": [], "unexpected_keys": []},
            actual={"missing_keys": list(missing), "unexpected_keys": list(unexpected)},
            reason_code="completed_strict_load_incompatible",
            **_error_context(contract),
        )
    return StrictLoadResult(
        status="SAM31_STRICT_COMPLETED_MODEL_LOAD_PASS",
        strict=True,
        missing_keys=(),
        unexpected_keys=(),
    )


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_state_cpu(
    path: Path,
    *,
    source_commit: str,
    checkpoint_sha256: str,
    contract: Sam31CheckpointContract = OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
) -> Mapping[str, Any]:
    """Read one verified checkpoint on CPU with no output or mutation path."""

    validate_pinned_identity(
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        contract=contract,
    )
    import torch

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise UnsupportedSam31CheckpointContractError(
                "checkpoint must be one regular single-link file",
                expected="regular single-link file",
                actual={"mode": before.st_mode, "nlink": before.st_nlink},
                reason_code="checkpoint_file_identity_invalid",
                **_error_context(contract),
            )
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after_hash = os.fstat(stream.fileno())
        observed_hash = digest.hexdigest()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (
                after_hash.st_dev,
                after_hash.st_ino,
                after_hash.st_size,
                after_hash.st_mtime_ns,
                after_hash.st_ctime_ns,
            )
            or observed_hash != checkpoint_sha256
        ):
            raise CheckpointHashMismatchError(
                "checkpoint file content differs from the declared pinned SHA-256",
                expected=checkpoint_sha256,
                actual=observed_hash,
                source_commit=source_commit,
                checkpoint_hash=checkpoint_sha256,
            )
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        after_load = os.fstat(stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after_load.st_dev,
            after_load.st_ino,
            after_load.st_size,
            after_load.st_mtime_ns,
            after_load.st_ctime_ns,
        ):
            raise CheckpointHashMismatchError(
                "checkpoint changed during strict load",
                expected=checkpoint_sha256,
                actual="file_identity_changed",
                source_commit=source_commit,
                checkpoint_hash=checkpoint_sha256,
            )
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        payload = payload["model"]
    if not isinstance(payload, Mapping):
        raise UnsupportedSam31CheckpointContractError(
            "checkpoint payload is not a state mapping",
            expected="mapping",
            actual=type(payload).__name__,
            reason_code="checkpoint_payload_not_mapping",
            **_error_context(contract),
        )
    return payload
