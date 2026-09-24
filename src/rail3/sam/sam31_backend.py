"""Real frozen SAM 3.1 backend for independent single-image text prompts.

This module deliberately contains no dataset-label or evaluation imports.  It
uses the reviewed strict builder and the pinned init-state compatibility shim;
each semantic text prompt resets upstream decoder state while reusing the one
image/session initialization for that image.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rail3.cache.candidates import (
    CandidateFailure,
    CandidateGenerationResult,
    CandidateNoResult,
    CandidateOutcomeTelemetry,
    CandidateRecord,
    encode_binary_mask_array,
)
from rail3.contracts import canonical_json_bytes, stable_id
from rail3.sam.checkpoint_compat import (
    PINNED_SAM31_CHECKPOINT_SHA256,
    PINNED_SAM31_COMMIT,
)
from rail3.sam.compat import start_session_with_compat
from rail3.sam.protocols import (
    BackendKind,
    EncodedImageState,
    FrozenSamBackend,
    ModelSpec,
    PromptKind,
    PromptRequest,
)
from rail3.sam.strict_builder import StrictPredictorBuild, build_strict_sam31_predictor


SAM31_ADAPTER_VERSION = "sam31-real-text-v1"
SAM31_CHECKPOINT_BYTES = 3_502_755_717
MAX_PEAK_VRAM_BYTES = 45 * 1024**3
MIN_FREE_VRAM_BYTES = 40 * 1024**3


@dataclass(frozen=True)
class PromptMetadata:
    class_id: int
    canonical_text: str
    family: str
    sequence_index: int

    def __post_init__(self) -> None:
        if self.class_id <= 0 or self.sequence_index < 0:
            raise ValueError("prompt metadata indices are invalid")
        if not self.canonical_text or not self.family:
            raise ValueError("prompt metadata text fields cannot be empty")


@dataclass
class _ImageSession:
    state: EncodedImageState
    dataset_id: str
    sample_id: str
    image_path: Path
    session_id: str
    owns_temporary: bool
    initialization_seconds: float


def official_sam31_model_spec() -> ModelSpec:
    return ModelSpec.create(
        model_name="sam3.1-object-multiplex",
        backend_kind=BackendKind.OFFICIAL,
        source_commit=PINNED_SAM31_COMMIT,
        checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
        preprocess_spec="official-single-image-1008-v1",
        adapter_version=SAM31_ADAPTER_VERSION,
    )


def real_backend_config_hash() -> str:
    return hashlib.sha256(canonical_json_bytes({
        "adapter_version": SAM31_ADAPTER_VERSION,
        "checkpoint_sha256": PINNED_SAM31_CHECKPOINT_SHA256,
        "compile_model": False,
        "default_output_prob_thresh": 0.5,
        "max_num_objects": 16,
        "multiplex_count": 16,
        "prompt_protocol": "all-canonical-foreground-text-prompts-v1",
        "semantic_prompt_state": "reset-per-text-prompt",
        "session_policy": "one-initialization-per-image",
        "source_commit": PINNED_SAM31_COMMIT,
        "use_fa3": False,
        "use_rope_real": True,
    })).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Sam31RealFrozenBackend(FrozenSamBackend):
    """One-process/one-model real backend with explicit per-image sessions."""

    def __init__(
        self,
        *,
        source: Path,
        checkpoint_path: Path,
        prompt_metadata: dict[str, PromptMetadata],
        builder: Callable[..., StrictPredictorBuild] = build_strict_sam31_predictor,
        torch_module: Any | None = None,
        enforce_runtime_environment: bool = True,
        physical_gpu_index: int = 1,
    ) -> None:
        self._model_spec = official_sam31_model_spec()
        self.config_hash = real_backend_config_hash()
        self.prompt_metadata = dict(prompt_metadata)
        self.model_build_count = 0
        self.checkpoint_load_count = 0
        self.image_session_initialization_count = 0
        self.prompt_call_count = 0
        self._states: dict[str, _ImageSession] = {}
        self._temporary_paths: set[Path] = set()
        if physical_gpu_index not in {0, 1}:
            raise ValueError("physical GPU index must be 0 or 1")
        self.physical_gpu_index = physical_gpu_index

        if checkpoint_path.name != "sam3.1_multiplex.pt":
            raise ValueError("real backend checkpoint filename differs from frozen identity")
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != SAM31_CHECKPOINT_BYTES:
            raise ValueError("real backend checkpoint file/byte identity mismatch")
        if enforce_runtime_environment:
            self._require_runtime_environment(physical_gpu_index)

        if torch_module is None:
            import torch

            torch_module = torch
        self._torch = torch_module
        if enforce_runtime_environment:
            if self._torch.cuda.is_initialized():
                raise RuntimeError("CUDA initialized before real SAM backend GPU admission gate")
            if self._torch.cuda.device_count() != 1:
                raise RuntimeError("real SAM backend requires exactly one visible CUDA device")
            free_bytes, _ = self._torch.cuda.mem_get_info(0)
            if free_bytes < MIN_FREE_VRAM_BYTES:
                raise RuntimeError("logical cuda:0 has less than 40 GiB free")
        self._torch.cuda.reset_peak_memory_stats(0)
        self._torch.cuda.synchronize(0)
        with self._torch.inference_mode():
            self._build = builder(
                source=source,
                checkpoint_path=checkpoint_path,
                warm_up=False,
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=True,
                compile_model=False,
                async_loading_frames=False,
                device="cuda",
            )
        self.model_build_count = 1
        self.checkpoint_load_count = self._build.evidence.final_strict_loads
        self._predictor = self._build.predictor
        if self._build.evidence.source_commit != PINNED_SAM31_COMMIT:
            raise RuntimeError("strict builder returned an unexpected source identity")
        if self._build.evidence.checkpoint_sha256 != PINNED_SAM31_CHECKPOINT_SHA256:
            raise RuntimeError("strict builder returned an unexpected checkpoint identity")
        if self._build.evidence.generated_buffer_count != 64:
            raise RuntimeError("strict builder did not complete exactly 64 deterministic buffers")
        if self._build.evidence.learned_parameter_coverage != 1.0:
            raise RuntimeError("strict builder learned-parameter coverage is not 100%")
        if self._build.evidence.intermediate_checkpoint_loads != 0:
            raise RuntimeError("strict builder performed an intermediate checkpoint load")
        if self.checkpoint_load_count != 1 or not self._build.strict_load.strict:
            raise RuntimeError("strict builder did not perform exactly one final strict load")
        if self._torch.cuda.max_memory_reserved(0) >= MAX_PEAK_VRAM_BYTES:
            raise RuntimeError("model construction reached the 45 GiB VRAM limit")
        self.assert_frozen()

    @staticmethod
    def _require_runtime_environment(physical_gpu_index: int = 1) -> None:
        required = {
            "RAIL3_STAGE": "AUTO_ANNOTATION",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu_index),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        drift = {
            name: {"expected": expected, "actual": os.environ.get(name)}
            for name, expected in required.items()
            if os.environ.get(name) != expected
        }
        if drift:
            raise RuntimeError(f"real SAM backend environment gate failed: {drift}")

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    @property
    def builder_evidence(self) -> dict[str, Any]:
        return self._build.evidence.as_dict()

    def assert_frozen(self) -> None:
        model = self._predictor.model
        if bool(model.training):
            raise RuntimeError("SAM31_BACKEND_MODEL_NOT_EVAL")
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if trainable:
            raise RuntimeError("SAM31_BACKEND_PARAMETERS_NOT_FROZEN")

    def encode_image(
        self,
        *,
        asset_id: str,
        image_bytes: bytes,
        width: int,
        height: int,
    ) -> EncodedImageState:
        if not image_bytes:
            raise ValueError("image bytes cannot be empty")
        suffix = ".png" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
        temporary = tempfile.NamedTemporaryFile(
            mode="wb", prefix="rail3-sam31-image-", suffix=suffix, delete=False
        )
        try:
            temporary.write(image_bytes)
            temporary.close()
            path = Path(temporary.name)
            self._temporary_paths.add(path)
            return self.encode_image_path(
                dataset_id="unavailable",
                sample_id=asset_id,
                asset_id=asset_id,
                image_path=path,
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                width=width,
                height=height,
                owns_temporary=True,
            )
        except Exception:
            temporary.close()
            Path(temporary.name).unlink(missing_ok=True)
            self._temporary_paths.discard(Path(temporary.name))
            raise

    def encode_image_path(
        self,
        *,
        dataset_id: str,
        sample_id: str | None = None,
        asset_id: str,
        image_path: Path,
        image_sha256: str,
        width: int,
        height: int,
        owns_temporary: bool = False,
    ) -> EncodedImageState:
        self.assert_frozen()
        if not image_path.is_file() or width <= 0 or height <= 0:
            raise ValueError("real image path and dimensions must be valid")
        if _sha256_file(image_path) != image_sha256:
            raise ValueError("real image content does not match manifest SHA-256")
        state_id = stable_id("encoded_state", {
            "model_spec_id": self.model_spec.model_spec_id,
            "asset_id": asset_id,
            "image_sha256": image_sha256,
            "width": width,
            "height": height,
        })
        if state_id in self._states:
            return self._states[state_id].state
        session_id = f"rail3-{state_id}"
        self._torch.cuda.synchronize(0)
        started = time.perf_counter()
        with self._torch.inference_mode():
            response, decision = start_session_with_compat(
                self._predictor,
                official_commit=PINNED_SAM31_COMMIT,
                resource_path=str(image_path),
                session_id=session_id,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )
        self._torch.cuda.synchronize(0)
        if response.get("session_id") != session_id:
            raise RuntimeError("upstream returned an unexpected session identity")
        if decision.dropped_parameters != ("offload_state_to_cpu",):
            raise RuntimeError("compatibility shim action differs from reviewed contract")
        state = EncodedImageState(
            state_id, asset_id, image_sha256, width, height, self.model_spec.model_spec_id
        )
        self.image_session_initialization_count += 1
        self._states[state_id] = _ImageSession(
            state=state,
            dataset_id=dataset_id,
            sample_id=sample_id or asset_id,
            image_path=image_path,
            session_id=session_id,
            owns_temporary=owns_temporary,
            initialization_seconds=time.perf_counter() - started,
        )
        return state

    def generate_candidates(
        self,
        state: EncodedImageState,
        prompts: tuple[PromptRequest, ...],
    ) -> tuple[CandidateGenerationResult, ...]:
        self.assert_frozen()
        context = self._states.get(state.state_id)
        if context is None or context.state != state:
            raise ValueError("encoded state was not produced by this active backend session")
        if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
            raise ValueError("prompt batch contains duplicate prompt identities")
        trajectory_id = stable_id("trajectory", {
            "model_spec_id": state.model_spec_id,
            "asset_id": state.asset_id,
            "state_id": state.state_id,
            "protocol": "independent-canonical-text-v1",
        })
        results: list[CandidateGenerationResult] = []
        try:
            for prompt in prompts:
                metadata = self.prompt_metadata.get(prompt.prompt_id)
                if prompt.kind is not PromptKind.TEXT or prompt.text is None or metadata is None:
                    results.append(CandidateGenerationResult(failure=CandidateFailure(
                        "UNAPPROVED_PROMPT", "prompt is not in the frozen canonical text protocol", False
                    )))
                    continue
                if prompt.text != metadata.canonical_text:
                    results.append(CandidateGenerationResult(failure=CandidateFailure(
                        "PROMPT_TEXT_DRIFT", "prompt text differs from frozen metadata", False
                    )))
                    continue
                results.append(self.run_action_prompt(
                    state,
                    prompt=prompt,
                    metadata=metadata,
                    trajectory_id=trajectory_id,
                ))
        finally:
            self._close_state(state.state_id)
        return tuple(results)

    def run_action_prompt(
        self,
        state: EncodedImageState,
        *,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        trajectory_id: str,
        upstream_object_id: int | None = None,
    ) -> CandidateGenerationResult:
        """Run one audited text, box, or point action without closing its session."""

        context = self._states.get(state.state_id)
        if context is None or context.state != state:
            raise ValueError("encoded state was not produced by this active backend session")
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": context.session_id,
            "frame_index": 0,
            "output_prob_thresh": 0.5,
        }
        if prompt.kind is PromptKind.TEXT:
            if prompt.text is None or prompt.text != metadata.canonical_text:
                raise ValueError("text action differs from frozen action metadata")
            request["text"] = prompt.text
        elif prompt.kind is PromptKind.BOX:
            if prompt.box is None:
                raise ValueError("box action has no geometry")
            x0, y0, x1, y1 = prompt.box
            if x1 > state.width or y1 > state.height:
                raise ValueError("box action exceeds image geometry")
            request.update({
                "bounding_boxes": [[
                    x0 / state.width,
                    y0 / state.height,
                    (x1 - x0) / state.width,
                    (y1 - y0) / state.height,
                ]],
                "bounding_box_labels": [1],
                "clear_old_boxes": True,
                "rel_coordinates": True,
            })
        elif prompt.kind is PromptKind.POINTS:
            if len(prompt.points) != 1 or upstream_object_id is None:
                raise ValueError("point action requires one point and an upstream object ID")
            x, y, label = prompt.points[0]
            if not (0 <= x < state.width and 0 <= y < state.height) or label != 1:
                raise ValueError("point action geometry or label is invalid")
            request.update({
                "points": [[(x + 0.5) / state.width, (y + 0.5) / state.height]],
                "point_labels": [label],
                "clear_old_points": True,
                "obj_id": upstream_object_id,
                "rel_coordinates": True,
            })
        else:
            raise ValueError("unsupported action prompt kind")

        self.prompt_call_count += 1
        self._torch.cuda.reset_peak_memory_stats(0)
        self._torch.cuda.synchronize(0)
        started = time.perf_counter()
        try:
            with self._torch.inference_mode():
                response = self._predictor.handle_request(request)
            self._torch.cuda.synchronize(0)
            runtime = time.perf_counter() - started
            peak_allocated = int(self._torch.cuda.max_memory_allocated(0))
            peak_reserved = int(self._torch.cuda.max_memory_reserved(0))
            if peak_reserved >= MAX_PEAK_VRAM_BYTES:
                raise RuntimeError("prompt call reached the 45 GiB VRAM limit")
            return self._convert_output(
                state=state,
                context=context,
                prompt=prompt,
                metadata=metadata,
                trajectory_id=trajectory_id,
                response=response,
                runtime_seconds=runtime,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
        except Exception as exc:
            runtime = time.perf_counter() - started
            peak_allocated = int(self._torch.cuda.max_memory_allocated(0))
            peak_reserved = int(self._torch.cuda.max_memory_reserved(0))
            return CandidateGenerationResult(failure=CandidateFailure(
                "SAM31_PROMPT_CALL_FAILED", f"{type(exc).__name__}: {exc}", True,
                self._outcome_telemetry(
                    state=state,
                    context=context,
                    prompt=prompt,
                    metadata=metadata,
                    runtime_seconds=runtime,
                    peak_allocated=peak_allocated,
                    peak_reserved=peak_reserved,
                ),
            ))

    def close_state(self, state: EncodedImageState) -> None:
        self._close_state(state.state_id)

    def _convert_output(
        self,
        *,
        state: EncodedImageState,
        context: _ImageSession,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        trajectory_id: str,
        response: Any,
        runtime_seconds: float,
        peak_allocated: int,
        peak_reserved: int,
    ) -> CandidateGenerationResult:
        import numpy as np

        if not isinstance(response, dict) or response.get("frame_index") != 0:
            raise ValueError("upstream prompt response envelope is invalid")
        outputs = response.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("upstream prompt outputs are unavailable")
        required = {"out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"}
        if not required.issubset(outputs):
            raise ValueError("upstream prompt output lacks required fields")
        object_ids = np.asarray(outputs["out_obj_ids"])
        scores = np.asarray(outputs["out_probs"], dtype=np.float64)
        boxes = np.asarray(outputs["out_boxes_xywh"], dtype=np.float64)
        masks = np.asarray(outputs["out_binary_masks"])
        count = int(masks.shape[0]) if masks.ndim == 3 else -1
        if (
            count < 0
            or tuple(masks.shape[1:]) != (state.height, state.width)
            or object_ids.shape != (count,)
            or scores.shape != (count,)
            or boxes.shape != (count, 4)
        ):
            raise ValueError("upstream prompt output shapes are inconsistent")
        if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
            raise ValueError("upstream prompt scores or boxes are non-finite")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("upstream out_probs falls outside [0, 1]")
        if not np.isin(masks, (False, True, 0, 1)).all():
            raise ValueError("upstream masks are not binary")
        if count == 0:
            return CandidateGenerationResult(no_result=CandidateNoResult(
                "UPSTREAM_NO_MASKS", "upstream completed the prompt and returned zero masks",
                self._outcome_telemetry(
                    state=state,
                    context=context,
                    prompt=prompt,
                    metadata=metadata,
                    runtime_seconds=runtime_seconds,
                    peak_allocated=peak_allocated,
                    peak_reserved=peak_reserved,
                ),
            ))
        retained_indices = tuple(
            candidate_index
            for candidate_index in range(count)
            if bool(masks[candidate_index].any())
        )
        if not retained_indices:
            return CandidateGenerationResult(no_result=CandidateNoResult(
                "UPSTREAM_NO_RETAINED_MASKS",
                "upstream completed the prompt; all returned empty masks were filtered",
                self._outcome_telemetry(
                    state=state,
                    context=context,
                    prompt=prompt,
                    metadata=metadata,
                    runtime_seconds=runtime_seconds,
                    peak_allocated=peak_allocated,
                    peak_reserved=peak_reserved,
                ),
            ))
        candidates: list[CandidateRecord] = []
        for candidate_index in retained_indices:
            mask = masks[candidate_index].astype(bool, copy=False)
            rle = encode_binary_mask_array(mask)
            mask_sha256 = hashlib.sha256(canonical_json_bytes({
                "width": state.width,
                "height": state.height,
                "rle": rle,
            })).hexdigest()
            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            candidate_id = stable_id("candidate", {
                "trajectory_id": trajectory_id,
                "step_index": metadata.sequence_index,
                "candidate_index": candidate_index,
                "mask_sha256": mask_sha256,
            })
            candidates.append(CandidateRecord(
                candidate_id=candidate_id,
                trajectory_id=trajectory_id,
                asset_id=state.asset_id,
                model_spec_id=state.model_spec_id,
                prompt_id=prompt.prompt_id,
                step_index=metadata.sequence_index,
                candidate_index=candidate_index,
                width=state.width,
                height=state.height,
                mask_rle=rle,
                mask_sha256=mask_sha256,
                bbox_xyxy=bbox,
                model_score=float(scores[candidate_index]),
                state_id=state.state_id,
                dataset_id=context.dataset_id,
                sample_id=context.sample_id,
                image_sha256=state.image_sha256,
                model_source_commit=PINNED_SAM31_COMMIT,
                checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
                prompt_class_id=metadata.class_id,
                canonical_prompt_text=metadata.canonical_text,
                prompt_family=metadata.family,
                prompt_sequence_index=metadata.sequence_index,
                upstream_score_field="out_probs",
                upstream_object_id=int(object_ids[candidate_index]),
                upstream_bbox_xywh=tuple(float(value) for value in boxes[candidate_index]),
                image_session_initialization_count=1,
                model_build_count=self.model_build_count,
                prompt_call_count=self.prompt_call_count,
                runtime_seconds=runtime_seconds,
                peak_allocated_vram_bytes=peak_allocated,
                peak_reserved_vram_bytes=peak_reserved,
                physical_gpu_index=self.physical_gpu_index,
                logical_device="cuda:0",
                config_hash=self.config_hash,
            ))
        return CandidateGenerationResult(
            candidate=candidates[0], additional_candidates=tuple(candidates[1:])
        )

    def _outcome_telemetry(
        self,
        *,
        state: EncodedImageState,
        context: _ImageSession,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        runtime_seconds: float,
        peak_allocated: int,
        peak_reserved: int,
    ) -> CandidateOutcomeTelemetry:
        return CandidateOutcomeTelemetry(
            dataset_id=context.dataset_id,
            sample_id=context.sample_id,
            asset_id=state.asset_id,
            image_sha256=state.image_sha256,
            model_source_commit=PINNED_SAM31_COMMIT,
            checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
            prompt_id=prompt.prompt_id,
            prompt_class_id=metadata.class_id,
            canonical_prompt_text=metadata.canonical_text,
            prompt_family=metadata.family,
            prompt_sequence_index=metadata.sequence_index,
            image_session_initialization_count=1,
            model_build_count=self.model_build_count,
            prompt_call_count=self.prompt_call_count,
            runtime_seconds=runtime_seconds,
            peak_allocated_vram_bytes=peak_allocated,
            peak_reserved_vram_bytes=peak_reserved,
            physical_gpu_index=self.physical_gpu_index,
            logical_device="cuda:0",
            config_hash=self.config_hash,
        )

    def _close_state(self, state_id: str) -> None:
        context = self._states.pop(state_id, None)
        if context is None:
            return
        try:
            self._predictor.handle_request({
                "type": "close_session",
                "session_id": context.session_id,
                "run_gc_collect": True,
                "clear_cache_threshold": 80,
            })
        finally:
            if context.owns_temporary:
                context.image_path.unlink(missing_ok=True)
                self._temporary_paths.discard(context.image_path)

    def close(self) -> None:
        for state_id in tuple(self._states):
            self._close_state(state_id)
        for path in tuple(self._temporary_paths):
            path.unlink(missing_ok=True)
            self._temporary_paths.discard(path)

    def __enter__(self) -> "Sam31RealFrozenBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
