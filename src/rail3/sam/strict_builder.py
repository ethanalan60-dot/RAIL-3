"""Thin project-local strict builder for one pinned SAM 3.1 pair.

The upstream predictor builder cannot be reused directly because it sends the
complete merged checkpoint through an intermediate tracker load before assembling
the final model.  This module uses the pinned upstream component constructors,
passes no checkpoint to intermediate construction, and performs one project-owned
final strict load after deterministic buffer completion.
"""

from __future__ import annotations

import gc
import pkg_resources
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rail3.sam.checkpoint_compat import (
    OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
    PINNED_SAM31_CHECKPOINT_SHA256,
    PINNED_SAM31_COMMIT,
    CheckpointContractAnalysis,
    PinnedSourceMismatchError,
    StrictLoadResult,
    complete_checkpoint_state_in_memory,
    load_checkpoint_state_cpu,
    load_completed_state_dict_strict,
    tensor_sha256,
)


@dataclass(frozen=True)
class StrictBuilderEvidence:
    source_commit: str
    checkpoint_sha256: str
    checkpoint_key_count: int
    final_model_key_count: int
    intermediate_checkpoint_loads: int
    final_strict_loads: int
    learned_parameter_coverage: float
    checkpoint_key_usage: float
    generated_buffer_count: int
    generated_buffer_aggregate_sha256: str
    generated_buffer_hashes: tuple[tuple[str, str], ...]
    construction_seconds: float
    checkpoint_cpu_load_seconds: float
    completion_validation_seconds: float
    strict_load_seconds: float

    def as_dict(self, *, include_individual_buffer_hashes: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_commit": self.source_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_key_count": self.checkpoint_key_count,
            "final_model_key_count": self.final_model_key_count,
            "intermediate_checkpoint_loads": self.intermediate_checkpoint_loads,
            "final_strict_loads": self.final_strict_loads,
            "learned_parameter_coverage": self.learned_parameter_coverage,
            "checkpoint_key_usage": self.checkpoint_key_usage,
            "generated_buffer_count": self.generated_buffer_count,
            "generated_buffer_aggregate_sha256": self.generated_buffer_aggregate_sha256,
            "construction_seconds": self.construction_seconds,
            "checkpoint_cpu_load_seconds": self.checkpoint_cpu_load_seconds,
            "completion_validation_seconds": self.completion_validation_seconds,
            "strict_load_seconds": self.strict_load_seconds,
        }
        if include_individual_buffer_hashes:
            result["generated_buffer_hashes"] = dict(self.generated_buffer_hashes)
        return result


@dataclass(frozen=True)
class StrictPredictorBuild:
    predictor: Any
    analysis: CheckpointContractAnalysis
    strict_load: StrictLoadResult
    evidence: StrictBuilderEvidence


def verify_pinned_source_repo(source: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1"], text=True
    ).strip()
    if commit != PINNED_SAM31_COMMIT:
        raise PinnedSourceMismatchError(
            "local upstream source is not at the pinned SAM 3.1 commit",
            expected=PINNED_SAM31_COMMIT,
            actual=commit,
            source_commit=commit,
            checkpoint_hash=PINNED_SAM31_CHECKPOINT_SHA256,
        )
    if status:
        raise PinnedSourceMismatchError(
            "local pinned upstream source worktree is not clean",
            expected="clean",
            actual="dirty",
            source_commit=commit,
            checkpoint_hash=PINNED_SAM31_CHECKPOINT_SHA256,
            reason_code="pinned_source_dirty",
        )
    return {"commit": commit, "tree": tree, "clean": True}


def build_final_sam31_structure(
    *,
    source: Path,
    checkpoint_path: Path,
    bpe_path: str | None = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = True,
    compile_model: bool = False,
    device: str = "cuda",
) -> Any:
    """Assemble the pinned final model with zero intermediate checkpoint loads."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError("pinned SAM 3.1 checkpoint is absent")
    if use_fa3 is not False or use_rope_real is not True or compile_model is not False:
        raise ValueError("strict SAM 3.1 builder parameters differ from the reviewed contract")
    if device not in {"cpu", "cuda"}:
        raise ValueError("strict SAM 3.1 builder device must be cpu or cuda")

    import sam3.model_builder as model_builder
    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity

    expected_builder = (source / "sam3" / "model_builder.py").resolve()
    imported_builder = Path(model_builder.__file__).resolve()
    if imported_builder != expected_builder:
        raise PinnedSourceMismatchError(
            "imported SAM builder is not the verified pinned source checkout",
            expected=expected_builder.as_posix(),
            actual=imported_builder.as_posix(),
            source_commit=PINNED_SAM31_COMMIT,
            checkpoint_hash=PINNED_SAM31_CHECKPOINT_SHA256,
            reason_code="pinned_source_import_path_mismatch",
        )

    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # This is the official tracker constructor with checkpoint loading disabled.
    tracker_model = model_builder.build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile=False,
        strict_state_dict_loading=True,
        device="cpu",
    )
    del tracker_model.backbone
    tracker_model.backbone = None
    tracker = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )

    # The remaining calls and constructor arguments are the pinned upstream
    # build_sam3_multiplex_video_predictor assembly surface. No upstream code is
    # patched, and no checkpoint is passed through any of these constructors.
    tri_neck = model_builder._create_multiplex_tri_backbone(
        compile_mode=None,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )
    text_encoder = model_builder._create_text_encoder(bpe_path)
    backbone = model_builder.SAM3VLBackboneTri(
        scalp=0,
        visual=tri_neck,
        text=text_encoder,
    )
    transformer = model_builder._create_sam3_transformer(use_fa3=use_fa3)
    segmentation_head = model_builder._create_segmentation_head(use_fa3=use_fa3)
    geometry_encoder = model_builder._create_geometry_encoder()
    dot_prod_scoring = model_builder._create_dot_product_scoring()
    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    final_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=tracker,
        detector=detector,
        score_threshold_detection=0.4,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=compile_model,
    )
    return final_model.to(device=device)


def build_strict_sam31_predictor(
    *,
    source: Path,
    checkpoint_path: Path,
    bpe_path: str | None = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = True,
    compile_model: bool = False,
    warm_up: bool = False,
    async_loading_frames: bool = False,
    session_expiration_sec: int = 1200,
    default_output_prob_thresh: float = 0.5,
    device: str = "cuda",
) -> StrictPredictorBuild:
    """Build, complete, and strictly load the one approved SAM 3.1 model."""

    source_result = verify_pinned_source_repo(source)
    if source_result["commit"] != PINNED_SAM31_COMMIT:
        raise AssertionError("source verification did not preserve the pinned commit")

    construction_started = time.perf_counter()
    model = build_final_sam31_structure(
        source=source,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile_model=compile_model,
        device=device,
    )
    construction_seconds = time.perf_counter() - construction_started

    checkpoint_started = time.perf_counter()
    checkpoint_state = load_checkpoint_state_cpu(
        checkpoint_path,
        source_commit=PINNED_SAM31_COMMIT,
        checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    completion_started = time.perf_counter()
    completed_state, analysis = complete_checkpoint_state_in_memory(
        model,
        checkpoint_state,
        source_commit=PINNED_SAM31_COMMIT,
        checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
    )
    completion_seconds = time.perf_counter() - completion_started

    strict_started = time.perf_counter()
    strict_result = load_completed_state_dict_strict(
        model,
        completed_state,
        source_commit=PINNED_SAM31_COMMIT,
        checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
    )
    strict_seconds = time.perf_counter() - strict_started

    del completed_state, checkpoint_state
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    buffer_map = dict(model.named_buffers())
    post_load_hashes = {
        spec.key: tensor_sha256(buffer_map[spec.key])
        for spec in OFFICIAL_SAM31_CHECKPOINT_CONTRACT.buffers
    }
    expected_hashes = dict(analysis.generated_buffer_hashes)
    if post_load_hashes != expected_hashes:
        raise RuntimeError("post-load generated RoPE buffers differ from validated values")

    from sam3.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor

    predictor = Sam3MultiplexVideoPredictor(
        model=model,
        session_expiration_sec=session_expiration_sec,
        default_output_prob_thresh=default_output_prob_thresh,
        async_loading_frames=async_loading_frames,
        warm_up=warm_up,
    )
    evidence = StrictBuilderEvidence(
        source_commit=PINNED_SAM31_COMMIT,
        checkpoint_sha256=PINNED_SAM31_CHECKPOINT_SHA256,
        checkpoint_key_count=len(analysis.shared_keys),
        final_model_key_count=(
            len(analysis.shared_keys) + len(analysis.generated_buffer_keys)
        ),
        intermediate_checkpoint_loads=0,
        final_strict_loads=1,
        learned_parameter_coverage=analysis.learned_parameter_coverage,
        checkpoint_key_usage=analysis.checkpoint_key_usage,
        generated_buffer_count=len(post_load_hashes),
        generated_buffer_aggregate_sha256=analysis.generated_buffer_aggregate_sha256,
        generated_buffer_hashes=tuple(sorted(post_load_hashes.items())),
        construction_seconds=construction_seconds,
        checkpoint_cpu_load_seconds=checkpoint_seconds,
        completion_validation_seconds=completion_seconds,
        strict_load_seconds=strict_seconds,
    )
    return StrictPredictorBuild(
        predictor=predictor,
        analysis=analysis,
        strict_load=strict_result,
        evidence=evidence,
    )
