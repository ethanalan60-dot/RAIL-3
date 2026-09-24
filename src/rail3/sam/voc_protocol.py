"""Frozen, label-free VOC text-prompt and cache-identity protocol."""

from __future__ import annotations

from typing import Any

from rail3.cache import CacheIdentity
from rail3.contracts import stable_id
from rail3.data.voc_taxonomy import VOC_FOREGROUND_CLASSES
from rail3.sam.protocols import PromptKind, PromptRequest
from rail3.sam.sam31_backend import PromptMetadata


SAM31_VOC_CACHE_SCHEMA = "rail3.sam31-voc-text-cache.v1"


def canonical_voc_prompts() -> tuple[tuple[PromptRequest, PromptMetadata], ...]:
    result = []
    for sequence_index, item in enumerate(VOC_FOREGROUND_CLASSES):
        prompt = PromptRequest.create(
            kind=PromptKind.TEXT,
            text=item.canonical_prompt,
            object_id=f"voc2012-class-{item.class_id}",
            seed=0,
        )
        result.append((prompt, PromptMetadata(
            class_id=item.class_id,
            canonical_text=item.canonical_prompt,
            family="canonical_voc_foreground_class_text_v1",
            sequence_index=sequence_index,
        )))
    return tuple(result)


def voc_state_id(item: dict[str, Any], model_spec_id: str) -> str:
    return stable_id("encoded_state", {
        "model_spec_id": model_spec_id,
        "asset_id": item["asset_id"],
        "image_sha256": item["image_sha256"],
        "width": int(item["width"]),
        "height": int(item["height"]),
    })


def voc_cache_identity(
    item: dict[str, Any], prompt: PromptRequest, *, model_spec_id: str
) -> CacheIdentity:
    return CacheIdentity(
        model_spec_id=model_spec_id,
        asset_id=item["asset_id"],
        image_sha256=item["image_sha256"],
        prompt_id=prompt.prompt_id,
        state_id=voc_state_id(item, model_spec_id),
        adapter_schema=SAM31_VOC_CACHE_SCHEMA,
    )
