"""Label-free canonical VOC2012 class and text-prompt vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VocClass:
    class_id: int
    source_name: str
    canonical_prompt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "source_name": self.source_name,
            "canonical_prompt": self.canonical_prompt,
        }


VOC_FOREGROUND_CLASSES = (
    VocClass(1, "aeroplane", "aeroplane"),
    VocClass(2, "bicycle", "bicycle"),
    VocClass(3, "bird", "bird"),
    VocClass(4, "boat", "boat"),
    VocClass(5, "bottle", "bottle"),
    VocClass(6, "bus", "bus"),
    VocClass(7, "car", "car"),
    VocClass(8, "cat", "cat"),
    VocClass(9, "chair", "chair"),
    VocClass(10, "cow", "cow"),
    VocClass(11, "diningtable", "dining table"),
    VocClass(12, "dog", "dog"),
    VocClass(13, "horse", "horse"),
    VocClass(14, "motorbike", "motorbike"),
    VocClass(15, "person", "person"),
    VocClass(16, "pottedplant", "potted plant"),
    VocClass(17, "sheep", "sheep"),
    VocClass(18, "sofa", "sofa"),
    VocClass(19, "train", "train"),
    VocClass(20, "tvmonitor", "tv/monitor"),
)
VOC_FOREGROUND_IDS = tuple(item.class_id for item in VOC_FOREGROUND_CLASSES)
VOC_BACKGROUND_ID = 0
VOC_IGNORE_ID = 255
VOC_ALLOWED_MASK_VALUES = frozenset(
    (VOC_BACKGROUND_ID, *VOC_FOREGROUND_IDS, VOC_IGNORE_ID)
)

