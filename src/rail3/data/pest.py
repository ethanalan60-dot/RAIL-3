"""Pest-domain taxonomy and bounding-box contracts with no dataset I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from rail3.contracts import SplitRole, stable_id
from rail3.contracts.serialization import to_primitive


PEST_TAXONOMY_SCHEMA_VERSION = "rail3.pest_taxonomy.v1"
PEST_BBOX_SCHEMA_VERSION = "rail3.pest_bbox.v1"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class BiologicalType(str, Enum):
    INSECT_PEST = "insect_pest"
    OTHER_ARTHROPOD_PEST = "other_arthropod_pest"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    AMBIGUOUS = "ambiguous"


_INCLUDABLE_TYPES = {
    BiologicalType.INSECT_PEST,
    BiologicalType.OTHER_ARTHROPOD_PEST,
}


@dataclass(frozen=True)
class PestTaxonomyClass:
    """One reviewed source class and its fail-closed pest decision."""

    class_id: str
    original_name: str
    normalized_name: str
    crop: str
    biological_type: BiologicalType
    evidence_source: str
    include_in_pest_subset: bool
    reason: str

    def __post_init__(self) -> None:
        for field_name in (
            "class_id",
            "original_name",
            "normalized_name",
            "crop",
            "evidence_source",
            "reason",
        ):
            _require_text(getattr(self, field_name), field_name)
        if not isinstance(self.biological_type, BiologicalType):
            raise ValueError("biological_type must be a BiologicalType value")
        if not isinstance(self.include_in_pest_subset, bool):
            raise ValueError("include_in_pest_subset must be a boolean")
        if self.include_in_pest_subset and self.biological_type not in _INCLUDABLE_TYPES:
            raise ValueError(
                "only confirmed insect or other arthropod pests may enter the pest subset"
            )

    @classmethod
    def create(
        cls,
        *,
        class_id: str,
        original_name: str,
        normalized_name: str,
        crop: str,
        biological_type: BiologicalType,
        evidence_source: str,
        reason: str,
        include_in_pest_subset: bool = False,
    ) -> "PestTaxonomyClass":
        return cls(
            class_id=class_id,
            original_name=original_name,
            normalized_name=normalized_name,
            crop=crop,
            biological_type=biological_type,
            evidence_source=evidence_source,
            include_in_pest_subset=include_in_pest_subset,
            reason=reason,
        )


def _taxonomy_identity_payload(
    *,
    source_dataset_id: str,
    source_dataset_version: str,
    parent_manifest_id: str,
    source_taxonomy_ref: str,
    classes: tuple[PestTaxonomyClass, ...],
) -> dict[str, object]:
    return {
        "schema_version": PEST_TAXONOMY_SCHEMA_VERSION,
        "source_dataset_id": source_dataset_id,
        "source_dataset_version": source_dataset_version,
        "parent_manifest_id": parent_manifest_id,
        "source_taxonomy_ref": source_taxonomy_ref,
        "classes": [to_primitive(item) for item in classes],
    }


@dataclass(frozen=True)
class PestTaxonomyManifest:
    """Immutable, content-addressed class audit derived from a parent manifest."""

    taxonomy_manifest_id: str
    source_dataset_id: str
    source_dataset_version: str
    parent_manifest_id: str
    source_taxonomy_ref: str
    classes: tuple[PestTaxonomyClass, ...]
    schema_version: str = PEST_TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "taxonomy_manifest_id",
            "source_dataset_id",
            "source_dataset_version",
            "parent_manifest_id",
            "source_taxonomy_ref",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.schema_version != PEST_TAXONOMY_SCHEMA_VERSION:
            raise ValueError("unsupported pest taxonomy schema version")
        if any(not isinstance(item, PestTaxonomyClass) for item in self.classes):
            raise ValueError("taxonomy classes must be PestTaxonomyClass records")
        class_ids = tuple(item.class_id for item in self.classes)
        if not class_ids:
            raise ValueError("taxonomy manifest must contain at least one reviewed class")
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("taxonomy class_id values must be unique")
        if class_ids != tuple(sorted(class_ids)):
            raise ValueError("taxonomy classes must be sorted by class_id")
        if self.taxonomy_manifest_id != self.expected_manifest_id():
            raise ValueError("taxonomy_manifest_id does not match canonical content")

    def expected_manifest_id(self) -> str:
        return stable_id(
            "pest_taxonomy",
            _taxonomy_identity_payload(
                source_dataset_id=self.source_dataset_id,
                source_dataset_version=self.source_dataset_version,
                parent_manifest_id=self.parent_manifest_id,
                source_taxonomy_ref=self.source_taxonomy_ref,
                classes=self.classes,
            ),
        )

    @property
    def content_sha256(self) -> str:
        return self.taxonomy_manifest_id.removeprefix("pest_taxonomy_")

    @property
    def included_class_ids(self) -> tuple[str, ...]:
        return tuple(
            item.class_id for item in self.classes if item.include_in_pest_subset
        )

    @classmethod
    def create(
        cls,
        *,
        source_dataset_id: str,
        source_dataset_version: str,
        parent_manifest_id: str,
        source_taxonomy_ref: str,
        classes: tuple[PestTaxonomyClass, ...],
    ) -> "PestTaxonomyManifest":
        ordered = tuple(sorted(classes, key=lambda item: item.class_id))
        identity = stable_id(
            "pest_taxonomy",
            _taxonomy_identity_payload(
                source_dataset_id=source_dataset_id,
                source_dataset_version=source_dataset_version,
                parent_manifest_id=parent_manifest_id,
                source_taxonomy_ref=source_taxonomy_ref,
                classes=ordered,
            ),
        )
        return cls(
            taxonomy_manifest_id=identity,
            source_dataset_id=source_dataset_id,
            source_dataset_version=source_dataset_version,
            parent_manifest_id=parent_manifest_id,
            source_taxonomy_ref=source_taxonomy_ref,
            classes=ordered,
        )


@dataclass(frozen=True)
class TaxonomyAuditSummary:
    taxonomy_manifest_id: str
    content_sha256: str
    reviewed_class_count: int
    included_pest_class_count: int
    excluded_class_count: int
    included_class_ids: tuple[str, ...]


def summarize_taxonomy(manifest: PestTaxonomyManifest) -> TaxonomyAuditSummary:
    included = manifest.included_class_ids
    return TaxonomyAuditSummary(
        taxonomy_manifest_id=manifest.taxonomy_manifest_id,
        content_sha256=manifest.content_sha256,
        reviewed_class_count=len(manifest.classes),
        included_pest_class_count=len(included),
        excluded_class_count=len(manifest.classes) - len(included),
        included_class_ids=included,
    )


@dataclass(frozen=True)
class CanonicalBoundingBox:
    """Integer box in zero-based, half-open ``[x_min, x_max)`` coordinates."""

    image_width: int
    image_height: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    coordinate_system: str = "zero_based_half_open"

    def __post_init__(self) -> None:
        if self.coordinate_system != "zero_based_half_open":
            raise ValueError("only zero_based_half_open boxes are canonical")
        coordinate_values = (
            self.image_width,
            self.image_height,
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in coordinate_values):
            raise ValueError("image dimensions and box coordinates must be integers")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not (0 <= self.x_min < self.x_max <= self.image_width):
            raise ValueError("x coordinates are empty or outside image bounds")
        if not (0 <= self.y_min < self.y_max <= self.image_height):
            raise ValueError("y coordinates are empty or outside image bounds")


class BoxOrigin(str, Enum):
    OFFICIAL_GROUND_TRUTH = "official_ground_truth"
    MASK_DERIVED_PREDICTION = "mask_derived_prediction"


@dataclass(frozen=True)
class IP102BoundingBoxRecord:
    box_record_id: str
    parent_asset_id: str
    group_id: str
    split_role: SplitRole
    class_id: str
    annotation_id: str
    box: CanonicalBoundingBox
    origin: BoxOrigin
    parent_candidate_id: str | None = None
    schema_version: str = PEST_BBOX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "box_record_id",
            "parent_asset_id",
            "group_id",
            "class_id",
            "annotation_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.schema_version != PEST_BBOX_SCHEMA_VERSION:
            raise ValueError("unsupported pest bbox schema version")
        if not isinstance(self.split_role, SplitRole):
            raise ValueError("split_role must be a SplitRole value")
        if not isinstance(self.origin, BoxOrigin):
            raise ValueError("origin must be a BoxOrigin value")
        if self.origin is BoxOrigin.OFFICIAL_GROUND_TRUTH and self.parent_candidate_id is not None:
            raise ValueError("official boxes cannot claim a model candidate parent")
        if self.origin is BoxOrigin.MASK_DERIVED_PREDICTION:
            if self.parent_candidate_id is None:
                raise ValueError("mask-derived boxes require parent_candidate_id")
            _require_text(self.parent_candidate_id, "parent_candidate_id")
        if self.box_record_id != self.expected_record_id():
            raise ValueError("box_record_id does not match canonical content")

    def expected_record_id(self) -> str:
        return stable_id(
            "ip102_box",
            {
                "schema_version": self.schema_version,
                "parent_asset_id": self.parent_asset_id,
                "group_id": self.group_id,
                "split_role": self.split_role,
                "class_id": self.class_id,
                "annotation_id": self.annotation_id,
                "box": self.box,
                "origin": self.origin,
                "parent_candidate_id": self.parent_candidate_id,
            },
        )

    @classmethod
    def create(
        cls,
        *,
        parent_asset_id: str,
        group_id: str,
        split_role: SplitRole,
        class_id: str,
        annotation_id: str,
        box: CanonicalBoundingBox,
        origin: BoxOrigin,
        parent_candidate_id: str | None = None,
    ) -> "IP102BoundingBoxRecord":
        payload = {
            "schema_version": PEST_BBOX_SCHEMA_VERSION,
            "parent_asset_id": parent_asset_id,
            "group_id": group_id,
            "split_role": split_role,
            "class_id": class_id,
            "annotation_id": annotation_id,
            "box": box,
            "origin": origin,
            "parent_candidate_id": parent_candidate_id,
        }
        identity = stable_id("ip102_box", payload)
        return cls(
            box_record_id=identity,
            parent_asset_id=parent_asset_id,
            group_id=group_id,
            split_role=split_role,
            class_id=class_id,
            annotation_id=annotation_id,
            box=box,
            origin=origin,
            parent_candidate_id=parent_candidate_id,
        )


def mask_to_bounding_box(mask: Sequence[Sequence[object]]) -> CanonicalBoundingBox | None:
    """Convert a rectangular synthetic/binary mask to a canonical tight box."""

    height = len(mask)
    if height == 0:
        return None
    width = len(mask[0])
    if width == 0:
        if any(len(row) != 0 for row in mask):
            raise ValueError("mask rows must have equal width")
        return None
    if any(len(row) != width for row in mask):
        raise ValueError("mask rows must have equal width")
    occupied = [
        (x, y)
        for y, row in enumerate(mask)
        for x, value in enumerate(row)
        if bool(value)
    ]
    if not occupied:
        return None
    xs, ys = zip(*occupied)
    return CanonicalBoundingBox(
        image_width=width,
        image_height=height,
        x_min=min(xs),
        y_min=min(ys),
        x_max=max(xs) + 1,
        y_max=max(ys) + 1,
    )


@dataclass(frozen=True)
class MetricApplicability:
    metric_id: str
    applicable: bool
    status: str
    reason: str | None


_PIXEL_MASK_METRICS = {"mask_miou", "miou", "dice", "boundary_f1"}


def ip102_metric_applicability(metric_id: str) -> MetricApplicability:
    _require_text(metric_id, "metric_id")
    normalized = metric_id.strip().lower()
    if normalized in _PIXEL_MASK_METRICS:
        return MetricApplicability(
            metric_id=normalized,
            applicable=False,
            status="N/A_NO_PIXEL_MASK_GROUND_TRUTH",
            reason="audited official IP102 sources provide classification labels and bounding boxes, not pixel masks",
        )
    return MetricApplicability(
        metric_id=normalized,
        applicable=True,
        status="APPLICABLE",
        reason=None,
    )
