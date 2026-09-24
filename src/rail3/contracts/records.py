"""Immutable M02 data, authorization, label, and split records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from rail3.contracts.serialization import stable_id, to_primitive


SCHEMA_VERSION = "rail3.manifest.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AuthorizationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"


class RecordStatus(str, Enum):
    VALID = "valid"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class SplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


class HiddenLabelStage(str, Enum):
    FIT_TARGET_BUILD = "FIT_TARGET_BUILD"
    CALIBRATION = "CALIBRATION"
    AUTO_ANNOTATION = "AUTO_ANNOTATION"
    FINAL_EVALUATION = "FINAL_EVALUATION"


class UseTag(str, Enum):
    INTERNAL_RESEARCH = "internal_research"
    MODEL_TRAINING = "model_training"
    MODEL_SELECTION = "model_selection"
    RISK_CALIBRATION = "risk_calibration"
    AUTO_ANNOTATION = "auto_annotation"
    FINAL_EVALUATION = "final_evaluation"
    HUMAN_VIEW = "human_view"
    DERIVED_LABEL_STORAGE = "derived_label_storage"
    AGGREGATE_PUBLICATION = "aggregate_publication"
    PER_SAMPLE_PUBLICATION = "per_sample_publication"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_relative_uri(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if "\\" in value or "://" in value or value.startswith("/"):
        raise ValueError(f"{field_name} must be a relative logical URI")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} contains an unsafe path component")


def _require_status_reason(status: RecordStatus, reason_code: str | None) -> None:
    if status is not RecordStatus.VALID and not reason_code:
        raise ValueError("non-valid records require a structured reason_code")
    if reason_code is not None and re.fullmatch(r"[A-Z][A-Z0-9_]*", reason_code) is None:
        raise ValueError("reason_code must be uppercase snake case")


def _unique_sorted(values: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(set(values), key=lambda value: getattr(value, "value", str(value))))


@dataclass(frozen=True)
class Provenance:
    parent_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    transform_spec: str | None = None
    code_commit: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.code_commit is not None and re.fullmatch(r"[0-9a-f]{40}", self.code_commit) is None:
            raise ValueError("code_commit must be a full lowercase Git commit")
        if self.transform_spec is not None:
            _require_text(self.transform_spec, "transform_spec")


@dataclass(frozen=True)
class AuthorizationRecord:
    authorization_record_id: str
    dataset_id: str
    canonical_name: str
    dataset_version: str
    status: AuthorizationStatus
    license_ref: str
    allowed_uses: tuple[UseTag, ...]
    reviewed_on: str | None = None
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("authorization_record_id", "dataset_id", "canonical_name", "dataset_version", "license_ref"):
            _require_text(getattr(self, name), name)
        if self.status is not AuthorizationStatus.APPROVED and not self.reason_code:
            raise ValueError("non-approved authorization requires reason_code")
        if self.reason_code is not None and re.fullmatch(r"[A-Z][A-Z0-9_]*", self.reason_code) is None:
            raise ValueError("authorization reason_code must be uppercase snake case")
        object.__setattr__(self, "allowed_uses", _unique_sorted(self.allowed_uses))

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        canonical_name: str,
        dataset_version: str,
        status: AuthorizationStatus,
        license_ref: str,
        allowed_uses: tuple[UseTag, ...],
        reviewed_on: str | None = None,
        reason_code: str | None = None,
    ) -> "AuthorizationRecord":
        identity = stable_id("authorization", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "canonical_name": canonical_name,
            "dataset_version": dataset_version,
            "license_ref": license_ref,
        })
        return cls(identity, dataset_id, canonical_name, dataset_version, status, license_ref, allowed_uses, reviewed_on, reason_code)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    canonical_name: str
    dataset_version: str
    authorization_record_id: str
    enabled: bool
    use_tags: tuple[UseTag, ...]
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_id", "canonical_name", "dataset_version", "authorization_record_id"):
            _require_text(getattr(self, name), name)
        if self.dataset_version.lower() == "latest":
            raise ValueError("dataset_version must be explicit, never 'latest'")
        object.__setattr__(self, "use_tags", _unique_sorted(self.use_tags))
        _require_status_reason(self.status, self.reason_code)


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    dataset_id: str
    grouping_spec: str
    source_group_key: str
    asset_ids: tuple[str, ...]
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("group_id", "dataset_id", "grouping_spec", "source_group_key"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "asset_ids", tuple(sorted(set(self.asset_ids))))
        _require_status_reason(self.status, self.reason_code)

    @classmethod
    def create(cls, *, dataset_id: str, grouping_spec: str, source_group_key: str, asset_ids: tuple[str, ...] = ()) -> "GroupRecord":
        group_id = stable_id("group", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "grouping_spec": grouping_spec,
            "source_group_key": source_group_key,
        })
        return cls(group_id, dataset_id, grouping_spec, source_group_key, asset_ids)


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    dataset_id: str
    dataset_version: str
    official_sample_key: str
    group_id: str
    image_uri: str
    label_uri: str | None
    content_sha256: str
    label_content_sha256: str | None
    label_spec_id: str | None
    official_split: SplitRole | None
    use_tags: tuple[UseTag, ...]
    annotation_uri: str | None = None
    width: int | None = None
    height: int | None = None
    image_mode: str | None = None
    image_channels: int | None = None
    present_class_ids: tuple[int, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("asset_id", "dataset_id", "dataset_version", "official_sample_key", "group_id"):
            _require_text(getattr(self, name), name)
        _require_relative_uri(self.image_uri, "image_uri")
        if self.label_uri is not None:
            _require_relative_uri(self.label_uri, "label_uri")
        if self.annotation_uri is not None:
            _require_relative_uri(self.annotation_uri, "annotation_uri")
        _require_sha256(self.content_sha256, "content_sha256")
        if self.label_content_sha256 is not None:
            _require_sha256(self.label_content_sha256, "label_content_sha256")
        object.__setattr__(self, "use_tags", _unique_sorted(self.use_tags))
        object.__setattr__(self, "present_class_ids", tuple(sorted(set(self.present_class_ids))))
        media_values = (self.width, self.height, self.image_mode, self.image_channels)
        if any(value is not None for value in media_values):
            if not all(value is not None for value in media_values):
                raise ValueError("asset image metadata must be either complete or absent")
            if self.width <= 0 or self.height <= 0 or self.image_channels <= 0:
                raise ValueError("asset image dimensions and channels must be positive")
            _require_text(self.image_mode, "image_mode")
        if any(value < 0 for value in self.present_class_ids):
            raise ValueError("present_class_ids cannot contain negative class IDs")
        _require_status_reason(self.status, self.reason_code)

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        dataset_version: str,
        official_sample_key: str,
        group_id: str,
        image_uri: str,
        content_sha256: str,
        use_tags: tuple[UseTag, ...],
        label_uri: str | None = None,
        label_content_sha256: str | None = None,
        label_spec_id: str | None = None,
        official_split: SplitRole | None = None,
        annotation_uri: str | None = None,
        width: int | None = None,
        height: int | None = None,
        image_mode: str | None = None,
        image_channels: int | None = None,
        present_class_ids: tuple[int, ...] = (),
        provenance: Provenance | None = None,
    ) -> "AssetRecord":
        asset_id = stable_id("asset", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "official_sample_key": official_sample_key,
            "content_sha256": content_sha256,
        })
        return cls(
            asset_id=asset_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            official_sample_key=official_sample_key,
            group_id=group_id,
            image_uri=image_uri,
            label_uri=label_uri,
            content_sha256=content_sha256,
            label_content_sha256=label_content_sha256,
            label_spec_id=label_spec_id,
            official_split=official_split,
            use_tags=use_tags,
            annotation_uri=annotation_uri,
            width=width,
            height=height,
            image_mode=image_mode,
            image_channels=image_channels,
            present_class_ids=present_class_ids,
            provenance=provenance or Provenance(),
        )


@dataclass(frozen=True)
class SplitRecord:
    split_record_id: str
    split_spec_id: str
    group_id: str
    role: SplitRole
    asset_ids: tuple[str, ...]
    source: str
    allowed_label_stages: tuple[HiddenLabelStage, ...]
    approved: bool
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("split_record_id", "split_spec_id", "group_id", "source"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "asset_ids", tuple(sorted(set(self.asset_ids))))
        object.__setattr__(self, "allowed_label_stages", _unique_sorted(self.allowed_label_stages))
        _require_status_reason(self.status, self.reason_code)

    @classmethod
    def create(
        cls,
        *,
        split_spec_id: str,
        group_id: str,
        role: SplitRole,
        asset_ids: tuple[str, ...],
        source: str,
        approved: bool = True,
        allowed_label_stages: tuple[HiddenLabelStage, ...] | None = None,
    ) -> "SplitRecord":
        defaults = {
            SplitRole.TRAIN: (
                HiddenLabelStage.FIT_TARGET_BUILD,
                HiddenLabelStage.FINAL_EVALUATION,
            ),
            SplitRole.VALIDATION: (HiddenLabelStage.FINAL_EVALUATION,),
            SplitRole.CALIBRATION: (
                HiddenLabelStage.CALIBRATION,
                HiddenLabelStage.FINAL_EVALUATION,
            ),
            SplitRole.TEST: (HiddenLabelStage.FINAL_EVALUATION,),
        }
        stages = defaults[role] if allowed_label_stages is None else allowed_label_stages
        split_record_id = stable_id("split", {
            "schema_version": SCHEMA_VERSION,
            "split_spec_id": split_spec_id,
            "group_id": group_id,
            "role": role,
            "asset_ids": sorted(set(asset_ids)),
        })
        return cls(split_record_id, split_spec_id, group_id, role, asset_ids, source, stages, approved)


@dataclass(frozen=True)
class DerivedAssetRecord:
    derived_asset_id: str
    parent_asset_id: str
    group_id: str
    split_role: SplitRole
    artifact_uri: str
    content_sha256: str
    transform_spec: str
    use_tags: tuple[UseTag, ...]
    provenance: Provenance
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("derived_asset_id", "parent_asset_id", "group_id", "transform_spec"):
            _require_text(getattr(self, name), name)
        _require_relative_uri(self.artifact_uri, "artifact_uri")
        _require_sha256(self.content_sha256, "content_sha256")
        object.__setattr__(self, "use_tags", _unique_sorted(self.use_tags))
        if self.parent_asset_id not in self.provenance.parent_ids:
            raise ValueError("derived provenance must include parent_asset_id")
        _require_status_reason(self.status, self.reason_code)

    @classmethod
    def create(
        cls,
        *,
        parent_asset_id: str,
        group_id: str,
        split_role: SplitRole,
        artifact_uri: str,
        content_sha256: str,
        transform_spec: str,
        use_tags: tuple[UseTag, ...],
        provenance: Provenance | None = None,
    ) -> "DerivedAssetRecord":
        identity = stable_id("derived_asset", {
            "schema_version": SCHEMA_VERSION,
            "parent_asset_id": parent_asset_id,
            "transform_spec": transform_spec,
            "content_sha256": content_sha256,
        })
        lineage = provenance or Provenance(parent_ids=(parent_asset_id,), transform_spec=transform_spec)
        return cls(identity, parent_asset_id, group_id, split_role, artifact_uri, content_sha256, transform_spec, use_tags, lineage)


@dataclass(frozen=True)
class LabelSpec:
    label_spec_id: str
    dataset_id: str
    task: str
    class_ids: tuple[int, ...]
    ignore_index: int
    void_values: tuple[int, ...]
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("label_spec_id", "dataset_id", "task"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "class_ids", tuple(sorted(set(self.class_ids))))
        object.__setattr__(self, "void_values", tuple(sorted(set(self.void_values))))
        if self.ignore_index in self.class_ids:
            raise ValueError("ignore_index cannot be a train/evaluation class ID")
        _require_status_reason(self.status, self.reason_code)

    @classmethod
    def create(cls, *, dataset_id: str, task: str, class_ids: tuple[int, ...], ignore_index: int, void_values: tuple[int, ...]) -> "LabelSpec":
        identity = stable_id("label_spec", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "task": task,
            "class_ids": sorted(set(class_ids)),
            "ignore_index": ignore_index,
            "void_values": sorted(set(void_values)),
        })
        return cls(identity, dataset_id, task, class_ids, ignore_index, void_values)


@dataclass(frozen=True)
class ManifestEnvelope:
    manifest_id: str
    dataset: DatasetSpec
    authorization: AuthorizationRecord
    groups: tuple[GroupRecord, ...]
    assets: tuple[AssetRecord, ...]
    splits: tuple[SplitRecord, ...]
    derived_assets: tuple[DerivedAssetRecord, ...] = ()
    label_specs: tuple[LabelSpec, ...] = ()
    status: RecordStatus = RecordStatus.VALID
    reason_code: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.manifest_id, "manifest_id")
        _require_status_reason(self.status, self.reason_code)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": to_primitive(self.dataset),
            "authorization": to_primitive(self.authorization),
            "groups": [to_primitive(item) for item in self.groups],
            "assets": [to_primitive(item) for item in self.assets],
            "splits": [to_primitive(item) for item in self.splits],
            "derived_assets": [to_primitive(item) for item in self.derived_assets],
            "label_specs": [to_primitive(item) for item in self.label_specs],
            "status": self.status,
            "reason_code": self.reason_code,
        }

    def expected_manifest_id(self) -> str:
        return stable_id("manifest", self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def create(
        cls,
        *,
        dataset: DatasetSpec,
        authorization: AuthorizationRecord,
        groups: tuple[GroupRecord, ...],
        assets: tuple[AssetRecord, ...],
        splits: tuple[SplitRecord, ...],
        derived_assets: tuple[DerivedAssetRecord, ...] = (),
        label_specs: tuple[LabelSpec, ...] = (),
        status: RecordStatus = RecordStatus.VALID,
        reason_code: str | None = None,
    ) -> "ManifestEnvelope":
        ordered = {
            "groups": tuple(sorted(groups, key=lambda item: item.group_id)),
            "assets": tuple(sorted(assets, key=lambda item: item.asset_id)),
            "splits": tuple(sorted(splits, key=lambda item: item.split_record_id)),
            "derived_assets": tuple(sorted(derived_assets, key=lambda item: item.derived_asset_id)),
            "label_specs": tuple(sorted(label_specs, key=lambda item: item.label_spec_id)),
        }
        provisional = cls("pending", dataset, authorization, status=status, reason_code=reason_code, **ordered)
        return cls(provisional.expected_manifest_id(), dataset, authorization, status=status, reason_code=reason_code, **ordered)
