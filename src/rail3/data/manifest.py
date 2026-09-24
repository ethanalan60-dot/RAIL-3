"""Typed manifest parsing and cross-record validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rail3.contracts import (
    AssetRecord,
    AuthorizationRecord,
    AuthorizationStatus,
    DatasetSpec,
    DerivedAssetRecord,
    GroupRecord,
    HiddenLabelStage,
    LabelSpec,
    ManifestEnvelope,
    Provenance,
    RecordStatus,
    SCHEMA_VERSION,
    SplitRecord,
    SplitRole,
    UseTag,
    stable_id,
)
from rail3.registry.datasets import evaluate_dataset_gate


@dataclass(frozen=True, order=True)
class ValidationIssue:
    code: str
    message: str
    object_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "object_ids": list(self.object_ids)}


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def status(self) -> str:
        return "PASS" if not self.issues else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "issue_count": len(self.issues), "issues": [item.to_dict() for item in self.issues]}


def _provenance(data: dict[str, Any]) -> Provenance:
    return Provenance(
        parent_ids=tuple(data.get("parent_ids", [])),
        source_refs=tuple(data.get("source_refs", [])),
        transform_spec=data.get("transform_spec"),
        code_commit=data.get("code_commit"),
        run_id=data.get("run_id"),
    )


def asset_record_from_dict(item: dict[str, Any], *, schema_version: str = SCHEMA_VERSION) -> AssetRecord:
    return AssetRecord(
        asset_id=item["asset_id"], dataset_id=item["dataset_id"], dataset_version=item["dataset_version"],
        official_sample_key=item["official_sample_key"], group_id=item["group_id"], image_uri=item["image_uri"],
        label_uri=item.get("label_uri"), content_sha256=item["content_sha256"],
        label_content_sha256=item.get("label_content_sha256"), label_spec_id=item.get("label_spec_id"),
        official_split=SplitRole(item["official_split"]) if item.get("official_split") else None,
        use_tags=tuple(UseTag(value) for value in item.get("use_tags", [])),
        annotation_uri=item.get("annotation_uri"),
        width=item.get("width"), height=item.get("height"), image_mode=item.get("image_mode"),
        image_channels=item.get("image_channels"),
        present_class_ids=tuple(int(value) for value in item.get("present_class_ids", [])),
        provenance=_provenance(item.get("provenance", {})), status=RecordStatus(item.get("status", "valid")),
        reason_code=item.get("reason_code"), schema_version=item.get("schema_version", schema_version),
    )


def manifest_from_dict(data: dict[str, Any]) -> ManifestEnvelope:
    authorization_data = data["authorization"]
    authorization = AuthorizationRecord(
        authorization_record_id=authorization_data["authorization_record_id"],
        dataset_id=authorization_data["dataset_id"],
        canonical_name=authorization_data["canonical_name"],
        dataset_version=authorization_data["dataset_version"],
        status=AuthorizationStatus(authorization_data["status"]),
        license_ref=authorization_data["license_ref"],
        allowed_uses=tuple(UseTag(value) for value in authorization_data.get("allowed_uses", [])),
        reviewed_on=authorization_data.get("reviewed_on"),
        reason_code=authorization_data.get("reason_code"),
        schema_version=authorization_data.get("schema_version", data.get("schema_version")),
    )
    dataset_data = data["dataset"]
    dataset = DatasetSpec(
        dataset_id=dataset_data["dataset_id"],
        canonical_name=dataset_data["canonical_name"],
        dataset_version=dataset_data["dataset_version"],
        authorization_record_id=dataset_data["authorization_record_id"],
        enabled=bool(dataset_data["enabled"]),
        use_tags=tuple(UseTag(value) for value in dataset_data.get("use_tags", [])),
        status=RecordStatus(dataset_data.get("status", "valid")),
        reason_code=dataset_data.get("reason_code"),
        schema_version=dataset_data.get("schema_version", data.get("schema_version")),
    )
    groups = tuple(GroupRecord(
        group_id=item["group_id"], dataset_id=item["dataset_id"], grouping_spec=item["grouping_spec"],
        source_group_key=item["source_group_key"], asset_ids=tuple(item.get("asset_ids", [])),
        status=RecordStatus(item.get("status", "valid")), reason_code=item.get("reason_code"),
        schema_version=item.get("schema_version", data.get("schema_version")),
    ) for item in data.get("groups", []))
    assets = tuple(
        asset_record_from_dict(item, schema_version=data.get("schema_version", SCHEMA_VERSION))
        for item in data.get("assets", [])
    )
    splits = tuple(SplitRecord(
        split_record_id=item["split_record_id"], split_spec_id=item["split_spec_id"], group_id=item["group_id"],
        role=SplitRole(item["role"]), asset_ids=tuple(item.get("asset_ids", [])), source=item["source"],
        allowed_label_stages=tuple(HiddenLabelStage(value) for value in item.get("allowed_label_stages", [])),
        approved=bool(item["approved"]), status=RecordStatus(item.get("status", "valid")),
        reason_code=item.get("reason_code"), schema_version=item.get("schema_version", data.get("schema_version")),
    ) for item in data.get("splits", []))
    derived = tuple(DerivedAssetRecord(
        derived_asset_id=item["derived_asset_id"], parent_asset_id=item["parent_asset_id"], group_id=item["group_id"],
        split_role=SplitRole(item["split_role"]), artifact_uri=item["artifact_uri"], content_sha256=item["content_sha256"],
        transform_spec=item["transform_spec"], use_tags=tuple(UseTag(value) for value in item.get("use_tags", [])),
        provenance=_provenance(item.get("provenance", {})), status=RecordStatus(item.get("status", "valid")),
        reason_code=item.get("reason_code"), schema_version=item.get("schema_version", data.get("schema_version")),
    ) for item in data.get("derived_assets", []))
    labels = tuple(LabelSpec(
        label_spec_id=item["label_spec_id"], dataset_id=item["dataset_id"], task=item["task"],
        class_ids=tuple(item.get("class_ids", [])), ignore_index=int(item["ignore_index"]),
        void_values=tuple(item.get("void_values", [])), status=RecordStatus(item.get("status", "valid")),
        reason_code=item.get("reason_code"), schema_version=item.get("schema_version", data.get("schema_version")),
    ) for item in data.get("label_specs", []))
    return ManifestEnvelope(
        manifest_id=data["manifest_id"], dataset=dataset, authorization=authorization, groups=groups, assets=assets,
        splits=splits, derived_assets=derived, label_specs=labels, status=RecordStatus(data.get("status", "valid")),
        reason_code=data.get("reason_code"), schema_version=data["schema_version"],
    )


def load_manifest(path: Path) -> ManifestEnvelope:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return manifest_from_dict(payload)


def validate_manifest(manifest: ManifestEnvelope) -> ValidationReport:
    from rail3.data.splits import audit_splits

    issues: list[ValidationIssue] = []
    versioned_records = (
        manifest,
        manifest.dataset,
        manifest.authorization,
        *manifest.groups,
        *manifest.assets,
        *manifest.splits,
        *manifest.derived_assets,
        *manifest.label_specs,
    )
    if any(item.schema_version != SCHEMA_VERSION for item in versioned_records):
        issues.append(ValidationIssue("SCHEMA_VERSION_MISMATCH", "all records must use the current schema version"))
    if manifest.manifest_id != manifest.expected_manifest_id():
        issues.append(ValidationIssue("MANIFEST_ID_MISMATCH", "manifest ID does not match canonical content", (manifest.manifest_id,)))
    if manifest.dataset.authorization_record_id != manifest.authorization.authorization_record_id:
        issues.append(ValidationIssue("AUTHORIZATION_REFERENCE_MISMATCH", "dataset references a different authorization record"))
    expected_authorization_id = stable_id("authorization", {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": manifest.authorization.dataset_id,
        "canonical_name": manifest.authorization.canonical_name,
        "dataset_version": manifest.authorization.dataset_version,
        "license_ref": manifest.authorization.license_ref,
    })
    if manifest.authorization.authorization_record_id != expected_authorization_id:
        issues.append(ValidationIssue("AUTHORIZATION_ID_MISMATCH", "authorization ID does not match stable identity fields", (manifest.authorization.authorization_record_id,)))
    if manifest.dataset.dataset_id != manifest.authorization.dataset_id or manifest.dataset.dataset_version != manifest.authorization.dataset_version:
        issues.append(ValidationIssue("AUTHORIZATION_DATASET_MISMATCH", "authorization identity does not match dataset"))
    gate = evaluate_dataset_gate(manifest.dataset, manifest.authorization)
    if not gate.allowed:
        issues.append(ValidationIssue(gate.reason_code, gate.message, (manifest.dataset.dataset_id,)))

    asset_ids = {item.asset_id for item in manifest.assets}
    group_ids = {item.group_id for item in manifest.groups}
    label_ids = {item.label_spec_id for item in manifest.label_specs}
    if len(asset_ids) != len(manifest.assets):
        issues.append(ValidationIssue("DUPLICATE_ASSET_ID", "asset IDs must be unique"))
    if len(group_ids) != len(manifest.groups):
        issues.append(ValidationIssue("DUPLICATE_GROUP_ID", "group IDs must be unique"))
    for asset in manifest.assets:
        expected_asset_id = stable_id("asset", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": asset.dataset_id,
            "dataset_version": asset.dataset_version,
            "official_sample_key": asset.official_sample_key,
            "content_sha256": asset.content_sha256,
        })
        if asset.asset_id != expected_asset_id:
            issues.append(ValidationIssue("ASSET_ID_MISMATCH", "asset ID does not match stable identity fields", (asset.asset_id,)))
        if asset.dataset_id != manifest.dataset.dataset_id or asset.dataset_version != manifest.dataset.dataset_version:
            issues.append(ValidationIssue("ASSET_DATASET_MISMATCH", "asset dataset identity differs from its envelope", (asset.asset_id,)))
        if asset.group_id not in group_ids:
            issues.append(ValidationIssue("UNKNOWN_ASSET_GROUP", "asset references an unknown group", (asset.asset_id, asset.group_id)))
        if asset.label_spec_id is not None and asset.label_spec_id not in label_ids:
            issues.append(ValidationIssue("UNKNOWN_LABEL_SPEC", "asset references an unknown label specification", (asset.asset_id, asset.label_spec_id)))
    for group in manifest.groups:
        expected_group_id = stable_id("group", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": group.dataset_id,
            "grouping_spec": group.grouping_spec,
            "source_group_key": group.source_group_key,
        })
        if group.group_id != expected_group_id:
            issues.append(ValidationIssue("GROUP_ID_MISMATCH", "group ID does not match stable identity fields", (group.group_id,)))
        if group.dataset_id != manifest.dataset.dataset_id:
            issues.append(ValidationIssue("GROUP_DATASET_MISMATCH", "group dataset identity differs from its envelope", (group.group_id,)))
        unknown = tuple(sorted(set(group.asset_ids) - asset_ids))
        if unknown:
            issues.append(ValidationIssue("UNKNOWN_GROUP_ASSET", "group references unknown assets", (group.group_id, *unknown)))
        wrong_group = tuple(sorted(
            asset_id for asset_id in group.asset_ids
            if asset_id in asset_ids and next(item for item in manifest.assets if item.asset_id == asset_id).group_id != group.group_id
        ))
        if wrong_group:
            issues.append(ValidationIssue("GROUP_ASSET_MEMBERSHIP_MISMATCH", "group contains assets assigned to a different group", (group.group_id, *wrong_group)))
    for split in manifest.splits:
        expected_split_id = stable_id("split", {
            "schema_version": SCHEMA_VERSION,
            "split_spec_id": split.split_spec_id,
            "group_id": split.group_id,
            "role": split.role,
            "asset_ids": sorted(set(split.asset_ids)),
        })
        if split.split_record_id != expected_split_id:
            issues.append(ValidationIssue("SPLIT_ID_MISMATCH", "split ID does not match stable identity fields", (split.split_record_id,)))
    for derived in manifest.derived_assets:
        expected_derived_id = stable_id("derived_asset", {
            "schema_version": SCHEMA_VERSION,
            "parent_asset_id": derived.parent_asset_id,
            "transform_spec": derived.transform_spec,
            "content_sha256": derived.content_sha256,
        })
        if derived.derived_asset_id != expected_derived_id:
            issues.append(ValidationIssue("DERIVED_ASSET_ID_MISMATCH", "derived asset ID does not match stable identity fields", (derived.derived_asset_id,)))
    for label in manifest.label_specs:
        expected_label_id = stable_id("label_spec", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": label.dataset_id,
            "task": label.task,
            "class_ids": sorted(set(label.class_ids)),
            "ignore_index": label.ignore_index,
            "void_values": sorted(set(label.void_values)),
        })
        if label.label_spec_id != expected_label_id:
            issues.append(ValidationIssue("LABEL_SPEC_ID_MISMATCH", "label specification ID does not match stable identity fields", (label.label_spec_id,)))
        if label.dataset_id != manifest.dataset.dataset_id:
            issues.append(ValidationIssue("LABEL_SPEC_DATASET_MISMATCH", "label specification belongs to another dataset", (label.label_spec_id,)))
    issues.extend(audit_splits(manifest).issues)
    return ValidationReport(tuple(sorted(set(issues))))
