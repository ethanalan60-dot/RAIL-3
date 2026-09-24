"""Hard split-leakage checks for group, lineage, hashes, and official test data."""

from __future__ import annotations

from collections import defaultdict

from rail3.contracts import HiddenLabelStage, ManifestEnvelope, SplitRole
from rail3.data.manifest import ValidationIssue, ValidationReport


def audit_splits(manifest: ManifestEnvelope) -> ValidationReport:
    issues: list[ValidationIssue] = []
    group_roles: dict[str, set[SplitRole]] = defaultdict(set)
    asset_roles: dict[str, set[SplitRole]] = defaultdict(set)
    group_assets = {item.group_id: set(item.asset_ids) for item in manifest.groups}

    for split in manifest.splits:
        group_roles[split.group_id].add(split.role)
        if not split.approved:
            issues.append(ValidationIssue("UNAPPROVED_SPLIT", "split record is not approved", (split.split_record_id,)))
        for asset_id in split.asset_ids:
            asset_roles[asset_id].add(split.role)
        expected = group_assets.get(split.group_id)
        if expected is None:
            issues.append(ValidationIssue("UNKNOWN_SPLIT_GROUP", "split references an unknown group", (split.split_record_id, split.group_id)))
        elif set(split.asset_ids) != expected:
            issues.append(ValidationIssue("SPLIT_GROUP_MEMBERSHIP_MISMATCH", "split members differ from the complete group membership", (split.split_record_id, split.group_id)))
        if split.role is SplitRole.CALIBRATION and HiddenLabelStage.FIT_TARGET_BUILD in split.allowed_label_stages:
            issues.append(ValidationIssue("CALIBRATION_USED_FOR_FIT", "calibration labels cannot build fit targets", (split.split_record_id,)))
        expected_stages = {
            SplitRole.TRAIN: {
                HiddenLabelStage.FIT_TARGET_BUILD,
                HiddenLabelStage.FINAL_EVALUATION,
            },
            SplitRole.VALIDATION: {HiddenLabelStage.FINAL_EVALUATION},
            SplitRole.CALIBRATION: {
                HiddenLabelStage.CALIBRATION,
                HiddenLabelStage.FINAL_EVALUATION,
            },
            SplitRole.TEST: {HiddenLabelStage.FINAL_EVALUATION},
        }[split.role]
        if set(split.allowed_label_stages) != expected_stages:
            issues.append(ValidationIssue("INVALID_LABEL_STAGE_POLICY", "split label stages differ from the frozen role policy", (split.split_record_id,)))

    for group_id, roles in group_roles.items():
        if len(roles) > 1:
            issues.append(ValidationIssue("GROUP_CROSSES_SPLITS", "one source group appears in multiple split roles", (group_id, *sorted(role.value for role in roles))))
    for asset_id, roles in asset_roles.items():
        if len(roles) > 1:
            issues.append(ValidationIssue("ASSET_CROSSES_SPLITS", "one asset appears in multiple split roles", (asset_id, *sorted(role.value for role in roles))))

    assets = {item.asset_id: item for item in manifest.assets}
    for asset in manifest.assets:
        roles = asset_roles.get(asset.asset_id, set())
        if not roles:
            issues.append(ValidationIssue("ASSET_WITHOUT_SPLIT", "asset is absent from split records", (asset.asset_id,)))
            continue
        assigned = next(iter(roles)) if len(roles) == 1 else None
        if asset.official_split is SplitRole.TEST and assigned is not SplitRole.TEST:
            issues.append(ValidationIssue("OFFICIAL_TEST_REASSIGNED", "official test asset cannot be reassigned", (asset.asset_id,)))
        group_role_set = group_roles.get(asset.group_id, set())
        if assigned is not None and group_role_set != {assigned}:
            issues.append(ValidationIssue("ASSET_GROUP_SPLIT_MISMATCH", "asset and source group split roles disagree", (asset.asset_id, asset.group_id)))

    content_roles: dict[str, set[SplitRole]] = defaultdict(set)
    content_objects: dict[str, list[str]] = defaultdict(list)
    for asset in manifest.assets:
        roles = asset_roles.get(asset.asset_id, set())
        content_roles[asset.content_sha256].update(roles)
        content_objects[asset.content_sha256].append(asset.asset_id)
    for derived in manifest.derived_assets:
        parent = assets.get(derived.parent_asset_id)
        if parent is None:
            issues.append(ValidationIssue("UNKNOWN_DERIVED_PARENT", "derived asset references an unknown parent", (derived.derived_asset_id, derived.parent_asset_id)))
            continue
        parent_roles = asset_roles.get(parent.asset_id, set())
        if parent_roles != {derived.split_role} or parent.group_id != derived.group_id:
            issues.append(ValidationIssue("DERIVED_SPLIT_MISMATCH", "derived asset must inherit parent group and split", (derived.derived_asset_id, parent.asset_id)))
        content_roles[derived.content_sha256].add(derived.split_role)
        content_objects[derived.content_sha256].append(derived.derived_asset_id)
    for digest, roles in content_roles.items():
        if len(roles) > 1:
            issues.append(ValidationIssue("CONTENT_HASH_CROSSES_SPLITS", "identical content appears in multiple split roles", tuple(sorted(content_objects[digest]))))

    return ValidationReport(tuple(sorted(set(issues))))
