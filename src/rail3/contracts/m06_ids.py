"""Stable M06 row identities built exclusively from non-target identity fields."""

from __future__ import annotations

import hashlib

from rail3.contracts import canonical_json_bytes, stable_id


M06_PROTOCOL_SHA256 = "4aae69f7953590b4c054420c71c7c5be2bd2da90e018a9544b87057904723f28"
FEATURE_VERSION = "rail3.m06.deployment-feature.v1"
TARGET_VERSION = "rail3.m06.qg-pal-target.v1"
FOLD_VERSION = "rail3.m06.group-fold.v1"


def state_action_id(
    *, dataset_id: str, dataset_version: str, asset_id: str, image_group_id: str,
    state_id: str, class_id: int, action_code: str, action_id: str,
) -> str:
    return stable_id("m06_state_action", {
        "protocol_sha256": M06_PROTOCOL_SHA256,
        "feature_version": FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "asset_id": asset_id,
        "image_group_id": image_group_id,
        "state_id": state_id,
        "class_id": int(class_id),
        "action_code": action_code,
        "action_id": action_id,
    })


def pairwise_id(
    *, state_id: str, state_action_a_id: str, state_action_b_id: str,
) -> str:
    ordered = sorted((state_action_a_id, state_action_b_id))
    return stable_id("m06_pairwise", {
        "protocol_sha256": M06_PROTOCOL_SHA256,
        "target_version": TARGET_VERSION,
        "state_id": state_id,
        "state_action_ids": ordered,
    })


def atom_action_id(
    *, dataset_id: str, dataset_version: str, asset_id: str, image_group_id: str,
    state_id: str, class_id: int, atom_id: str, action_code: str, action_id: str,
) -> str:
    return stable_id("m06_atom_action", {
        "protocol_sha256": M06_PROTOCOL_SHA256,
        "feature_version": FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "asset_id": asset_id,
        "image_group_id": image_group_id,
        "state_id": state_id,
        "class_id": int(class_id),
        "atom_id": atom_id,
        "action_code": action_code,
        "action_id": action_id,
    })


def state_tie_group_id(*, state_id: str, image_group_id: str) -> str:
    """Identify the comparison group without encoding winners or target values."""

    return stable_id("m06_tie_group", {
        "protocol_sha256": M06_PROTOCOL_SHA256,
        "target_version": TARGET_VERSION,
        "state_id": state_id,
        "image_group_id": image_group_id,
    })


def seeded_digest(seed: int, *values: str) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "seed": int(seed), "values": list(values), "fold_version": FOLD_VERSION,
    })).hexdigest()
