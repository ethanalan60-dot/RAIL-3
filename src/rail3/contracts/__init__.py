"""Versioned manifest contracts and deterministic identity helpers."""

from rail3.contracts.records import (
    SCHEMA_VERSION,
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
    SplitRecord,
    SplitRole,
    UseTag,
)
from rail3.contracts.serialization import canonical_json_bytes, stable_id

__all__ = [
    "SCHEMA_VERSION",
    "AssetRecord",
    "AuthorizationRecord",
    "AuthorizationStatus",
    "DatasetSpec",
    "DerivedAssetRecord",
    "GroupRecord",
    "HiddenLabelStage",
    "LabelSpec",
    "ManifestEnvelope",
    "Provenance",
    "RecordStatus",
    "SplitRecord",
    "SplitRole",
    "UseTag",
    "canonical_json_bytes",
    "stable_id",
]
