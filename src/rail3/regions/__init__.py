"""Prompt-induced region contracts."""

from rail3.regions.atomic import (
    ATOM_SCHEMA_VERSION,
    AtomicPartition,
    CandidateMaskInput,
    attach_lineage,
    build_atomic_partition,
    decode_label_map,
    deterministic_merge_view,
    partition_statistics,
)

__all__ = [
    "ATOM_SCHEMA_VERSION",
    "AtomicPartition",
    "CandidateMaskInput",
    "attach_lineage",
    "build_atomic_partition",
    "decode_label_map",
    "deterministic_merge_view",
    "partition_statistics",
]
