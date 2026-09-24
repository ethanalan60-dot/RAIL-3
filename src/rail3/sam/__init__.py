"""Frozen SAM integration contracts that do not import the upstream package."""

from rail3.sam.compat import (
    PINNED_SAM31_COMMIT,
    CompatibilityDecision,
    UnsupportedOfficialInterfaceError,
    init_state_with_compat,
    prepare_init_state_call,
    start_session_with_compat,
)
from rail3.sam.checkpoint import Sam31CheckpointIdentity, verify_checkpoint_file
from rail3.sam.checkpoint_compat import (
    OFFICIAL_SAM31_CHECKPOINT_CONTRACT,
    CheckpointDtypeMismatchError,
    CheckpointHashMismatchError,
    CheckpointShapeMismatchError,
    GeneratedBufferDeterminismError,
    PinnedSourceMismatchError,
    UnexpectedCheckpointKeyError,
    UnexpectedMissingKeyError,
    UnsupportedSam31CheckpointContractError,
    complete_checkpoint_state_in_memory,
    load_completed_state_dict_strict,
)
from rail3.sam.protocols import BackendKind, EncodedImageState, FrozenSamBackend, ModelSpec, PromptKind, PromptRequest
from rail3.sam.sam31_backend import PromptMetadata, Sam31RealFrozenBackend, official_sam31_model_spec

__all__ = [
    "PINNED_SAM31_COMMIT",
    "CompatibilityDecision",
    "UnsupportedOfficialInterfaceError",
    "init_state_with_compat",
    "prepare_init_state_call",
    "start_session_with_compat",
    "Sam31CheckpointIdentity",
    "verify_checkpoint_file",
    "OFFICIAL_SAM31_CHECKPOINT_CONTRACT",
    "CheckpointDtypeMismatchError",
    "CheckpointHashMismatchError",
    "CheckpointShapeMismatchError",
    "GeneratedBufferDeterminismError",
    "PinnedSourceMismatchError",
    "UnexpectedCheckpointKeyError",
    "UnexpectedMissingKeyError",
    "UnsupportedSam31CheckpointContractError",
    "complete_checkpoint_state_in_memory",
    "load_completed_state_dict_strict",
    "BackendKind",
    "EncodedImageState",
    "FrozenSamBackend",
    "ModelSpec",
    "PromptKind",
    "PromptRequest",
    "PromptMetadata",
    "Sam31RealFrozenBackend",
    "official_sam31_model_spec",
]
