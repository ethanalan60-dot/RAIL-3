"""Content-addressed candidate cache with immutable atomic objects."""

from rail3.cache.candidates import (
    CacheCorruptionError,
    CacheExecution,
    CacheIdentity,
    CandidateCache,
    CandidateFailure,
    CandidateGenerationResult,
    CandidateNoResult,
    CandidateOutcomeTelemetry,
    CandidateRecord,
)

__all__ = [
    "CacheCorruptionError",
    "CacheExecution",
    "CacheIdentity",
    "CandidateCache",
    "CandidateFailure",
    "CandidateGenerationResult",
    "CandidateNoResult",
    "CandidateOutcomeTelemetry",
    "CandidateRecord",
]
