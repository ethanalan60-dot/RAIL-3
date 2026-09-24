"""Fail-closed adapter for one pinned SAM 3.1 facade signature defect.

This module intentionally has no dependency on SAM, torch, or CUDA.  It validates
the final model's bound ``init_state`` signature before dropping the single false
facade default that the pinned final model cannot accept.
"""

from __future__ import annotations

import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from rail3.sam.protocols import Sam31InitStateModel


PINNED_SAM31_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
_DROPPABLE_PARAMETER = "offload_state_to_cpu"
_EXPECTED_PARAMETERS = (
    ("resource_path", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("offload_video_to_cpu", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("async_loading_frames", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("use_torchcodec", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("use_cv2", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("input_is_mp4", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
)


class UnsupportedOfficialInterfaceError(RuntimeError):
    """A structured, fail-closed rejection of an unreviewed official interface."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        official_commit: str,
        observed_signature: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.official_commit = official_commit
        self.expected_commit = PINNED_SAM31_COMMIT
        self.observed_signature = observed_signature

    def as_dict(self) -> dict[str, str | None]:
        return {
            "error_type": type(self).__name__,
            "reason_code": self.reason_code,
            "message": str(self),
            "official_commit": self.official_commit,
            "expected_commit": self.expected_commit,
            "observed_signature": self.observed_signature,
        }


@dataclass(frozen=True)
class CompatibilityDecision:
    """Auditable description of the deterministic compatibility action."""

    status: str
    official_commit: str
    observed_signature: str
    dropped_parameters: tuple[str, ...]

    def as_dict(self) -> dict[str, str | list[str]]:
        return {
            "status": self.status,
            "official_commit": self.official_commit,
            "observed_signature": self.observed_signature,
            "dropped_parameters": list(self.dropped_parameters),
        }


def _validate_commit(official_commit: str) -> None:
    if official_commit != PINNED_SAM31_COMMIT:
        raise UnsupportedOfficialInterfaceError(
            "unknown_official_commit",
            "compatibility shim is disabled for an unreviewed SAM commit",
            official_commit=official_commit,
        )


def _validated_signature(model: Sam31InitStateModel, official_commit: str) -> inspect.Signature:
    try:
        signature = inspect.signature(model.init_state)
    except (TypeError, ValueError) as exc:
        raise UnsupportedOfficialInterfaceError(
            "uninspectable_init_state",
            "final model init_state signature cannot be inspected",
            official_commit=official_commit,
        ) from exc

    observed = tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in signature.parameters.values()
    )
    if observed != _EXPECTED_PARAMETERS:
        raise UnsupportedOfficialInterfaceError(
            "unknown_init_state_signature",
            "final model init_state signature differs from the pinned reviewed contract",
            official_commit=official_commit,
            observed_signature=str(signature),
        )
    return signature


def prepare_init_state_call(
    model: Sam31InitStateModel,
    *,
    official_commit: str,
    init_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], CompatibilityDecision]:
    """Validate and copy facade kwargs for the pinned final model.

    The caller's mapping and the model object are never mutated.  Only an exact
    boolean ``False`` value for ``offload_state_to_cpu`` may be removed.
    """

    _validate_commit(official_commit)
    signature = _validated_signature(model, official_commit)
    forwarded = dict(init_kwargs)
    dropped: tuple[str, ...] = ()

    if _DROPPABLE_PARAMETER in forwarded:
        requested = forwarded[_DROPPABLE_PARAMETER]
        if requested is not False:
            raise UnsupportedOfficialInterfaceError(
                "unsupported_state_offload",
                "offload_state_to_cpu must be exactly False for the pinned compatibility shim",
                official_commit=official_commit,
                observed_signature=str(signature),
            )
        del forwarded[_DROPPABLE_PARAMETER]
        dropped = (_DROPPABLE_PARAMETER,)

    try:
        signature.bind(**forwarded)
    except TypeError as exc:
        raise UnsupportedOfficialInterfaceError(
            "incompatible_init_state_arguments",
            "facade arguments do not bind to the pinned final model signature",
            official_commit=official_commit,
            observed_signature=str(signature),
        ) from exc

    return forwarded, CompatibilityDecision(
        status="shim-tested-with-fakes",
        official_commit=official_commit,
        observed_signature=str(signature),
        dropped_parameters=dropped,
    )


def init_state_with_compat(
    model: Sam31InitStateModel,
    *,
    official_commit: str,
    init_kwargs: Mapping[str, Any],
) -> tuple[Any, CompatibilityDecision]:
    """Call the pinned final model after a successful compatibility decision.

    Exceptions raised by the model call, including ``TypeError``, are deliberately
    not caught or rewritten so the original type, message, and traceback survive.
    """

    forwarded, decision = prepare_init_state_call(
        model,
        official_commit=official_commit,
        init_kwargs=init_kwargs,
    )
    return model.init_state(**forwarded), decision


def start_session_with_compat(
    predictor: Any,
    *,
    official_commit: str,
    resource_path: Any,
    session_id: str | None = None,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> tuple[dict[str, str], CompatibilityDecision]:
    """Create one upstream session through the pinned compatibility boundary.

    The predictor and model are not patched. The function preserves the reviewed
    upstream session-state envelope while routing the final ``init_state`` call
    through :func:`init_state_with_compat`.
    """

    model = getattr(predictor, "model", None)
    sessions = getattr(predictor, "_all_inference_states", None)
    if model is None or not isinstance(sessions, dict):
        raise TypeError("predictor lacks the reviewed model/session surface")
    init_kwargs: dict[str, Any] = {
        "resource_path": resource_path,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    if hasattr(predictor, "video_loader_type"):
        init_kwargs["video_loader_type"] = predictor.video_loader_type

    inference_state, decision = init_state_with_compat(
        model,
        official_commit=official_commit,
        init_kwargs=init_kwargs,
    )
    selected_session_id = session_id or str(uuid.uuid4())
    if selected_session_id in sessions:
        raise ValueError("session ID already exists")
    now = time.time()
    sessions[selected_session_id] = {
        "state": inference_state,
        "session_id": selected_session_id,
        "start_time": now,
        "last_use_time": now,
    }
    return {"session_id": selected_session_id}, decision
