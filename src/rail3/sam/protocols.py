"""Frozen SAM backend and checkpoint-free compatibility protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rail3.contracts import stable_id

if TYPE_CHECKING:
    from rail3.cache.candidates import CandidateGenerationResult


_SHA256 = re.compile(r"[0-9a-f]{64}")


class BackendKind(str, Enum):
    MOCK = "mock"
    OFFICIAL = "official"


class PromptKind(str, Enum):
    TEXT = "text"
    POINTS = "points"
    BOX = "box"


@dataclass(frozen=True)
class ModelSpec:
    model_spec_id: str
    model_name: str
    backend_kind: BackendKind
    source_commit: str
    checkpoint_sha256: str | None
    preprocess_spec: str
    adapter_version: str
    frozen: bool = True
    schema_version: str = "rail3.model.v1"

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("source_commit must be a full lowercase Git commit")
        if self.backend_kind is BackendKind.OFFICIAL:
            if self.checkpoint_sha256 is None or _SHA256.fullmatch(self.checkpoint_sha256) is None:
                raise ValueError("official model specs require a checkpoint SHA-256")
        elif self.checkpoint_sha256 is not None:
            raise ValueError("mock model specs cannot claim a checkpoint")
        if not self.frozen:
            raise ValueError("RAIL-3 model specs must be frozen")
        expected_id = stable_id("model_spec", {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "backend_kind": self.backend_kind,
            "source_commit": self.source_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocess_spec": self.preprocess_spec,
            "adapter_version": self.adapter_version,
            "frozen": self.frozen,
        })
        if self.model_spec_id != expected_id:
            raise ValueError("model_spec_id does not match canonical model identity")

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        backend_kind: BackendKind,
        source_commit: str,
        checkpoint_sha256: str | None,
        preprocess_spec: str,
        adapter_version: str,
    ) -> "ModelSpec":
        payload = {
            "schema_version": "rail3.model.v1",
            "model_name": model_name,
            "backend_kind": backend_kind,
            "source_commit": source_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "preprocess_spec": preprocess_spec,
            "adapter_version": adapter_version,
            "frozen": True,
        }
        return cls(stable_id("model_spec", payload), model_name, backend_kind, source_commit, checkpoint_sha256, preprocess_spec, adapter_version)


@dataclass(frozen=True)
class PromptRequest:
    prompt_id: str
    kind: PromptKind
    text: str | None = None
    points: tuple[tuple[int, int, int], ...] = ()
    box: tuple[int, int, int, int] | None = None
    object_id: str | None = None
    seed: int = 0
    schema_version: str = "rail3.prompt.v1"

    def __post_init__(self) -> None:
        populated = sum((self.text is not None, bool(self.points), self.box is not None))
        if populated != 1:
            raise ValueError("exactly one prompt payload must be populated")
        if self.kind is PromptKind.TEXT and self.text is None:
            raise ValueError("text prompt requires text")
        if self.kind is PromptKind.POINTS and not self.points:
            raise ValueError("points prompt requires points")
        if self.kind is PromptKind.BOX and self.box is None:
            raise ValueError("box prompt requires a box")
        if any(label not in {-1, 0, 1} for _, _, label in self.points):
            raise ValueError("point labels must be -1, 0, or 1")
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError("box must use a positive half-open integer window")
        expected_id = stable_id("prompt", {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "text": self.text,
            "points": self.points,
            "box": self.box,
            "object_id": self.object_id,
            "seed": self.seed,
        })
        if self.prompt_id != expected_id:
            raise ValueError("prompt_id does not match canonical prompt identity")

    @classmethod
    def create(
        cls,
        *,
        kind: PromptKind,
        text: str | None = None,
        points: tuple[tuple[int, int, int], ...] = (),
        box: tuple[int, int, int, int] | None = None,
        object_id: str | None = None,
        seed: int = 0,
    ) -> "PromptRequest":
        payload = {
            "schema_version": "rail3.prompt.v1",
            "kind": kind,
            "text": text,
            "points": points,
            "box": box,
            "object_id": object_id,
            "seed": seed,
        }
        return cls(stable_id("prompt", payload), kind, text, points, box, object_id, seed)


@dataclass(frozen=True)
class EncodedImageState:
    state_id: str
    asset_id: str
    image_sha256: str
    width: int
    height: int
    model_spec_id: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.image_sha256) is None:
            raise ValueError("image_sha256 must be lowercase SHA-256")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("encoded image dimensions must be positive")
        expected_id = stable_id("encoded_state", {
            "model_spec_id": self.model_spec_id,
            "asset_id": self.asset_id,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
        })
        if self.state_id != expected_id:
            raise ValueError("state_id does not match canonical encoded-state identity")


@runtime_checkable
class FrozenSamBackend(Protocol):
    @property
    def model_spec(self) -> ModelSpec:
        """Return the immutable model identity."""

    def assert_frozen(self) -> None:
        """Fail if any backend parameter is trainable."""

    def encode_image(
        self,
        *,
        asset_id: str,
        image_bytes: bytes,
        width: int,
        height: int,
    ) -> EncodedImageState:
        """Encode an image once and return reusable state."""

    def generate_candidates(
        self,
        state: EncodedImageState,
        prompts: tuple[PromptRequest, ...],
    ) -> tuple["CandidateGenerationResult", ...]:
        """Generate deterministic candidates for prompts sharing one state."""


class Sam31InitStateModel(Protocol):
    """The exact public shape expected from the pinned final SAM 3.1 model."""

    def init_state(
        self,
        resource_path: Any,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
    ) -> Any:
        """Initialize state for an image or video resource."""
