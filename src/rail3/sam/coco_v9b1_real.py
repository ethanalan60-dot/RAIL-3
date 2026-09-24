"""Lazy real-SAM adapter for the V9B1 result-blind trajectory driver.

Importing this module does not import torch, construct a model, load a
checkpoint, or open a panel image.  ``build`` is the explicit real-execution
boundary and must only be called by an admitted per-GPU worker.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import io
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any

from rail3.contracts import stable_id
from rail3.sam.coco_v9b1_trajectory import (
    IMAGE_ROOT,
    LabelFreeSessionBackend,
    PanelImageIdentity,
    SessionHandle,
    V9B1ContractError,
    validate_small_authorities,
)
from rail3.sam.protocols import PromptRequest
from rail3.sam.sam31_backend import PromptMetadata


EXPECTED_SOURCE_PATH = Path("artifacts/source/sam3")
EXPECTED_CHECKPOINT_PATH = Path("artifacts/checkpoints/sam3/sam3.1_multiplex.pt")
EXPECTED_SOURCE_REVISION = "96914d2425f90a64f45ca977c2b5165418099543"
EXPECTED_SOURCE_TREE = "573deb167702e014829a5b830de8ae62abe891d5"
EXPECTED_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
EXPECTED_CHECKPOINT_BYTES = 3_502_755_717


def bind_pinned_sam_import_precedence(source: Path) -> Path:
    """Bind future upstream imports to one verified cleanroom source tree."""

    already_imported = sorted(
        name for name in sys.modules
        if name == "sam3" or name.startswith("sam3.")
    )
    if already_imported:
        raise V9B1ContractError(
            "V9B1_UPSTREAM_SAM_IMPORTED_BEFORE_SOURCE_BIND",
            str(already_imported),
        )
    try:
        metadata = source.lstat()
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise V9B1ContractError(
            "V9B1_SAM_SOURCE_IMPORT_ROOT_INVALID", str(source)
        ) from exc
    if source.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise V9B1ContractError(
            "V9B1_SAM_SOURCE_IMPORT_ROOT_INVALID", str(source)
        )
    spec = importlib.machinery.PathFinder.find_spec("sam3", [str(resolved)])
    locations = () if spec is None or spec.submodule_search_locations is None else tuple(
        Path(location).resolve() for location in spec.submodule_search_locations
    )
    expected_package = (resolved / "sam3").resolve()
    if locations != (expected_package,):
        raise V9B1ContractError(
            "V9B1_PINNED_SAM_PACKAGE_NOT_DISCOVERABLE", str(locations)
        )
    retained = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() != resolved
    ]
    sys.path[:] = [str(resolved), *retained]
    if Path(sys.path[0]).resolve() != resolved:
        raise V9B1ContractError(
            "V9B1_PINNED_SAM_IMPORT_PRECEDENCE_FAILED", str(source)
        )
    return resolved


def decoded_rgb_sha256(rgb_image: Any) -> str:
    """Compute the frozen TMLR V9 decoded-pixel identity framing."""

    width, height = rgb_image.size
    digest = hashlib.sha256()
    digest.update(b"TMLR_V9_RGB_V1\0")
    digest.update(struct.pack(">II", int(width), int(height)))
    digest.update(rgb_image.tobytes())
    return digest.hexdigest()


class RealSam31TrajectoryBackend(LabelFreeSessionBackend):
    """One admitted GPU worker wrapping the frozen SAM 3.1 backend."""

    def __init__(self, backend: Any, *, image_root: Path, physical_gpu_index: int) -> None:
        self._backend = backend
        self.model_spec_id = backend.model_spec.model_spec_id
        self.physical_gpu_index = physical_gpu_index
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._image_root_fd = os.open(image_root, flags)
        except OSError as exc:
            raise V9B1ContractError("V9B1_IMAGE_ROOT_OPEN_FAILED", str(image_root)) from exc
        self._prepared_image: PanelImageIdentity | None = None
        self._original_bytes: bytes | None = None
        self._rgb_image: Any | None = None

    @classmethod
    def build(
        cls,
        *,
        repo_root: Path,
        config: dict[str, Any],
        physical_gpu_index: int,
    ) -> "RealSam31TrajectoryBackend":
        """Cross the explicit model-construction/checkpoint-load boundary."""

        validate_small_authorities(repo_root, config)
        sam = config.get("sam", {})
        expected = {
            "source_path": str(EXPECTED_SOURCE_PATH),
            "source_revision": EXPECTED_SOURCE_REVISION,
            "source_tree": EXPECTED_SOURCE_TREE,
            "checkpoint_path": str(EXPECTED_CHECKPOINT_PATH),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
            "fallback_allowed": False,
            "runtime_download_allowed": False,
            "real_backend_only": True,
        }
        if any(sam.get(key) != value for key, value in expected.items()):
            raise V9B1ContractError("V9B1_SAM_CONFIG_IDENTITY_DRIFT", "SAM authority drift")
        source = repo_root / EXPECTED_SOURCE_PATH
        checkpoint = repo_root / EXPECTED_CHECKPOINT_PATH
        image_root = repo_root / IMAGE_ROOT
        from rail3.sam.coco_v9b1_asset_gate import validate_sam_assets_before_model

        # This performs the exact receipt/source/checkpoint audit while neither
        # torch nor upstream SAM has been imported by this worker.
        validate_sam_assets_before_model(repo_root, config)
        pinned_source = bind_pinned_sam_import_precedence(source)
        # Deliberately lazy: this import/constructor is the only real model-load path.
        from rail3.sam.sam31_backend import Sam31RealFrozenBackend

        backend = Sam31RealFrozenBackend(
            source=pinned_source,
            checkpoint_path=checkpoint,
            prompt_metadata={},
            physical_gpu_index=physical_gpu_index,
        )
        expected_rail_backend = (repo_root / "src/rail3/sam/sam31_backend.py").resolve()
        actual_rail_backend = Path(sys.modules["rail3.sam.sam31_backend"].__file__).resolve()
        model_builder = sys.modules.get("sam3.model_builder")
        expected_model_builder = (pinned_source / "sam3/model_builder.py").resolve()
        if (
            actual_rail_backend != expected_rail_backend
            or model_builder is None
            or Path(model_builder.__file__).resolve() != expected_model_builder
        ):
            backend.close()
            raise V9B1ContractError(
                "V9B1_REAL_BACKEND_IMPORT_ORIGIN_DRIFT", "pinned import origin mismatch"
            )
        return cls(backend, image_root=image_root, physical_gpu_index=physical_gpu_index)

    def _read_exact_image(self, image: PanelImageIdentity) -> bytes:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(image.filename, flags, dir_fd=self._image_root_fd)
        except OSError as exc:
            raise V9B1ContractError("V9B1_PANEL_IMAGE_OPEN_FAILED", image.filename) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise V9B1ContractError("V9B1_PANEL_IMAGE_NOT_SINGLE_LINK_REGULAR", image.filename)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise V9B1ContractError("V9B1_PANEL_IMAGE_CHANGED_DURING_READ", image.filename)
            encoded = b"".join(chunks)
        finally:
            os.close(descriptor)
        if hashlib.sha256(encoded).hexdigest() != image.file_sha256:
            raise V9B1ContractError("V9B1_PANEL_IMAGE_FILE_HASH_MISMATCH", image.filename)
        return encoded

    def prepare_image(self, image: PanelImageIdentity) -> None:
        if self._prepared_image is not None:
            raise V9B1ContractError("V9B1_IMAGE_ALREADY_PREPARED", image.canonical_image_id)
        encoded = self._read_exact_image(image)
        try:
            from PIL import Image

            with Image.open(io.BytesIO(encoded)) as opened:
                rgb = opened.convert("RGB")
                rgb.load()
        except Exception as exc:
            raise V9B1ContractError("V9B1_PANEL_IMAGE_DECODE_FAILED", image.filename) from exc
        if rgb.size != (image.width, image.height):
            raise V9B1ContractError("V9B1_PANEL_IMAGE_DIMENSION_MISMATCH", image.filename)
        if decoded_rgb_sha256(rgb) != image.pixel_sha256:
            raise V9B1ContractError("V9B1_PANEL_IMAGE_PIXEL_HASH_MISMATCH", image.filename)
        self._prepared_image = image
        self._original_bytes = encoded
        self._rgb_image = rgb

    def release_image(self, image: PanelImageIdentity) -> None:
        if self._prepared_image != image:
            raise V9B1ContractError("V9B1_RELEASE_IMAGE_IDENTITY_MISMATCH", image.canonical_image_id)
        self._prepared_image = None
        self._original_bytes = None
        if self._rgb_image is not None:
            self._rgb_image.close()
        self._rgb_image = None

    def _require_prepared(self, image: PanelImageIdentity) -> None:
        if self._prepared_image != image or self._original_bytes is None or self._rgb_image is None:
            raise V9B1ContractError("V9B1_IMAGE_NOT_PREPARED", image.canonical_image_id)

    @staticmethod
    def _png_bytes(image: Any) -> bytes:
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=False, compress_level=9)
        return stream.getvalue()

    def open_original(self, image: PanelImageIdentity, *, purpose: str) -> SessionHandle:
        self._require_prepared(image)
        state = self._backend.encode_image(
            asset_id=image.asset_id,
            image_bytes=self._original_bytes,
            width=image.width,
            height=image.height,
        )
        return SessionHandle(state, purpose)

    def open_flip(self, image: PanelImageIdentity) -> SessionHandle:
        self._require_prepared(image)
        from PIL import Image

        flipped = self._rgb_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        try:
            encoded = self._png_bytes(flipped)
        finally:
            flipped.close()
        asset_id = stable_id("v9b1_physical_input", {
            "asset_id": image.asset_id,
            "transform": "horizontal_flip",
        })
        state = self._backend.encode_image(
            asset_id=asset_id,
            image_bytes=encoded,
            width=image.width,
            height=image.height,
        )
        return SessionHandle(state, "A3_FLIP")

    def open_crop(
        self,
        image: PanelImageIdentity,
        crop_box_xyxy: tuple[int, int, int, int],
        *,
        action_id: str,
    ) -> SessionHandle:
        self._require_prepared(image)
        x0, y0, x1, y1 = crop_box_xyxy
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise V9B1ContractError("V9B1_CROP_BOX_INVALID", action_id)
        crop = self._rgb_image.crop(crop_box_xyxy)
        try:
            encoded = self._png_bytes(crop)
        finally:
            crop.close()
        asset_id = stable_id("v9b1_physical_input", {
            "asset_id": image.asset_id,
            "transform": "crop",
            "crop_box_xyxy": crop_box_xyxy,
            "action_id": action_id,
        })
        state = self._backend.encode_image(
            asset_id=asset_id,
            image_bytes=encoded,
            width=x1 - x0,
            height=y1 - y0,
        )
        return SessionHandle(state, f"A4:{action_id}")

    def run_prompt(
        self,
        session: SessionHandle,
        *,
        prompt: PromptRequest,
        metadata: PromptMetadata,
        trajectory_id: str,
        upstream_object_id: int | None = None,
    ) -> Any:
        return self._backend.run_action_prompt(
            session.state,
            prompt=prompt,
            metadata=metadata,
            trajectory_id=trajectory_id,
            upstream_object_id=upstream_object_id,
        )

    def close_session(self, session: SessionHandle) -> None:
        self._backend.close_state(session.state)

    def counters(self) -> dict[str, int]:
        return {
            "model_builds": int(self._backend.model_build_count),
            "checkpoint_loads": int(self._backend.checkpoint_load_count),
            "session_initializations": int(self._backend.image_session_initialization_count),
            "prompt_calls": int(self._backend.prompt_call_count),
        }

    def evidence(self) -> dict[str, Any]:
        return {
            "model_spec_id": self.model_spec_id,
            "model_source_commit": self._backend.model_spec.source_commit,
            "checkpoint_sha256": self._backend.model_spec.checkpoint_sha256,
            "sam_config_hash": self._backend.config_hash,
        }

    def close(self) -> None:
        self._backend.close()
        if self._rgb_image is not None:
            self._rgb_image.close()
            self._rgb_image = None
        if getattr(self, "_image_root_fd", None) is not None:
            os.close(self._image_root_fd)
            self._image_root_fd = None

    def __enter__(self) -> "RealSam31TrajectoryBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
