"""Pure, fail-closed COCO category-union target construction.

The caller owns authorization, source reads, panel enumeration and provenance.
This module does no file I/O and never infers class presence to select states.
Rasterization is exclusively the official pycocotools implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        raise ValueError(f"{field} must be {'positive' if positive else 'nonnegative'}")
    return result


def _csv_integer(value: Any, field: str) -> int:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    return _integer(value, field, positive=True)


def _record(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def validate_categories(
    coco_categories: Iterable[Mapping[str, Any]],
    frozen_mapping: Iterable[Mapping[str, Any]],
) -> dict[int, int]:
    """Validate frozen CSV mapping rows against an in-memory COCO catalog.

    Mapping rows use ``voc_index``, ``coco_category_id``,
    ``expected_coco_name``, ``coco_supercategory`` and ``status`` from the
    frozen taxonomy-verification CSV. CSV integer strings are accepted only
    in these mapping rows. The returned dictionary maps VOC class to COCO
    category ID. The caller verifies the mapping's frozen identity and grid.
    """
    by_id: dict[int, Mapping[str, Any]] = {}
    names: set[str] = set()
    for item in coco_categories:
        category = _record(item, "category")
        category_id = _integer(category.get("id"), "category.id")
        name, supercategory = category.get("name"), category.get("supercategory")
        if not isinstance(name, str) or not name:
            raise ValueError("category.name must be a nonempty string")
        if not isinstance(supercategory, str) or not supercategory:
            raise ValueError("category.supercategory must be a nonempty string")
        if category_id in by_id or name in names:
            raise ValueError("duplicate category ID or name")
        by_id[category_id] = category
        names.add(name)

    result: dict[int, int] = {}
    mapped_categories: set[int] = set()
    for item in frozen_mapping:
        row = _record(item, "frozen mapping row")
        class_id = _csv_integer(row.get("voc_index"), "mapping.voc_index")
        category_id = _csv_integer(row.get("coco_category_id"), "mapping.coco_category_id")
        if class_id in result or category_id in mapped_categories:
            raise ValueError("duplicate frozen class or category mapping")
        if row.get("status") != "PASS":
            raise ValueError("frozen category mapping is not PASS")
        category = by_id.get(category_id)
        if category is None:
            raise ValueError(f"mapped COCO category {category_id} is absent")
        if category["name"] != row.get("expected_coco_name"):
            raise ValueError(f"mapped COCO category {category_id} name mismatch")
        if category["supercategory"] != row.get("coco_supercategory"):
            raise ValueError(f"mapped COCO category {category_id} supercategory mismatch")
        result[class_id] = category_id
        mapped_categories.add(category_id)
    if not result:
        raise ValueError("frozen category mapping is empty")
    return result


def _validate_compressed_counts(counts: str | bytes, pixels: int) -> bytes:
    """Check run lengths before passing an encoded string to native code.

    This validates the COCO signed, delta-encoded count stream only; it does
    not create a raster or change the official decoding convention. Checking
    the sum prevents malformed short streams from yielding uninitialized
    pixels in the native decoder.
    """
    try:
        encoded = counts.encode("ascii") if isinstance(counts, str) else counts
    except UnicodeEncodeError as exc:
        raise ValueError("compressed RLE counts must be ASCII") from exc
    if not encoded:
        raise ValueError("compressed RLE counts are empty")
    runs: list[int] = []
    position = 0
    total = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            if position >= len(encoded):
                raise ValueError("truncated compressed RLE count")
            digit = encoded[position] - 48
            position += 1
            if not 0 <= digit <= 63:
                raise ValueError("invalid compressed RLE count character")
            value |= (digit & 31) << shift
            shift += 5
            if not digit & 32:
                if digit & 16:
                    value -= 1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("negative compressed RLE run length")
        runs.append(value)
        total += value
        if total > pixels:
            raise ValueError("RLE run lengths exceed image size")
    if total != pixels:
        raise ValueError("RLE run lengths do not cover image size")
    return encoded


def _decode_instance(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, Mapping):
        size = _sequence(segmentation.get("size"), "RLE.size")
        if len(size) != 2:
            raise ValueError("RLE.size must contain height and width")
        rle_height = _integer(size[0], "RLE.size[0]", positive=True)
        rle_width = _integer(size[1], "RLE.size[1]", positive=True)
        if (rle_height, rle_width) != (height, width):
            raise ValueError("RLE size does not match exact image height and width")
        counts = segmentation.get("counts")
        if isinstance(counts, (str, bytes)):
            encoded = _validate_compressed_counts(counts, height * width)
            rle: Any = {"size": [height, width], "counts": encoded}
        else:
            counts = _sequence(counts, "RLE.counts")
            if not counts:
                raise ValueError("uncompressed RLE counts are empty")
            values = [_integer(value, "RLE run length") for value in counts]
            if sum(values) != height * width:
                raise ValueError("RLE run lengths do not cover image size")
            try:
                rle = coco_mask.frPyObjects({"size": [height, width], "counts": values}, height, width)
            except Exception as exc:
                raise ValueError("official uncompressed RLE conversion failed") from exc
    else:
        polygons = _sequence(segmentation, "polygon segmentation")
        if not polygons:
            raise ValueError("polygon segmentation is empty")
        coordinates: list[list[float]] = []
        for polygon in polygons:
            points = _sequence(polygon, "polygon")
            if len(points) < 6 or len(points) % 2:
                raise ValueError("polygon requires at least three coordinate pairs")
            coordinates.append([_finite_number(point, "polygon coordinate") for point in points])
        try:
            rle = coco_mask.frPyObjects(coordinates, height, width)
        except Exception as exc:
            raise ValueError("official polygon rasterization failed") from exc
    try:
        decoded = coco_mask.decode(rle)
    except Exception as exc:
        raise ValueError("official COCO mask decoding failed") from exc
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise ValueError("decoded instance has the wrong image shape")
    decoded = np.asarray(decoded, dtype=np.bool_)
    if not decoded.any():
        raise ValueError("zero decoded area instance")
    return decoded


def decode_category_unions(
    image_record: Mapping[str, Any],
    annotations_for_image: Iterable[Mapping[str, Any]],
    category_ids: Iterable[int],
) -> dict[int, np.ndarray]:
    """Return full-image boolean unions for every requested COCO category.

    IDs, image linkage and crowd flags are validated for every supplied
    annotation. Only requested-category geometry is decoded. Both crowd and
    non-crowd instances enter the union. Missing categories retain all-zero
    masks. Optional bbox/area metadata is checked but never used to clip,
    replace or reweight a segmentation.
    """
    image = _record(image_record, "image")
    image_id = _integer(image.get("id"), "image.id")
    height = _integer(image.get("height"), "image.height", positive=True)
    width = _integer(image.get("width"), "image.width", positive=True)
    requested = [_integer(value, "requested category ID") for value in category_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate requested category ID")
    unions = {category_id: np.zeros((height, width), dtype=np.bool_) for category_id in requested}
    seen_annotations: set[int] = set()
    for item in annotations_for_image:
        annotation = _record(item, "annotation")
        annotation_id = _integer(annotation.get("id"), "annotation.id")
        if annotation_id in seen_annotations:
            raise ValueError(f"duplicate annotation ID {annotation_id}")
        seen_annotations.add(annotation_id)
        linked_image_id = _integer(annotation.get("image_id"), "annotation.image_id")
        if linked_image_id != image_id:
            raise ValueError(f"annotation {annotation_id} has the wrong image link")
        category_id = _integer(annotation.get("category_id"), "annotation.category_id")
        crowd = _integer(annotation.get("iscrowd"), "annotation.iscrowd")
        if crowd not in (0, 1):
            raise ValueError("annotation.iscrowd must be 0 or 1")
        if "area" in annotation and _finite_number(annotation["area"], "annotation.area") < 0:
            raise ValueError("annotation.area must be nonnegative")
        if "bbox" in annotation:
            bbox = _sequence(annotation["bbox"], "annotation.bbox")
            if len(bbox) != 4:
                raise ValueError("annotation.bbox must contain x, y, width, height")
            box = [_finite_number(value, "bbox coordinate") for value in bbox]
            if box[2] < 0 or box[3] < 0:
                raise ValueError("bbox width and height must be nonnegative")
        if category_id in unions:
            try:
                decoded = _decode_instance(annotation.get("segmentation"), height, width)
            except ValueError as exc:
                raise ValueError(f"annotation {annotation_id}: {exc}") from exc
            unions[category_id] |= decoded
    return unions
