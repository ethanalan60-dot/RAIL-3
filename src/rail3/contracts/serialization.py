"""Canonical JSON and stable SHA-256 identifiers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings require string keys")
            result[key] = to_primitive(item)
        return result
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_primitive(item) for item in value), key=canonical_json_bytes)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON forbids NaN and infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_primitive(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def stable_id(prefix: str, payload: Any) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("stable ID prefix must be lowercase snake_case")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"{prefix}_{digest}"
