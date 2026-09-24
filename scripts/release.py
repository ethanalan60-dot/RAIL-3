#!/usr/bin/env python3
"""Verify the supplied snapshot or inspect a non-executing input-location plan.

Standard library only. No imports from research code, dataset access, input
payload hashing, download, model loading, training, evaluation or subprocesses.
This gateway never invokes the historical research drivers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]


def contained_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("invalid relative manifest path")
    result = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in (result, *result.parents) if p != root.parent):
        raise ValueError("symlink in supplied manifest path")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes snapshot")
    return result


def verify(root: Path = ROOT) -> dict:
    manifest_raw = (root / "RELEASE_FILES.json").read_bytes()
    manifest = json.loads(manifest_raw)
    records = manifest["files"]
    names = [item["path"] for item in records]
    if len(set(names)) != len(names):
        raise ValueError("duplicate manifest path")
    if any(name in ("RELEASE_FILES.json", "SHA256SUMS") for name in names):
        raise ValueError("recursive checksum membership")
    expected = {}
    python_files = 0
    for item in records:
        name = item["path"]
        raw = contained_file(root, name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != item["bytes"] or digest != item["sha256"]:
            raise ValueError("supplied file identity mismatch: " + name)
        expected[name] = digest
        if name.endswith(".py"):
            ast.parse(raw, filename=name)  # Parses only; imports/executes nothing.
            python_files += 1
    expected["RELEASE_FILES.json"] = hashlib.sha256(manifest_raw).hexdigest()
    observed = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in observed:
            raise ValueError("duplicate checksum path")
        observed[name] = digest
    if observed != expected:
        raise ValueError("checksum inventory differs from manifest")
    return {"status": "PASS", "scope": "manifest integrity and Python syntax only",
            "files": len(records), "python_files_parsed": python_files,
            "scientific_reproduction": "NOT_RUN", "authentication": "NOT_ESTABLISHED"}


def plan(config: dict, run_id: str, output: Path, root: Path = ROOT) -> dict:
    if config.get("schema_version") != "rail3.release-input-locations.v1":
        raise ValueError("unsupported release input-location schema")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run ID must be a new 1–64 character alphanumeric/hyphen/underscore label")
    destination = output.resolve()
    root = root.resolve()
    if destination.exists() or output.is_symlink():
        raise ValueError("output must not already exist")
    if destination == root or destination in root.parents:
        raise ValueError("output cannot be the snapshot or its ancestor")
    if destination.is_relative_to(root):
        relative = destination.relative_to(root)
        if relative.parts[0] not in ("release-runs", "build") or len(relative.parts) < 2:
            raise ValueError("inside snapshot, use a fresh build/ or release-runs/ child")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("explicit input-location mapping required")
    unbound = []
    mapped = []
    for name, item in inputs.items():
        if not isinstance(item, dict):
            raise ValueError("input entries must be objects")
        location = item.get("path")
        if location is None:
            unbound.append(name)
            continue
        if not isinstance(location, str) or not location:
            raise ValueError("input location must be a nonempty path or null")
        source = Path(location)
        if not source.is_absolute():
            source = root / source
        # Lexical normalization only: never stat/readlink a real input path.
        # This checks the plan, not symlink resolution or physical input identity.
        source = Path(os.path.abspath(source))
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("input and output paths must not overlap")
        # Deliberately never open, list, hash, load or check existence of inputs.
        mapped.append(name)
        if item.get("identity_status") != "VERIFIED" or not item.get("sha256"):
            unbound.append(name)
    return {
        "status": "PLAN_ONLY", "scientific_execution": "BLOCKED",
        "reason": "This release gateway has no scientific execution mode. Historical authority and missing inputs remain unresolved.",
        "run_id": run_id, "output": str(destination), "output_created": False,
        "mapped_input_names": mapped, "unbound_input_names": unbound,
        "input_payloads_accessed": False, "historical_ids_overwritten": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Hash only supplied manifest files and parse Python syntax")
    p = sub.add_parser("plan", help="Inspect locations/new output without accessing any input payload")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        integrity = verify()
        if args.command == "verify":
            result = integrity
        else:
            result = plan(json.loads(args.config.read_text()), args.run_id, args.output)
            result["release_integrity"] = integrity["status"]
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Release check refused: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
