#!/usr/bin/env python3
"""Inspect release entry points without importing or running scientific drivers.

This is a release adapter, not one of the recovered historical scripts.
It checks shipped source/configuration identities and an explicit location plan.
It never opens input payloads, verifies an original experimental transaction,
creates a run directory, or launches training, inference, evaluation or an atlas.
Original scientific execution remains in the historical implementations, subject
to their original dependencies and identity/authority checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_registry(root=ROOT):
    manifest = json.loads((root / "RELEASE_FILES.json").read_text())
    identities = {row["path"]: row for row in manifest["files"]}
    check_source_files(root, ["ENTRYPOINT_CAPABILITIES.json"], identities)
    registry = json.loads((root / "ENTRYPOINT_CAPABILITIES.json").read_text())
    if registry.get("schema_version") != "rail3.entrypoint-capabilities.v1":
        raise ValueError("unsupported capability registry")
    return registry["entrypoints"]


def check_source_files(root, paths, identities):
    """Only caller-selected shipped .py/.json files; no imports or external I/O."""
    checked, missing = [], []
    for name in paths:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix not in (".py", ".json"):
            raise ValueError("invalid source/configuration path in capability registry")
        local = root / path
        if local.is_symlink() or not local.resolve().is_relative_to(root.resolve()):
            raise ValueError("source path escapes candidate")
        if not local.is_file():
            missing.append(name)
            continue
        if name not in identities:
            raise ValueError("source file is not in release identity manifest: " + name)
        raw = local.read_bytes()
        expected = identities[name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("release source identity mismatch: " + name)
        if path.suffix == ".py":
            ast.parse(raw, filename=name)
        else:
            json.loads(raw)
        checked.append(name)
    return checked, missing


def inspect_bindings(config, required_roles):
    """Inspect user-supplied location fields only, not the files they describe."""
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("configuration must contain input-location objects")
    unbound, supplied = [], []
    for role in required_roles:
        entry = inputs.get(role)
        if entry is None or (isinstance(entry, dict) and entry.get("path") is None):
            unbound.append(role)
        elif not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("invalid input-location field for " + role)
        else:
            supplied.append(role)
    return supplied, unbound


def validate(root, entry, config, run_id, output, distribution_version=importlib.metadata.version):
    manifest = json.loads((root / "RELEASE_FILES.json").read_text())
    identities = {row["path"]: row for row in manifest["files"]}
    required = list(dict.fromkeys(["scripts/release.py", *entry["required_source_files"]]))
    checked, missing_source = check_source_files(root, required, identities)
    if "scripts/release.py" in missing_source:
        raise ValueError("shipped location planner is absent")
    # The existing planner is standard-library only. It preserves its own role,
    # performs lexical input checks, and never imports a scientific module.
    planner_path = root / "scripts/release.py"
    planner_raw = planner_path.read_bytes()
    if hashlib.sha256(planner_raw).hexdigest() != identities["scripts/release.py"]["sha256"]:
        raise ValueError("release planner identity changed before use")
    namespace = {"__file__": str(planner_path), "__name__": "rail3_release_location_plan"}
    exec(compile(planner_raw, str(planner_path), "exec"), namespace)
    plan = namespace["plan"](config, run_id, output, root=root)
    supplied, unbound = inspect_bindings(config, entry["required_input_roles"])
    installed, missing_distributions = {}, []
    for package in entry["runtime_distributions"]:
        try:
            installed[package] = distribution_version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_distributions.append(package)
    incomplete = bool(missing_source or missing_distributions or unbound)
    return {
        "status": "RESOURCE_BINDINGS_INCOMPLETE" if incomplete else "ENGINEERING_VALIDATION_ONLY",
        "engineering_checks": "PASS",
        "entrypoint": entry["id"], "historical_script": entry["historical_script"],
        "source_files_checked": checked, "missing_source_files": missing_source,
        "installed_runtime_distributions": installed,
        "missing_runtime_distributions": missing_distributions,
        "runtime_dependency_scope": entry.get("runtime_dependency_scope", "PRESENCE_METADATA_ONLY"),
        "runtime_compatibility_verification": "NOT_PERFORMED_PRESENCE_METADATA_ONLY",
        "supplied_input_location_roles": supplied, "unbound_input_roles": unbound,
        "input_payloads_opened": False, "input_filesystem_probed": False,
        "location_checks": "LEXICAL_ONLY_NOT_PHYSICAL_IDENTITY_VERIFICATION",
        "historical_identity_verification": "NOT_PERFORMED",
        "historical_authority_limits": entry["historical_authority_limits"],
        "scientific_readiness": "NOT_ESTABLISHED",
        "new_release_run_adapter": entry["new_release_run_adapter"],
        "scientific_execution": "NOT_ATTEMPTED",
        "run_id": plan["run_id"], "output_created": False,
        "exit_code": 3 if incomplete else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List historical source capabilities without importing them")
    parser.add_argument("--entrypoint", help="Canonical release entry-point key (see --list)")
    parser.add_argument("--validate-only", action="store_true", help="Check source/configuration and location fields only")
    parser.add_argument("--config", type=Path, default=ROOT / "release_engineering_inputs.example.json")
    parser.add_argument("--run-id", help="Explicit new run label; no run is executed")
    parser.add_argument("--output", type=Path, help="New output location; no directory is created")
    args = parser.parse_args()  # --help exits before registry/planner source reads.
    try:
        entries = read_registry()
        if args.list:
            if args.validate_only or args.entrypoint:
                parser.error("use --list by itself")
            print(json.dumps({"entrypoints": entries, "scientific_execution": "NOT_ATTEMPTED"}, indent=2))
            return 0
        if not args.validate_only or not args.entrypoint or not args.run_id or args.output is None:
            parser.error("provide --list, or --entrypoint NAME --validate-only --run-id ID --output PATH")
        by_id = {e["id"]: e for e in entries}
        if args.entrypoint not in by_id:
            parser.error("unknown entrypoint; inspect --list")
        config = json.loads(args.config.read_text())
        result = validate(ROOT, by_id[args.entrypoint], config, args.run_id, args.output)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as error:
        # An engineering failure is a nonzero result, never a scientific PASS.
        print(json.dumps({"status": "ENGINEERING_VALIDATION_FAILED", "reason": str(error),
                          "scientific_execution": "NOT_ATTEMPTED"}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
