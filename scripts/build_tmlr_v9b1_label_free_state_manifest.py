#!/usr/bin/env python3
"""Validate or create the frozen label-free 1,000 x 20 COCO state manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from rail3.contracts import canonical_json_bytes
from rail3.sam.coco_v9b1_trajectory import (
    EXPECTED_SEMANTIC_RECORDS,
    EXPECTED_STATES,
    build_semantic_states,
    committed_implementation_identity,
    freeze_state_manifest,
    freeze_state_manifest_csv,
    load_config,
    load_frozen_panel_images,
    load_taxonomy,
    shard_manifest_payload,
    state_manifest_csv_bytes,
    state_manifest_payload,
    validate_small_authorities,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="create frozen CSV and shard summary")
    arguments = parser.parse_args()
    repo_root = REPOSITORY
    if Path.cwd().resolve() != repo_root:
        raise RuntimeError("V9B1_STATE_MANIFEST_CWD_DRIFT")
    config = load_config(repo_root)
    validate_small_authorities(repo_root, config)
    images = load_frozen_panel_images(repo_root)
    taxonomy = load_taxonomy(repo_root)
    states = build_semantic_states(images, taxonomy)
    payload = state_manifest_payload(states)
    if len(states) != EXPECTED_STATES or payload["counts"]["semantic_records_expected"] != EXPECTED_SEMANTIC_RECORDS:
        raise RuntimeError("V9B1 frozen state grid count drift")
    csv_bytes = state_manifest_csv_bytes(states)
    result = {
        "images": len(images),
        "states": len(states),
        "semantic_records_expected": payload["counts"]["semantic_records_expected"],
        "shard_images": payload["counts"]["shard_images"],
        "shard_semantic_records": payload["counts"]["shard_semantic_records"],
        "state_manifest_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "wrote_files": False,
    }
    if arguments.write:
        implementation = committed_implementation_identity(
            repo_root, require_global_clean=True
        )
        output = config["outputs"]
        root = repo_root / output["source_data_root"]
        state_path = root / output["state_manifest"]
        shard_path = root / output["shard_manifest"]
        result["state_manifest_sha256"] = freeze_state_manifest_csv(state_path, states)
        shard_payload = shard_manifest_payload(
            state_manifest_path=state_path.relative_to(repo_root),
            state_manifest_sha256=result["state_manifest_sha256"],
            state_payload=payload,
            implementation=implementation,
        )
        result["shard_manifest_sha256"] = freeze_state_manifest(shard_path, shard_payload)
        result["wrote_files"] = True
    print((canonical_json_bytes(result) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
