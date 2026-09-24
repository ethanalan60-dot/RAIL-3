#!/usr/bin/env python3
"""Evaluate frozen Track A once; never rerun prediction or training."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from rail3.evaluation.tmlr_v9c_execution import IdentityDrift, execute, load_locked_panel, prediction_inputs_from_panel, verify_frozen_files


def main():
    os.chdir(REPOSITORY)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", choices=["TMLR_V9C_TRACK_A_COCO_GT_EVALUATION_FINAL"])
    modes.add_argument("--evaluate-saved-targets", action="store_true")
    parser.add_argument("--budget-semantics", choices=["FORCED_K"],
                        help="optional assertion of the SHA-bound FORCED_K primary semantics")
    args = parser.parse_args()
    try:
        if args.preflight:
            verified = verify_frozen_files(REPOSITORY)
            panel = load_locked_panel(REPOSITORY)
            prediction_inputs_from_panel(panel, verified["prediction_manifest"], verified["sparse_binding"])
            print(json.dumps({"status": "FINAL_PRE_GT_IDENTITIES_PASS",
                              "checkpoint_count": verified["checkpoint_count"],
                              "prediction_artifacts": verified["prediction_artifact_count"],
                              "panel_states": len(panel["states"]),
                              "COCO_GT_READS_ADDED": 0,
                              "E4_PRIMARY_SPARSE_SEMANTICS": verified["sparse_binding"]["primary_semantics"],
                              "POSITIVE_GAIN_CAP_ROLE": verified["sparse_binding"]["secondary_role"],
                              "E4_sparse_binding": verified["files"]["configs/experiments/tmlr_v9c_track_a_e4_sparse_binding.json"],
                              "FORCED_K_SELECTION_IDENTITIES_VERIFIED": True,
                              "cache_aggregates": panel["cache_aggregates"]}, sort_keys=True))
        else:
            result = execute(REPOSITORY, args.budget_semantics, targets_already_saved=args.evaluate_saved_targets)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            if result["status"] != "COMPLETE":
                return 2
    except IdentityDrift as error:
        print(f"STOP-BLOCKED_TMLR_V9C_FINAL_IDENTITY_DRIFT: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
