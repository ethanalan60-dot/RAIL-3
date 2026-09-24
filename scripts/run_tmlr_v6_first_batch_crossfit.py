#!/usr/bin/env python3
"""Run registered TMLR V6 first-batch jobs on one physical GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from rail3.models.tmlr_v6.training import (
    FIRST_BATCH_MODEL_IDS,
    FOLDS,
    SEEDS,
    _validated_visible_device,
    assert_cuda_determinism_environment,
    load_protocol,
    run_crossfit,
)


def _repository_root() -> Path:
    value = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(value).resolve()


def main() -> int:
    assert_cuda_determinism_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", required=True, choices=FOLDS)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=FIRST_BATCH_MODEL_IDS,
        default=list(FIRST_BATCH_MODEL_IDS),
    )
    args = parser.parse_args()
    if len(args.folds) != len(set(args.folds)):
        raise ValueError("TMLR V6 fold list is duplicated")
    if len(args.models) != len(set(args.models)):
        raise ValueError("TMLR V6 model list is duplicated")
    repository = _repository_root()
    protocol = load_protocol(repository)
    # Validate the entire worker launch before the first per-job API can
    # create a lock or output.  Mixed physical-GPU fold lists fail atomically.
    for outer_fold in args.folds:
        _validated_visible_device(protocol, outer_fold)
    reports = []
    for outer_fold in args.folds:
        for model_id in args.models:
            for seed in SEEDS:
                report = run_crossfit(
                    model_id=model_id,
                    outer_fold=outer_fold,
                    seed=seed,
                    repository=repository,
                )
                reports.append(report)
                print(json.dumps({
                    key: report[key]
                    for key in (
                        "run_id",
                        "status",
                        "model_id",
                        "outer_fold",
                        "seed",
                        "gpu_wall_seconds",
                        "peak_vram_bytes",
                    )
                }, sort_keys=True), flush=True)
    print(json.dumps({
        "status": "TMLR_V6_FIRST_BATCH_WORKER_COMPLETE",
        "run_count": len(reports),
        "gpu_wall_seconds": sum(
            float(item["gpu_wall_seconds"]) for item in reports
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
