#!/usr/bin/env python3
"""Run registered phase-2 F0/F1 jobs on one exclusive physical GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from rail3.models.tmlr_v6.phase2_f0_f1 import run_f0_f1_crossfit
from rail3.models.tmlr_v6.phase2_io import FILTRATIONS, FOLDS, SCALES, SEEDS
from rail3.models.tmlr_v6.training import (
    assert_cuda_determinism_environment,
    load_protocol,
)


def _repository_root() -> Path:
    value = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(value).resolve()


def main() -> int:
    assert_cuda_determinism_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", required=True, choices=FOLDS)
    parser.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    parser.add_argument(
        "--filtrations", nargs="+", choices=FILTRATIONS,
        default=list(FILTRATIONS),
    )
    args = parser.parse_args()
    for name, values in (
        ("folds", args.folds), ("scales", args.scales),
        ("filtrations", args.filtrations),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"TMLR V6 phase-2 {name} list is duplicated")
    repository = _repository_root()
    protocol = load_protocol(repository)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assignments = {
        str(device): set(folds)
        for device, folds in protocol["training"]["gpu_assignment"].items()
    }
    if visible not in assignments or not set(args.folds) <= assignments[visible]:
        raise RuntimeError("TMLR V6 phase-2 worker violates physical-GPU assignment")
    reports = []
    for scale in args.scales:
        for filtration in args.filtrations:
            for outer_fold in args.folds:
                for seed in SEEDS:
                    report = run_f0_f1_crossfit(
                        scale=scale,
                        filtration_id=filtration,
                        outer_fold=outer_fold,
                        seed=seed,
                        repository=repository,
                    )
                    reports.append(report)
                    print(json.dumps({
                        key: report[key] for key in (
                            "run_id", "status", "model_id", "filtration_id",
                            "scale", "outer_fold", "seed", "gpu_wall_seconds",
                            "peak_vram_bytes",
                        )
                    }, sort_keys=True), flush=True)
    print(json.dumps({
        "status": "TMLR_V6_PHASE2_F0_F1_WORKER_COMPLETE",
        "run_count": len(reports),
        "gpu_wall_seconds": sum(float(item["gpu_wall_seconds"]) for item in reports),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
