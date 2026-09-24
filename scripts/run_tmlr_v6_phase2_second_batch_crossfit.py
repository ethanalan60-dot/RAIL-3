#!/usr/bin/env python3
"""Run lock-authorized TMLR V6 phase-two second-batch jobs on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from rail3.models.tmlr_v6.phase2_contract import (
    Phase2Job,
    expected_second_batch_jobs,
    validate_phase2_launch_lock,
)
from rail3.models.tmlr_v6.phase2_models import SECOND_BATCH_MODEL_IDS
from rail3.models.tmlr_v6.phase2_objectives import PHASE2_LAMBDAS
from rail3.models.tmlr_v6.phase2_training import run_phase2_second_batch_crossfit
from rail3.models.tmlr_v6.training import (
    FOLDS,
    assert_cuda_determinism_environment,
    load_protocol,
)


def _repository_root() -> Path:
    value = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return Path(value).resolve()


def selected_jobs(
    *, folds: Sequence[str], models: Sequence[str],
    lambdas: Sequence[float],
) -> tuple[Phase2Job, ...]:
    fold_set = set(folds)
    model_set = set(models)
    lambda_set = set(float(value) for value in lambdas)
    result = tuple(
        job for job in expected_second_batch_jobs()
        if job.outer_fold in fold_set
        and job.model_id in model_set
        and (job.cost_lambda is None or job.cost_lambda in lambda_set)
    )
    expected = sum(
        15 if model not in {"L2D_P", "SPO_PLUS_P"}
        else 15 * len(lambda_set)
        for model in model_set
    ) * len(fold_set) // 5
    if len(result) != expected or len(set(result)) != expected:
        raise RuntimeError("TMLR V6 phase-two worker job selection drift")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    assert_cuda_determinism_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True, choices=FOLDS)
    parser.add_argument(
        "--models", nargs="+", choices=SECOND_BATCH_MODEL_IDS,
        default=list(SECOND_BATCH_MODEL_IDS),
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float, choices=PHASE2_LAMBDAS,
        default=list(PHASE2_LAMBDAS),
        help="lambda-specific direct jobs; R2/R3/Q2 remain joint/single jobs",
    )
    args = parser.parse_args(argv)
    if (
        len(args.folds) != len(set(args.folds))
        or len(args.models) != len(set(args.models))
        or len(args.lambdas) != len(set(args.lambdas))
    ):
        raise ValueError("TMLR V6 phase-two worker arguments are duplicated")

    repository = _repository_root()
    validate_phase2_launch_lock(repository)
    protocol = load_protocol(repository)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assignments = {
        str(device): set(folds)
        for device, folds in protocol["training"]["gpu_assignment"].items()
    }
    if visible not in assignments or not set(args.folds) <= assignments[visible]:
        raise RuntimeError(
            "TMLR V6 phase-two worker violates physical-GPU assignment"
        )
    jobs = selected_jobs(
        folds=args.folds, models=args.models, lambdas=args.lambdas,
    )
    reports = []
    for job in jobs:
        report = run_phase2_second_batch_crossfit(
            job=job, repository=repository,
        )
        reports.append(report)
        print(json.dumps({
            key: report[key] for key in (
                "run_id", "status", "model_id", "outer_fold", "seed",
                "cost_lambda", "gpu_wall_seconds", "peak_vram_bytes",
            )
        }, sort_keys=True), flush=True)
    print(json.dumps({
        "status": "TMLR_V6_PHASE2_SECOND_BATCH_WORKER_COMPLETE",
        "run_count": len(reports),
        "gpu_wall_seconds": sum(
            float(item["gpu_wall_seconds"]) for item in reports
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
