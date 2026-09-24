"""Finite-action LL4TTA/L2D/SPO+ adaptations for the diagnostic core."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rail3.contracts import stable_id
from rail3.models.m06e.data import TensorBundle, sha256_file
from rail3.models.m06e.features import fit_scaler
from rail3.models.m06e.losses import normalized_action_cost
from rail3.models.m06e.models import parameter_count
from rail3.models.m06e.training import (
    FOLDS, _atomic_json, _atomic_npz, _device, _initialize_cuda_telemetry,
    _inner_split, _scale_view, _seed, _snapshot, _tensor,
)


MODEL_IDS = ("L2D_MULTI_ACTION", "DFL_SPO_FINITE")
LAMBDAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
SEEDS = (13, 37, 71)
PROTOCOL_PATH = Path("configs/experiments/rail_diagnostic_core_v1.json")
IMPLEMENTATION_PATHS = (
    "src/rail3/diagnostic/baselines.py",
    "scripts/train_rail_diagnostic_baselines.py",
)
RUN_SECONDS_LIMIT = 1800.0
VRAM_LIMIT = 40 * 1024**3


class MultiActionNetwork(nn.Module):
    """One state-level network with five action outputs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(128, 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def deterministic_masked_argmin(
    values: torch.Tensor, feasible: torch.Tensor,
) -> torch.Tensor:
    """Choose the first feasible minimum in the frozen action order."""

    if values.shape != feasible.shape or values.ndim != 2:
        raise ValueError("values and feasibility must be aligned state/action matrices")
    if not feasible.any(dim=1).all():
        raise ValueError("every state must contain a feasible action")
    return values.masked_fill(~feasible, torch.inf).argmin(dim=1)


def l2d_oracle_action(
    residual: torch.Tensor, normalized_cost: torch.Tensor,
    feasible: torch.Tensor, cost_lambda: float,
) -> torch.Tensor:
    """Training-fold oracle class for the frozen cost-sensitive action set."""

    eligible = feasible & torch.isfinite(residual)
    return deterministic_masked_argmin(
        torch.nan_to_num(residual, nan=torch.inf)
        + float(cost_lambda) * normalized_cost,
        eligible,
    )


def l2d_loss(
    logits: torch.Tensor, residual: torch.Tensor,
    normalized_cost: torch.Tensor, feasible: torch.Tensor,
    state_mask: torch.Tensor, cost_lambda: float,
) -> torch.Tensor:
    """Unweighted masked multiclass cross-entropy; no focal reweighting."""

    eligible = feasible & torch.isfinite(residual)
    active = state_mask & eligible.any(dim=1)
    if not active.any():
        raise ValueError("L2D loss mask contains no target-defined state")
    targets = l2d_oracle_action(
        residual[active], normalized_cost[active], eligible[active], cost_lambda,
    )
    masked_logits = logits[active].masked_fill(~eligible[active], -torch.inf)
    return F.cross_entropy(masked_logits, targets)


def spo_plus_loss(
    prediction: torch.Tensor, true_value: torch.Tensor,
    feasible: torch.Tensor, state_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Published SPO+ minimization convention for finite one-hot actions.

    For ``w*(c)=argmin_w c^T w`` the loss is

    ``-min_w (2*c_hat-c)^T w + 2*c_hat^T w*(c) - c^T w*(c)``.
    """

    eligible_all = feasible & torch.isfinite(true_value)
    active = state_mask & eligible_all.any(dim=1)
    if not active.any():
        raise ValueError("SPO+ loss mask contains no target-defined state")
    prediction = prediction[active]
    true_value = true_value[active]
    eligible = eligible_all[active]
    true_best = deterministic_masked_argmin(true_value, eligible)
    adversarial_best = deterministic_masked_argmin(
        2.0 * prediction - true_value, eligible,
    )
    rows = torch.arange(prediction.shape[0], device=prediction.device)
    loss = (
        -(2.0 * prediction[rows, adversarial_best]
          - true_value[rows, adversarial_best])
        + 2.0 * prediction[rows, true_best]
        - true_value[rows, true_best]
    )
    # The algebraic expression is exactly zero when there is only one legal
    # action, but float32 cancellation need not be.  Preserve that exact
    # finite-action contract while retaining a zero gradient connection.
    loss = torch.where(
        eligible.sum(dim=1) == 1,
        prediction.sum(dim=1) * 0.0,
        loss,
    )
    return loss.mean()


def finite_spo_subgradient(
    prediction: torch.Tensor, true_value: torch.Tensor,
    feasible: torch.Tensor,
) -> torch.Tensor:
    """A deterministic SPO+ subgradient used by exact sign tests."""

    eligible = feasible & torch.isfinite(true_value)
    true_best = deterministic_masked_argmin(true_value, eligible)
    adversarial_best = deterministic_masked_argmin(
        2.0 * prediction - true_value, eligible,
    )
    gradient = torch.zeros_like(prediction)
    rows = torch.arange(prediction.shape[0], device=prediction.device)
    gradient[rows, true_best] += 2.0
    gradient[rows, adversarial_best] -= 2.0
    return gradient


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _implementation_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *IMPLEMENTATION_PATHS],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise RuntimeError("diagnostic baseline implementation is not committed")
    return result


def _lambda_token(value: float) -> str:
    return f"lambda_{value:.3f}".replace(".", "p")


def _verify_existing(
    run_dir: Path, expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not run_dir.exists():
        return None
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.is_file() or any(run_dir.glob("*.tmp")):
        raise RuntimeError(f"incomplete diagnostic baseline run: {run_dir}")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"diagnostic baseline identity mismatch: {run_dir}")
    for key in ("prediction", "checkpoint"):
        item = report[key]
        path = Path(item["path"])
        if (
            not path.is_file() or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"completed diagnostic baseline {key} drift")
    return report


def _global_features(
    view: Mapping[str, Any], outer_train: np.ndarray,
    normalized_cost: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state_scaler = fit_scaler(np.asarray(view["state_x"])[outer_train])
    raw_action = np.asarray(view["action_x"])
    action_scaler = fit_scaler(raw_action[outer_train].reshape(-1, raw_action.shape[-1]))
    state = state_scaler.transform(np.asarray(view["state_x"]))
    action = action_scaler.transform(raw_action.reshape(-1, raw_action.shape[-1])).reshape(raw_action.shape)
    action_with_cost = np.concatenate((action, normalized_cost[:, :, None]), axis=2)
    features = np.concatenate((state, action_with_cost.reshape(len(state), -1)), axis=1).astype(np.float32)
    scaler = {
        "state_mean": state_scaler.mean, "state_scale": state_scaler.scale,
        "action_mean": action_scaler.mean, "action_scale": action_scaler.scale,
    }
    return features, scaler


def run_crossfit(
    *, bundle_root: Path, fold_manifest_path: Path, model_id: str,
    outer_fold: str, seed: int, cost_lambda: float,
    output_root: Path, repository: Path,
) -> dict[str, Any]:
    """Fit one immutable S1364 direct-baseline OOF run."""

    if (
        model_id not in MODEL_IDS or outer_fold not in FOLDS
        or seed not in SEEDS or cost_lambda not in LAMBDAS
    ):
        raise ValueError("unregistered diagnostic baseline run")
    if fold_manifest_path.name != "voc2012-m06e-s_max-folds.json":
        raise ValueError("direct baselines are frozen to S1364")
    device = _device()
    protocol_path = repository / PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if list(LAMBDAS) != protocol["actions"]["cost_grid"]:
        raise RuntimeError("diagnostic baseline lambda-grid drift")
    frozen = protocol["frozen_inputs"]
    if sha256_file(fold_manifest_path) != frozen["folds"]["S1364"]["sha256"]:
        raise RuntimeError("diagnostic baseline fold identity drift")
    if sha256_file(bundle_root / "manifest.json") != frozen["f0_bundle_manifest"]["sha256"]:
        raise RuntimeError("diagnostic baseline bundle identity drift")
    implementation_commit = _implementation_commit(repository)
    protocol_sha256 = sha256_file(protocol_path)
    run_dir = (
        output_root / model_id / _lambda_token(cost_lambda)
        / outer_fold / f"seed_{seed}"
    )
    run_id = stable_id("diagnostic_run", {
        "model_id": model_id, "scale": "S_MAX", "lambda": cost_lambda,
        "outer_fold": outer_fold, "seed": seed,
        "implementation_commit": implementation_commit,
        "protocol_sha256": protocol_sha256,
    })
    expected = {
        "schema_version": "rail3.diagnostic.direct-baseline-run.v1",
        "status": "PASS", "run_id": run_id, "phase": "outer_crossfit",
        "scale": "S_MAX", "model_id": model_id,
        "cost_lambda": cost_lambda, "outer_fold": outer_fold, "seed": seed,
        "implementation_commit": implementation_commit,
        "protocol_sha256": protocol_sha256,
    }
    existing = _verify_existing(run_dir, expected)
    if existing is not None:
        return existing

    started = time.perf_counter()
    started_at = _utc_now()
    _seed(seed)
    telemetry_index = _initialize_cuda_telemetry(device)
    bundle = TensorBundle(bundle_root)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    view = _scale_view(bundle, fold_manifest)
    train_groups, stop_groups, heldout_groups = _inner_split(
        fold_manifest, outer_fold, seed,
    )
    train_np = np.isin(view["group_ids"], tuple(train_groups))
    stop_np = np.isin(view["group_ids"], tuple(stop_groups))
    heldout_np = np.isin(view["group_ids"], tuple(heldout_groups))
    outer_train_np = train_np | stop_np
    cost_t = _tensor(view["cost"], device, torch.float32)
    outer_train_t = _tensor(outer_train_np, device, torch.bool)
    cost_norm_t, cost_median = normalized_action_cost(cost_t, outer_train_t)
    features_np, scalers = _global_features(
        view, outer_train_np, cost_norm_t.detach().cpu().numpy(),
    )
    features = _tensor(features_np, device, torch.float32)
    residual = _tensor(view["state_target"], device, torch.float32)
    feasible = _tensor(view["feasible"], device, torch.bool) & torch.isfinite(residual)
    train_mask = _tensor(train_np, device, torch.bool)
    stop_mask = _tensor(stop_np, device, torch.bool)
    model = MultiActionNetwork(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    def objective(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = model(features)
        if model_id == "L2D_MULTI_ACTION":
            loss = l2d_loss(
                output, residual, cost_norm_t, feasible, mask, cost_lambda,
            )
        else:
            true_value = residual + float(cost_lambda) * cost_norm_t
            loss = spo_plus_loss(output, true_value, feasible, mask)
        return loss, output

    best_state: dict[str, torch.Tensor] | None = None
    best_value, best_epoch, stale, completed = math.inf, 0, 0, 0
    for epoch in range(1, 121):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(train_mask)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite diagnostic baseline loss")
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(objective(stop_mask)[0].item())
        completed = epoch
        if value < best_value - 1e-8:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= 15:
                break
        if time.perf_counter() - started > RUN_SECONDS_LIMIT:
            raise RuntimeError("diagnostic baseline run exceeded wall limit")
    if best_state is None:
        raise RuntimeError("diagnostic baseline produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        raw = model(features)
    predicted_values = -raw if model_id == "L2D_MULTI_ACTION" else raw
    state_indices = np.flatnonzero(heldout_np)
    prediction_path = run_dir / "predictions.npz"
    _atomic_npz(
        prediction_path,
        state_index=view["state_global"][state_indices].astype(np.int32),
        predicted_values=predicted_values[state_indices].cpu().numpy().astype(np.float32),
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save({
        "model": best_state, "model_id": model_id, "lambda": cost_lambda,
        "cost_median_seconds": float(cost_median.item()),
        "scalers": scalers,
    }, temporary)
    temporary.replace(checkpoint_path)
    wall = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_reserved(telemetry_index))
    if wall > RUN_SECONDS_LIMIT or peak >= VRAM_LIMIT:
        raise RuntimeError("diagnostic baseline resource limit exceeded")
    report = {
        **expected,
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES')}",
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_group_ids": sorted(train_groups),
        "inner_stop_group_ids": sorted(stop_groups),
        "heldout_group_ids": sorted(heldout_groups),
        "parameter_count": parameter_count(model), "input_dimension": int(features.shape[1]),
        "best_epoch": best_epoch, "epochs_completed": completed,
        "inner_stop_loss": best_value,
        "cost_median_seconds": float(cost_median.item()),
        "started_at_utc": started_at, "ended_at_utc": _utc_now(),
        "wall_seconds": wall, "gpu_wall_seconds": wall,
        "peak_vram_bytes": peak,
        "prediction": {"path": prediction_path.as_posix(), "bytes": prediction_path.stat().st_size, "sha256": sha256_file(prediction_path)},
        "checkpoint": {"path": checkpoint_path.as_posix(), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
        "counts": {"inner_train_states": int(train_np.sum()), "inner_stop_states": int(stop_np.sum()), "heldout_states": int(heldout_np.sum())},
        "feasibility_masked": True,
        "deterministic_tie_order": ["STOP", "A3", "A4", "A5", "A6"],
        "post_hoc_class_weighting": False,
        "focal_loss": False,
        "heldout_access": {"validation40": 0, "calibration30": 0, "pilot_test30": 0, "official_voc_val": 0},
    }
    _atomic_json(run_dir / "run-manifest.json", report)
    return report
