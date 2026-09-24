"""Deterministic single-GPU outer cross-fit for M06-E residual models."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from rail3.contracts import canonical_json_bytes, stable_id
from rail3.models.m06e.data import TensorBundle, sha256_file
from rail3.models.m06e.features import fit_scaler
from rail3.models.m06e.losses import (
    atomic_state_losses, normalized_action_cost, pareto_integrated_decision_regret,
)
from rail3.models.m06e.models import (
    DirectGainBaseline, GlobalResidual, LinearAtomicResidual, SharedAtomicResidual,
    aggregate_atomic, parameter_count,
)


MODEL_IDS = ("R0_GLOBAL_RESIDUAL", "R1_ATOMIC_RESIDUAL", "R2_ATOMIC_PIDR", "Q2_DIRECT_GAIN", "P1_LINEAR_PAL")
SEEDS = (13, 37, 71)
FOLDS = tuple(f"fold_{index}" for index in range(5))
VRAM_LIMIT = 40 * 1024**3
RUN_SECONDS_LIMIT = 1800.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M06-E worker requires exactly one visible CUDA device")
    return torch.device("cuda:0")


def _initialize_cuda_telemetry(device: torch.device) -> int:
    """Materialize the lazy CUDA context before resetting peak statistics."""

    current = torch.cuda.current_device()
    if device.index is None or current != device.index:
        raise RuntimeError("M06-E CUDA telemetry device drift")
    torch.cuda.reset_peak_memory_stats(current)
    return current


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _inner_split(fold_manifest: Mapping[str, Any], outer_fold: str, seed: int) -> tuple[set[str], set[str], set[str]]:
    if outer_fold not in FOLDS:
        raise ValueError("unknown M06-E outer fold")
    heldout = {str(item["image_group_id"]) for item in fold_manifest["groups"] if item["fold_id"] == outer_fold}
    candidates = sorted(str(item["image_group_id"]) for item in fold_manifest["groups"] if item["fold_id"] != outer_fold)
    stop_count = len(candidates) // 5
    def key(group_id: str) -> tuple[str, str]:
        payload = canonical_json_bytes({
            "namespace": f"m06e-inner-{fold_manifest['scale']}-v1", "seed": seed,
            "outer_fold": outer_fold, "image_group_id": group_id,
        })
        return hashlib.sha256(payload).hexdigest(), group_id
    stop = set(sorted(candidates, key=key)[:stop_count])
    train = set(candidates) - stop
    if train & stop or train & heldout or stop & heldout or len(train | stop | heldout) != len(fold_manifest["groups"]):
        raise RuntimeError("M06-E group split isolation failed")
    return train, stop, heldout


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def _tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        result = result.to(dtype)
    return result.to(device)


def _state_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    defined = torch.isfinite(target) & mask[:, None]
    per_action = F.huber_loss(prediction, torch.nan_to_num(target), reduction="none", delta=0.05)
    per_state = (per_action * defined).sum(1) / defined.sum(1).clamp_min(1)
    selected = mask & defined.any(1)
    return per_state[selected].mean()


def _scale_view(bundle: TensorBundle, fold_manifest: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {str(item["image_group_id"]) for item in fold_manifest["groups"]}
    group_ids = bundle.manifest["group_ids"]
    state_global = np.flatnonzero(np.asarray([value in allowed for value in group_ids], dtype=bool))
    if state_global.size != int(fold_manifest["group_count"]) * 20:
        raise RuntimeError("M06-E scale state cardinality mismatch")
    lengths_all = bundle.array("state_lengths")
    offsets = np.r_[0, np.cumsum(lengths_all)]
    atom_global = np.concatenate([
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64) for index in state_global
    ])
    lengths = lengths_all[state_global]
    return {
        "state_global": state_global.astype(np.int32), "atom_global": atom_global.astype(np.int32),
        "lengths": lengths.astype(np.int64),
        "group_ids": np.asarray([group_ids[index] for index in state_global]),
        "atom_x": bundle.array("atom_x")[atom_global], "state_x": bundle.array("state_x")[state_global],
        "action_x": bundle.array("action_x")[state_global], "atom_target": bundle.array("atom_target")[atom_global],
        "atom_weights": bundle.array("atom_weights")[atom_global], "state_target": bundle.array("state_target")[state_global],
        "cost": bundle.array("cost_seconds")[state_global], "feasible": bundle.array("feasible")[state_global],
    }


def _scalers(view: Mapping[str, Any], train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(view["lengths"])
    atom_train = np.repeat(train_mask, lengths)
    atom_scaler = fit_scaler(view["atom_x"][atom_train])
    state_scaler = fit_scaler(view["state_x"][train_mask])
    action_scaler = fit_scaler(view["action_x"][train_mask].reshape(-1, view["action_x"].shape[-1]))
    return (
        atom_scaler.transform(view["atom_x"]), state_scaler.transform(view["state_x"]),
        action_scaler.transform(view["action_x"].reshape(-1, view["action_x"].shape[-1])).reshape(view["action_x"].shape),
    )


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _verify_existing(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_path = path / "run-manifest.json"
    if not path.exists():
        return None
    if not manifest_path.is_file() or any(path.glob("*.tmp")):
        raise RuntimeError(f"incomplete M06-E run exists: {path}")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing M06-E run identity mismatch: {path}")
    prediction = Path(report["prediction"]["path"])
    if prediction.stat().st_size != int(report["prediction"]["bytes"]) or sha256_file(prediction) != report["prediction"]["sha256"]:
        raise RuntimeError("existing M06-E prediction identity drift")
    return report


def run_crossfit(
    *, bundle_root: Path, fold_manifest_path: Path, model_id: str,
    outer_fold: str, seed: int, output_root: Path, repository: Path,
) -> dict[str, Any]:
    if model_id not in MODEL_IDS or outer_fold not in FOLDS or seed not in SEEDS:
        raise ValueError("unregistered M06-E run")
    device = _device()
    bundle = TensorBundle(bundle_root)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    head = _git_head(repository)
    protocol_path = repository / "configs/experiments/voc_m06e_atomic_residual_v1.json"
    data_manifest_path = repository / "data/manifests/voc2012-m06e-fit1364.json"
    run_dir = output_root / fold_manifest["scale"] / model_id / outer_fold / f"seed_{seed}"
    run_id = stable_id("m06e_run", {
        "scale": fold_manifest["scale"], "model_id": model_id,
        "outer_fold": outer_fold, "seed": seed, "git_commit": head,
        "bundle_sha256": sha256_file(bundle_root / "manifest.json"),
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
    })
    expected = {
        "schema_version": "rail3.m06e.model-run-manifest.v1", "status": "PASS",
        "run_id": run_id, "phase": "outer_crossfit",
        "scale": fold_manifest["scale"], "model_id": model_id,
        "outer_fold": outer_fold, "seed": seed, "git_commit": head,
        "bundle_sha256": sha256_file(bundle_root / "manifest.json"),
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
    }
    existing = _verify_existing(run_dir, expected)
    if existing is not None:
        return existing
    started = time.perf_counter()
    started_utc = _utc_now()
    _seed(seed)
    telemetry_device_index = _initialize_cuda_telemetry(device)
    view = _scale_view(bundle, fold_manifest)
    train_groups, stop_groups, heldout_groups = _inner_split(fold_manifest, outer_fold, seed)
    train_np = np.isin(view["group_ids"], tuple(train_groups))
    stop_np = np.isin(view["group_ids"], tuple(stop_groups))
    heldout_np = np.isin(view["group_ids"], tuple(heldout_groups))
    outer_train_np = train_np | stop_np
    if not train_np.any() or not stop_np.any() or not heldout_np.any():
        raise RuntimeError("empty M06-E train/stop/heldout partition")
    atom_x_np, state_x_np, action_x_np = _scalers(view, outer_train_np)
    cost_t = _tensor(view["cost"], device, torch.float32)
    train_t = _tensor(train_np, device, torch.bool)
    stop_t = _tensor(stop_np, device, torch.bool)
    outer_train_t = _tensor(outer_train_np, device, torch.bool)
    cost_norm, cost_median = normalized_action_cost(cost_t, outer_train_t)
    action_x_np = np.concatenate((action_x_np, cost_norm.detach().cpu().numpy()[:, :, None]), axis=2)
    atom_x = _tensor(atom_x_np, device, torch.float32)
    state_x = _tensor(state_x_np, device, torch.float32)
    action_x = _tensor(action_x_np, device, torch.float32)
    atom_target = _tensor(view["atom_target"], device, torch.float32)
    atom_weights = _tensor(view["atom_weights"], device, torch.float32)
    state_target = _tensor(view["state_target"], device, torch.float32)
    lengths = _tensor(view["lengths"], device, torch.long)
    global_x = torch.cat((
        state_x[:, None, :].expand(-1, 5, -1), action_x,
    ), dim=2)
    atom_states = torch.repeat_interleave(torch.arange(state_x.shape[0], device=device), lengths)
    p1_x = None
    if model_id == "P1_LINEAR_PAL":
        p1_x = torch.cat((
            atom_x[:, None, :].expand(-1, 5, -1),
            state_x[atom_states, None, :].expand(-1, 5, -1), action_x[atom_states],
        ), dim=2)
    if model_id == "R0_GLOBAL_RESIDUAL":
        model: torch.nn.Module = GlobalResidual(global_x.shape[2])
    elif model_id in {"R1_ATOMIC_RESIDUAL", "R2_ATOMIC_PIDR"}:
        model = SharedAtomicResidual(atom_x.shape[1], state_x.shape[1], action_x.shape[2])
    elif model_id == "Q2_DIRECT_GAIN":
        model = DirectGainBaseline(global_x.shape[2])
    else:
        assert p1_x is not None
        model = LinearAtomicResidual(p1_x.shape[2])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    def forward_loss(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if model_id == "R0_GLOBAL_RESIDUAL":
            state_prediction = model(global_x.reshape(-1, global_x.shape[-1])).reshape(-1, 5)
            return _state_huber(state_prediction, state_target, mask), state_prediction, None
        if model_id == "Q2_DIRECT_GAIN":
            gain_prediction = model(global_x.reshape(-1, global_x.shape[-1])).reshape(-1, 5)
            gain_target = state_target[:, :1] - state_target
            return _state_huber(gain_prediction, gain_target, mask), gain_prediction, None
        if model_id == "P1_LINEAR_PAL":
            assert p1_x is not None
            atomic_prediction = model(p1_x.reshape(-1, p1_x.shape[-1])).reshape(-1, 5)
            state_prediction = aggregate_atomic(atomic_prediction, atom_weights, lengths)
            atom_loss, _ = atomic_state_losses(
                atomic_prediction, state_prediction, atom_target, state_target,
                atom_weights, lengths, mask,
            )
            return atom_loss, state_prediction, atomic_prediction
        atomic_prediction, state_prediction = model(atom_x, state_x, action_x, lengths, atom_weights)
        atom_loss, state_loss = atomic_state_losses(
            atomic_prediction, state_prediction, atom_target, state_target,
            atom_weights, lengths, mask,
        )
        loss = atom_loss + state_loss
        if model_id == "R2_ATOMIC_PIDR":
            loss = loss + 0.5 * pareto_integrated_decision_regret(
                state_prediction, state_target, cost_norm, mask,
            )
        return loss, state_prediction, atomic_prediction

    best_state: dict[str, torch.Tensor] | None = None
    best_value, best_epoch, stale, completed = math.inf, 0, 0, 0
    for epoch in range(1, 121):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward_loss(train_t)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite M06-E training loss")
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(forward_loss(stop_t)[0].item())
        completed = epoch
        if value < best_value - 1e-8:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = _snapshot(model)
        else:
            stale += 1
            if stale >= 15:
                break
        if time.perf_counter() - started > RUN_SECONDS_LIMIT:
            raise RuntimeError("M06-E single-run wall limit exceeded")
    if best_state is None:
        raise RuntimeError("M06-E early stopping produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        _, state_prediction, atomic_prediction = forward_loss(stop_t)
    state_indices = np.flatnonzero(heldout_np)
    output: dict[str, np.ndarray] = {
        "state_index": view["state_global"][state_indices].astype(np.int32),
        "state_prediction": state_prediction[state_indices].detach().cpu().numpy().astype(np.float32),
    }
    if atomic_prediction is not None:
        local_offsets = np.r_[0, np.cumsum(view["lengths"])]
        atom_local = np.concatenate([np.arange(local_offsets[index], local_offsets[index + 1]) for index in state_indices])
        output["atom_index"] = view["atom_global"][atom_local].astype(np.int32)
        output["atom_prediction"] = atomic_prediction[atom_local].detach().cpu().numpy().astype(np.float32)
    prediction_path = run_dir / "predictions.npz"
    _atomic_npz(prediction_path, **output)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save({"model": best_state}, temporary); temporary.replace(checkpoint_path)
    wall = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_reserved(telemetry_device_index))
    if wall > RUN_SECONDS_LIMIT or peak >= VRAM_LIMIT:
        raise RuntimeError("M06-E per-run resource limit exceeded")
    report = {
        **expected, "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES')}",
        "train_group_ids": sorted(train_groups), "inner_stop_group_ids": sorted(stop_groups),
        "heldout_group_ids": sorted(heldout_groups),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "feature_manifest_sha256": bundle.manifest["feature_report_sha256"],
        "target_manifest_sha256": bundle.manifest["target_report_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "started_at_utc": started_utc, "ended_at_utc": _utc_now(),
        "parameter_count": parameter_count(model), "best_epoch": best_epoch,
        "epochs_completed": completed, "inner_stop_loss": best_value,
        "cost_median_seconds": float(cost_median.item()), "wall_seconds": wall,
        "peak_vram_bytes": peak, "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "prediction": {"path": prediction_path.as_posix(), "bytes": prediction_path.stat().st_size, "sha256": sha256_file(prediction_path)},
        "checkpoint": {"logical_path": checkpoint_path.as_posix(), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
        "counts": {"inner_train_states": int(train_np.sum()), "inner_stop_states": int(stop_np.sum()), "heldout_states": int(heldout_np.sum())},
        "validation_row_access": False, "heldout_access": {"validation40": 0, "calibration30": 0, "pilot_test30": 0, "official_voc_val": 0},
        "single_residual_predictor": model_id in {"R0_GLOBAL_RESIDUAL", "R1_ATOMIC_RESIDUAL", "R2_ATOMIC_PIDR", "P1_LINEAR_PAL"},
        "independent_heads": [], "independent_q_head_in_R_models": False,
        "opportunity_head_in_R_models": False, "gpu_wall_seconds": wall,
    }
    _atomic_json(run_dir / "run-manifest.json", report)
    return report
