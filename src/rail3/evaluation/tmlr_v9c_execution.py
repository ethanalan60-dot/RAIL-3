"""Audited I/O for the frozen V9C evaluation; never trains or predicts.

The SHA-bound pre-GT supplement supplies the sparse-budget binding. Annotation access
is a one-shot operation preceded by an immutable identity record. Saved targets
can be evaluated again without reopening the annotation source.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import numpy as np

from rail3.cache.candidates import CandidateGenerationResult, decode_binary_mask_array
from rail3.contracts import canonical_json_bytes
from rail3.evaluation.tmlr_v9c_coco_targets import decode_category_unions, validate_categories
from rail3.sam.coco_v9b1_cache import secure_create_once, secure_create_or_verify, secure_read_bytes
from rail3.sam.coco_v9b1_trajectory import (
    build_semantic_states, load_frozen_panel_images, load_taxonomy,
)
from rail3.sam.voc_actions import ActionOutcome


SOURCE = Path("artifacts/paper/source_data/tmlr_v9c")
TARGET_ROOT = Path("artifacts/targets/tmlr-v9c/track-a")
ANNOTATIONS = Path("data/public/coco2017/raw/annotations/instances_val2017.json")
PREDICTION_LOCK = SOURCE / "track_a_coco_prediction_lock.json"
E4 = Path("configs/experiments/tmlr_v9c_track_a_e4_decision_contract.json")
E5 = Path("configs/experiments/tmlr_v9c_track_a_e5_decision_contract.json")
SPARSE_BINDING = Path("configs/experiments/tmlr_v9c_track_a_e4_sparse_binding.json")
SPARSE_BINDING_SHA256 = "034fea1cc0a154dbefb25f95253f0ba9154103c8a06198e201aa57b7183a5f07"
TRAJECTORY_LOCK = SOURCE / "final_trajectory_lock.json"
PRE_READ = SOURCE / "track_a_before_first_gt_read.json"
FIRST_READ = SOURCE / "track_a_first_gt_read.json"
TARGET_INVENTORY = SOURCE / "track_a_coco_target_inventory.json"
START_COMMIT = "edb781b5ea9118326404cc02a152d5a29e106a61"
ACTIONS = ("STOP", "A3", "A4", "A5", "A6")
FAMILIES = ("R0_SMALL_P", "R0_CM_P", "R1_P", "RECT_P", "UNION_P", "R3_P")


class IdentityDrift(RuntimeError):
    """A frozen identity failed; the caller must not repair or read GT."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def evaluation_environment() -> dict[str, str]:
    from importlib.metadata import version

    return {"numpy": np.__version__, "pycocotools": version("pycocotools")}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = secure_read_bytes(path)
    return {"path": path.as_posix(), "bytes": len(raw), "sha256": sha(raw)}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(secure_read_bytes(Path(path)))


def create_json(path: str | Path, payload: Any) -> None:
    secure_create_once(Path(path), canonical_json_bytes(payload) + b"\n")


def create_result_json(path: str | Path, payload: Any) -> None:
    """Resume generated results only when their existing bytes are identical."""
    secure_create_or_verify(Path(path), canonical_json_bytes(payload) + b"\n")


def _verify(path: str | Path, binding: dict[str, Any]) -> dict[str, Any]:
    observed = identity(path)
    if observed["sha256"] != binding["sha256"] or (
        "bytes" in binding and observed["bytes"] != int(binding["bytes"])
    ):
        raise IdentityDrift(f"frozen identity drift: {path}")
    return observed


def _git_bytes(repo: Path, commit: str, path: str | Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{Path(path).as_posix()}"], cwd=repo)


def load_sparse_binding() -> dict[str, Any]:
    """Authenticate the user-authorized supplement before any GT access."""
    raw = secure_read_bytes(SPARSE_BINDING)
    if sha(raw) != SPARSE_BINDING_SHA256:
        raise IdentityDrift("E4 sparse binding SHA differs from evaluation authority")
    return json.loads(raw)


def verify_frozen_files(repo: Path) -> dict[str, Any]:
    """Hash exact locked files, including 18 checkpoints and 24 predictions."""
    sparse_binding = load_sparse_binding()
    e4_lock_path = SOURCE / "track_a_e4_decision_contract_lock.json"
    if secure_read_bytes(e4_lock_path) != _git_bytes(repo, START_COMMIT, e4_lock_path):
        raise IdentityDrift("E4 lock changed from edb781b")
    e4_lock = read_json(e4_lock_path)
    bindings = {**e4_lock["frozen_authorities_unchanged"], **e4_lock["artifacts"]}
    observed = {path: _verify(path, binding) for path, binding in bindings.items()}
    observed[SPARSE_BINDING.as_posix()] = identity(SPARSE_BINDING)
    for path, binding in sparse_binding["unchanged_authorities"].items():
        observed[path] = _verify(path, binding)
    selection_authority = sparse_binding["selection_identity_authority"]["record"]
    observed[selection_authority["path"]] = _verify(selection_authority["path"], selection_authority)
    for path, commit in ((E4, START_COMMIT), (E5, "e21b4fa28238e6644414e84db7dbf9d5f57ebfb6")):
        if secure_read_bytes(path) != _git_bytes(repo, commit, path):
            raise IdentityDrift(f"contract Git identity drift: {path}")
    prediction = read_json(PREDICTION_LOCK)
    if prediction["status"] != "TMLR_V9C_TRACK_A_PREDICTION_LOCK_COMPLETE":
        raise IdentityDrift("prediction freeze incomplete")
    if prediction["counters_at_lock"]["COCO_GT_READS"] != 0 or prediction["counters_at_lock"]["PERFORMANCE_RESULTS"] != 0:
        raise IdentityDrift("prediction lock was not pre-GT")
    if prediction["track_b_no_go"]["status"] != "NO_GO":
        raise IdentityDrift("Track B disposition changed")
    checkpoints = prediction["checkpoint_identities"]
    if {(row["family"], row["seed"]) for row in checkpoints} != {
        (family, seed) for family in FAMILIES for seed in (13, 37, 71)
    } or len(checkpoints) != 18:
        raise IdentityDrift("18-checkpoint grid changed")
    for row in checkpoints:
        p = row["checkpoint"]["path"]
        observed[p] = _verify(p, row["checkpoint"])
    for key in ("checkpoint_lock", "feature_inventory", "prediction_inventory", "trajectory_inventory", "trajectory_lock"):
        binding = prediction[key]
        observed[binding["path"]] = _verify(binding["path"], binding)
    for p, digest in prediction["implementation"]["sha256"].items():
        observed[p] = _verify(p, {"sha256": digest})
    manifests = prediction["prediction_manifests"]
    for build in ("build1", "build2"):
        binding = manifests[build]
        observed[binding["path"]] = _verify(binding["path"], binding)
    if secure_read_bytes(Path(manifests["build1"]["path"])) != secure_read_bytes(Path(manifests["build2"]["path"])):
        raise IdentityDrift("prediction manifests differ")
    manifest = read_json(manifests["build1"]["path"])
    if len(manifest["prediction_artifacts"]) != 24:
        raise IdentityDrift("prediction artifact count changed")
    for row in manifest["prediction_artifacts"]:
        observed[row["path"]] = _verify(row["path"], row)
    feature_inventory = read_json(prediction["feature_inventory"]["path"])
    for row in feature_inventory["roles"]:
        observed[row["path"]] = _verify(row["path"], row)
    return {"files": observed, "checkpoint_count": 18, "prediction_artifact_count": 24,
            "prediction_lock": prediction, "prediction_manifest": manifest,
            "sparse_binding": sparse_binding}


def _cache_path(root: Path, key: str) -> Path:
    suffix = key.rsplit("_", 1)[-1]
    if len(suffix) != 64 or any(c not in "0123456789abcdef" for c in suffix) or "/" in key:
        raise IdentityDrift("invalid locked cache key")
    return root / suffix[:2] / f"{key}.json"


def load_locked_panel(repo: Path) -> dict[str, Any]:
    """Read only frozen panel/plans/cache hashes, retaining all 20,000 states."""
    images = load_frozen_panel_images(repo)
    taxonomy = load_taxonomy(repo)
    states = build_semantic_states(images, taxonomy)
    inventory = read_json("artifacts/paper/source_data/tmlr_v9b1/coco_external_trajectory_inventory.json")
    missing = read_json(SOURCE / "technical_missingness_report.json")
    roots = {key: Path(value) for key, value in inventory["final_cache_roots"].items()}
    plans, plan_entries = {}, []
    plan_bytes = 0
    keys: dict[str, set[str]] = {"a0": set(), "actions": set()}
    for image in images:
        digest = sha(image.asset_id.encode())
        path = roots["plans"] / digest[:2] / f"image_plan_{digest}.json"
        raw = secure_read_bytes(path)
        plan = json.loads(raw)
        if plan["canonical_image_id"] != image.canonical_image_id or plan["asset_id"] != image.asset_id:
            raise IdentityDrift("plan image identity drift")
        plans[image.image_id] = plan
        plan_entries.append({"canonical_image_id": image.canonical_image_id, "plan_lock_sha256": sha(raw)})
        plan_bytes += len(raw)
        keys["a0"].update(row["cache_key"] for row in plan["a0"])
        keys["actions"].update(row["cache_key"] for row in plan["actions"])
    plan_sha = sha(canonical_json_bytes(sorted(plan_entries, key=lambda row: row["canonical_image_id"])))
    if plan_sha != inventory["plan_locks"]["plan_lock_aggregate_sha256"] or plan_bytes != inventory["plan_locks"]["plan_lock_bytes"]:
        raise IdentityDrift("trajectory plan aggregate drift")
    cache_bindings, cache_aggregates = {}, {}
    for kind, expected_key in (("a0", "a0_cache"), ("actions", "action_cache")):
        entries = []
        for key in sorted(keys[kind]):
            binding = identity(_cache_path(roots[kind], key))
            cache_bindings[key] = binding
            entries.append({"cache_key": key, "bytes": binding["bytes"], "sha256": binding["sha256"]})
        aggregate = {"objects": len(entries), "bytes": sum(row["bytes"] for row in entries),
                     "aggregate_sha256": sha(canonical_json_bytes(entries))}
        if aggregate != inventory[expected_key]:
            raise IdentityDrift(f"{kind} cache aggregate drift")
        cache_aggregates[kind] = aggregate
    missing_states = set(missing["state_level"]["semantic_state_ids"])
    missing_actions = {(row["semantic_state_id"], row["action_code"]) for row in inventory["action_technical_missingness"]["rows"]}
    if len(missing_states) != 1 or len(missing_actions) != 3 or len(missing_states | {s for s, _ in missing_actions}) != 4:
        raise IdentityDrift("technical missingness grid drift")
    return {"images": images, "states": states, "plans": plans, "taxonomy": taxonomy,
            "roots": roots, "cache_bindings": cache_bindings, "cache_aggregates": cache_aggregates,
            "plan_sha256": plan_sha, "missing_states": missing_states, "missing_actions": missing_actions}


def read_annotations_once(annotation_path: Path, pre_read_path: Path, receipt_path: Path,
                          before_read: dict[str, Any], *, recovery_snapshot: Path | None = None) -> bytes:
    """Create immutable evidence before opening the source exactly once.

    A failed attempt is retained and cannot silently restart. Tests use separate
    synthetic files; the execution entry point supplies only ANNOTATIONS.
    """
    if pre_read_path.exists() or receipt_path.exists():
        raise RuntimeError("FIRST_GT_READ_ALREADY_ATTEMPTED; do not reopen annotations")
    create_json(pre_read_path, before_read)
    first_timestamp = utc_now()
    descriptor = None
    receipt = {"schema_version": "rail3.tmlr-v9c-first-gt-read.v1", "source": annotation_path.as_posix(),
               "COCO_GT_FIRST_READ_TIMESTAMP": None, "COCO_GT_READS": 0,
               "pre_read_record": identity(pre_read_path), "status": "OPEN_FAILED"}
    try:
        descriptor = os.open(annotation_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        receipt.update(COCO_GT_FIRST_READ_TIMESTAMP=first_timestamp, COCO_GT_READS=1, status="READ_FAILED")
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("annotation source is not a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != metadata.st_size:
            raise ValueError("annotation source changed during first read")
        if recovery_snapshot is not None:
            secure_create_once(recovery_snapshot, raw)
            receipt["untracked_recovery_snapshot"] = identity(recovery_snapshot)
        receipt.update(status="READ_COMPLETE", bytes=len(raw), sha256=sha(raw), read_completed_utc=utc_now())
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        create_json(receipt_path, receipt)


def _payload(binding: dict[str, Any], cache_key: str) -> dict[str, Any]:
    raw = secure_read_bytes(Path(binding["path"]))
    if sha(raw) != binding["sha256"] or len(raw) != binding["bytes"]:
        raise IdentityDrift("cache changed after pre-GT verification")
    envelope = json.loads(raw)
    payload = envelope["payload"]
    if envelope["cache_key"] != cache_key or envelope["payload_sha256"] != sha(canonical_json_bytes(payload)):
        raise IdentityDrift("cache envelope drift")
    return payload


def candidate_union(candidates: Any, height: int, width: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=bool)
    for candidate in candidates:
        if (candidate.height, candidate.width) != (height, width):
            raise ValueError("cached mask dimensions differ from frozen image")
        union |= decode_binary_mask_array(candidate.mask_rle, width, height)
    return union


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def construct_targets(raw_annotations: bytes, panel: dict[str, Any]) -> dict[str, Any]:
    """Build every category union and residual; no policy filtering or imputation."""
    coco = json.loads(raw_annotations)
    mapping_path = Path("artifacts/paper/source_data/tmlr_v9b0_r2/coco_voc20_taxonomy_verification.csv")
    mapping_rows = list(csv.DictReader(StringIO(secure_read_bytes(mapping_path).decode())))
    mapping = validate_categories(coco["categories"], mapping_rows)
    if set(mapping) != set(range(1, 21)):
        raise IdentityDrift("VOC20 mapping changed")
    image_records = {}
    for row in coco["images"]:
        if row["id"] in image_records:
            raise ValueError("duplicate COCO image ID")
        image_records[row["id"]] = row
    selected_ids = {image.image_id for image in panel["images"]}
    annotations = {image_id: [] for image_id in selected_ids}
    annotation_ids = set()
    for row in coco["annotations"]:
        if row["id"] in annotation_ids:
            raise ValueError("duplicate COCO annotation ID")
        annotation_ids.add(row["id"])
        if row["image_id"] in annotations:
            annotations[row["image_id"]].append(row)
    states = panel["states"]
    state_index = {state.semantic_state_id: index for index, state in enumerate(states)}
    target = np.full((len(states), 5), np.nan, dtype=np.float64)
    feasible = np.zeros((len(states), 5), dtype=bool)
    entries = []
    for image in panel["images"]:
        image_row = image_records[image.image_id]
        if (image_row["width"], image_row["height"], image_row["file_name"]) != (image.width, image.height, image.filename):
            raise IdentityDrift("GT image metadata differs from frozen panel")
        references = decode_category_unions(image_row, annotations[image.image_id], mapping.values())
        packed = np.stack([np.packbits(references[mapping[c]].reshape(-1), bitorder="big") for c in range(1, 21)])
        mask_path = TARGET_ROOT / "unions" / f"{image.canonical_image_id}.npz"
        secure_create_or_verify(mask_path, _npz_bytes(reference_unions_packed=packed,
                           shape=np.asarray([image.height, image.width], dtype=np.int64),
                           class_ids=np.arange(1, 21, dtype=np.int16),
                           category_ids=np.asarray([mapping[c] for c in range(1, 21)], dtype=np.int16)))
        entries.append({"image_id": image.canonical_image_id, **identity(mask_path),
                        "packed_payload_sha256": sha(packed.tobytes()), "classes": 20})
        plan = panel["plans"][image.image_id]
        for row in plan["a0"]:
            state_id, key = row["semantic_state_id"], row["cache_key"]
            if state_id in panel["missing_states"]:
                raise IdentityDrift("missing trajectory gained an A0 plan")
            index = state_index[state_id]
            result = CandidateGenerationResult.from_dict(_payload(panel["cache_bindings"][key], key))
            if result.failure is not None:
                raise ValueError("unexpected A0 failure")
            mask = candidate_union(result.all_candidates, image.height, image.width)
            reference = references[mapping[row["class_id"]]]
            target[index, 0] = np.count_nonzero(mask != reference) / mask.size
            feasible[index, 0] = True
        for row in plan["actions"]:
            action = row["action_code"]
            if action not in ACTIONS:
                continue
            state_id, key = row["semantic_state_id"], row["cache_key"]
            index, action_index = state_index[state_id], ACTIONS.index(action)
            feasible[index, action_index] = row["feasible"]
            result = ActionOutcome.from_dict(_payload(panel["cache_bindings"][key], key))
            if (result.action_id, result.action_code, result.feasible) != (row["action_id"], action, row["feasible"]):
                raise IdentityDrift("action outcome differs from frozen plan")
            if (state_id, action) in panel["missing_actions"]:
                if result.outcome != "failure":
                    raise IdentityDrift("frozen A4 missingness changed")
                continue
            if not result.feasible:
                if result.outcome != "infeasible":
                    raise ValueError("infeasible action has unexpected outcome")
                continue
            if result.outcome not in ("result", "no_result"):
                raise ValueError("unexpected action failure")
            mask = candidate_union(result.candidates, image.height, image.width)
            reference = references[mapping[row["class_id"]]]
            target[index, action_index] = np.count_nonzero(mask != reference) / mask.size
        if (image.image_index + 1) % 100 == 0:
            print(json.dumps({"target_images_completed": image.image_index + 1, "panel_images": 1000}), flush=True)
    unavailable = {states[index].semantic_state_id for index in np.flatnonzero(~np.isfinite(target[:, 0]))}
    undefined_feasible = {(states[i].semantic_state_id, ACTIONS[a]) for i, a in zip(*np.nonzero(feasible & ~np.isfinite(target)))}
    if unavailable != panel["missing_states"] or undefined_feasible != panel["missing_actions"]:
        raise IdentityDrift("target missingness differs from frozen exact-state/action rules")
    arrays = {"targets": target, "plan_feasible": feasible,
              "state_ids": np.asarray([state.semantic_state_id for state in states]),
              "image_group_ids": np.asarray([state.image.canonical_image_id for state in states])}
    array_path = TARGET_ROOT / "residual_targets.npz"
    secure_create_or_verify(array_path, _npz_bytes(**arrays))
    target_inventory = {
        "schema_version": "rail3.tmlr-v9c-coco-target-inventory.v1", "status": "TARGETS_COMPLETE",
        "COCO_IMAGES": len(panel["images"]), "PANEL_STATES": len(states),
        "COCO_GT_FIRST_READ_TIMESTAMP": read_json(FIRST_READ)["COCO_GT_FIRST_READ_TIMESTAMP"],
        "COCO_GT_READS": 1, "annotation_source": {"path": ANNOTATIONS.as_posix(), "bytes": len(raw_annotations), "sha256": sha(raw_annotations)},
        "first_read_receipt": identity(FIRST_READ), "mapping": identity(mapping_path),
        "target_contract": read_json("configs/experiments/tmlr_v9_coco_external_v2.json")["future_target_contract"],
        "target_arrays": identity(array_path), "reference_union_artifacts": entries,
        "missing_trajectory_states": sorted(unavailable),
        "missing_feasible_action_pairs": [list(row) for row in sorted(undefined_feasible)],
        "raw_masks_and_target_arrays_committed": False,
    }
    create_result_json(TARGET_INVENTORY, target_inventory)
    return arrays


def load_frozen_predictions(state_ids: np.ndarray, manifest: dict[str, Any],
                            missing_state_ids: set[str], selection_available: np.ndarray) -> tuple[dict, dict]:
    lookup = {str(value): index for index, value in enumerate(state_ids)}
    predictions, selected = {}, {}
    for row in manifest["prediction_artifacts"]:
        if row["artifact_kind"] != "ENSEMBLE_CONTINUOUS_AND_LAMBDA0_SELECTION":
            continue
        _verify(row["path"], row)
        with np.load(BytesIO(secure_read_bytes(Path(row["path"]))), allow_pickle=False) as saved:
            ids = saved["state_ids"].astype(str)
            family = row["family"]
            if family in predictions or family not in FAMILIES:
                raise IdentityDrift("duplicate or unexpected ensemble family")
            if ids.ndim != 1 or len(set(ids)) != len(ids) or set(lookup) - set(ids) != missing_state_ids or not set(ids) <= set(lookup):
                raise IdentityDrift("unexpected or duplicate prediction states")
            raw_predictions, raw_actions = saved["prediction"], saved["selected_action"]
            if raw_predictions.shape != (len(ids), 5) or raw_predictions.dtype != np.float64 or not np.isfinite(raw_predictions).all():
                raise IdentityDrift("ensemble prediction shape/dtype/finite drift")
            if raw_actions.shape != (len(ids),) or raw_actions.dtype.kind not in "US" or not set(raw_actions.astype(str)) <= set(ACTIONS):
                raise IdentityDrift("ensemble selected-action shape/labels drift")
            indices = np.asarray([lookup[value] for value in ids])
            frozen_feasible = saved["feasibility"]
            if frozen_feasible.shape != (len(ids), 5) or not np.array_equal(frozen_feasible, selection_available[indices]):
                raise IdentityDrift("frozen selection availability differs from pre-GT plans/missingness")
            decoded_actions = raw_actions.astype(str)
            action_indices = np.asarray([ACTIONS.index(value) for value in decoded_actions], dtype=np.int8)
            expected = np.where(selection_available[indices], raw_predictions, np.inf).argmin(axis=1)
            if not np.array_equal(action_indices, expected):
                raise IdentityDrift("frozen selected actions differ from continuous first-argmin rule")
            if saved["selected_prediction"].shape != (len(ids),) or not np.array_equal(saved["selected_prediction"], raw_predictions[np.arange(len(ids)), action_indices]):
                raise IdentityDrift("frozen selected prediction values drift")
            prediction = np.full((len(state_ids), 5), np.nan, dtype=np.float64)
            prediction[indices] = raw_predictions
            actions = np.full(len(state_ids), -1, dtype=np.int8)
            actions[indices] = action_indices
            predictions[family], selected[family] = prediction, actions
    if set(predictions) != set(FAMILIES):
        raise IdentityDrift("six ensemble prediction families not exact")
    return predictions, selected


def verify_pre_gt_sparse_selections(state_ids: np.ndarray, prediction: np.ndarray,
                                   selection_available: np.ndarray, excluded_states: set[str],
                                   binding: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the original ranked FORCED_K sequences using predictions only."""
    eligible = (np.asarray([state_id not in excluded_states for state_id in state_ids])
                & np.isfinite(prediction).all(axis=1) & selection_available[:, 0]
                & selection_available[:, 1:].any(axis=1))
    authority = binding["selection_identity_authority"]
    if int(eligible.sum()) != authority["eligible_states"]:
        raise IdentityDrift("pre-GT sparse eligible-state identity drift")
    ids = state_ids[eligible]
    masked = np.where(selection_available[eligible], prediction[eligible], np.inf)
    best_nonstop = masked[:, 1:].argmin(axis=1) + 1
    score = masked[:, 0] - masked[np.arange(len(ids)), best_nonstop]
    order = np.lexsort((ids, -score))
    observed = []
    for frozen in authority["FORCED_K"]:
        cap = max(1, math.ceil(frozen["budget"] * len(order)))
        chosen = order[:cap]
        pairs = [[str(ids[index]), ACTIONS[int(best_nonstop[index])]] for index in chosen]
        digest = sha(json.dumps(pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        row = {"budget": frozen["budget"], "K": cap, "used": len(chosen), "selection_sha256": digest}
        if row != frozen:
            raise IdentityDrift("pre-GT FORCED_K sparse selection identity drift")
        observed.append(row)
    return observed


def prediction_inputs_from_panel(panel: dict[str, Any], manifest: dict[str, Any],
                                 sparse_binding: dict[str, Any]) -> tuple[dict, dict]:
    state_ids = np.asarray([state.semantic_state_id for state in panel["states"]])
    index = {state_id: position for position, state_id in enumerate(state_ids)}
    selection_available = np.zeros((len(state_ids), 5), dtype=bool)
    for plan in panel["plans"].values():
        for row in plan["a0"]:
            selection_available[index[row["semantic_state_id"]], 0] = True
        for row in plan["actions"]:
            if row["action_code"] in ACTIONS:
                selection_available[index[row["semantic_state_id"]], ACTIONS.index(row["action_code"])] = row["feasible"]
    for state_id, action in panel["missing_actions"]:
        selection_available[index[state_id], ACTIONS.index(action)] = False
    predictions, selected = load_frozen_predictions(state_ids, manifest, panel["missing_states"], selection_available)
    excluded = panel["missing_states"] | {state_id for state_id, _ in panel["missing_actions"]}
    verify_pre_gt_sparse_selections(state_ids, predictions["R3_P"], selection_available, excluded, sparse_binding)
    return predictions, selected


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, sort_keys=True, separators=(",", ":"))
                               if isinstance(value, (dict, list)) else value)
                         for key, value in row.items()})
    return stream.getvalue().encode()


def _number(value: Any) -> str:
    return "null" if value is None else f"{value:.12g}" if isinstance(value, float) else str(value)


def verify_primary_sparse_results(result: dict[str, Any], binding: dict[str, Any]) -> None:
    """Require the original FORCED_K selections at the result-writing boundary."""
    e4 = result["e1_e5_registry"]["E4"]
    if e4["budget_semantics"] != binding["primary_semantics"]:
        raise IdentityDrift("E4 result uses secondary sparse semantics")
    rows = e4["metrics"]["sparse"]
    expected = binding["selection_identity_authority"]["FORCED_K"]
    if len(rows) != len(expected) or sorted(row["budget"] for row in rows) != binding["budgets"]:
        raise IdentityDrift("E4 primary sparse budget rows differ from binding")
    for frozen in expected:
        row = next(row for row in rows if row["budget"] == frozen["budget"])
        if (row["model"] != "R3_P" or row["budget_semantics"] != "FORCED_K"
                or row["cap"] != frozen["K"] or row["used"] != frozen["used"]
                or row["selection_sha256"] != frozen["selection_sha256"]):
            raise IdentityDrift("E4 FORCED_K selection identity differs from before-GT evidence")


def write_results(result: dict[str, Any], verification: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    """Write all numbers first; derive the prose report from those same records."""
    sparse_binding = load_sparse_binding()
    verify_primary_sparse_results(result, sparse_binding)
    receipt = read_json(FIRST_READ)
    registry = result["e1_e5_registry"]
    registry_artifact = {"schema_version": "rail3.tmlr-v9c-track-a-external-hypothesis-results.v1",
                         "status": result["evaluation_status"], "hypotheses": registry,
                         "COCO_GT_FIRST_READ_TIMESTAMP": receipt["COCO_GT_FIRST_READ_TIMESTAMP"],
                         "COCO_GT_READS": receipt["COCO_GT_READS"], "PERFORMANCE_RESULTS": len(result["core_rows"]),
                         "performance_result_count_unit": "core source-table rows",
                         "first_read_record": identity(FIRST_READ), "pre_read_record": identity(PRE_READ),
                         "target_inventory": identity(TARGET_INVENTORY),
                         "E4_contract": identity(E4), "E5_contract": identity(E5),
                         "E4_sparse_binding": identity(SPARSE_BINDING),
                         "E4_PRIMARY_SPARSE_SEMANTICS": sparse_binding["primary_semantics"],
                         "POSITIVE_GAIN_CAP_ROLE": sparse_binding["secondary_role"],
                         "TRACK_B_STATUS": "NO_GO", "TRACK_B_EVALUATED": False,
                         "TRACK_B_SCIENTIFIC_FAILURE": False, "E6": "N/A_NOT_EVALUATED",
                         "SECONDARY_UTILITY_LOSS_SENSITIVITY": "NOT_AUTHORIZED_NOT_RUN", "holds": result["holds"]}
    core_path = SOURCE / "track_a_external_core_results.csv"
    secure_create_or_verify(core_path, _csv_bytes(result["core_rows"]))
    create_result_json(SOURCE / "track_a_external_bootstrap_results.json", result["bootstrap_results"])
    create_result_json(SOURCE / "track_a_e1_e5_registry.json", registry_artifact)
    missingness = {**result["missingness_evaluation"],
                   "frozen_exact_missingness": read_json(SOURCE / "technical_missingness_report.json"),
                   "primary_denominator_rule": "Only undefined exact action cells and STOP-relative pairs excluded; unaffected actions and image groups retained",
                   "complete_oracle_denominator_rule": "Exclude the four exact trajectory-incomplete states; preserve panel membership and all image groups",
                   "scientific_outcome_imputed": False}
    create_result_json(SOURCE / "track_a_technical_missingness_evaluation.json", missingness)
    create_result_json(SOURCE / "track_a_external_qa.json", qa)
    provenance = {"schema_version": "rail3.tmlr-v9c-track-a-evaluation-provenance.v1",
                  "frozen_files": verification["files"], "reproducibility": result["reproducibility"],
                  "primary_metric_implementation": identity("src/rail3/evaluation/tmlr_v9c_metrics.py"),
                  "target_implementation": identity("src/rail3/evaluation/tmlr_v9c_coco_targets.py"),
                  "execution_implementation": identity("src/rail3/evaluation/tmlr_v9c_execution.py"),
                  "E4_sparse_binding": identity(SPARSE_BINDING),
                  "secondary_sparse_semantics": {"semantics": "POSITIVE_GAIN_CAP", "role": "SECONDARY_SENSITIVITY",
                                                  "determines_E4_SUPPORTED": False, "determines_E5": False},
                  "inherited_formula_sources": {
                      p: identity(p) for p in (
                          "src/rail3/analysis/tmlr_v6_phase2.py",
                          "configs/experiments/tmlr_v6_p0_phase2_semantic_addendum.json")},
                  "bootstrap_recomputed_twice": True, "COCO_GT_source_open_count": 1,
                  "primary_fidelity_missingness": "Current V9C action-level missingness applied to mathematically defined MAE cells and DRRE pairs",
                  "secondary_scope": "POSITIVE_GAIN_CAP is descriptive sparse sensitivity only; alternative target-loss sensitivity remains VOC FIT-only and was not run"}
    create_result_json(SOURCE / "track_a_external_evaluation_provenance.json", provenance)
    report = ["# TMLR V9C Track A COCO external results", "",
              f"Evaluation status: `{result['evaluation_status']}`. Every numerical statement below is generated from the machine-readable source tables and registry.", "",
              f"First COCO GT read: `{receipt['COCO_GT_FIRST_READ_TIMESTAMP']}`; source opens: **{receipt['COCO_GT_READS']}**.",
              "The frozen panel contains 1,000 images and all 20,000 image/class states. Track B remains NO_GO, not evaluated, and is not classified as a scientific failure.", "",
              "## Frozen hypotheses", "",
              "| Hypothesis | Contrast | Effect | 95% image-group CI | Status |",
              "|---|---|---:|---|---|"]
    for key in ("E1", "E2"):
        row = registry[key]
        report.append(f"| {key} | {row['metric']}: {row['left']} minus {row['right']} | {_number(row['difference'])} | [{_number(row['ci']['lower'])}, {_number(row['ci']['upper'])}] | {row['status']} |")
    report += ["", "E3 reports the inherited matched-local comparisons with R1_P on the left and RECT_P/UNION_P on the right, for both MAE and DRRE. The full paired CI must fit within ±5% of the right model's full-sample point metric; a zero reference requires exact equality. No overall E3 support rule is added, and E3 does not enter E5.", "",
               "| Pair | Metric | Difference | 95% CI | Practical-tie status |", "|---|---|---:|---|---|"]
    for row in registry["E3"]["comparisons"].values():
        report.append(f"| {row['left']} − {row['right']} | {row['metric']} | {_number(row['difference'])} | [{_number(row['ci']['lower'])}, {_number(row['ci']['upper'])}] | {row['status']} |")
    e4 = registry["E4"]
    metrics = e4["metrics"]
    intervention = next(row for row in result["core_rows"] if row["model"] == "R3_P" and row["row_type"] == "intervention")
    report += ["", "## R3_P intervention utility at lambda zero", "",
               f"Baseline: STOP. Normalized regret: **{_number(metrics['normalized_regret'])}**. Utility improvement, `1 − normalized_regret`, is **{_number(e4['utility_improvement']['point'])}**, with paired 95% CI **[{_number(metrics['utility_improvement_ci']['lower'])}, {_number(metrics['utility_improvement_ci']['upper'])}]**.", "",
               f"Predicted non-STOP: **{intervention['predicted_non_STOP_count']} / {intervention['defined_prediction_states']}** ({_number(intervention['predicted_non_STOP_fraction'])}). This diagnostic includes predictable states excluded from complete-oracle metrics.", "",
               f"E4 budget binding: `{e4['budget_semantics']}`, supplement SHA-256 `{SPARSE_BINDING_SHA256}`. FORCED_K supplies the three primary sparse rows. POSITIVE_GAIN_CAP is reported only as SECONDARY_SENSITIVITY and cannot determine E4 or E5.", "",
               "| Budget | Realized gain | Oracle gain denominator | Gain capture |", "|---|---:|---:|---:|"]
    for row in metrics["sparse"]:
        report.append(f"| {row['budget']:.0%} | {_number(row['realized_gain'])} | {_number(row['oracle_gain_denominator'])} | {_number(row['gain_capture'])} |")
    report += ["", f"False STOP: {intervention['false_stop_states']} (rate {_number(intervention['false_stop_rate'])}); false intervention: {intervention['false_intervention_states']} (rate {_number(intervention['false_intervention_rate'])}); negative intervention: {intervention['negative_intervention_states']} (rate {_number(intervention['negative_intervention_rate'])}). False-event rates use all complete-oracle evaluable states; negative-intervention rate uses selected non-STOP states.", "",
               f"E4 predicates: `{json.dumps(e4.get('predicates'), sort_keys=True)}`. E4 status: **{e4['status']}**. Descriptive clearly harmful: `{e4.get('E4_CLEARLY_HARMFUL')}`; this flag does not enter E5.", "",
               f"E5 / final classification: **{registry['E5']['status']}**.",
               registry["E5"].get("interpretation", "Required inferential input is undefined; classification remains null."), "",
               "## Denominators and QA", "",
               f"MAE states: {missingness['primary_mae_evaluable_states']}; defined action cells: {missingness['primary_mae_defined_state_actions']}. DRRE states: {missingness['primary_drre_evaluable_states']}; defined STOP-relative pairs: {missingness['primary_drre_defined_stop_pairs']}. Complete-oracle states: {missingness['complete_oracle_evaluable_states']}. Retained image groups: {missingness['retained_image_groups']}.", "",
               "The one unavailable trajectory and three A4 failures remain technical missingness. No STOP, class-absence, empty-mask or residual value is imputed. Per-action primary metrics retain unaffected actions; complete-oracle metrics exclude the four exact affected states. The panel and image groups are retained.", "",
               "Bootstrap uses the same sorted image-group population, paired PCG64 draws, seed 13, 2,000 replicates and linear 2.5%/97.5% quantiles. The computation was repeated from the saved inputs, and all outputs and bootstrap array hashes matched exactly. Undefined rates remain null; no nonfinite number is serialized.", "",
               "Prediction files, all 18 checkpoints, frozen features/scalers, trajectory locks/caches, and E4/E5 contracts passed post-evaluation identity checks. No retraining, prediction change, threshold tuning, panel change, or missingness change occurred.", "",
               "Alternative target-loss sensitivity was not run: its inherited permission is VOC FIT-only. The separately authorized POSITIVE_GAIN_CAP sparse sensitivity is descriptive only. No Track B recovery, new primary experiment, manuscript rewrite, merge, or push is part of this evaluation.", "",
               "Machine-readable records are under `artifacts/paper/source_data/tmlr_v9c/`: target inventory, core results, bootstrap results, E1–E5 registry, missingness evaluation, QA, first-read records, and provenance. Reference masks and residual arrays remain untracked under `artifacts/targets/tmlr-v9c/track-a/`.", ""]
    report_path = Path("docs/experiments/TMLR_V9C_TRACK_A_COCO_EXTERNAL_RESULTS.md")
    secure_create_or_verify(report_path, "\n".join(report).encode())
    # Append only: historical ledger contents are not inspected or interpreted.
    ledger = Path("docs/paper/RESULTS_LEDGER.md")
    addition = ("\n\n## TMLR V9C Track A COCO external evaluation\n\n"
                f"First GT read: `{receipt['COCO_GT_FIRST_READ_TIMESTAMP']}`; frozen panel: 1,000 images × 20 classes. "
                f"E1: `{registry['E1']['status']}`; E2: `{registry['E2']['status']}`; "
                f"E3: `{registry['E3']['status']}`; E4: `{e4['status']}`; E5: `{registry['E5']['status']}`. "
                "Track B remains NO_GO / NOT_EVALUATED / NOT_SCIENTIFIC_FAILURE.\n\n"
                "Source: [COCO external results](../experiments/TMLR_V9C_TRACK_A_COCO_EXTERNAL_RESULTS.md); "
                "`artifacts/paper/source_data/tmlr_v9c/track_a_e1_e5_registry.json`. "
                "No post-GT model, prediction, threshold, panel, or missingness change.\n").encode()
    intent_path = SOURCE / "track_a_results_ledger_append_intent.json"
    if intent_path.exists():
        intent = read_json(intent_path)
        old_ledger_identity = intent["preserved_prefix"]
        if intent["appended_sha256"] != sha(addition):
            raise IdentityDrift("ledger append content changed")
    else:
        old_ledger_identity = identity(ledger)
        create_json(intent_path, {"preserved_prefix": old_ledger_identity, "appended_sha256": sha(addition)})
    raw_ledger = secure_read_bytes(ledger)
    prefix_size = old_ledger_identity["bytes"]
    if sha(raw_ledger[:prefix_size]) != old_ledger_identity["sha256"]:
        raise IdentityDrift("historical ledger prefix changed")
    existing_addition = raw_ledger[prefix_size:]
    if not addition.startswith(existing_addition):
        raise IdentityDrift("unexpected concurrent ledger addition")
    remaining = addition[len(existing_addition):]
    if remaining:
        fd = os.open(ledger, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(remaining)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
    create_result_json(SOURCE / "track_a_results_ledger_append_audit.json", {
        "preserved_prefix": old_ledger_identity, "appended_bytes": len(addition),
        "appended_sha256": sha(addition), "new_identity": identity(ledger),
        "historical_ledger_values_inspected": False})
    return registry_artifact


def execute(repo: Path, budget_semantics: str | None = None, *, targets_already_saved: bool = False) -> dict[str, Any]:
    from rail3.evaluation.tmlr_v9c_metrics import evaluate_metrics

    sparse_binding = load_sparse_binding()
    if budget_semantics is not None and budget_semantics != sparse_binding["primary_semantics"]:
        raise ValueError("E4 primary sparse binding is FORCED_K; POSITIVE_GAIN_CAP is secondary only")
    budget_semantics = sparse_binding["primary_semantics"]
    verification = verify_frozen_files(repo)
    panel = load_locked_panel(repo)
    predictions, selected = prediction_inputs_from_panel(panel, verification["prediction_manifest"], sparse_binding)
    expected_state_ids = np.asarray([state.semantic_state_id for state in panel["states"]])
    expected_groups = np.asarray([state.image.canonical_image_id for state in panel["states"]])
    if targets_already_saved:
        before = read_json(PRE_READ)
        receipt = read_json(FIRST_READ)
        if receipt["status"] != "READ_COMPLETE" or receipt["COCO_GT_READS"] != 1 or not receipt["COCO_GT_FIRST_READ_TIMESTAMP"]:
            raise IdentityDrift("first-read receipt incomplete")
        _verify(PRE_READ, receipt["pre_read_record"])
        if before["frozen_files"] != verification["files"]:
            raise IdentityDrift("frozen files differ from before-read evidence")
        if before["E4_sparse_binding"] != identity(SPARSE_BINDING):
            raise IdentityDrift("E4 sparse binding differs from before-read evidence")
        for path, binding in before["evaluation_source"].items():
            _verify(path, binding)
        if before["environment"] != evaluation_environment():
            raise IdentityDrift("evaluation environment differs from before-read evidence")
        if before["E4_budget_semantics"] != budget_semantics:
            raise ValueError("cannot change E4 sparse binding after GT")
        if not TARGET_INVENTORY.exists():
            snapshot = receipt["untracked_recovery_snapshot"]
            _verify(snapshot["path"], snapshot)
            if snapshot["sha256"] != receipt["sha256"] or snapshot["bytes"] != receipt["bytes"]:
                raise IdentityDrift("first-read recovery snapshot identity drift")
            construct_targets(secure_read_bytes(Path(snapshot["path"])), panel)
        target_inventory = read_json(TARGET_INVENTORY)
        _verify(FIRST_READ, target_inventory["first_read_receipt"])
        expected_annotation = {key: receipt[key] for key in ("bytes", "sha256")} | {"path": ANNOTATIONS.as_posix()}
        if target_inventory["annotation_source"] != expected_annotation:
            raise IdentityDrift("target inventory annotation identity drift")
        for binding in target_inventory["reference_union_artifacts"]:
            _verify(binding["path"], binding)
        _verify(target_inventory["target_arrays"]["path"], target_inventory["target_arrays"])
        with np.load(BytesIO(secure_read_bytes(Path(target_inventory["target_arrays"]["path"]))), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    else:
        before = {"schema_version": "rail3.tmlr-v9c-before-first-gt-read.v1",
                  "PREDICTIONS_FROZEN_BEFORE_GT": True, "UTC_timestamp": utc_now(),
                  "current_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  "prediction_lock": identity(PREDICTION_LOCK), "trajectory_lock": identity(TRAJECTORY_LOCK),
                  "E4_contract": identity(E4), "E5_contract": identity(E5),
                  "E4_sparse_binding": identity(SPARSE_BINDING),
                  "PRE_GT_FORCED_K_SELECTION_IDENTITIES_VERIFIED": True,
                  "COCO_GT_FIRST_READ_TIMESTAMP": None, "COCO_GT_READS": 0, "PERFORMANCE_RESULTS": 0,
                  "E4_budget_semantics": budget_semantics,
                  "checkpoint_count_verified": 18, "frozen_files": verification["files"],
                  "environment": evaluation_environment(),
                  "cache_aggregates": panel["cache_aggregates"], "plan_aggregate_sha256": panel["plan_sha256"],
                  "evaluation_source": {p: identity(p) for p in (
                      "src/rail3/evaluation/tmlr_v9c_execution.py",
                      "src/rail3/evaluation/tmlr_v9c_metrics.py",
                      "src/rail3/evaluation/tmlr_v9c_coco_targets.py",
                      "scripts/run_tmlr_v9c_track_a_evaluation.py")}}
        if ANNOTATIONS.resolve().is_relative_to(repo.resolve()) is False:
            raise ValueError("annotation source escapes repository")
        raw = read_annotations_once(ANNOTATIONS, PRE_READ, FIRST_READ, before,
                                    recovery_snapshot=TARGET_ROOT / "first_read_source_snapshot.json")
        arrays = construct_targets(raw, panel)
    if not np.array_equal(arrays["state_ids"], expected_state_ids) or not np.array_equal(arrays["image_group_ids"], expected_groups):
        raise IdentityDrift("target state/image alignment differs from frozen panel")
    arguments = (predictions, arrays["targets"], arrays["plan_feasible"], selected,
                 arrays["state_ids"], arrays["image_group_ids"], read_json(E4), read_json(E5))
    result = evaluate_metrics(*arguments, budget_semantics=budget_semantics)
    repeated = evaluate_metrics(*arguments, budget_semantics=budget_semantics)
    if canonical_json_bytes(result) != canonical_json_bytes(repeated):
        raise RuntimeError("bootstrap or metric reproducibility failed")
    after = verify_frozen_files(repo)
    if after != verification:
        raise IdentityDrift("post-GT frozen identity drift")
    for binding in panel["cache_bindings"].values():
        _verify(binding["path"], binding)
    missing = result["missingness_evaluation"]
    if missing["retained_image_groups"] != 1000 or missing["panel_states"] != 20000 or missing["complete_oracle_evaluable_states"] != 19996:
        raise ValueError("final panel/group/oracle population differs from frozen grid")
    qa = {"status": "PASS", "PREDICTION_LOCK_UNCHANGED": True,
          "PREDICTION_FILES_UNCHANGED": True, "CHECKPOINT_FILES_UNCHANGED": True,
          "TRAJECTORY_LOCK_UNCHANGED": True, "TRAJECTORY_CACHE_FILES_UNCHANGED": True,
          "E4_CONTRACT_UNCHANGED": True, "E5_CONTRACT_UNCHANGED": True,
          "E4_SPARSE_BINDING_UNCHANGED": True,
          "POST_GT_RETRAINING": False, "POST_GT_PREDICTION_CHANGE": False,
          "POST_GT_THRESHOLD_TUNING": False, "POST_GT_PANEL_CHANGE": False,
          "POST_GT_MISSINGNESS_CHANGE": False, "BOOTSTRAP_REPRODUCIBLE": True,
          "unexpected_missing": 0, "duplicate_state_ids": 0, "nonfinite_serialized_metrics": 0,
          "first_read_timestamp": read_json(FIRST_READ)["COCO_GT_FIRST_READ_TIMESTAMP"],
          "frozen_files_sha256": sha(canonical_json_bytes(verification["files"]))}
    return write_results(result, verification, qa)
