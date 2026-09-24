"""Deterministic five-fold group manifests for nested M06-C scales."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable, Mapping

from rail3.contracts import canonical_json_bytes
from rail3.contracts.m06_ids import seeded_digest


FOLD_IDS = tuple(f"fold_{index}" for index in range(5))


def build_scale_folds(
    groups: Iterable[Mapping[str, Any]], *, scale: str, seed: int,
    heldout_group_ids: Iterable[str], protocol_sha256: str,
) -> dict[str, Any]:
    selected = [dict(item) for item in groups]
    count = len(selected)
    if count not in {100, 250, 500, 1000} or count % 5:
        raise ValueError("M06 folds require 100, 250, 500, or 1000 groups")
    ids = [str(item["image_group_id"]) for item in selected]
    if len(set(ids)) != count:
        raise ValueError("M06-C fold groups are not unique")
    heldout = {str(value) for value in heldout_group_ids}
    if set(ids) & heldout:
        raise ValueError("M06-C fold groups overlap held-out groups")
    capacity = count // 5
    class_totals: Counter[int] = Counter()
    for group in selected:
        classes = sorted({int(value) for value in group["positive_class_ids"]})
        group["positive_class_ids"] = classes
        class_totals.update(classes)

    def order_key(group: Mapping[str, Any]) -> tuple[Any, ...]:
        classes = list(group["positive_class_ids"])
        rarest = min((class_totals[value] for value in classes), default=10**9)
        return rarest, -len(classes), seeded_digest(seed, str(group["image_group_id"]))

    fold_groups = {fold: [] for fold in FOLD_IDS}
    fold_classes = {fold: Counter() for fold in FOLD_IDS}
    fold_positive: Counter[str] = Counter()
    assignments = []
    for group in sorted(selected, key=order_key):
        classes = list(group["positive_class_ids"])
        candidates = []
        for fold in FOLD_IDS:
            if len(fold_groups[fold]) >= capacity:
                continue
            delta = 0.0
            for class_id in classes:
                target = class_totals[class_id] / len(FOLD_IDS)
                before = fold_classes[fold][class_id] - target
                after = fold_classes[fold][class_id] + 1 - target
                delta += after * after - before * before
            candidates.append((
                delta, fold_positive[fold] + len(classes), len(fold_groups[fold]),
                seeded_digest(seed, str(group["image_group_id"]), fold), fold,
            ))
        fold = min(candidates)[-1]
        fold_groups[fold].append(group)
        fold_classes[fold].update(classes)
        fold_positive[fold] += len(classes)
        assignments.append({
            "image_group_id": str(group["image_group_id"]),
            "asset_id": str(group["asset_id"]), "sample_id": str(group["sample_id"]),
            "fold_id": fold, "positive_class_ids": classes,
        })
    assignments.sort(key=lambda item: item["image_group_id"])
    folds = {
        fold: {
            "group_count": len(fold_groups[fold]),
            "positive_state_count": fold_positive[fold],
            "class_support": {str(class_id): fold_classes[fold][class_id] for class_id in range(1, 21)},
            "classes_represented": sum(fold_classes[fold][value] > 0 for value in range(1, 21)),
        }
        for fold in FOLD_IDS
    }
    checks = {
        "exactly_five_folds": len(folds) == 5,
        "group_count_exact": len(assignments) == count,
        "each_group_once": len({item["image_group_id"] for item in assignments}) == count,
        "equal_groups_per_fold": all(value["group_count"] == capacity for value in folds.values()),
        "no_heldout_group": not bool(set(ids) & heldout),
    }
    semantic = {
        "schema_version": "rail3.m06c.fold-manifest.v1", "scale": scale,
        "protocol_sha256": protocol_sha256, "seed": seed,
        "algorithm": "deterministic capacity-constrained greedy multilabel balancing",
        "fold_count": 5, "group_count": count, "groups": assignments,
        "folds": folds, "checks": checks,
    }
    return {**semantic, "semantic_sha256": hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()}
