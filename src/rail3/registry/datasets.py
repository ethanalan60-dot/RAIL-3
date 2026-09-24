"""Authorization gates for dataset activation and licensed uses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rail3.contracts.records import AuthorizationRecord, AuthorizationStatus, DatasetSpec, RecordStatus, UseTag


def _dataset_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_SACO_GOLD_KEYS = {"sacogold", "saco", "saco/gold"}


@dataclass(frozen=True)
class DatasetGateDecision:
    allowed: bool
    reason_code: str
    message: str


class DatasetAuthorizationError(RuntimeError):
    def __init__(self, decision: DatasetGateDecision) -> None:
        super().__init__(decision.message)
        self.reason_code = decision.reason_code
        self.decision = decision


def evaluate_dataset_gate(dataset: DatasetSpec, authorization: AuthorizationRecord) -> DatasetGateDecision:
    normalized = _dataset_key(dataset.dataset_id + dataset.canonical_name)
    if "sacogold" in normalized and authorization.status is not AuthorizationStatus.APPROVED:
        return DatasetGateDecision(False, "SACO_GOLD_IDENTITY_AND_LICENSE_BLOCKED", "SA-Co/Gold remains blocked until identity and authorization are approved")
    if dataset.status is not RecordStatus.VALID:
        return DatasetGateDecision(False, dataset.reason_code or "DATASET_NOT_VALID", "dataset record is not valid")
    if dataset.enabled and authorization.status is not AuthorizationStatus.APPROVED:
        return DatasetGateDecision(False, "AUTHORIZATION_NOT_APPROVED", "enabled dataset requires approved authorization")
    missing = set(dataset.use_tags) - set(authorization.allowed_uses)
    if dataset.enabled and missing:
        return DatasetGateDecision(False, "USE_NOT_AUTHORIZED", "dataset requests uses absent from the authorization record")
    if not dataset.enabled:
        return DatasetGateDecision(False, "DATASET_DISABLED", "dataset is disabled")
    return DatasetGateDecision(True, "AUTHORIZED", "dataset and requested uses are approved")


class DatasetRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[DatasetSpec, AuthorizationRecord]] = {}

    def register(self, dataset: DatasetSpec, authorization: AuthorizationRecord) -> None:
        if dataset.dataset_id in self._entries:
            raise ValueError(f"dataset already registered: {dataset.dataset_id}")
        decision = evaluate_dataset_gate(dataset, authorization)
        if not decision.allowed:
            raise DatasetAuthorizationError(decision)
        self._entries[dataset.dataset_id] = (dataset, authorization)

    def require_use(self, dataset_id: str, use: UseTag) -> DatasetSpec:
        try:
            dataset, authorization = self._entries[dataset_id]
        except KeyError as exc:
            raise DatasetAuthorizationError(DatasetGateDecision(False, "DATASET_NOT_REGISTERED", "dataset is not registered")) from exc
        if use not in dataset.use_tags or use not in authorization.allowed_uses:
            raise DatasetAuthorizationError(DatasetGateDecision(False, "USE_NOT_AUTHORIZED", "requested dataset use is not authorized"))
        return dataset
