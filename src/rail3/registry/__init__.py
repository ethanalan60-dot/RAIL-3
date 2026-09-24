"""Fail-closed dataset authorization registry."""

from rail3.registry.datasets import DatasetAuthorizationError, DatasetGateDecision, DatasetRegistry, evaluate_dataset_gate

__all__ = ["DatasetAuthorizationError", "DatasetGateDecision", "DatasetRegistry", "evaluate_dataset_gate"]
