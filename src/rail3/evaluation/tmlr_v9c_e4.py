"""Evaluate the frozen E4 contract against already computed metrics.

This module performs no I/O or statistical computation. Invalid execution inputs
raise ``ValueError``; valid inputs produce exactly one of the two E4 decisions.
"""

from collections.abc import Mapping
import math
import operator
from typing import Any


_OPERATORS = {"<": operator.lt, ">": operator.gt}
_SCALAR_PREDICATES = {
    "A": ("AGGREGATE_BETTER_THAN_STOP", "normalized_regret"),
    "B": ("GROUP_ROBUST_IMPROVEMENT", "utility_improvement_ci.lower"),
    "C": ("NONTRIVIAL_INTERVENTION", "predicted_non_STOP_count"),
}
_SPARSE_METRICS = {"realized_gain", "oracle_gain_denominator", "gain_capture"}


def _mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _field(value: Mapping, key: str, label: str) -> Any:
    if key not in value:
        raise ValueError(f"{label}.{key} is required")
    return value[key]


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _budgets(value: Any, label: str) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three unique budgets")
    budgets = [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(budgets)) != 3:
        raise ValueError(f"{label} must contain exactly three unique budgets")
    return budgets


def _clause(value: Any, label: str) -> tuple[str, str, int | float]:
    clause = _mapping(value, label)
    metric = _field(clause, "metric", label)
    comparison = _field(clause, "operator", label)
    if not isinstance(metric, str):
        raise ValueError(f"{label}.metric must be a string")
    if not isinstance(comparison, str) or comparison not in _OPERATORS:
        raise ValueError(f"{label}.operator must be '<' or '>'")
    threshold = _number(_field(clause, "value", label), f"{label}.value")
    return metric, comparison, threshold


def _clauses(value: Any, expected: set[str], label: str) -> list[tuple[str, str, int | float]]:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"{label} must contain exactly the required metric clauses")
    clauses = [_clause(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if {clause[0] for clause in clauses} != expected:
        raise ValueError(f"{label} must contain exactly the required metric clauses")
    return clauses


def _passes(clause: tuple[str, str, int | float], values: Mapping) -> bool:
    metric, comparison, threshold = clause
    value = values[metric]
    # An explicitly undefined capture is not replaced with a numeric value.
    return value is not None and _OPERATORS[comparison](value, threshold)


def evaluate_e4(contract: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return E4 predicates, scientific status, harmful description, and E5 U.

    Comparators and thresholds come from ``contract``. All three configured
    budgets must have unique metric rows. A zero oracle denominator may carry
    ``gain_capture=None``; every other required numeric value must be finite.
    Neither input is mutated.
    """
    contract = _mapping(contract, "contract")
    metrics = _mapping(metrics, "metrics")
    model = _mapping(_field(contract, "model", "contract"), "contract.model")
    model_lambda = _number(_field(model, "lambda", "contract.model"), "contract.model.lambda")
    if model.get("family") != "R3_P" or model.get("baseline") != "STOP" or model_lambda != 0:
        raise ValueError("contract.model must identify R3_P versus STOP at lambda=0")

    budgets = _budgets(_field(contract, "sparse_budgets", "contract"), "contract.sparse_budgets")
    predicates = _mapping(_field(contract, "predicates", "contract"), "contract.predicates")
    if set(predicates) != {"A", "B", "C", "D"}:
        raise ValueError("contract.predicates must contain exactly A, B, C, and D")
    scalar_clauses = {}
    for key, (name, metric) in _SCALAR_PREDICATES.items():
        label = f"contract.predicates.{key}"
        predicate = _mapping(predicates[key], label)
        clause = _clause(predicate, label)
        if predicate.get("name") != name or clause[0] != metric:
            raise ValueError(f"{label} must identify its prescribed predicate and metric")
        scalar_clauses[key] = clause

    sparse_predicate = _mapping(predicates["D"], "contract.predicates.D")
    if sparse_predicate.get("name") != "SPARSE_UTILITY_POSITIVE":
        raise ValueError("contract.predicates.D must identify SPARSE_UTILITY_POSITIVE")
    sparse_budgets = _budgets(
        _field(sparse_predicate, "exists_budget_in", "contract.predicates.D"),
        "contract.predicates.D.exists_budget_in",
    )
    if set(sparse_budgets) != set(budgets):
        raise ValueError("predicate D budgets must match contract.sparse_budgets")
    sparse_clauses = _clauses(
        _field(sparse_predicate, "all", "contract.predicates.D"),
        _SPARSE_METRICS,
        "contract.predicates.D.all",
    )

    decision = _mapping(_field(contract, "decision", "contract"), "contract.decision")
    required_predicates = _field(decision, "all", "contract.decision")
    if (
        not isinstance(required_predicates, list)
        or len(required_predicates) != 4
        or any(not isinstance(key, str) for key in required_predicates)
        or set(required_predicates) != set(predicates)
    ):
        raise ValueError("contract.decision.all must contain A, B, C, and D exactly once")
    if decision.get("supported") != "E4_SUPPORTED" or decision.get("otherwise") != "E4_NOT_SUPPORTED":
        raise ValueError("contract.decision must provide the prescribed E4 status labels")

    harmful = _mapping(
        _field(contract, "descriptive_harmful", "contract"), "contract.descriptive_harmful"
    )
    if harmful.get("supplies_E5") is not False:
        raise ValueError("contract.descriptive_harmful.supplies_E5 must be false")
    harmful_clauses = _clauses(
        _field(harmful, "all", "contract.descriptive_harmful"),
        {"normalized_regret", "utility_improvement_ci.upper"},
        "contract.descriptive_harmful.all",
    )

    ci = _mapping(_field(metrics, "utility_improvement_ci", "metrics"), "metrics.utility_improvement_ci")
    lower = _number(_field(ci, "lower", "metrics.utility_improvement_ci"), "metrics.utility_improvement_ci.lower")
    upper = _number(_field(ci, "upper", "metrics.utility_improvement_ci"), "metrics.utility_improvement_ci.upper")
    if lower > upper:
        raise ValueError("metrics.utility_improvement_ci.lower must not exceed upper")
    count = _field(metrics, "predicted_non_STOP_count", "metrics")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("metrics.predicted_non_STOP_count must be a nonnegative integer")
    values = {
        "normalized_regret": _number(
            _field(metrics, "normalized_regret", "metrics"), "metrics.normalized_regret"
        ),
        "utility_improvement_ci.lower": lower,
        "utility_improvement_ci.upper": upper,
        "predicted_non_STOP_count": count,
    }

    sparse = _field(metrics, "sparse", "metrics")
    if not isinstance(sparse, list) or len(sparse) != len(budgets):
        raise ValueError("metrics.sparse must contain exactly one row for each configured budget")
    rows = {}
    for index, item in enumerate(sparse):
        label = f"metrics.sparse[{index}]"
        row = _mapping(item, label)
        budget = _number(_field(row, "budget", label), f"{label}.budget")
        if budget not in budgets or budget in rows:
            raise ValueError("metrics.sparse must contain exactly one row for each configured budget")
        denominator = _number(_field(row, "oracle_gain_denominator", label), f"{label}.oracle_gain_denominator")
        if denominator < 0:
            raise ValueError(f"{label}.oracle_gain_denominator must be nonnegative")
        capture = _field(row, "gain_capture", label)
        if capture is not None or denominator != 0:
            capture = _number(capture, f"{label}.gain_capture")
        rows[budget] = {
            "realized_gain": _number(_field(row, "realized_gain", label), f"{label}.realized_gain"),
            "oracle_gain_denominator": denominator,
            "gain_capture": capture,
        }

    flags = {key: _passes(clause, values) for key, clause in scalar_clauses.items()}
    flags["D"] = any(
        all(_passes(clause, rows[budget]) for clause in sparse_clauses)
        for budget in sparse_budgets
    )
    supported = all(flags[key] for key in required_predicates)
    return {
        "predicates": flags,
        "E4_SUPPORTED": supported,
        "status": decision["supported"] if supported else decision["otherwise"],
        "E4_CLEARLY_HARMFUL": all(_passes(clause, values) for clause in harmful_clauses),
        "E5_U": supported,
    }
