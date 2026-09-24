"""Synthetic CPU contract tests; not a reproduction of scientific results.

All arrays, identifiers and metric values below are constructed in this file.
The sole file read is the shipped E4 decision-contract JSON. Its referenced
paths are never followed. No dataset, mask asset, checkpoint, prediction,
result table, model, SAM backend or bootstrap implementation is loaded.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from rail3.contracts.serialization import canonical_json_bytes, stable_id
from rail3.evaluation.tmlr_v9c_e4 import evaluate_e4
from rail3.regions.atomic import CandidateMaskInput, build_atomic_partition


class CanonicalSerializationTests(unittest.TestCase):
    def test_nested_mapping_order_has_one_canonical_encoding(self) -> None:
        left = {"z": [3, {"b": True, "a": None}], "a": "synthetic"}
        right = {"a": "synthetic", "z": [3, {"a": None, "b": True}]}
        expected = b'{"a":"synthetic","z":[3,{"a":null,"b":true}]}'
        self.assertEqual(canonical_json_bytes(left), expected)
        self.assertEqual(canonical_json_bytes(right), expected)
        self.assertNotEqual(canonical_json_bytes([1, 2]), canonical_json_bytes([2, 1]))

    def test_nonfinite_values_and_nonstring_keys_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json_bytes({"nested": [{"value": value}]})
        with self.assertRaises(TypeError):
            canonical_json_bytes({1: "ambiguous key"})

    def test_stable_ids_keep_namespace_and_canonical_payload_identity(self) -> None:
        payload = {"b": 2, "a": 1}
        first = stable_id("synthetic_asset", payload)
        reordered = stable_id("synthetic_asset", {"a": 1, "b": 2})
        other_namespace = stable_id("synthetic_state", payload)
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, other_namespace)
        self.assertRegex(first, r"^synthetic_asset_[0-9a-f]{64}$")
        self.assertNotEqual(first, stable_id("synthetic_asset", {"a": 1, "b": 3}))
        for invalid in ("", "Uppercase", "not-a-namespace", "1state"):
            with self.subTest(prefix=invalid), self.assertRaises(ValueError):
                stable_id(invalid, payload)


def _candidate(name: str, mask: np.ndarray) -> CandidateMaskInput:
    return CandidateMaskInput.create(
        candidate_id=f"synthetic-{name}",
        action_id=f"synthetic-action-{name}",
        action_code="A0",
        class_id=1,
        prompt_source="synthetic-unit-test",
        mask=mask,
    )


def _partition(candidates):
    return build_atomic_partition(
        asset_id="synthetic-asset-only",
        image_sha256="a" * 64,  # Artificial identifier, not a file digest.
        candidates=candidates,
    )


class SyntheticAtomicPartitionTests(unittest.TestCase):
    def test_disconnected_equal_signatures_remain_separate_cells(self) -> None:
        mask = np.zeros((5, 7), dtype=bool)
        mask[1, 1] = True
        mask[3, 5] = True
        partition = _partition((_candidate("two-islands", mask),))
        left = int(partition.label_map[1, 1])
        right = int(partition.label_map[3, 5])
        self.assertNotEqual(left, right)
        self.assertEqual(partition.atoms[left]["member_candidate_ids"],
                         partition.atoms[right]["member_candidate_ids"])
        self.assertEqual(partition.atoms[left]["area"], 1)
        self.assertEqual(partition.atoms[right]["area"], 1)
        self.assertTrue(any(atom["all_zero_background_signature"]
                            for atom in partition.atoms))

    def test_candidate_order_preserves_labels_and_cell_identity(self) -> None:
        left = np.zeros((5, 6), dtype=bool)
        right = np.zeros_like(left)
        left[1:4, 1:4] = True
        right[2:5, 3:5] = True
        candidates = (_candidate("left", left), _candidate("right", right))
        forward = _partition(candidates)
        backward = _partition(tuple(reversed(candidates)))
        np.testing.assert_array_equal(forward.label_map, backward.label_map)
        self.assertEqual(forward.atoms, backward.atoms)
        self.assertEqual(forward.candidate_to_atoms, backward.candidate_to_atoms)

    def test_pixel_coverage_and_candidate_reconstruction_are_lossless(self) -> None:
        ring = np.zeros((6, 7), dtype=bool)
        bar = np.zeros_like(ring)
        ring[1:5, 1:6] = True
        ring[2:4, 2:5] = False
        bar[:, 3] = True
        candidates = (_candidate("ring", ring), _candidate("bar", bar))
        partition = _partition(candidates)
        coverage = np.zeros(ring.shape, dtype=np.int8)
        reconstructed = {candidate.candidate_id: np.zeros_like(ring)
                         for candidate in candidates}
        for index, atom in enumerate(partition.atoms):
            support = np.zeros_like(ring)
            for y, x0, x1 in atom["runs"]:
                support[y, x0:x1] = True
            coverage += support.astype(np.int8)
            self.assertEqual(int(support.sum()), atom["area"])
            self.assertTrue(np.all(partition.label_map[support] == index))
            for candidate_id in atom["member_candidate_ids"]:
                reconstructed[candidate_id] |= support
        np.testing.assert_array_equal(coverage, np.ones(ring.shape, dtype=np.int8))
        for candidate in candidates:
            np.testing.assert_array_equal(reconstructed[candidate.candidate_id],
                                          candidate.mask)
        self.assertEqual(partition.reconstruction_error_pixels, 0)


class SyntheticE4RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        contract_path = root / "configs/experiments/tmlr_v9c_track_a_e4_decision_contract.json"
        if contract_path.is_symlink() or not contract_path.resolve().is_relative_to(root):
            raise RuntimeError("E4 synthetic test requires the local shipped contract")
        cls.contract = json.loads(contract_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        # Deliberately artificial values; never read from experiment outputs.
        # Only the last budget has a positive gain. D requires at least one.
        budgets = self.contract["sparse_budgets"]
        self.metrics = {
            "normalized_regret": 0.6,
            "utility_improvement_ci": {"lower": 0.1, "upper": 0.5},
            "predicted_non_STOP_count": 3,
            "sparse": [
                {"budget": budgets[0], "realized_gain": -1.0,
                 "oracle_gain_denominator": 4.0, "gain_capture": -0.25},
                {"budget": budgets[1], "realized_gain": 0.0,
                 "oracle_gain_denominator": 4.0, "gain_capture": 0.0},
                {"budget": budgets[2], "realized_gain": 1.0,
                 "oracle_gain_denominator": 4.0, "gain_capture": 0.25},
            ],
        }

    def test_one_positive_budget_suffices_when_all_other_conditions_hold(self) -> None:
        before_metrics = copy.deepcopy(self.metrics)
        before_contract = copy.deepcopy(self.contract)
        decision = evaluate_e4(self.contract, self.metrics)
        self.assertEqual(decision["predicates"], dict.fromkeys("ABCD", True))
        self.assertTrue(decision["E4_SUPPORTED"])
        self.assertEqual(decision["status"], "E4_SUPPORTED")
        self.assertTrue(decision["E5_U"])
        self.assertFalse(decision["E4_CLEARLY_HARMFUL"])
        self.assertEqual(self.metrics, before_metrics)
        self.assertEqual(self.contract, before_contract)

    def test_ci_lower_bound_is_strictly_positive_without_tolerance(self) -> None:
        for lower, expected in ((-0.1, False), (0.0, False), (1e-12, True)):
            with self.subTest(lower=lower):
                self.metrics["utility_improvement_ci"]["lower"] = lower
                decision = evaluate_e4(self.contract, self.metrics)
                self.assertEqual(decision["predicates"]["B"], expected)
                self.assertEqual(decision["E4_SUPPORTED"], expected)

    def test_each_required_condition_can_prevent_support(self) -> None:
        for condition in "ABCD":
            with self.subTest(condition=condition):
                metrics = copy.deepcopy(self.metrics)
                if condition == "A":
                    metrics["normalized_regret"] = 1.0
                elif condition == "B":
                    metrics["utility_improvement_ci"]["lower"] = 0.0
                elif condition == "C":
                    metrics["predicted_non_STOP_count"] = 0
                else:
                    metrics["sparse"][-1]["realized_gain"] = 0.0
                    metrics["sparse"][-1]["gain_capture"] = 0.0
                decision = evaluate_e4(self.contract, metrics)
                self.assertEqual(decision["predicates"],
                                 {name: name != condition for name in "ABCD"})
                self.assertFalse(decision["E4_SUPPORTED"])
                self.assertFalse(decision["E5_U"])
                self.assertEqual(decision["status"], "E4_NOT_SUPPORTED")

    def test_undefined_zero_denominator_cannot_supply_positive_sparse_utility(self) -> None:
        self.metrics["sparse"][-1].update(
            realized_gain=0.0, oracle_gain_denominator=0.0, gain_capture=None,
        )
        decision = evaluate_e4(self.contract, self.metrics)
        self.assertFalse(decision["predicates"]["D"])
        self.assertFalse(decision["E4_SUPPORTED"])
        self.assertIsNone(self.metrics["sparse"][-1]["gain_capture"])

    def test_missing_or_duplicate_budget_rows_and_nonfinite_inputs_are_rejected(self) -> None:
        missing = copy.deepcopy(self.metrics)
        missing["sparse"].pop()
        duplicate = copy.deepcopy(self.metrics)
        duplicate["sparse"][1]["budget"] = duplicate["sparse"][0]["budget"]
        nonfinite = copy.deepcopy(self.metrics)
        nonfinite["utility_improvement_ci"]["upper"] = float("nan")
        for name, metrics in (("missing", missing), ("duplicate", duplicate),
                              ("nonfinite", nonfinite)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_e4(self.contract, metrics)
