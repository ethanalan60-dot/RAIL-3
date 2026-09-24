#!/usr/bin/env python3
"""Run only the release's explicit synthetic CPU contract tests.

Help uses the standard library only. Research/NumPy imports occur only after
argument parsing. The script writes no output files and runs no historical
test discovery, model, SAM backend, evaluation dataset or bootstrap routine.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run tiny synthetic CPU tests of serialization, atomic geometry "
            "and E4 boolean rules. Requires NumPy for the test run. "
            "This does not reproduce or verify experimental results."
        ),
    )
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2,
                        help="unittest output verbosity (default: 2)")
    args = parser.parse_args(argv)  # --help exits before any project/NumPy import.

    import importlib.util
    import json
    from pathlib import Path
    import sys
    import unittest

    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    test_file = root / "tests/test_release_synthetic.py"
    for path in (source, test_file):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            parser.error("release smoke paths must stay inside this release directory")
    if not source.is_dir() or not test_file.is_file():
        parser.error("the local release src/ and synthetic test file are required")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("rail3_release_synthetic_tests", test_file)
    if spec is None or spec.loader is None:
        parser.error("cannot load the explicit local synthetic test file")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        print(json.dumps({
            "status": "BLOCKED",
            "reason": "required import unavailable",
            "missing_module": error.name,
            "scope": "synthetic CPU contract tests only",
            "scientific_reproduction": "NOT_RUN",
        }, sort_keys=True))
        return 2

    forbidden_prefixes = (
        "torch", "sam3", "rail3.sam", "rail3.models",
        "rail3.evaluation.tmlr_v9c_metrics",
        "rail3.evaluation.tmlr_v9c_execution",
        "rail3.evaluation.tmlr_v9c_coco_targets",
    )
    unexpected = sorted(name for name in sys.modules if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden_prefixes
    ))
    if unexpected:
        print(json.dumps({"status": "BLOCKED", "reason": "unexpected backend/science import",
                          "modules": unexpected, "scientific_reproduction": "NOT_RUN"},
                         sort_keys=True))
        return 2

    suite = unittest.TestSuite()
    for class_name in (
        "CanonicalSerializationTests",
        "SyntheticAtomicPartitionTests",
        "SyntheticE4RuleTests",
    ):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(
            getattr(module, class_name),
        ))
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "scope": "synthetic CPU contract tests only; not scientific verification",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "scientific_reproduction": "NOT_RUN",
    }, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
