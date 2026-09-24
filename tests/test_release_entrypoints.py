"""Artificial adapter fixtures; no real experiment, payload or runtime module."""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_entrypoints_adapter", ROOT / "scripts/release_entrypoints.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class EntryPointAdapterTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "build"
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch, prefix="entrypoint-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/release.py").write_bytes((ROOT / "scripts/release.py").read_bytes())
        # A top-level exception would fire if the historical fixture were imported.
        (self.root / "scripts/history_fixture.py").write_text('raise RuntimeError("must never import historical fixture")\n')
        rows = []
        for name in ("scripts/release.py", "scripts/history_fixture.py"):
            raw = (self.root / name).read_bytes()
            rows.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        (self.root / "RELEASE_FILES.json").write_text(json.dumps({"files": rows}))
        self.entry = {"id": "synthetic_fixture", "historical_script": "scripts/history_fixture.py",
                      "required_source_files": ["scripts/history_fixture.py"],
                      "required_input_roles": ["artificial_payload"], "runtime_distributions": ["synthetic-runtime"],
                      "historical_authority_limits": ["Synthetic authority not verified"],
                      "new_release_run_adapter": "SOURCE_SNAPSHOT_ONLY"}
        self.config = {"schema_version": "rail3.release-input-locations.v1",
                       "inputs": {"artificial_payload": {"path": None, "sha256": None,
                                                          "identity_status": "UNKNOWN"}}}

    def validate(self, version=lambda name: "0.0.synthetic"):
        return adapter.validate(self.root, self.entry, self.config, "synthetic-new",
                                self.root / "release-runs/synthetic-new", version)

    def test_unbound_inputs_nonzero_but_source_check_passes_without_import(self):
        result = self.validate()
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["unbound_input_roles"], ["artificial_payload"])
        self.assertEqual(result["engineering_checks"], "PASS")
        self.assertFalse((self.root / "release-runs").exists())

    def test_supplied_path_is_not_opened_or_certified(self):
        self.config["inputs"]["artificial_payload"]["path"] = "never-created/payload.npz"
        # A metadata probe would fail this test, even though no payload exists.
        original_stat = Path.stat
        original_open = Path.open
        def guard(original):
            def checked(path, *args, **kwargs):
                if "never-created" in path.parts:
                    raise AssertionError("input payload was probed or opened")
                return original(path, *args, **kwargs)
            return checked
        with patch.object(Path, "stat", guard(original_stat)), patch.object(Path, "open", guard(original_open)):
            result = self.validate()
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["input_payloads_opened"])
        self.assertFalse(result["input_filesystem_probed"])
        self.assertEqual(result["historical_identity_verification"], "NOT_PERFORMED")
        self.assertEqual(result["scientific_readiness"], "NOT_ESTABLISHED")

    def test_missing_runtime_distribution_is_reported(self):
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)
        result = self.validate(missing)
        self.assertEqual(result["missing_runtime_distributions"], ["synthetic-runtime"])
        self.assertEqual(result["exit_code"], 3)

    def test_missing_and_modified_code_are_distinct(self):
        self.entry["required_source_files"].append("scripts/missing_fixture.py")
        result = self.validate()
        self.assertEqual(result["missing_source_files"], ["scripts/missing_fixture.py"])
        (self.root / "scripts/history_fixture.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.validate()

    def test_modified_planner_is_rejected_before_execution(self):
        (self.root / "scripts/release.py").write_text('raise RuntimeError("should not execute")\n')
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.validate()

    def test_existing_output_and_input_overlap_are_rejected(self):
        output = self.root / "release-runs/synthetic-new"
        output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.validate()
        # A separate artificial configuration probes only lexical overlap.
        self.config["inputs"]["artificial_payload"]["path"] = "release-runs"
        with self.assertRaisesRegex(ValueError, "overlap"):
            adapter.validate(self.root, self.entry, self.config, "other",
                             self.root / "release-runs/other", lambda name: "0.synthetic")

    def test_raw_payload_types_and_escaping_source_paths_are_refused(self):
        for path in ("predictions.npz", "../history_fixture.py", "/history_fixture.py"):
            with self.assertRaisesRegex(ValueError, "invalid source"):
                adapter.check_source_files(self.root, [path], {})


if __name__ == "__main__":
    unittest.main()
