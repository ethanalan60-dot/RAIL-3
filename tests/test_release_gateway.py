"""Synthetic path-layer checks only; no actual research inputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "release_gateway", Path(__file__).resolve().parents[1] / "scripts/release.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        # All artificial files remain inside this candidate's ignored build/.
        self.base = Path(__file__).resolve().parents[1] / "build"
        self.base.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=self.base, prefix="gateway-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.config = {"schema_version": "rail3.release-input-locations.v1",
                       "inputs": {"synthetic": {"path": None, "sha256": None,
                                                  "identity_status": "UNKNOWN"}}}

    def test_unbound_plan_does_not_create_output(self):
        output = self.root / "release-runs" / "new"
        result = gateway.plan(self.config, "synthetic-new", output, self.root)
        self.assertEqual(result["scientific_execution"], "BLOCKED")
        self.assertFalse(output.exists())
        self.assertFalse(result["input_payloads_accessed"])

    def test_refuses_existing_output_and_source_subtree(self):
        for output in (self.root, self.root / "src" / "bad"):
            with self.assertRaises(ValueError):
                gateway.plan(self.config, "synthetic-new", output, self.root)

    def test_input_output_overlap_is_rejected(self):
        self.config["inputs"]["synthetic"]["path"] = "release-runs"
        with self.assertRaises(ValueError):
            gateway.plan(self.config, "synthetic-new", self.root / "release-runs/new", self.root)

    def test_invalid_run_id_and_manifest_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            gateway.plan(self.config, "../old", self.root / "release-runs/new", self.root)
        for name in ("../outside", "/outside"):
            with self.assertRaises(ValueError):
                gateway.contained_file(self.root, name)


if __name__ == "__main__":
    unittest.main()
