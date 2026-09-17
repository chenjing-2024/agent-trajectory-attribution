import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills/agent3sigma-annotation/scripts/safety_refusal"


class RefusalQuality(unittest.TestCase):
    def test_pipeline_propagates_quality_using_real_builder_and_validator(self):
        scenarios = [({}, "ok", 0), ({"target_review": True}, "completed_with_review", 1),
                     ({"primary_review": True}, "completed_with_review", 1),
                     ({"target_review": True, "primary_review": True}, "completed_with_review", 1),
                     ({"invalid": True}, "completed_with_review", 1),
                     ({"target_failed": 1}, "failed", 2),
                     ({"no_target": True}, "no_eligible_cases", 0),
                     ({"no_target": True, "target_review": True}, "completed_with_review", 1)]
        for fixture, status, code in scenarios:
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory(prefix="refusal quality ") as tmp:
                root = Path(tmp)
                data = root / "cases"; data.mkdir()
                (data / "example.json").write_text(json.dumps({"trajectory_id": "example", "trajectory": [
                    {"role": "user", "component_id": "C1", "content": "Synthetic unsafe request."},
                    {"role": "assistant", "component_id": "C2", "content": "I cannot help with that."}]}))
                fixture_path = root / "fixture.json"; fixture_path.write_text(json.dumps(fixture))
                shim = root / "model-shim"
                helper = Path(__file__).with_name("fake_refusal_model_stage.py")
                shim.write_text(f"#!{sys.executable}\n" + helper.read_text())
                shim.chmod(0o755)
                env = dict(os.environ, REFUSAL_TEST_FIXTURE=str(fixture_path))
                env.pop("OPENAI_API_KEY", None)
                output = root / "run"
                result = subprocess.run([sys.executable, str(SCRIPTS / "run_pipeline.py"),
                                         "--data_dir", str(data), "--output_root", str(output),
                                         "--model", "fixture-model", "--python_bin", str(shim)],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["status"], status)
                if not fixture.get("no_target"):
                    record = json.loads((output / "02_primary/cases/example.json").read_text())
                    for stage in ["target", "primary"]:
                        if fixture.get(f"{stage}_review"):
                            self.assertTrue(record["needs_review"])
                            self.assertIn(stage, record["_metadata"]["review_sources"])
                    validation = json.loads((output / "03_validation/summary.json").read_text())
                    if fixture.get("target_review") or fixture.get("primary_review") or fixture.get("invalid"):
                        self.assertEqual(validation["review"], 1)
                        direct = subprocess.run([sys.executable, str(SCRIPTS / "validate_primary.py"),
                                                 "--cases_dir", str(output / "02_primary/cases"),
                                                 "--output_dir", str(root / "direct-validation")], capture_output=True)
                        self.assertEqual(direct.returncode, 1)

    def test_report_only_does_not_hide_runtime_or_input_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run([sys.executable, str(SCRIPTS / "validate_primary.py"),
                                     "--cases_dir", str(root / "missing"), "--output_dir", str(root / "out"),
                                     "--report_only"], capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
