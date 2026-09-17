import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[2] / "skills/canary-annotation"


class CanaryDryRuns(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="canary dry run ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        (self.project / "results/canary/cases").mkdir(parents=True)
        self.config = self.root / "config.env"
        self.config.write_text(
            (SKILL / "config/annotation.example.env").read_text().replace(
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_BASE_URL=https://private.example.invalid/v1",
            )
        )
        self.env = dict(os.environ, PROJECT_ROOT=str(self.project),
                        MODEL_ID="dry-run-placeholder", ANNOTATION_CONFIG=str(self.config))
        self.env.pop("OPENAI_API_KEY", None)
        # Annotation scripts use python. Any accidental execution must fail.
        self.shims = self.root / "bin"
        self.shims.mkdir()
        sentinel = self.shims / "python"
        sentinel.write_text('#!/bin/sh\necho "Annotation execution forbidden" >&2\nexit 97\n')
        sentinel.chmod(0o755)
        self.env["PATH"] = str(self.shims) + os.pathsep + self.env["PATH"]

    def invoke(self, workflow, plan, *extra):
        return subprocess.run(
            ["bash", str(SKILL / "bin/annotation_skill.sh"), workflow,
             "--dry_run", "--plan", str(plan), *extra],
            env=self.env, text=True, capture_output=True,
        )

    def test_all_workflows_plan_without_credentials_or_annotation_execution(self):
        counts = {"unsafe-all": 15, "safety-refusal-all": 6,
                  "safety-refusal": 4, "initial": 6, "validate-unsafe": 2,
                  "repair-unsafe": 3, "finalize-unsafe": 1,
                  "finalize-refusal": 1, "validate-final-unsafe": 3,
                  "validate-final-refusal": 2, "finalize-all": 7}
        for workflow, count in counts.items():
            with self.subTest(workflow=workflow):
                plan = self.root / f"{workflow}.json"
                result = self.invoke(workflow, plan)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(plan.read_text())
                self.assertEqual(record["status"], "dry_run_complete")
                self.assertEqual(len(record["stages"]), count)
                self.assertTrue(all(s["status"] == "planned" for s in record["stages"]))
                self.assertTrue(any(d["status"] == "deferred" for d in record["dependencies"]))
                self.assertNotIn("private.example.invalid", plan.read_text() + result.stdout)
        self.assertEqual(list(self.project.rglob("*.json")), [])
        self.assertEqual(sorted(p.name for p in (self.project / "results").iterdir()), ["canary"])
        refusal = json.loads((self.root / "safety-refusal-all.json").read_text())
        self.assertTrue(refusal["stages"][1]["command"][1].endswith("annotate_primary_targets.py"))
        self.assertTrue(refusal["stages"][-2]["command"][1].endswith("validate_safety_refusal_annotations_deterministic.py"))

    def test_existing_plan_is_preserved(self):
        plan = self.root / "existing.json"
        plan.write_text("existing content")
        self.assertNotEqual(self.invoke("unsafe-all", plan).returncode, 0)
        self.assertEqual(plan.read_text(), "existing content")

    def test_missing_script_marks_plan_failed(self):
        with self.config.open("a") as stream:
            stream.write(f'\nPIPELINE_DIR="{self.root / "missing-scripts"}"\n')
        plan = self.root / "failed.json"
        result = self.invoke("unsafe-all", plan)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(plan.read_text())["status"], "failed")

    def test_missing_input_is_rejected(self):
        self.env["PROJECT_ROOT"] = str(self.root / "missing-input")
        result = self.invoke("safety-refusal-all", self.root / "missing.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input directory not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
