import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class InputPreparation(unittest.TestCase):
    def run_prepare(self, skill, source, output, kind):
        return subprocess.run([sys.executable, str(ROOT / "skills" / skill / "scripts/common/prepare_annotation_input.py"),
                               "--input", str(source), "--input_format", kind, "--output_root", str(output)],
                              text=True, capture_output=True)

    def test_raw_detailed_and_preservation_in_both_skills(self):
        for skill in ["agent3sigma-annotation", "canary-annotation"]:
            with self.subTest(skill=skill), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = root / "detailed.json"
                raw.write_text(json.dumps({"results": [{"item": {"id": "refused"}, "turns": [{"user": "Synthetic request", "agent": "I refuse."}]}]}))
                result = self.run_prepare(skill, raw, root / "normalized", "raw")
                self.assertEqual(result.returncode, 0, result.stderr)
                case = json.loads((root / "normalized/cases/refused.json").read_text())
                self.assertEqual([c["component_id"] for c in case["trajectory"] if "component_id" in c], ["C1", "C2"])
                result = self.run_prepare(skill, root / "normalized/cases", root / "preserved", "normalized")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads((root / "preserved/cases/refused.json").read_text()), case)
                self.assertNotEqual(self.run_prepare(skill, raw, root / "normalized", "raw").returncode, 0)

    def test_invalid_or_duplicate_ids_fail_without_partial_cases(self):
        for skill in ["agent3sigma-annotation", "canary-annotation"]:
            for invalid in ["duplicate_trajectory", "duplicate_component", "wrong_format"]:
                with self.subTest(skill=skill, invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); source = root / "input"; source.mkdir()
                    case = {"trajectory_id": "example", "trajectory": [{"role": "user", "content": "hello", "component_id": "C7"}]}
                    if invalid == "duplicate_component":
                        case["trajectory"].append(dict(case["trajectory"][0]))
                    if invalid == "wrong_format":
                        case = {"messages": []}
                    (source / "a.json").write_text(json.dumps(case))
                    if invalid == "duplicate_trajectory":
                        (source / "b.json").write_text(json.dumps(case))
                    result = self.run_prepare(skill, source, root / "output", "normalized")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((root / "output/cases").exists())

    def test_raw_normalization_is_first_in_refusal_and_canary_plans(self):
        with tempfile.TemporaryDirectory(prefix="raw input ") as tmp:
            root = Path(tmp); raw = root / "raw.json"
            raw.write_text(json.dumps({"item": {"id": "example"}, "turns": [{"user": "request", "agent": "refusal"}]}))
            result = subprocess.run([sys.executable, str(ROOT / "skills/agent3sigma-annotation/scripts/safety_refusal/run_pipeline.py"),
                                     "--raw_input", str(raw), "--output_root", str(root / "refusal"), "--model", "unused", "--dry_run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "refusal/run_manifest.json").read_text())
            self.assertEqual(manifest["stages"][0]["name"], "normalize_raw_input")
            self.assertFalse((root / "refusal/00_normalized").exists())
            canary = ROOT / "skills/canary-annotation"
            config = root / "config.env"; config.write_text((canary / "config/annotation.example.env").read_text())
            env = dict(os.environ, PROJECT_ROOT=str(root / "project"), MODEL_ID="unused", ANNOTATION_CONFIG=str(config), INPUT_FORMAT="raw", RAW_INPUT=str(raw))
            env.pop("OPENAI_API_KEY", None)
            for workflow in ["unsafe-all", "safety-refusal-all"]:
                plan = root / f"{workflow}.json"
                result = subprocess.run(["bash", str(canary / "bin/annotation_skill.sh"), workflow, "--dry_run", "--plan", str(plan)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(plan.read_text())["stages"][0]["name"], "Normalize raw input")
            self.assertFalse((root / "project/results/canary").exists())


if __name__ == "__main__":
    unittest.main()
