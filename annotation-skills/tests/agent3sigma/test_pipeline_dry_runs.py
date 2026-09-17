import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "skills"
SKILL = ROOT / "agent3sigma-annotation"
UNSAFE_RUNNER = SKILL / "scripts" / "unsafe" / "run_full_pipeline.py"
MERGE_RUNNER = SKILL / "scripts" / "unsafe" / "merge_annotations.py"


class PipelineDryRuns(unittest.TestCase):
    def test_merge_strict_ignores_stage_specific_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_names = (
                "target",
                "primary",
                "attack_chain",
                "execution_chain",
            )
            records = {
                "target": {
                    "trajectory_id": "case-1",
                    "target_status": "completed_unsafe_action",
                    "attack_success": True,
                    "target_component_id": "C4",
                    "confidence": 0.9,
                    "needs_review": False,
                },
                "primary": {
                    "trajectory_id": "case-1",
                    "primary_component_id": "C1",
                    "confidence": 0.8,
                    "needs_review": False,
                },
                "attack_chain": {
                    "trajectory_id": "case-1",
                    "attack_chain": [
                        {"component_id": "C1", "role_in_attack": "source"}
                    ],
                    "confidence": 0.7,
                    "needs_review": False,
                },
                "execution_chain": {
                    "trajectory_id": "case-1",
                    "execution_chain": [
                        {
                            "component_id": "C4",
                            "role_in_execution": "unsafe_content_generation",
                        }
                    ],
                    "confidence": 0.6,
                    "needs_review": True,
                    "review_reason": "execution check",
                },
            }
            command = [
                sys.executable,
                str(MERGE_RUNNER),
                "--run_root",
                str(root),
                "--output_root",
                str(root / "combined"),
                "--strict",
            ]
            for source_name in source_names:
                source_dir = root / source_name / "cases"
                source_dir.mkdir(parents=True)
                (source_dir / f"case-1.{source_name}.json").write_text(
                    json.dumps(records[source_name]),
                    encoding="utf-8",
                )
                command.extend(
                    [f"--{source_name}_dir", str(source_dir)]
                )

            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads(
                (root / "combined" / "merge_summary.json").read_text()
            )
            self.assertEqual(summary["counts"]["indexing_errors"], 0)
            self.assertEqual(summary["counts"]["problematic_cases"], 0)
            merged_file = next((root / "combined" / "cases").glob("*.json"))
            merged = json.loads(merged_file.read_text())
            self.assertEqual(merged["confidence"], 0.6)
            self.assertTrue(merged["needs_review"])

    def test_unsafe_reports_no_eligible_detailed_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "detailed.json"
            source.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "item": {
                                    "index": 0,
                                    "id": "benign-0",
                                    "name": "benign",
                                    "data_type": "benign",
                                },
                                "judgment": {"utility_verdict": "failed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(UNSAFE_RUNNER),
                    "--input",
                    str(source),
                    "--output_root",
                    str(output),
                    "--model",
                    "test-model",
                    "--provider",
                    "compatible",
                    "--base_url",
                    "https://example.invalid/v1",
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("无符合条件的案例", result.stdout)
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "no_eligible_cases")
            self.assertFalse((output / "01_normalized").exists())

    def test_unsafe_prepared_cases_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cases = base / "cases"
            output = base / "unsafe"
            cases.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILL
                        / "scripts"
                        / "unsafe"
                        / "run_full_pipeline.py"
                    ),
                    "--input",
                    str(cases),
                    "--output_root",
                    str(output),
                    "--model",
                    "test-model",
                    "--provider",
                    "compatible",
                    "--base_url",
                    "https://example.invalid/v1",
                    "--limit",
                    "2",
                    "--dry_run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(len(manifest["stages"]), 15)
            self.assertEqual([stage["name"] for stage in manifest["stages"] if "--limit" in stage["command"]], ["select_run_cases"])
            self.assertTrue(all("--limit_per_category" not in stage["command"] for stage in manifest["stages"]))
            self.assertNotIn(
                "https://example.invalid/v1",
                json.dumps(manifest),
            )
            target_stage = next(
                stage
                for stage in manifest["stages"]
                if stage["name"] == "annotate_target"
            )
            timeout_index = target_stage["command"].index("--timeout")
            self.assertEqual(
                target_stage["command"][timeout_index + 1],
                "600",
            )
            merge_stage = next(
                stage
                for stage in manifest["stages"]
                if stage["name"] == "merge_annotations"
            )
            self.assertNotIn("--strict", merge_stage["command"])
            self.assertTrue(
                manifest["paths"]["final_annotations"].endswith(
                    "09_repairs/final_annotations/cases"
                )
            )

    def test_safety_refusal_data_dir_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cases = base / "cases"
            output = base / "refusal"
            cases.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILL
                        / "scripts"
                        / "safety_refusal"
                        / "run_pipeline.py"
                    ),
                    "--data_dir",
                    str(cases),
                    "--output_root",
                    str(output),
                    "--model",
                    "test-model",
                    "--base_url",
                    "https://example.invalid/v1",
                    "--dry_run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(len(manifest["stages"]), 4)
            self.assertNotIn(
                "https://example.invalid/v1",
                json.dumps(manifest),
            )


if __name__ == "__main__":
    unittest.main()
