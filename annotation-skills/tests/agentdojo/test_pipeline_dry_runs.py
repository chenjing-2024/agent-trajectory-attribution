import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2] / "skills"
SKILL_ROOT = REPO_ROOT / "agentdojo-annotation"


class PipelineDryRunTest(unittest.TestCase):
    def test_unsafe_pipeline_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            output = root / "unsafe"
            raw.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILL_ROOT
                        / "scripts"
                        / "unsafe"
                        / "run_initial_pipeline.py"
                    ),
                    "--raw_root",
                    str(raw),
                    "--output_root",
                    str(output),
                    "--model",
                    "test-model",
                    "--limit",
                    "10",
                    "--dry_run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            validate_stage = next(
                stage
                for stage in manifest["stages"]
                if stage["stage"] == "validate_c1_annotations"
            )
            self.assertEqual(
                validate_stage["command"][-2:],
                ["--limit", "10"],
            )

    def test_full_unsafe_pipeline_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            output = root / "unsafe-full"
            raw.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILL_ROOT
                        / "scripts"
                        / "unsafe"
                        / "run_full_pipeline.py"
                    ),
                    "--raw_root",
                    str(raw),
                    "--output_root",
                    str(output),
                    "--model",
                    "test-model",
                    "--base_url",
                    "https://example.invalid/v1",
                    "--limit",
                    "10",
                    "--dry_run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "full_run_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(len(manifest["stages"]), 7)
            self.assertNotIn(
                "https://example.invalid/v1",
                json.dumps(manifest),
            )


    def test_task_alignment_pipeline_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            output = root / "task-alignment"
            raw.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        SKILL_ROOT
                        / "scripts"
                        / "task_alignment"
                        / "run_pipeline.py"
                    ),
                    "--raw_root",
                    str(raw),
                    "--output_root",
                    str(output),
                    "--dry_run",
                    "--write_skipped",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(manifest["min_components"], 4)
            self.assertEqual(len(manifest["stages"]), 3)
            for stage in manifest["stages"]:
                self.assertEqual(stage["status"], "planned")
                self.assertTrue(Path(stage["command"][1]).is_file())
            self.assertFalse((output / "annotations").exists())


if __name__ == "__main__":
    unittest.main()
