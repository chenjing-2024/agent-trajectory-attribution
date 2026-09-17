import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2] / "skills"
SCRIPT = (
    REPO_ROOT
    / "agentdojo-annotation"
    / "scripts"
    / "unsafe"
    / "merge.py"
)


class MergeRepairedAnnotationsTest(unittest.TestCase):
    def test_sparse_repairs_replace_matching_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "original"
            repaired = root / "repaired"
            output = root / "final"
            (original / "nested").mkdir(parents=True)
            (repaired / "nested").mkdir(parents=True)
            (original / "nested" / "case.json").write_text('{"v":"old"}')
            (original / "keep.json").write_text('{"v":"keep"}')
            (repaired / "nested" / "case.json").write_text('{"v":"new"}')

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--original_root",
                    str(original),
                    "--repaired_root",
                    str(repaired),
                    "--output_root",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                json.loads((output / "nested" / "case.json").read_text())["v"],
                "new",
            )
            self.assertEqual(
                json.loads((output / "keep.json").read_text())["v"], "keep"
            )
            self.assertEqual(
                len((output / "merge_manifest.jsonl").read_text().splitlines()), 1
            )


    def test_missing_original_repair_requires_known_component(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("original", "repaired", "components"):
                (root / name).mkdir()
            (root / "repaired/a.json").write_text('{"attack_success": false}')
            (root / "components/a.json").write_text('{}')
            base = [sys.executable, str(SCRIPT), "--original_root", str(root / "original"),
                    "--repaired_root", str(root / "repaired"), "--components_root", str(root / "components")]
            result = subprocess.run(base + ["--output_root", str(root / "accepted")], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "accepted/a.json").exists())
            (root / "repaired/unexpected.json").write_text('{}')
            result = subprocess.run(base + ["--output_root", str(root / "rejected")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
