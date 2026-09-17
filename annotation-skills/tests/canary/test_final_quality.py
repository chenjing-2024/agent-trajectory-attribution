import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills/canary-annotation"
GATE = SKILL / "bin/check_final_quality.py"


def report(kind, counts):
    schemas = {"refusal": "safety_refusal_deterministic_validation_report_v1",
               "deterministic": "unsafe_action_deterministic_validation_report_v3",
               "semantic": "unsafe_action_semantic_validation_report_v3_root_cause"}
    return {"schema_version": schemas[kind], "summary": {"num_cases": sum(counts.values()), "status_counts": counts}}


class FinalQuality(unittest.TestCase):
    def test_report_status_matrix(self):
        for counts, expected, code in [({"pass": 1}, "ok", 0), ({"fail": 1}, "completed_with_review", 1),
                                       ({"uncertain": 1}, "completed_with_review", 1),
                                       ({"validator_error": 1}, "failed", 2), ({}, "no_eligible_cases", 0)]:
            with self.subTest(counts=counts), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                det = root / "det.json"; sem = root / "sem.json"; out = root / "quality.json"
                det.write_text(json.dumps(report("deterministic", {"pass": sum(counts.values())})))
                sem.write_text(json.dumps(report("semantic", counts)))
                result = subprocess.run([sys.executable, str(GATE), "--kind", "unsafe", "--deterministic", str(det),
                                         "--semantic", str(sem), "--output", str(out)], capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(out.read_text())["status"], expected)

    def test_missing_malformed_and_inconsistent_reports_fail_closed(self):
        for value in [None, "not-json", {}, report("refusal", {"pass": -1}),
                      {"schema_version": "safety_refusal_deterministic_validation_report_v1", "summary": {"num_cases": 2, "status_counts": {"pass": 1}}}]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / "report.json"; out = root / "quality.json"
                if value is not None:
                    source.write_text(value if isinstance(value, str) else json.dumps(value))
                result = subprocess.run([sys.executable, str(GATE), "--kind", "refusal", "--deterministic", str(source),
                                         "--output", str(out)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(out.read_text())["status"], "failed")

    def test_real_final_refusal_wrapper_returns_nonzero_for_invalid_annotation(self):
        with tempfile.TemporaryDirectory(prefix="canary final quality ") as tmp:
            root = Path(tmp); project = root / "project"
            cases = project / "results/canary/cases"; cases.mkdir(parents=True)
            (cases / "example.json").write_text(json.dumps({"trajectory_id": "example", "trajectory": [
                {"component_id": "C1", "role": "user", "content": "Synthetic request"},
                {"component_id": "C2", "role": "assistant", "content": "I refuse."}]}))
            annotations = project / "results/canary_safety_refusal_target_primary_annotations_final/cases"
            annotations.mkdir(parents=True); (annotations / "example.json").write_text("{}")
            config = root / "config.env"; config.write_text((SKILL / "config/annotation.example.env").read_text())
            shim = root / "bin"; shim.mkdir()
            python = shim / "python"
            python.write_text(f"#!{sys.executable}\nimport os,sys\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n")
            python.chmod(0o755)
            env = dict(os.environ, PROJECT_ROOT=str(project), ANNOTATION_CONFIG=str(config), MODEL_ID="unused",
                        PATH=str(shim) + os.pathsep + os.environ["PATH"])
            env.pop("OPENAI_API_KEY", None)
            result = subprocess.run(["bash", str(SKILL / "bin/annotation_skill.sh"), "validate-final-refusal"],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            quality = json.loads(next(project.rglob("final_quality.json")).read_text())
            self.assertEqual(quality["status"], "completed_with_review")
            self.assertEqual(quality["summaries"]["deterministic"]["num_fail"], 1)

    def test_unsafe_wrapper_propagates_gate_exit_with_offline_report_fixtures(self):
        for deterministic, semantic, expected in [({"warning": 1}, {"pass": 1}, 0),
                                                  ({"fail": 1}, {"pass": 1}, 1),
                                                  ({"pass": 1}, {"uncertain": 1}, 1),
                                                  ({"pass": 1}, {"validator_error": 1}, 2)]:
            with self.subTest(semantic=semantic, deterministic=deterministic), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); project = root / "project"
                (project / "results/canary/cases").mkdir(parents=True)
                (project / "results/canary_full_annotations_unsafe_action_final/cases").mkdir(parents=True)
                (project / "results/canary_full_annotations_unsafe_action_final/cases/a.json").write_text(json.dumps({"trajectory_id": "a", "target_type": "unsafe_action"}))
                fixture = root / "fixture.json"
                fixture.write_text(json.dumps({"deterministic": report("deterministic", deterministic),
                                                "semantic": report("semantic", semantic)}))
                config = root / "config.env"; config.write_text((SKILL / "config/annotation.example.env").read_text())
                shim = root / "bin"; shim.mkdir()
                python = shim / "python"
                python.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
script = Path(sys.argv[1]).name
assert script in {"validate_unsafe_action_annotations_deterministic_v3.py", "validate_unsafe_action_annotations_semantic_v3_root_cause.py"}
kind = "semantic" if "semantic" in script else "deterministic"
output = Path(sys.argv[sys.argv.index("--output_root") + 1])
output.mkdir(parents=True)
fixture = json.loads(Path(os.environ["QUALITY_TEST_FIXTURE"]).read_text())
(output / (kind + "_validation_report.json")).write_text(json.dumps(fixture[kind]))
status = next(iter(fixture[kind]["summary"]["status_counts"]))
(output / "all_cases.jsonl").write_text(json.dumps({"trajectory_id": "a", "semantic_status" if kind == "semantic" else "status": status}) + "\\n")
''')
                python.chmod(0o755)
                env = dict(os.environ, PROJECT_ROOT=str(project), ANNOTATION_CONFIG=str(config), MODEL_ID="unused",
                            OPENAI_API_KEY="synthetic-fixture-only", QUALITY_TEST_FIXTURE=str(fixture),
                            PATH=str(shim) + os.pathsep + os.environ["PATH"])
                result = subprocess.run(["bash", str(SKILL / "bin/annotation_skill.sh"), "validate-final-unsafe"],
                                        env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                quality = json.loads(next(project.rglob("final_quality.json")).read_text())
                self.assertEqual(quality["returncode"], expected)


if __name__ == "__main__":
    unittest.main()
