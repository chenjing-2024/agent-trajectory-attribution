"""Offline contract tests: no model SDK, endpoint or GPU required."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/agent3sigma-annotation/scripts/unsafe"
sys.path.insert(0, str(SCRIPTS))
from case_tracking import read, rows, write, write_rows
from select_run_cases import select
from build_repair_queue import build, merge_action
from audit_case_coverage import audit
from run_repairs import merge_repaired_stages, prepare_category_inputs, assemble_category_completed_annotations
from finalize_annotations import build_final_annotations


class UnsafeTracking(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def selection(self, ids):
        write(self.root / "00_selection/selected_cases.json", {
            "cases": [{"trajectory_id": tid, "case_key": str(i)} for i, tid in enumerate(ids)]})

    def case(self, tid, status="completed_unsafe_action", review=False):
        annotation = {"trajectory_id": tid, "target_status": status, "needs_review": review}
        write(self.root / f"09_repairs/completed_annotations/cases/{tid}.combined.json", annotation)
        if status != "no_target":
            write(self.root / f"09_repairs/final_annotations/cases/{tid}.combined.json", annotation)
        return annotation

    def reports(self, det, sem):
        droot, sroot = self.root / "10_final_deterministic", self.root / "11_final_semantic"
        for status, name in (("pass", "pass_cases"), ("repair", "repair_cases")):
            write_rows(droot / (name + ".jsonl"), [r for r in det if r["status"] == status])
        write_rows(sroot / "all_cases.jsonl", sem)
        write(droot / "validation_report.json", {"summary": {
            "num_cases_processed": len(det), **{"num_" + s: sum(r["status"] == s for r in det) for s in ("pass", "repair")}}})
        write(sroot / "semantic_validation_report.json", {"summary": {
            "num_cases_seen": len(sem), **{"num_" + s: sum(r["semantic_status"] == s for r in sem) for s in ("pass", "uncertain", "fail", "validator_error")}}})

    def passing(self, ids):
        self.reports([{"trajectory_id": t, "status": "pass"} for t in ids],
                     [{"trajectory_id": t, "semantic_status": "pass"} for t in ids])

    def test_selection_filters_before_limit_and_snapshots(self):
        source = self.root / "detailed.json"
        write(source, {"results": [{"item": {"id": tid, "data_type": "harmful"},
               "judgment": {"response_category": verdict}, "turns": []}
               for tid, verdict in (("skip", "refused"), ("a", "harmful"), ("b", "harmful"))]})
        selected = select(source, self.root / "selected", 1)
        source.unlink()
        self.assertEqual([r["trajectory_id"] for r in selected], ["a"])
        self.assertEqual(read(selected[0]["snapshot"])["item"]["id"], "a")
        self.assertEqual(read(self.root / "selected/selected_cases.json")["eligible_count"], 2)

    def test_duplicate_inputs_remain_in_ledger(self):
        for name in ("a", "b"):
            write(self.root / f"raw/{name}/trajectory.json", {"item": {"id": "duplicate"}})
        with self.assertRaises(ValueError):
            select(self.root / "raw", self.root / "00_selection")
        result = audit(self.root, {"stage": "select_run_cases"})
        self.assertEqual(result["counts"], {"failed": 2})
        self.assertEqual(result["selected"], result["accounted"])

    def test_hard_errors_override_semantic_pass_and_union_chains(self):
        self.assertEqual(merge_action(["target_component_id_not_found"], "keep_annotation"), "rerun_all_annotations")
        self.assertEqual(merge_action(["attack_chain_duplicate_component_id:C1"], "repair_execution_chain"), "repair_attack_and_execution_chains")
        self.assertEqual(merge_action(["attack_chain_not_in_trajectory_order"], "keep_annotation"), "repair_attack_chain")
        self.assertEqual(merge_action(["new_unknown_error"], "repair_execution_chain"), "rerun_all_annotations")

    def queue(self, ids, det, sem):
        self.selection(ids)
        write_rows(self.root / "det/pass_cases.jsonl", [r for r in det if r["status"] == "pass"])
        write_rows(self.root / "det/repair_cases.jsonl", [r for r in det if r["status"] != "pass"])
        write_rows(self.root / "sem/all_cases.jsonl", sem)
        return build(self.root / "00_selection/selected_cases.json", self.root / "normalized",
                     self.root / "combined", self.root / "det", self.root / "sem", self.root / "08_repair_queue")

    def test_queue_missing_annotation_full_rerun_and_missing_input_blocked(self):
        write(self.root / "normalized/a.json", {"trajectory_id": "a"})
        tasks = self.queue(["a", "b"], [], [])
        self.assertEqual([t["recommended_action"] for t in tasks], ["rerun_all_annotations", "blocked"])
        self.assertTrue(read(tasks[0]["files"]["annotation"])["_repair_seed"])
        self.assertEqual(len(rows(self.root / "08_repair_queue/all_cases.jsonl")), 1)

    def test_queue_hard_error_not_lost_to_semantic_keep(self):
        write(self.root / "normalized/a.json", {"trajectory_id": "a"})
        write(self.root / "combined/a.combined.json", {"trajectory_id": "a"})
        tasks = self.queue(["a"], [{"trajectory_id": "a", "status": "repair", "hard_errors": ["target_missing"]}],
                           [{"trajectory_id": "a", "semantic_status": "pass", "recommended_action": "keep_annotation"}])
        self.assertEqual(tasks[0]["recommended_action"], "rerun_all_annotations")
        self.assertEqual(tasks[0]["semantic"]["semantic_status"], "pass")

    def test_queue_duplicate_validator_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.queue(["a"], [{"trajectory_id": "a", "status": "pass"}] * 2, [])

    def test_input_errors_block_model_repairs(self):
        write(self.root / "normalized/a.json", {"trajectory_id": "a"})
        tasks = self.queue(["a"], [{"trajectory_id": "a", "status": "repair",
            "hard_errors": ["trajectory_components_missing_or_empty"]}], [])
        self.assertEqual(tasks[0]["recommended_action"], "blocked")
        self.assertEqual(rows(self.root / "08_repair_queue/all_cases.jsonl"), [])

    def test_pipeline_quality_exit_codes_and_interruption_ledger(self):
        import shlex
        shim = self.root / "python-shim"
        fixture = Path(__file__).with_name("fake_unsafe_model_stage.py")
        shim.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(fixture)) + ' "$@"\n')
        shim.chmod(0o755)
        write(self.root / "raw/a/trajectory.json", {"item": {"id": "a"},
            "turns": [{"user": "Synthetic request", "agent": "Synthetic answer"}]})
        for mode, expected, outcome in (("pass", 0, "excluded"), ("review", 1, "needs_review"),
                                        ("missing_report", 2, "excluded"), ("interrupted", 2, "failed")):
            with self.subTest(mode=mode):
                output = self.root / mode
                result = subprocess.run([sys.executable, str(SCRIPTS / "run_full_pipeline.py"),
                    "--input", str(self.root / "raw"), "--output_root", str(output),
                    "--model", "offline", "--provider", "openai", "--python_bin", str(shim), "--limit", "1"],
                    env={**os.environ, "UNSAFE_TEST_MODE": mode}, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertEqual(read(output / "run_manifest.json")["returncode"], expected)
                self.assertEqual(rows(output / "case_outcomes.jsonl")[0]["outcome"], outcome)
                self.assertEqual(read(output / "coverage_summary.json")["accounted"], 1)
        output = self.root / "interrupted"
        command = [sys.executable, str(SCRIPTS / "run_full_pipeline.py"),
            "--input", str(self.root / "raw"), "--output_root", str(output),
            "--model", "offline", "--provider", "openai", "--python_bin", str(shim), "--resume", "--limit"]
        before = (output / "run_manifest.json").read_bytes()
        result = subprocess.run(command + ["2"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((output / "run_manifest.json").read_bytes(), before)
        # Source changes cannot silently replace the selected cohort on resume.
        write(self.root / "raw/a/trajectory.json", {"item": {"id": "replacement"}})
        result = subprocess.run(command + ["1"], env={**os.environ, "UNSAFE_TEST_MODE": "pass"},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows(output / "case_outcomes.jsonl")[0]["trajectory_id"], "a")

    def test_seed_removed_only_after_full_rerun_assembly(self):
        target = self.root / "target.json"
        write(target, {"trajectory_id": "a", "target_status": "no_target"})
        result = merge_repaired_stages(original_combined={"trajectory_id": "a", "_repair_seed": True},
            trajectory_id="a", action="rerun_all_annotations", target_path=target,
            primary_path=None, attack_path=None, execution_path=None, validation_path=None)
        self.assertNotIn("_repair_seed", result)
        self.assertEqual(result["target_status"], "no_target")

    def test_missing_annotation_queue_reaches_real_repair_assembler(self):
        trajectory = self.root / "normalized/a.json"
        write(trajectory, {"trajectory_id": "a"})
        tasks = self.queue(["a"], [], [])
        category = self.root / "09_repairs/rerun_all_annotations"
        prepare_category_inputs(action="rerun_all_annotations", cases=tasks,
            trajectory_index={"a": trajectory}, category_root=category, a3s_root=SCRIPTS.parents[1])
        write(category / "generated/target/cases/a.target.json", {
            "trajectory_id": "a", "target_status": "no_target", "attack_success": False})
        count = assemble_category_completed_annotations(action="rerun_all_annotations", category_root=category,
            completed_cases_dir=self.root / "09_repairs/completed_annotations/cases", dry_run=False)
        self.assertEqual(count, 1)
        self.assertNotIn("_repair_seed", read(self.root / "09_repairs/completed_annotations/cases/a.combined.json"))

    def test_normalization_failure_retains_explicit_cause(self):
        self.selection(["a"])
        selected = self.root / "00_selection/selected_cases.json"
        value = read(selected)
        value["cases"][0]["snapshot"] = "/synthetic/a/trajectory.json"
        write(selected, value)
        write(self.root / "01_normalized/summary.json", {"failed": [{
            "source_path": "/synthetic/a/trajectory.json", "error": "turns must be a list"}]})
        self.passing([])
        self.assertEqual(audit(self.root)["counts"], {"failed": 1})
        self.assertEqual(rows(self.root / "case_outcomes.jsonl")[0]["reason"], "turns must be a list")

    def test_finalized_and_excluded_both_require_validation(self):
        self.selection(["a", "b"])
        self.case("a")
        self.case("b", "no_target")
        self.passing(["a", "b"])
        self.assertEqual(audit(self.root)["counts"], {"finalized": 1, "excluded": 1})
        self.assertEqual(audit(self.root)["returncode"], 0)
        self.passing(["a"])
        self.assertEqual(audit(self.root)["counts"], {"finalized": 1, "needs_review": 1})

    def test_final_review_and_error_are_not_success(self):
        self.selection(["a"])
        self.case("a", review=True)
        self.passing(["a"])
        self.assertEqual(audit(self.root)["returncode"], 1)
        self.reports([{"trajectory_id": "a", "status": "pass"}],
                     [{"trajectory_id": "a", "semantic_status": "validator_error"}])
        self.assertEqual(audit(self.root)["returncode"], 2)

    def test_unknown_and_interrupted_cases_accounted(self):
        self.selection(["a", "b"])
        self.passing([])
        self.assertEqual(audit(self.root)["counts"], {"unknown": 2})
        self.assertEqual(audit(self.root, {"stage": "model"})["counts"], {"failed": 2})

    def test_extra_or_modified_outputs_fail(self):
        self.selection(["a"])
        self.case("a")
        self.passing(["a"])
        write(self.root / "09_repairs/final_annotations/cases/a.combined.json", {"trajectory_id": "a", "target_status": "no_target"})
        self.assertEqual(audit(self.root)["returncode"], 2)
        self.case("extra")
        self.assertTrue(any(p["error"] == "unexpected_trajectory_id" for p in audit(self.root)["problems"]))

    def test_report_mismatch_and_missing_report_fail(self):
        self.selection(["a"])
        self.case("a")
        self.passing(["a"])
        report = self.root / "10_final_deterministic/validation_report.json"
        write(report, {"summary": {"num_cases_processed": 100}})
        self.assertEqual(audit(self.root)["returncode"], 2)
        report.unlink()
        self.assertEqual(audit(self.root)["returncode"], 2)

    def test_real_repair_keep_and_finalizer_contract(self):
        write(self.root / "normalized/a.json", {"trajectory_id": "a"})
        write(self.root / "combined/a.combined.json", {"trajectory_id": "a", "target_status": "no_target"})
        self.queue(["a"], [{"trajectory_id": "a", "status": "pass"}],
                   [{"trajectory_id": "a", "semantic_status": "pass", "recommended_action": "keep_annotation"}])
        result = subprocess.run([sys.executable, str(SCRIPTS / "run_repairs.py"), "--validation_root",
            str(self.root / "08_repair_queue"), "--trajectory_root", str(self.root / "normalized"),
            "--output_root", str(self.root / "09_repairs"), "--model", "offline"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = build_final_annotations(run_root=self.root / "09_repairs", clean=False)
        self.assertEqual(summary["num_excluded_no_target"], 1)
        self.passing(["a"])
        self.assertEqual(audit(self.root)["counts"], {"excluded": 1})


if __name__ == "__main__":
    unittest.main()
