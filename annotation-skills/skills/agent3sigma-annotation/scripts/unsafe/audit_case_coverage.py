#!/usr/bin/env python3
"""Account for every selected unsafe case using final evidence, not file counts."""
import argparse
from collections import Counter
from pathlib import Path

from case_tracking import index_cases, read, rows, write, write_rows


def audit(run_root, pipeline_error=None):
    root = Path(run_root)
    selection_path = root / "00_selection/selected_cases.json"
    problems, outcomes = [], []
    if not selection_path.is_file():
        problems.append({"error": "missing_selection_manifest"})
        selected = []
    else:
        selected = read(selection_path)["cases"]
    if not selected and not problems and not pipeline_error:
        summary = {"schema_version": "unsafe_case_coverage_v1", "status": "no_eligible_cases", "returncode": 0,
                   "selected": 0, "accounted": 0, "counts": {}, "problems": [], "pipeline_error": None}
        write_rows(root / "case_outcomes.jsonl", [])
        write(root / "coverage_summary.json", summary)
        return summary
    expected = {row.get("trajectory_id") for row in selected}
    final, errors = index_cases(root / "09_repairs/final_annotations/cases")
    completed, completed_errors = index_cases(root / "09_repairs/completed_annotations/cases")
    normalized, normalized_errors = index_cases(root / "01_normalized/cases")
    problems.extend(errors + completed_errors + normalized_errors)
    for stage, records in (("final", final), ("completed", completed), ("normalized", normalized)):
        problems.extend({"error": "unexpected_trajectory_id", "stage": stage, "trajectory_id": tid} for tid in records if tid not in expected)
    def evidence(paths):
        result = {}
        for path in paths:
            try:
                for record in rows(path):
                    tid = record.get("trajectory_id")
                    if tid not in expected or tid in result:
                        problems.append({"error": "unexpected_or_duplicate_evidence_id", "path": str(path), "trajectory_id": tid})
                    else:
                        result[tid] = record
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                problems.append({"error": str(exc), "path": str(path)})
        return result
    deterministic = evidence([root / "10_final_deterministic/pass_cases.jsonl", root / "10_final_deterministic/repair_cases.jsonl"])
    semantic = evidence([root / "11_final_semantic/all_cases.jsonl"])
    tasks = evidence([root / "08_repair_queue/repair_tasks.jsonl"])
    reports = {}
    for kind, path, records, status_key, total_key, statuses in (
        ("deterministic", root / "10_final_deterministic/validation_report.json", deterministic,
         "status", "num_cases_processed", ("pass", "repair")),
        ("semantic", root / "11_final_semantic/semantic_validation_report.json", semantic,
         "semantic_status", "num_cases_seen", ("pass", "uncertain", "fail", "validator_error")),
    ):
        try:
            report = read(path)
            counts = report["summary"]
            if any(row.get(status_key) not in statuses for row in records.values()):
                raise ValueError("Unknown validation status in case evidence")
            if counts[total_key] != len(records):
                raise ValueError("Validation report total disagrees with case evidence")
            for status in statuses:
                if counts["num_" + status] != sum(r.get(status_key) == status for r in records.values()):
                    raise ValueError("Validation report status counts disagree with case evidence")
            reports[kind] = report
        except (ValueError, OSError, KeyError, TypeError) as exc:
            problems.append({"error": str(exc), "path": str(path)})
    for tid, (_, annotation) in final.items():
        if tid not in completed or completed[tid][1] != annotation:
            problems.append({"error": "final_differs_from_validated_completed_annotation", "trajectory_id": tid})

    failed_normalization = {}
    normalization_summary = root / "01_normalized/summary.json"
    if normalization_summary.exists():
        failed_normalization = {r["source_path"]: r.get("error", "normalization failed") for r in read(normalization_summary).get("failed", [])}
    for row in selected:
        tid = row.get("trajectory_id")
        d, s, task = deterministic.get(tid, {}), semantic.get(tid, {}), tasks.get(tid, {})
        candidate = final.get(tid) or completed.get(tid)
        history = {"normalized": str(normalized[tid][0]) if tid in normalized else None,
                   "repair_task": task or None, "deterministic": d or None, "semantic": s or None}
        if row.get("input_error"):
            outcome, reason = "failed", row["input_error"]
        elif row.get("snapshot") in failed_normalization:
            outcome, reason = "failed", failed_normalization[row["snapshot"]]
        elif candidate:
            path, annotation = candidate
            history["annotation"] = str(path)
            if annotation.get("_repair_seed"):
                outcome, reason = "failed", "repair_seed_not_replaced"
            elif s.get("semantic_status") == "validator_error":
                outcome, reason = "failed", "final_semantic_validator_error"
            elif not d or not s:
                outcome, reason = "needs_review", "missing_final_validation_evidence"
            elif d.get("status") != "pass" or s.get("semantic_status") != "pass" or annotation.get("needs_review") is True:
                outcome, reason = "needs_review", "final_validation_or_annotation_requires_review"
            elif annotation.get("target_status") == "no_target" and tid not in final:
                outcome, reason = "excluded", "validated_no_target"
            elif tid in final and annotation.get("target_status") in {"completed_unsafe_action", "high_risk_unsafe_fallback"}:
                outcome, reason = "finalized", "final_output_and_both_validations_passed"
            else:
                outcome, reason = "needs_review", "inconsistent_final_output_or_target_status"
        elif task.get("recommended_action") == "blocked":
            outcome, reason = task["disposition"], task["reason"]
        elif pipeline_error:
            outcome, reason = "failed", "pipeline_interrupted_before_case_resolution"
        else:
            outcome, reason = "unknown", "no_final_output_or_explicit_disposition"
        outcomes.append({**row, "outcome": outcome, "reason": reason, "evidence": history})
    counts = dict(Counter(row["outcome"] for row in outcomes))
    if problems or pipeline_error or counts.get("failed") or counts.get("unknown") or not selected:
        status, code = "failed", 2
    elif counts.get("needs_review"):
        status, code = "completed_with_review", 1
    else:
        status, code = "ok", 0
    summary = {"schema_version": "unsafe_case_coverage_v1", "status": status, "returncode": code,
               "selected": len(selected), "accounted": len(outcomes), "counts": counts,
               "problems": problems, "pipeline_error": pipeline_error, "final_quality": reports}
    write_rows(root / "case_outcomes.jsonl", outcomes)
    write(root / "coverage_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.run_root)["returncode"])
