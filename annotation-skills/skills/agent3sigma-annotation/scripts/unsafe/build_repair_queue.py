#!/usr/bin/env python3
"""Combine deterministic and semantic findings into one auditable task per input."""
import argparse
from pathlib import Path

from case_tracking import index_cases, read, rows, write, write_rows

ACTIONS = {"keep_annotation", "rerun_all_annotations", "repair_attack_chain",
           "repair_execution_chain", "repair_attack_and_execution_chains"}


def merge_action(hard, semantic_action):
    if not hard:
        return semantic_action if semantic_action in ACTIONS else "rerun_all_annotations"
    # Only known chain-specific failures can safely reuse target and primary.
    if any(not error.startswith(("attack_chain_", "execution_chain_")) for error in hard):
        return "rerun_all_annotations"
    if semantic_action == "rerun_all_annotations":
        return semantic_action
    attack = any(e.startswith("attack_chain_") for e in hard) or semantic_action in {"repair_attack_chain", "repair_attack_and_execution_chains"}
    execution = any(e.startswith("execution_chain_") for e in hard) or semantic_action in {"repair_execution_chain", "repair_attack_and_execution_chains"}
    if attack and execution:
        return "repair_attack_and_execution_chains"
    return "repair_attack_chain" if attack else "repair_execution_chain"


def build(selection, normalized, annotations, deterministic, semantic, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = read(selection)["cases"]
    expected = {r["trajectory_id"] for r in selected}
    trajectories, trajectory_errors = index_cases(normalized)
    combined, annotation_errors = index_cases(annotations)
    if trajectory_errors or any(tid not in expected for tid in trajectories) or any(tid not in expected for tid in combined):
        raise ValueError("Ambiguous or unexpected input/annotation IDs; repair cannot safely proceed")
    # Duplicate annotations must never be replaced by an arbitrary match.
    if any(p.get("error") == "duplicate_id" for p in annotation_errors):
        raise ValueError("Duplicate annotation IDs")
    def keyed(records):
        result = {}
        for record in records:
            tid = record.get("trajectory_id")
            if tid not in expected or tid in result:
                raise ValueError(f"Unexpected or duplicate validator ID: {tid}")
            result[tid] = record
        return result
    det = keyed(rows(Path(deterministic) / "pass_cases.jsonl", True) + rows(Path(deterministic) / "repair_cases.jsonl", True))
    sem = keyed(rows(Path(semantic) / "all_cases.jsonl", True))
    tasks, executable = [], []
    for selected_case in selected:
        tid = selected_case["trajectory_id"]
        d, s = det.get(tid, {}), sem.get(tid, {})
        task = {"trajectory_id": tid, "case_key": selected_case["case_key"],
                "deterministic": d, "semantic": s, "hard_errors": d.get("hard_errors", [])}
        if tid not in trajectories:
            task.update(recommended_action="blocked", reason="missing_normalized_trajectory", disposition="failed")
        elif any(error.startswith("trajectory_") for error in task["hard_errors"]):
            task.update(recommended_action="blocked", reason="invalid_normalized_trajectory", disposition="failed")
        elif s.get("semantic_status") == "validator_error" and not task["hard_errors"]:
            task.update(recommended_action="blocked", reason="semantic_validator_error", disposition="needs_review")
        else:
            norm_path, _ = trajectories[tid]
            annotation = combined.get(tid)
            hard = list(task["hard_errors"])
            if not d or d.get("status") != "pass" and not hard:
                hard.append("missing_or_invalid_deterministic_result")
            if annotation is None:
                hard.append("missing_or_unreadable_annotation")
                # The legacy full-rerun assembler requires a base file. This is
                # explicitly a seed, never an accepted or keepable annotation.
                seed = output / "seed_annotations" / f"{norm_path.stem}.combined.json"
                write(seed, {"trajectory_id": tid, "_repair_seed": True})
                annotation = (seed, {})
            action = merge_action(hard, s.get("recommended_action"))
            if s.get("semantic_status") != "pass" and action == "keep_annotation":
                action = "rerun_all_annotations"
            task.update(hard_errors=hard, recommended_action=action,
                        reason="deterministic_hard_errors_take_precedence" if hard else "semantic_recommendation",
                        case_id=norm_path.stem, files={"annotation": str(annotation[0]), "trajectory": str(norm_path)})
            executable.append(task)
        tasks.append(task)
    write_rows(output / "repair_tasks.jsonl", tasks)
    # Preserve the legacy repair runner's input contract.
    write_rows(output / "all_cases.jsonl", executable)
    write(output / "queue_summary.json", {"selected": len(selected), "executable": len(executable),
          "blocked": len(tasks) - len(executable), "annotation_discovery_errors": annotation_errors})
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "normalized", "annotations", "deterministic", "semantic", "output_root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    build(args.selection, args.normalized, args.annotations, args.deterministic, args.semantic, args.output_root)
