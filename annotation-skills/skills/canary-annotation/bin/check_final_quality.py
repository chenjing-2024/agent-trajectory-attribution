#!/usr/bin/env python3
"""Turn final validation report counts into an explicit status and exit code."""

import argparse
import json
from pathlib import Path


SCHEMAS = {
    "refusal": "safety_refusal_deterministic_validation_report_v1",
    "unsafe_deterministic": "unsafe_action_deterministic_validation_report_v3",
    "unsafe_semantic": "unsafe_action_semantic_validation_report_v3_root_cause",
}


def read_summary(path, kind):
    report = json.loads(Path(path).read_text())
    if kind in SCHEMAS and report.get("schema_version") != SCHEMAS[kind]:
        raise ValueError(f"Unexpected {kind} report schema")
    summary = report["summary"]
    total, counts = summary["num_cases"], summary["status_counts"]
    if type(total) is not int or total < 0 or not isinstance(counts, dict):
        raise ValueError("Invalid total/counts in validation report")
    allowed = {"pass", "fail", "validator_error"}
    allowed |= {"warning", "review"} if kind == "unsafe_deterministic" else {"uncertain"} if kind == "unsafe_semantic" else {"review"}
    if set(counts) - allowed or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Invalid validation status or count")
    if sum(counts.values()) != total:
        raise ValueError("Validation counts do not add up to num_cases")
    return summary


def evaluate(deterministic, semantic=None, kind="refusal"):
    summaries = {"deterministic": read_summary(deterministic, "unsafe_deterministic" if kind == "unsafe" else "refusal")}
    if kind == "unsafe":
        if semantic is None:
            raise ValueError("Unsafe final quality requires a semantic report")
        summaries["semantic"] = read_summary(semantic, "unsafe_semantic")
        if summaries["semantic"]["num_cases"] != summaries["deterministic"]["num_cases"]:
            raise ValueError("Final deterministic and semantic case counts differ")
    unresolved = {stage: {key: value for key, value in summary["status_counts"].items()
                          if key not in {"pass", "warning"} and value}
                  for stage, summary in summaries.items()}
    if any(s["status_counts"].get("validator_error", 0) for s in summaries.values()):
        status, code = "failed", 2
    elif any(unresolved.values()):
        status, code = "completed_with_review", 1
    elif summaries["deterministic"]["num_cases"] == 0:
        status, code = "no_eligible_cases", 0
    else:
        status, code = "ok", 0
    return {"status": status, "returncode": code, "summaries": summaries, "unresolved": unresolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["unsafe", "refusal"], required=True)
    parser.add_argument("--deterministic", required=True)
    parser.add_argument("--semantic")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.deterministic, args.semantic, args.kind)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        result = {"status": "failed", "returncode": 2, "error": str(exc)}
    result["schema_version"] = "canary_final_quality_v1"
    result["kind"] = args.kind
    result["reports"] = {"deterministic": args.deterministic, "semantic": args.semantic}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Final quality: {result['status']} (exit {result['returncode']})\nReport: {destination}")
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
