#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VALIDATION_SCHEMA_VERSION = "safety_refusal_primary_validation_v1"
SUMMARY_SCHEMA_VERSION = "safety_refusal_primary_validation_summary_v1"
EXPECTED_INPUT_SCHEMA_VERSION = "safety_refusal_primary_root_cause_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministically validate per-case safety-refusal primary-root-cause annotations.")
    p.add_argument("--cases_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--glob", default="*.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report_only", action="store_true", help="Write quality findings without exit 1 for review; runtime/input errors still fail. For orchestrators that aggregate the report.")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_case(case_path: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    metrics = {"primary_index": None, "target_index": None, "distance": None}

    try:
        ann = load_json(case_path)
        checks["annotation_json_valid"] = True
    except Exception as exc:
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "trajectory_id": case_path.stem,
            "status": "review",
            "input_path": str(case_path),
            "source_path": None,
            "target_component_id": None,
            "primary_component_id": None,
            "checks": {"annotation_json_valid": False},
            "review_reasons": [f"Could not parse annotation JSON: {type(exc).__name__}: {exc}"],
            "metrics": metrics,
        }

    if not isinstance(ann, dict):
        ann = {}
        checks["annotation_is_object"] = False
        reasons.append("Annotation JSON is not an object.")
    else:
        checks["annotation_is_object"] = True

    required = [
        "schema_version", "trajectory_id", "source_path", "label",
        "target_component_id", "target_summary", "primary_attribution_component",
        "assistant_generated_cause", "confidence", "needs_review", "review_reason",
    ]
    missing = [k for k in required if k not in ann]
    checks["required_fields_present"] = not missing
    if missing:
        reasons.append(f"Missing required fields: {missing}")

    checks["schema_version_valid"] = ann.get("schema_version") == EXPECTED_INPUT_SCHEMA_VERSION
    if not checks["schema_version_valid"]:
        reasons.append(f"Unexpected schema_version: {ann.get('schema_version')!r}")

    trajectory_id = ann.get("trajectory_id") if isinstance(ann.get("trajectory_id"), str) else case_path.stem
    checks["trajectory_id_valid"] = isinstance(ann.get("trajectory_id"), str) and bool(ann.get("trajectory_id").strip())
    if not checks["trajectory_id_valid"]:
        reasons.append("Missing or invalid trajectory_id.")

    checks["filename_matches_trajectory_id"] = case_path.stem == trajectory_id
    if not checks["filename_matches_trajectory_id"]:
        reasons.append(f"Filename {case_path.stem!r} does not match trajectory_id {trajectory_id!r}.")

    checks["label_valid"] = ann.get("label") == "safety_refusal"
    if not checks["label_valid"]:
        reasons.append(f"Expected label='safety_refusal', found {ann.get('label')!r}.")

    target_id = ann.get("target_component_id") if isinstance(ann.get("target_component_id"), str) else None
    checks["target_component_id_valid"] = bool(target_id)
    if not checks["target_component_id_valid"]:
        reasons.append("Missing or invalid target_component_id.")

    primary_block = ann.get("primary_attribution_component")
    checks["primary_block_valid"] = isinstance(primary_block, dict)
    if not isinstance(primary_block, dict):
        primary_block = {}
        reasons.append("primary_attribution_component is missing or invalid.")

    primary_id = primary_block.get("component_id") if isinstance(primary_block.get("component_id"), str) else None
    primary_role = primary_block.get("role") if isinstance(primary_block.get("role"), str) else None
    primary_summary = primary_block.get("summary")

    checks["primary_component_id_valid"] = bool(primary_id)
    if not checks["primary_component_id_valid"]:
        reasons.append("Missing or invalid primary component_id.")

    checks["primary_role_valid"] = primary_role in {"user", "assistant", "tool", "system"}
    if not checks["primary_role_valid"]:
        reasons.append(f"Missing or invalid primary role: {primary_role!r}.")

    checks["primary_summary_valid"] = isinstance(primary_summary, str) and bool(primary_summary.strip())
    if not checks["primary_summary_valid"]:
        reasons.append("Primary summary is missing or empty.")

    target_summary = ann.get("target_summary")
    checks["target_summary_valid"] = target_summary is None or (isinstance(target_summary, str) and bool(target_summary.strip()))
    if not checks["target_summary_valid"]:
        reasons.append("target_summary must be a non-empty string or null.")

    assistant_generated = ann.get("assistant_generated_cause")
    checks["assistant_generated_type_valid"] = isinstance(assistant_generated, bool)
    if not checks["assistant_generated_type_valid"]:
        reasons.append("assistant_generated_cause must be boolean.")

    confidence = ann.get("confidence")
    checks["confidence_valid"] = is_number(confidence) and math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0
    if not checks["confidence_valid"]:
        reasons.append(f"confidence must be in [0,1], found {confidence!r}.")

    needs_review = ann.get("needs_review")
    review_reason = ann.get("review_reason")
    checks["review_fields_valid"] = (
        isinstance(needs_review, bool)
        and ((needs_review and isinstance(review_reason, str) and bool(review_reason.strip())) or (not needs_review and review_reason is None))
    )
    if not checks["review_fields_valid"]:
        reasons.append("Review fields are inconsistent.")
    if needs_review is True:
        reasons.append(f"Annotation requests review: {review_reason or 'reason not provided'}")

    source_value = ann.get("source_path") if isinstance(ann.get("source_path"), str) else None
    checks["source_path_field_valid"] = bool(source_value)
    if not checks["source_path_field_valid"]:
        reasons.append("Missing or invalid source_path.")

    original = None
    if source_value:
        source_path = Path(source_value)
        checks["source_path_exists"] = source_path.is_file()
        if not checks["source_path_exists"]:
            reasons.append(f"source_path does not exist: {source_path}")
        else:
            try:
                original = load_json(source_path)
                checks["source_json_valid"] = True
            except Exception as exc:
                checks["source_json_valid"] = False
                reasons.append(f"Could not parse source JSON: {type(exc).__name__}: {exc}")

    if original is not None:
        checks["source_is_object"] = isinstance(original, dict)
        if not isinstance(original, dict):
            reasons.append("Source trajectory JSON is not an object.")
        else:
            checks["source_trajectory_id_match"] = original.get("trajectory_id") == trajectory_id
            if not checks["source_trajectory_id_match"]:
                reasons.append(f"Source trajectory_id {original.get('trajectory_id')!r} does not match {trajectory_id!r}.")

            trajectory = original.get("trajectory")
            checks["trajectory_list_valid"] = isinstance(trajectory, list)
            if not isinstance(trajectory, list):
                reasons.append("Source file has no valid trajectory list.")
            else:
                cmap: dict[str, dict[str, Any]] = {}
                pos: dict[str, int] = {}
                dup: list[str] = []
                for i, comp in enumerate(trajectory):
                    if not isinstance(comp, dict):
                        continue
                    cid = comp.get("component_id")
                    if not isinstance(cid, str):
                        continue
                    if cid in cmap:
                        dup.append(cid)
                        continue
                    cmap[cid] = comp
                    pos[cid] = i

                checks["component_ids_unique"] = not dup
                if dup:
                    reasons.append(f"Duplicate component IDs: {sorted(set(dup))}")

                checks["target_exists"] = target_id in cmap if target_id else False
                checks["primary_exists"] = primary_id in cmap if primary_id else False
                if not checks["target_exists"]:
                    reasons.append(f"Target component {target_id!r} not found.")
                if not checks["primary_exists"]:
                    reasons.append(f"Primary component {primary_id!r} not found.")

                if checks["target_exists"]:
                    metrics["target_index"] = pos[target_id]
                    target_role = cmap[target_id].get("role")
                    checks["target_role_is_assistant"] = target_role == "assistant"
                    if not checks["target_role_is_assistant"]:
                        reasons.append(f"Target role is {target_role!r}, expected 'assistant'.")

                if checks["primary_exists"]:
                    metrics["primary_index"] = pos[primary_id]
                    actual_role = cmap[primary_id].get("role")
                    checks["primary_role_match"] = actual_role == primary_role
                    if not checks["primary_role_match"]:
                        reasons.append(f"Primary role mismatch: annotation={primary_role!r}, source={actual_role!r}.")

                if checks["target_exists"] and checks["primary_exists"]:
                    ti = pos[target_id]
                    pi = pos[primary_id]
                    checks["primary_not_target"] = primary_id != target_id
                    checks["primary_before_target"] = pi < ti
                    metrics["distance"] = ti - pi
                    if not checks["primary_not_target"]:
                        reasons.append("Primary component equals target component.")
                    if not checks["primary_before_target"]:
                        reasons.append(f"Primary {primary_id} does not appear before target {target_id}.")

                if primary_role is not None and isinstance(assistant_generated, bool):
                    expected = primary_role == "assistant"
                    checks["assistant_generated_cause_match"] = assistant_generated == expected
                    if not checks["assistant_generated_cause_match"]:
                        reasons.append("assistant_generated_cause is inconsistent with primary role.")

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "status": "pass" if not reasons else "review",
        "input_path": str(case_path),
        "source_path": source_value,
        "target_component_id": target_id,
        "primary_component_id": primary_id,
        "checks": checks,
        "review_reasons": reasons,
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    if not args.cases_dir.is_dir():
        print(f"ERROR: cases_dir not found: {args.cases_dir}", file=sys.stderr)
        return 2

    paths = sorted(p for p in args.cases_dir.glob(args.glob) if p.is_file())
    if not paths:
        print(f"ERROR: no files matching {args.glob!r} in {args.cases_dir}", file=sys.stderr)
        return 2

    pass_dir = args.output_dir / "pass"
    review_dir = args.output_dir / "review"
    pass_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    distances: list[int] = []
    review_ids: list[str] = []

    for i, path in enumerate(paths, 1):
        result = validate_case(path)
        tid = result.get("trajectory_id") or path.stem
        status = result["status"]
        out = (pass_dir if status == "pass" else review_dir) / f"{tid}.json"

        if out.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            print(f"[{i}/{len(paths)}] {tid} SKIP existing")
            continue

        save_json(out, result)
        counts[status] += 1
        if status == "review":
            review_ids.append(str(tid))

        try:
            ann = load_json(path)
            role = ann.get("primary_attribution_component", {}).get("role")
            if isinstance(role, str):
                roles[role] += 1
        except Exception:
            pass

        d = result.get("metrics", {}).get("distance")
        if isinstance(d, int):
            distances.append(d)

        print(f"[{i}/{len(paths)}] {tid} status={status}")

    distance_summary = {
        "count": len(distances),
        "min": min(distances) if distances else None,
        "median": statistics.median(distances) if distances else None,
        "mean": statistics.mean(distances) if distances else None,
        "max": max(distances) if distances else None,
    }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "cases_dir": str(args.cases_dir),
        "input_cases": len(paths),
        "pass": counts["pass"],
        "review": counts["review"],
        "skipped_existing": counts["skipped_existing"],
        "pass_dir": str(pass_dir),
        "review_dir": str(review_dir),
        "review_case_ids": review_ids,
        "primary_role_counts": dict(sorted(roles.items())),
        "distance": distance_summary,
    }
    save_json(args.output_dir / "summary.json", summary)

    print("\n" + "=" * 80)
    print("Safety-refusal primary deterministic validation completed")
    print("=" * 80)
    print(f"Input cases      : {len(paths)}")
    print(f"Pass             : {counts['pass']}")
    print(f"Review           : {counts['review']}")
    print(f"Skipped existing : {counts['skipped_existing']}")
    print(f"Pass directory   : {pass_dir}")
    print(f"Review directory : {review_dir}")
    print(f"Summary          : {args.output_dir / 'summary.json'}")
    print("=" * 80)

    return 1 if counts["review"] and not args.report_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
