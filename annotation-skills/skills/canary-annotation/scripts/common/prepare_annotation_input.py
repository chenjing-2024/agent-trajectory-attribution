#!/usr/bin/env python3
"""Normalize raw item/turns cases, or validate normalized cases without renumbering."""

import argparse
import json
from pathlib import Path
import re

from normalize_components import normalize_single_trajectory


def validate_normalized(case):
    if not isinstance(case, dict):
        raise ValueError("Expected a case object")
    tid = case.get("trajectory_id")
    if not isinstance(tid, str) or not tid.strip() or tid in {".", ".."} or any(c in tid for c in "/\\\x00"):
        raise ValueError("trajectory_id must be a nonempty, filename-safe string")
    trajectory = case.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("Expected a nonempty normalized trajectory")
    ids = []
    for component in trajectory:
        if not isinstance(component, dict):
            raise ValueError("Trajectory components must be objects")
        if component.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported component role")
        if not isinstance(component.get("content"), str):
            raise ValueError("Normalized component content must be a string")
        cid = component.get("component_id")
        if cid is None and component["role"] == "system":
            continue
        if not isinstance(cid, str) or not re.fullmatch(r"C(?:0|[1-9]\d*)", cid):
            raise ValueError("Non-system components require C0/C1-style nonnegative component IDs")
        ids.append(cid)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate component IDs")
    metadata = case.get("components_metadata")
    if metadata is not None:
        if not isinstance(metadata, list) or any(not isinstance(c, dict) for c in metadata):
            raise ValueError("Invalid components_metadata")
        if [c.get("component_id") for c in metadata] != ids:
            raise ValueError("components_metadata IDs/order disagree with trajectory")
    return tid


def prepare(source, input_format, destination=None):
    source = Path(source).resolve()
    files = [source] if source.is_file() else sorted(source.rglob("*.json"))
    # Prepared-case directories also contain judgment/summary JSON files.
    if input_format == "raw" and source.is_dir():
        trajectories = [p for p in files if p.name == "trajectory.json"]
        if trajectories:
            files = trajectories
    if not files:
        raise ValueError("No input JSON cases found")
    cases, failures, seen = [], [], set()
    for path in files:
        try:
            value = json.loads(path.read_text())
            records = value.get("results") if input_format == "raw" and isinstance(value, dict) and "results" in value else [value]
            if not isinstance(records, list) or not records:
                raise ValueError("Expected nonempty detailed.json results")
            for value in records:
                if input_format == "raw":
                    if not isinstance(value, dict) or not isinstance(value.get("item"), dict) or not isinstance(value.get("turns"), list):
                        raise ValueError("Raw case requires item object and turns list; normalized input uses --input_format normalized")
                    value = normalize_single_trajectory(value, str(path))
                tid = validate_normalized(value)
                if tid in seen:
                    raise ValueError(f"Duplicate trajectory_id: {tid}")
                seen.add(tid)
                cases.append(value)
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            failures.append({"source": str(path), "error": str(exc)})
    report = {"status": "failed" if failures else "ok", "input": str(source),
              "input_format": input_format, "source_files": len(files),
              "validated_cases": len(cases), "failures": failures,
              "component_ids": "assigned from raw turns" if input_format == "raw" else "preserved"}
    if destination:
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if not failures:
            (destination / "cases").mkdir()
            for case in cases:
                (destination / "cases" / f"{case['trajectory_id']}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    if failures:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input_format", choices=["raw", "normalized"], required=True)
    parser.add_argument("--output_root")
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    if args.check_only == bool(args.output_root):
        parser.error("Choose exactly one of --output_root or --check_only")
    print(json.dumps(prepare(args.input, args.input_format, args.output_root), indent=2))


if __name__ == "__main__":
    main()
