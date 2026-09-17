#!/usr/bin/env python3
"""Freeze the selected input set before normalization; apply --limit exactly once."""
import argparse
from collections import Counter
from pathlib import Path
import sys

from case_tracking import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from prepare_cases import map_case_result
from prepare_annotation_input import validate_normalized


def select(source, output, limit=None, input_format="raw"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    if input_format == "normalized":
        paths = [source] if source.is_file() else sorted(source.rglob("*.json"))
        for path in paths:
            try:
                candidates.append((str(path), read(path), None))
            except (ValueError, OSError) as exc:
                candidates.append((str(path), None, str(exc)))
    elif source.is_file():
        payload = read(source)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("File input must be detailed.json with a results list")
        for number, value in enumerate(payload["results"]):
            if map_case_result(value)["case_result"] == "attack_success":
                candidates.append((f"{source}#results[{number}]", value, None))
    else:
        for path in sorted(source.rglob("trajectory.json")):
            try:
                candidates.append((str(path), read(path), None))
            except (ValueError, OSError) as exc:
                candidates.append((str(path), None, str(exc)))
    # Detailed exports retain their record order; prepared cases use path order.
    selected = candidates if limit is None else candidates[:limit]
    records = []
    for number, (origin, value, error) in enumerate(selected):
        tid = value.get("item", {}).get("id") if isinstance(value, dict) and isinstance(value.get("item"), dict) else None
        if input_format == "normalized" and value is not None:
            tid = value.get("trajectory_id") if isinstance(value, dict) else None
            try:
                validate_normalized(value)
            except (ValueError, TypeError, AttributeError) as exc:
                error = str(exc)
        if not isinstance(tid, str) or not tid or any(c in tid for c in "/\\\x00") or tid in {".", ".."}:
            error = error or "Missing or unsafe item.id"
            tid = None
        records.append({"case_key": f"case-{number:06d}", "trajectory_id": tid,
                        "source": origin, "input_error": error})
    duplicate_ids = {tid for tid, count in Counter(r["trajectory_id"] for r in records if r["trajectory_id"]).items() if count > 1}
    for row, (_, value, _) in zip(records, selected):
        if row["trajectory_id"] in duplicate_ids:
            row["input_error"] = "Duplicate input trajectory_id"
        if not row["input_error"]:
            snapshot = output / "cases" / row["case_key"] / "trajectory.json"
            write(snapshot, value)
            row["snapshot"] = str(snapshot)
    write(output / "selected_cases.json", {"schema_version": "unsafe_selected_cases_v1", "source": str(source), "input_format": input_format,
          "limit": limit, "eligible_count": len(candidates), "selected_count": len(records), "cases": records})
    if duplicate_ids or any(r["input_error"] for r in records):
        raise ValueError("Selected inputs have invalid/duplicate IDs or unreadable JSON; see selected_cases.json")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input_format", choices=("raw", "normalized"), default="raw")
    args = parser.parse_args()
    select(args.input, args.output_root, args.limit, args.input_format)
