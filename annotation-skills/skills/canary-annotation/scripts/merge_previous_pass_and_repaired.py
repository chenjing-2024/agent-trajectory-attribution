#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge previously passed complete annotations with repaired "
            "complete annotations. Repaired annotations override previous "
            "annotations with the same trajectory_id."
        )
    )

    parser.add_argument(
        "--pass_dir",
        type=Path,
        required=True,
        help="Directory containing previous pass annotation JSON files.",
    )
    parser.add_argument(
        "--repaired_dir",
        type=Path,
        required=True,
        help="Directory containing repaired merged annotation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root. Final annotations are written to output_dir/cases.",
    )
    parser.add_argument(
        "--expected_pass",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expected_repaired",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expected_final",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--required_target_type",
        default="unsafe_action",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected JSON object at {path}, "
            f"got {type(obj).__name__}"
        )

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def discover_annotation_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)

    files = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
    )

    annotation_files = []

    for path in files:
        try:
            obj = load_json(path)
        except Exception:
            continue

        if not isinstance(obj.get("trajectory_id"), str):
            continue

        if not isinstance(obj.get("target_action"), dict):
            continue

        if not isinstance(
            obj.get("primary_attribution_component"),
            dict,
        ):
            continue

        if not isinstance(obj.get("attack_chain"), list):
            continue

        if not isinstance(obj.get("execution_chain"), list):
            continue

        annotation_files.append(path)

    return annotation_files


def validate_annotation(
    obj: dict[str, Any],
    required_target_type: str | None,
) -> list[str]:
    errors = []

    trajectory_id = obj.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        errors.append("missing trajectory_id")

    if (
        required_target_type is not None
        and obj.get("target_type") != required_target_type
    ):
        errors.append(
            f"target_type={obj.get('target_type')!r}, "
            f"expected {required_target_type!r}"
        )

    target_action = obj.get("target_action")
    if not isinstance(target_action, dict):
        errors.append("missing target_action")
    elif not isinstance(target_action.get("component_id"), str):
        errors.append("missing target_action.component_id")

    primary = obj.get("primary_attribution_component")
    if not isinstance(primary, dict):
        errors.append("missing primary_attribution_component")
    elif not isinstance(primary.get("component_id"), str):
        errors.append(
            "missing primary_attribution_component.component_id"
        )

    if not isinstance(obj.get("attack_chain"), list):
        errors.append("missing attack_chain list")

    if not isinstance(obj.get("execution_chain"), list):
        errors.append("missing execution_chain list")

    return errors


def safe_filename(
    trajectory_id: str,
    source_path: Path,
) -> str:
    name = source_path.name

    if name.endswith(".annotation.json"):
        return name

    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in trajectory_id
    )

    return f"{safe_id}.annotation.json"


def load_collection(
    root: Path,
    source_name: str,
    required_target_type: str | None,
    excluded: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[str, tuple[Path, dict[str, Any]]],
    list[dict[str, Any]],
]:
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    validation_failures: list[dict[str, Any]] = []

    files = discover_annotation_files(root)

    for path in files:
        obj = load_json(path)
        trajectory_id = str(obj["trajectory_id"]).strip()
        if (excluded is not None and source_name == "repaired" and required_target_type == "unsafe_action"
                and obj.get("target_type") in {"safety_refusal", "no_target"}):
            excluded.append({"trajectory_id": trajectory_id, "source_path": str(path),
                             "reason": "reclassified_" + obj["target_type"], "needs_review": bool(obj.get("needs_review"))})
            continue

        errors = validate_annotation(
            obj,
            required_target_type,
        )

        if errors:
            validation_failures.append({
                "source": source_name,
                "path": str(path),
                "trajectory_id": trajectory_id,
                "errors": errors,
            })
            continue

        if trajectory_id in records:
            first_path, first_obj = records[trajectory_id]

            if first_obj == obj:
                continue

            raise ValueError(
                f"Conflicting duplicate trajectory_id "
                f"{trajectory_id!r} in {source_name}:\n"
                f"First:  {first_path}\n"
                f"Second: {path}"
            )

        records[trajectory_id] = (path, obj)

    return records, validation_failures


def main() -> None:
    args = parse_args()

    cases_dir = args.output_dir / "cases"

    if cases_dir.exists() and args.force:
        shutil.rmtree(cases_dir)

    cases_dir.mkdir(parents=True, exist_ok=True)

    pass_records, pass_failures = load_collection(
        args.pass_dir,
        "previous_pass",
        args.required_target_type,
    )

    excluded_reclassified = []
    repaired_records, repaired_failures = load_collection(
        args.repaired_dir,
        "repaired",
        args.required_target_type,
        excluded_reclassified,
    )
    for excluded in excluded_reclassified:
        pass_records.pop(excluded["trajectory_id"], None)
    (args.output_dir / "excluded_reclassified.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded_reclassified), encoding="utf-8")

    if pass_failures or repaired_failures:
        report = {
            "status": "validation_failed",
            "pass_validation_failures": pass_failures,
            "repaired_validation_failures": repaired_failures,
        }

        write_json(
            args.output_dir / "merge_report.json",
            report,
        )

        raise SystemExit(
            "ERROR: Invalid annotation files found. "
            "See merge_report.json."
        )

    if (
        args.expected_pass is not None
        and len(pass_records) != args.expected_pass
    ):
        raise SystemExit(
            f"ERROR: Previous pass count is {len(pass_records)}, "
            f"expected {args.expected_pass}."
        )

    if (
        args.expected_repaired is not None
        and len(repaired_records) != args.expected_repaired
    ):
        raise SystemExit(
            f"ERROR: Repaired count is {len(repaired_records)}, "
            f"expected {args.expected_repaired}."
        )

    final_records: dict[
        str,
        tuple[str, Path, dict[str, Any]],
    ] = {}

    for trajectory_id, (path, obj) in pass_records.items():
        final_records[trajectory_id] = (
            "previous_pass",
            path,
            obj,
        )

    overridden_ids = []
    new_repaired_ids = []

    for trajectory_id, (path, obj) in repaired_records.items():
        if trajectory_id in final_records:
            overridden_ids.append(trajectory_id)
        else:
            new_repaired_ids.append(trajectory_id)

        final_records[trajectory_id] = (
            "repaired",
            path,
            obj,
        )

    if (
        args.expected_final is not None
        and len(final_records) != args.expected_final
    ):
        raise SystemExit(
            f"ERROR: Final count is {len(final_records)}, "
            f"expected {args.expected_final}.\n"
            f"Previous pass: {len(pass_records)}\n"
            f"Repaired: {len(repaired_records)}\n"
            f"Overrides: {len(overridden_ids)}\n"
            f"New repaired additions: {len(new_repaired_ids)}"
        )

    manifest = []

    for trajectory_id in sorted(final_records):
        source_type, source_path, obj = final_records[trajectory_id]

        output_name = safe_filename(
            trajectory_id,
            source_path,
        )

        output_path = cases_dir / output_name

        if output_path.exists():
            output_name = (
                f"{trajectory_id.replace('/', '_')}"
                ".annotation.json"
            )
            output_path = cases_dir / output_name

        shutil.copy2(source_path, output_path)

        manifest.append({
            "trajectory_id": trajectory_id,
            "source_type": source_type,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "target_component_id": (
                obj.get("target_action") or {}
            ).get("component_id"),
            "primary_component_id": (
                obj.get(
                    "primary_attribution_component"
                ) or {}
            ).get("component_id"),
            "attack_chain_length": len(
                obj.get("attack_chain", [])
            ),
            "execution_chain_length": len(
                obj.get("execution_chain", [])
            ),
        })

    report = {
        "status": "success",
        "inputs": {
            "pass_dir": str(args.pass_dir),
            "repaired_dir": str(args.repaired_dir),
        },
        "counts": {
            "num_previous_pass": len(pass_records),
            "num_repaired": len(repaired_records),
            "num_overridden": len(overridden_ids),
            "num_new_repaired_additions": len(
                new_repaired_ids
            ),
            "num_final": len(final_records),
        },
        "overridden_trajectory_ids": sorted(overridden_ids),
        "new_repaired_trajectory_ids": sorted(
            new_repaired_ids
        ),
        "outputs": {
            "cases_dir": str(cases_dir),
            "manifest": str(
                args.output_dir / "final_manifest.jsonl"
            ),
        },
    }

    write_jsonl(
        args.output_dir / "final_manifest.jsonl",
        manifest,
    )

    write_json(
        args.output_dir / "merge_report.json",
        report,
    )

    print("=" * 80)
    print("Previous pass + repaired merge finished")
    print("=" * 80)
    print(f"Previous pass:          {len(pass_records)}")
    print(f"Repaired:               {len(repaired_records)}")
    print(f"Overrides:              {len(overridden_ids)}")
    print(
        f"New repaired additions: {len(new_repaired_ids)}"
    )
    print(f"Final annotations:      {len(final_records)}")
    print(f"Cases directory:        {cases_dir}")
    print(
        "Report:                 "
        f"{args.output_dir / 'merge_report.json'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
