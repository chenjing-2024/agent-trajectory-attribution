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
            "Merge existing safety-refusal annotations with cases "
            "reclassified as safety_refusal during unsafe-action repair."
        )
    )

    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--primary_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--expected_base",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expected_final",
        type=int,
        default=None,
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
        raise ValueError(f"Expected JSON object at {path}")

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def discover_records(
    root: Path,
    required_fields: tuple[str, ...] = (),
) -> dict[str, tuple[Path, dict[str, Any]]]:
    records: dict[str, tuple[Path, dict[str, Any]]] = {}

    if not root.is_dir():
        raise NotADirectoryError(root)

    for path in sorted(root.rglob("*.json")):
        try:
            obj = load_json(path)
        except Exception:
            continue

        trajectory_id = obj.get("trajectory_id")

        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            continue

        if any(field not in obj for field in required_fields):
            continue

        trajectory_id = trajectory_id.strip()

        if trajectory_id in records:
            previous_path, previous_obj = records[trajectory_id]

            if previous_obj == obj:
                continue

            raise ValueError(
                f"Conflicting duplicate trajectory_id "
                f"{trajectory_id!r}:\n"
                f"First:  {previous_path}\n"
                f"Second: {path}"
            )

        records[trajectory_id] = (path, obj)

    return records


def output_name(
    trajectory_id: str,
    source_path: Path,
) -> str:
    name = source_path.name

    for suffix in (
        ".target.json",
        ".primary.json",
        ".annotation.json",
        ".json",
    ):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if base:
                return f"{base}.annotation.json"

    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in trajectory_id
    )

    return f"{safe_id}.annotation.json"


def main() -> None:
    args = parse_args()

    cases_dir = args.output_dir / "cases"

    if cases_dir.exists() and args.force:
        shutil.rmtree(cases_dir)

    cases_dir.mkdir(parents=True, exist_ok=True)

    base_records = discover_records(
        args.base_dir,
        required_fields=(
            "target_action",
            "primary_attribution_component",
        ),
    )

    target_records = discover_records(
        args.target_dir,
        required_fields=("target_type",),
    )

    primary_records = discover_records(
        args.primary_dir,
        required_fields=("primary_attribution_component",),
    )

    base_safety_records = {
        trajectory_id: value
        for trajectory_id, value in base_records.items()
        if value[1].get("target_type") == "safety_refusal"
    }

    if (
        args.expected_base is not None
        and len(base_safety_records) != args.expected_base
    ):
        raise SystemExit(
            f"ERROR: Base safety-refusal count is "
            f"{len(base_safety_records)}, expected "
            f"{args.expected_base}."
        )

    final_records: dict[
        str,
        tuple[str, Path, dict[str, Any]],
    ] = {}

    for trajectory_id, (path, obj) in base_safety_records.items():
        final_records[trajectory_id] = (
            "previous_safety_refusal",
            path,
            obj,
        )

    additions: list[str] = []
    skipped_invalid: list[dict[str, Any]] = []
    missing_primary: list[dict[str, Any]] = []

    for trajectory_id, (target_path, target) in target_records.items():
        if target.get("target_type") != "safety_refusal":
            continue

        primary_entry = primary_records.get(trajectory_id)

        if primary_entry is None:
            missing_primary.append({
                "trajectory_id": trajectory_id,
                "target_path": str(target_path),
            })
            continue

        primary_path, primary = primary_entry

        target_action = target.get("target_action")
        primary_component = primary.get(
            "primary_attribution_component"
        )

        if not isinstance(target_action, dict):
            skipped_invalid.append({
                "trajectory_id": trajectory_id,
                "reason": "missing_target_action",
                "target_path": str(target_path),
            })
            continue

        if not isinstance(primary_component, dict):
            skipped_invalid.append({
                "trajectory_id": trajectory_id,
                "reason": "missing_primary_component",
                "primary_path": str(primary_path),
            })
            continue

        confidence_values = [
            float(value)
            for value in (
                target.get("confidence"),
                primary.get("confidence"),
            )
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]

        merged = {
            "trajectory_id": trajectory_id,
            "target_type": "safety_refusal",
            "attack_success": target.get(
                "attack_success",
                False,
            ),
            "target_action": target_action,
            "primary_attribution_component": primary_component,
            "confidence": (
                min(confidence_values)
                if confidence_values
                else None
            ),
            "needs_review": bool(
                target.get("needs_review", False)
                or primary.get("needs_review", False)
            ),
            "review_reason": (
                target.get("review_reason")
                or primary.get("review_reason")
            ),
            "_metadata": {
                "reclassified_from": (
                    "unsafe_action_repair_set"
                ),
                "target_annotation_path": str(target_path),
                "primary_annotation_path": str(primary_path),
                "target_metadata": target.get(
                    "_metadata",
                    {},
                ),
                "primary_metadata": primary.get(
                    "_metadata",
                    {},
                ),
            },
        }

        final_records[trajectory_id] = (
            "reclassified_safety_refusal",
            target_path,
            merged,
        )

        additions.append(trajectory_id)

    if missing_primary:
        raise SystemExit(
            "ERROR: Reclassified safety-refusal targets are "
            "missing primary annotations:\n"
            + "\n".join(
                item["trajectory_id"]
                for item in missing_primary
            )
        )

    if skipped_invalid:
        raise SystemExit(
            "ERROR: Invalid reclassified safety-refusal "
            f"annotations: {skipped_invalid}"
        )

    if (
        args.expected_final is not None
        and len(final_records) != args.expected_final
    ):
        raise SystemExit(
            f"ERROR: Final safety-refusal count is "
            f"{len(final_records)}, expected "
            f"{args.expected_final}.\n"
            f"Base: {len(base_safety_records)}\n"
            f"Reclassified additions: {len(additions)}"
        )

    manifest: list[dict[str, Any]] = []

    for trajectory_id in sorted(final_records):
        source_type, source_path, obj = final_records[
            trajectory_id
        ]

        destination = (
            cases_dir
            / output_name(
                trajectory_id,
                source_path,
            )
        )

        write_json(destination, obj)

        manifest.append({
            "trajectory_id": trajectory_id,
            "source_type": source_type,
            "source_path": str(source_path),
            "output_path": str(destination),
            "target_component_id": (
                obj.get("target_action") or {}
            ).get("component_id"),
            "primary_component_id": (
                obj.get(
                    "primary_attribution_component"
                ) or {}
            ).get("component_id"),
        })

    manifest_path = args.output_dir / "final_manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(
                json.dumps(item, ensure_ascii=False)
                + "\n"
            )

    report = {
        "status": "success",
        "counts": {
            "num_previous_safety_refusal": len(
                base_safety_records
            ),
            "num_reclassified_additions": len(additions),
            "num_final": len(final_records),
        },
        "reclassified_trajectory_ids": sorted(additions),
        "outputs": {
            "cases_dir": str(cases_dir),
            "manifest": str(manifest_path),
        },
    }

    write_json(
        args.output_dir / "merge_report.json",
        report,
    )

    print("=" * 80)
    print("Final safety-refusal merge finished")
    print("=" * 80)
    print(
        "Previous safety refusal: "
        f"{len(base_safety_records)}"
    )
    print(
        "Reclassified additions:  "
        f"{len(additions)}"
    )
    print(
        "Final safety refusal:    "
        f"{len(final_records)}"
    )
    print(f"Cases directory:        {cases_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
