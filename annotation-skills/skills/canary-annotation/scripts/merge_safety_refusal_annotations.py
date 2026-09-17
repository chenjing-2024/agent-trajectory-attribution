#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(obj).__name__}."
        )

    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def case_key(path: Path) -> str:
    """
    Normalize matching target and primary annotation filenames.

    Examples:
      case.target.json  -> case
      case.primary.json -> case
      case.json         -> case
    """
    name = path.name

    for suffix in (
        ".primary.json",
        ".target.json",
        ".annotation.json",
        ".json",
    ):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return path.stem


def discover_json_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    for path in sorted(directory.glob("*.json")):
        key = case_key(path)

        if key in files:
            raise ValueError(
                f"Duplicate normalized case key {key!r}: "
                f"{files[key]} and {path}"
            )

        files[key] = path

    return files


def merge_one(
    *,
    target_path: Path,
    primary_path: Path,
) -> dict[str, Any]:
    target = load_json(target_path)
    primary = load_json(primary_path)

    target_type = target.get("target_type")
    primary_target_type = primary.get("target_type")

    if target_type != "safety_refusal":
        raise ValueError(
            f"Target annotation is not safety_refusal: "
            f"{target_path} has target_type={target_type!r}"
        )

    if (
        primary_target_type is not None
        and primary_target_type != "safety_refusal"
    ):
        raise ValueError(
            f"Primary annotation has inconsistent target_type: "
            f"{primary_path} has target_type={primary_target_type!r}"
        )

    target_trajectory_id = target.get("trajectory_id")
    primary_trajectory_id = primary.get("trajectory_id")

    if (
        target_trajectory_id is not None
        and primary_trajectory_id is not None
        and str(target_trajectory_id) != str(primary_trajectory_id)
    ):
        raise ValueError(
            "trajectory_id mismatch: "
            f"target={target_trajectory_id!r}, "
            f"primary={primary_trajectory_id!r}"
        )

    target_action = target.get("target_action")
    if not isinstance(target_action, dict):
        raise ValueError(
            f"Missing target_action object in {target_path}"
        )

    primary_component = primary.get(
        "primary_attribution_component"
    )
    if not isinstance(primary_component, dict):
        raise ValueError(
            f"Missing primary_attribution_component object in {primary_path}"
        )

    target_component_id = target_action.get("component_id")
    primary_target_component_id = primary.get("target_component_id")

    if (
        target_component_id is not None
        and primary_target_component_id is not None
        and str(target_component_id) != str(primary_target_component_id)
    ):
        raise ValueError(
            "target component mismatch: "
            f"target={target_component_id!r}, "
            f"primary={primary_target_component_id!r}"
        )

    return {
        "trajectory_id": (
            target_trajectory_id
            if target_trajectory_id is not None
            else primary_trajectory_id
        ),
        "target_type": "safety_refusal",
        "attack_success": target.get(
            "attack_success",
            primary.get("attack_success", False),
        ),
        "target_action": target_action,
        "primary_attribution_component": primary_component,
        "confidence": primary.get("confidence"),
        "needs_review": bool(target.get("needs_review") or primary.get("needs_review")),
        "review_reason": "; ".join(str(r) for r in (target.get("review_reason"), primary.get("review_reason")) if r) or None,
        "_metadata": {
            "target_annotation_path": str(target_path),
            "primary_annotation_path": str(primary_path),
            "review_sources": {"target": {"needs_review": bool(target.get("needs_review")), "reason": target.get("review_reason")}, "primary": {"needs_review": bool(primary.get("needs_review")), "reason": primary.get("review_reason")}},
            "target_metadata": target.get("_metadata", {}),
            "primary_metadata": primary.get("_metadata", {}),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge safety-refusal target annotations with their "
            "primary root-attribution annotations."
        )
    )

    parser.add_argument(
        "--target_dir",
        required=True,
        help="Directory containing safety-refusal *.target.json files.",
    )
    parser.add_argument(
        "--primary_dir",
        required=True,
        help="Directory containing safety-refusal *.primary.json files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for merged *.annotation.json files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged annotation files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_dir = Path(args.target_dir)
    primary_dir = Path(args.primary_dir)
    output_dir = Path(args.output_dir)

    if not target_dir.is_dir():
        raise NotADirectoryError(
            f"Target directory not found: {target_dir}"
        )

    if not primary_dir.is_dir():
        raise NotADirectoryError(
            f"Primary directory not found: {primary_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    target_files = discover_json_files(target_dir)
    primary_files = discover_json_files(primary_dir)

    all_keys = sorted(set(target_files) | set(primary_files))

    merged_count = 0
    skipped_existing = 0
    missing_target: list[str] = []
    missing_primary: list[str] = []
    failures: list[dict[str, str]] = []

    for index, key in enumerate(all_keys, 1):
        target_path = target_files.get(key)
        primary_path = primary_files.get(key)

        if target_path is None:
            missing_target.append(key)
            print(f"[{index}/{len(all_keys)}] MISSING TARGET: {key}")
            continue

        if primary_path is None:
            missing_primary.append(key)
            print(f"[{index}/{len(all_keys)}] MISSING PRIMARY: {key}")
            continue

        output_path = output_dir / f"{key}.annotation.json"

        if output_path.exists() and not args.force:
            skipped_existing += 1
            print(
                f"[{index}/{len(all_keys)}] SKIP EXISTING: "
                f"{output_path.name}"
            )
            continue

        try:
            merged = merge_one(
                target_path=target_path,
                primary_path=primary_path,
            )
            write_json(output_path, merged)
            merged_count += 1

            print(
                f"[{index}/{len(all_keys)}] OK: "
                f"{output_path.name}"
            )

        except Exception as exc:
            failures.append({
                "case_key": key,
                "target_path": str(target_path),
                "primary_path": str(primary_path),
                "error": repr(exc),
            })

            print(
                f"[{index}/{len(all_keys)}] FAILED: "
                f"{key}: {exc!r}"
            )

    report = {
        "target_dir": str(target_dir),
        "primary_dir": str(primary_dir),
        "output_dir": str(output_dir),
        "num_target_files": len(target_files),
        "num_primary_files": len(primary_files),
        "num_merged": merged_count,
        "num_skipped_existing": skipped_existing,
        "num_missing_target": len(missing_target),
        "num_missing_primary": len(missing_primary),
        "num_failures": len(failures),
        "missing_target": missing_target,
        "missing_primary": missing_primary,
        "failures": failures,
    }

    write_json(
        output_dir.parent / "merge_report.json",
        report,
    )

    print("=" * 80)
    print("Safety-refusal annotation merge finished")
    print("=" * 80)
    print(f"Target files:       {len(target_files)}")
    print(f"Primary files:      {len(primary_files)}")
    print(f"Merged:             {merged_count}")
    print(f"Skipped existing:   {skipped_existing}")
    print(f"Missing target:     {len(missing_target)}")
    print(f"Missing primary:    {len(missing_primary)}")
    print(f"Failures:           {len(failures)}")
    print(f"Output directory:   {output_dir}")
    print(
        "Report:             "
        f"{output_dir.parent / 'merge_report.json'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
