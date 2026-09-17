#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


# =============================================================================
# Basic utilities
# =============================================================================


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final Agent3Sigma annotations by filtering "
            "no_target cases from completed annotations."
        )
    )

    parser.add_argument(
        "--run_root",
        type=Path,
        required=True,
        help=(
            "Repair run root containing "
            "completed_annotations/cases."
        ),
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete the existing final_annotations directory "
            "before rebuilding it."
        ),
    )

    parser.add_argument("--allow_empty", action="store_true", help="Allow an empty candidate set; the full runner must still audit all selected cases.")
    return parser.parse_args()


# =============================================================================
# Final annotation construction
# =============================================================================


def build_final_annotations(
    *,
    run_root: Path,
    clean: bool,
    allow_empty: bool = False,
) -> dict[str, Any]:
    completed_cases_dir = (
        run_root / "completed_annotations" / "cases"
    )
    final_root = run_root / "final_annotations"
    final_cases_dir = final_root / "cases"
    summary_path = final_root / "filter_summary.json"

    if not completed_cases_dir.is_dir():
        raise NotADirectoryError(
            "Completed annotation directory not found: "
            f"{completed_cases_dir}"
        )

    if clean and final_root.exists():
        shutil.rmtree(final_root)

    final_cases_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(
        completed_cases_dir.glob("*.combined.json")
    )

    if not input_paths and not allow_empty:
        raise FileNotFoundError(
            "No completed combined annotations found in: "
            f"{completed_cases_dir}"
        )

    kept: list[dict[str, Any]] = []
    excluded_no_target: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    expected_output_names: set[str] = set()

    for source_path in input_paths:
        try:
            annotation = load_json(source_path)
        except Exception as exc:
            invalid.append(
                {
                    "file": source_path.name,
                    "error": str(exc),
                }
            )
            continue

        trajectory_id = annotation.get("trajectory_id")
        target_status = annotation.get("target_status")

        if target_status == "no_target":
            excluded_no_target.append(
                {
                    "file": source_path.name,
                    "trajectory_id": trajectory_id,
                    "target_status": target_status,
                    "review_reason": annotation.get(
                        "review_reason"
                    ),
                }
            )
            continue

        destination_path = (
            final_cases_dir / source_path.name
        )
        copy_file(source_path, destination_path)
        expected_output_names.add(source_path.name)

        kept.append(
            {
                "file": source_path.name,
                "trajectory_id": trajectory_id,
                "target_status": target_status,
            }
        )

    # Remove stale files when rebuilding without --clean.
    for output_path in final_cases_dir.glob(
        "*.combined.json"
    ):
        if output_path.name not in expected_output_names:
            output_path.unlink()

    summary = {
        "schema_version": (
            "agent3sigma_final_annotations_v1"
        ),
        "run_root": str(run_root),
        "completed_annotations_dir": str(
            completed_cases_dir
        ),
        "final_annotations_dir": str(
            final_cases_dir
        ),
        "num_completed_annotations": len(
            input_paths
        ),
        "num_final_annotations": len(kept),
        "num_excluded_no_target": len(
            excluded_no_target
        ),
        "num_invalid": len(invalid),
        "kept": kept,
        "excluded_no_target": excluded_no_target,
        "invalid": invalid,
    }

    write_json(summary_path, summary)

    return summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    run_root = args.run_root.resolve()

    if not run_root.is_dir():
        raise NotADirectoryError(
            f"Run root not found: {run_root}"
        )

    summary = build_final_annotations(
        run_root=run_root,
        clean=args.clean,
        allow_empty=args.allow_empty,
    )

    print()
    print("=" * 80)
    print("Final annotation construction completed")
    print("=" * 80)
    print(
        "Completed annotations : "
        f"{summary['num_completed_annotations']}"
    )
    print(
        "Final annotations     : "
        f"{summary['num_final_annotations']}"
    )
    print(
        "Excluded no_target    : "
        f"{summary['num_excluded_no_target']}"
    )
    print(
        "Invalid files         : "
        f"{summary['num_invalid']}"
    )
    print(
        "Final cases           : "
        f"{summary['final_annotations_dir']}"
    )
    print(
        "Filter summary        : "
        f"{Path(summary['final_annotations_dir']).parent / 'filter_summary.json'}"
    )

    if summary["excluded_no_target"]:
        print()
        print("Excluded no_target cases:")

        for item in summary["excluded_no_target"]:
            print(
                f"  {item['trajectory_id']}: "
                f"{item['file']}"
            )

    if summary["invalid"]:
        print()
        print("Invalid files:")

        for item in summary["invalid"]:
            print(
                f"  {item['file']}: "
                f"{item['error']}"
            )


if __name__ == "__main__":
    main()
