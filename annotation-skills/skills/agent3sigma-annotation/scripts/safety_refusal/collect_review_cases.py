#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAJECTORY_ID_PATTERN = re.compile(r"(syn-\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect normalized case files corresponding to trajectory IDs "
            "listed in no_target_reviews.json."
        )
    )

    parser.add_argument(
        "--reviews_file",
        type=Path,
        required=True,
        help="Path to no_target_reviews.json.",
    )
    parser.add_argument(
        "--cases_dir",
        type=Path,
        required=True,
        help="Directory containing normalized case JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to which matched case files will be copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory_ids(reviews_file: Path) -> list[str]:
    data = load_json(reviews_file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {reviews_file}, "
            f"but found {type(data).__name__}."
        )

    trajectory_ids: list[str] = []
    seen: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            print(
                f"[WARNING] Skipping review entry {index}: "
                f"expected object, got {type(item).__name__}"
            )
            continue

        trajectory_id = item.get("trajectory_id")

        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            print(
                f"[WARNING] Skipping review entry {index}: "
                "missing valid trajectory_id"
            )
            continue

        trajectory_id = trajectory_id.strip().lower()

        if trajectory_id not in seen:
            seen.add(trajectory_id)
            trajectory_ids.append(trajectory_id)

    return trajectory_ids


def extract_trajectory_id(filename: str) -> str | None:
    match = TRAJECTORY_ID_PATTERN.search(filename)

    if match is None:
        return None

    return match.group(1).lower()


def index_case_files(cases_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)

    for case_file in sorted(cases_dir.glob("*.json")):
        trajectory_id = extract_trajectory_id(case_file.name)

        if trajectory_id is None:
            print(
                f"[WARNING] Could not extract trajectory ID from: "
                f"{case_file.name}"
            )
            continue

        index[trajectory_id].append(case_file)

    return dict(index)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()

    reviews_file: Path = args.reviews_file.resolve()
    cases_dir: Path = args.cases_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not reviews_file.is_file():
        raise FileNotFoundError(
            f"Reviews file does not exist: {reviews_file}"
        )

    if not cases_dir.is_dir():
        raise NotADirectoryError(
            f"Cases directory does not exist: {cases_dir}"
        )

    trajectory_ids = load_trajectory_ids(reviews_file)
    case_index = index_case_files(cases_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    skipped_existing: list[str] = []
    missing_ids: list[str] = []
    duplicate_matches: dict[str, list[str]] = {}

    for trajectory_id in trajectory_ids:
        matches = case_index.get(trajectory_id, [])

        if not matches:
            missing_ids.append(trajectory_id)
            print(f"[MISSING] {trajectory_id}")
            continue

        if len(matches) > 1:
            duplicate_matches[trajectory_id] = [
                str(path) for path in matches
            ]
            print(
                f"[DUPLICATE] {trajectory_id}: "
                f"{len(matches)} files matched"
            )

        for source_file in matches:
            destination_file = output_dir / source_file.name

            if destination_file.exists() and not args.overwrite:
                skipped_existing.append(source_file.name)
                print(f"[SKIPPED EXISTING] {source_file.name}")
                continue

            shutil.copy2(source_file, destination_file)
            copied_files.append(source_file.name)
            print(f"[COPIED] {source_file.name}")

    report = {
        "reviews_file": str(reviews_file),
        "cases_dir": str(cases_dir),
        "output_dir": str(output_dir),
        "review_entries": len(trajectory_ids),
        "files_copied": len(copied_files),
        "files_skipped_existing": len(skipped_existing),
        "missing_trajectory_ids": len(missing_ids),
        "duplicate_trajectory_ids": len(duplicate_matches),
        "copied_files": copied_files,
        "skipped_existing_files": skipped_existing,
        "missing_ids": missing_ids,
        "duplicate_matches": duplicate_matches,
    }

    report_file = output_dir / "collection_report.json"
    save_json(report, report_file)

    print()
    print("=" * 80)
    print("No-target review case collection completed")
    print("=" * 80)
    print(f"Trajectory IDs       : {len(trajectory_ids)}")
    print(f"Files copied         : {len(copied_files)}")
    print(f"Skipped existing     : {len(skipped_existing)}")
    print(f"Missing IDs          : {len(missing_ids)}")
    print(f"Duplicate IDs        : {len(duplicate_matches)}")
    print(f"Output directory     : {output_dir}")
    print(f"Collection report    : {report_file}")


if __name__ == "__main__":
    main()
