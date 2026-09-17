#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected JSON object at {path}, "
            f"got {type(obj).__name__}"
        )

    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc!r}"
                ) from exc

            if isinstance(obj, dict):
                output.append(obj)

    return output


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(value + "\n")


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_name(value: Any) -> str:
    text = str(value or "").strip()

    chars: list[str] = []

    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("_")

    output = "".join(chars).strip("_")

    while "__" in output:
        output = output.replace("__", "_")

    return output or "unknown"


def normalize_case_key(filename: str) -> str:
    for suffix in (
        ".annotation.json",
        ".semantic_validation.json",
        ".target.json",
        ".primary.json",
        ".json",
    ):
        if filename.endswith(suffix):
            return filename[:-len(suffix)]

    return Path(filename).stem


def build_file_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}

    if not directory.is_dir():
        return index

    for path in sorted(directory.glob("*.json")):
        key = normalize_case_key(path.name)
        index[key] = path

        # Also index the safe form because some pipelines change "__" to "_".
        index[safe_name(key)] = path

    return index


def first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue

        path = Path(str(value))

        if path.is_file():
            return path

    return None


def find_by_case_key(
    *,
    case_id: str,
    index: dict[str, Path],
) -> Path | None:
    candidates = [
        case_id,
        safe_name(case_id),
        case_id.replace("chain_task_", "chain__task_", 1),
        case_id.replace("chain__task_", "chain_task_", 1),
    ]

    for candidate in candidates:
        if candidate in index:
            return index[candidate]

        safe_candidate = safe_name(candidate)

        if safe_candidate in index:
            return index[safe_candidate]

    return None


def copy_file(
    source: Path,
    destination: Path,
    force: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if not force:
            return

        destination.unlink()

    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all non-pass semantic-validation cases for "
            "targeted unsafe-action reannotation."
        )
    )

    parser.add_argument(
        "--semantic_all_cases_jsonl",
        required=True,
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        help="Original merged unsafe-action annotation directory.",
    )
    parser.add_argument(
        "--normalized_dir",
        required=True,
        help="Original normalized trajectory directory.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    semantic_path = Path(args.semantic_all_cases_jsonl)
    annotations_dir = Path(args.annotations_dir)
    normalized_dir = Path(args.normalized_dir)
    output_root = Path(args.output_root)

    if not semantic_path.is_file():
        raise FileNotFoundError(semantic_path)

    if not annotations_dir.is_dir():
        raise NotADirectoryError(annotations_dir)

    if not normalized_dir.is_dir():
        raise NotADirectoryError(normalized_dir)

    output_root.mkdir(parents=True, exist_ok=True)

    annotations_index = build_file_index(annotations_dir)
    normalized_index = build_file_index(normalized_dir)

    semantic_cases = read_jsonl(semantic_path)

    repair_cases = [
        case
        for case in semantic_cases
        if (
            case.get("semantic_status") != "pass"
            or case.get("recommended_action") != "keep_annotation"
        )
    ]

    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    manifest: list[dict[str, Any]] = []
    missing_annotations: list[str] = []
    missing_normalized: list[str] = []

    for index, case in enumerate(repair_cases, 1):
        case_id = str(
            case.get("case_id")
            or case.get("trajectory_id")
            or f"unknown_{index}"
        )

        trajectory_id = case.get("trajectory_id")
        semantic_status = str(
            case.get("semantic_status") or "unknown"
        )
        action = str(
            case.get("recommended_action") or "repair_annotation"
        )

        action_counts[action] += 1
        status_counts[semantic_status] += 1

        files = case.get("files")

        if not isinstance(files, dict):
            files = {}

        annotation_path = first_existing_path(
            files.get("annotation"),
            case.get("annotation_path"),
        )

        if annotation_path is None:
            annotation_path = find_by_case_key(
                case_id=case_id,
                index=annotations_index,
            )

        normalized_path = first_existing_path(
            files.get("normalized"),
            case.get("normalized_path"),
        )

        if normalized_path is None:
            normalized_path = find_by_case_key(
                case_id=case_id,
                index=normalized_index,
            )

        case_folder = (
            output_root
            / "by_action"
            / safe_name(action)
            / safe_name(case_id)
        )

        copied_annotation_path: Path | None = None
        copied_normalized_path: Path | None = None

        if annotation_path is not None:
            copied_annotation_path = (
                output_root
                / "annotations"
                / annotation_path.name
            )

            copy_file(
                annotation_path,
                copied_annotation_path,
                args.force,
            )

            copy_file(
                annotation_path,
                case_folder / annotation_path.name,
                args.force,
            )
        else:
            missing_annotations.append(case_id)

        if normalized_path is not None:
            copied_normalized_path = (
                output_root
                / "normalized_cases"
                / normalized_path.name
            )

            copy_file(
                normalized_path,
                copied_normalized_path,
                args.force,
            )

            copy_file(
                normalized_path,
                case_folder / normalized_path.name,
                args.force,
            )
        else:
            missing_normalized.append(case_id)

        semantic_validation_file = first_existing_path(
            case.get("semantic_validation_file"),
            case.get("validation_file"),
        )

        if semantic_validation_file is not None:
            copy_file(
                semantic_validation_file,
                case_folder / semantic_validation_file.name,
                args.force,
            )

        manifest_item = {
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "semantic_status": semantic_status,
            "recommended_action": action,
            "weighted_average": case.get("weighted_average"),
            "scores": case.get("scores", {}),
            "failed_dimensions": case.get(
                "failed_dimensions",
                [],
            ),
            "possible_alternative_target_component_id": (
                case.get(
                    "possible_alternative_target_component_id"
                )
            ),
            "possible_alternative_primary_component_id": (
                case.get(
                    "possible_alternative_primary_component_id"
                )
            ),
            "annotation_path": (
                str(annotation_path)
                if annotation_path is not None
                else None
            ),
            "normalized_path": (
                str(normalized_path)
                if normalized_path is not None
                else None
            ),
            "copied_annotation_path": (
                str(copied_annotation_path)
                if copied_annotation_path is not None
                else None
            ),
            "copied_normalized_path": (
                str(copied_normalized_path)
                if copied_normalized_path is not None
                else None
            ),
            "case_folder": str(case_folder),
            "summary": case.get("summary"),
            "rationales": case.get("rationales", {}),
            "deterministic": case.get("deterministic"),
            "semantic": case.get("semantic"),
            "hard_errors": case.get("hard_errors", []),
        }

        manifest.append(manifest_item)

        write_json(
            case_folder / "repair_case.json",
            manifest_item,
        )

        print(
            f"[{index}/{len(repair_cases)}] "
            f"{semantic_status} {action} {case_id}"
        )

    case_ids = sorted({
        str(item["case_id"])
        for item in manifest
    })

    trajectory_ids = sorted({
        str(item["trajectory_id"])
        for item in manifest
        if item.get("trajectory_id") is not None
    })

    write_jsonl(
        output_root / "repair_manifest.jsonl",
        manifest,
    )

    write_json(
        output_root / "repair_manifest.json",
        manifest,
    )

    write_lines(
        output_root / "case_ids.txt",
        case_ids,
    )

    write_lines(
        output_root / "trajectory_ids.txt",
        trajectory_ids,
    )

    for action in sorted(action_counts):
        action_items = [
            item
            for item in manifest
            if item["recommended_action"] == action
        ]

        write_jsonl(
            output_root
            / "manifests_by_action"
            / f"{safe_name(action)}.jsonl",
            action_items,
        )

        write_lines(
            output_root
            / "ids_by_action"
            / f"{safe_name(action)}.txt",
            [
                str(item["case_id"])
                for item in action_items
            ],
        )

    report = {
        "semantic_all_cases_jsonl": str(semantic_path),
        "annotations_dir": str(annotations_dir),
        "normalized_dir": str(normalized_dir),
        "output_root": str(output_root),
        "num_semantic_cases": len(semantic_cases),
        "num_repair_cases": len(repair_cases),
        "status_counts": dict(status_counts),
        "recommended_action_counts": dict(action_counts),
        "num_missing_annotations": len(missing_annotations),
        "num_missing_normalized": len(missing_normalized),
        "missing_annotations": missing_annotations,
        "missing_normalized": missing_normalized,
        "outputs": {
            "repair_annotations": str(
                output_root / "annotations"
            ),
            "normalized_cases": str(
                output_root / "normalized_cases"
            ),
            "by_action": str(
                output_root / "by_action"
            ),
            "repair_manifest_jsonl": str(
                output_root / "repair_manifest.jsonl"
            ),
            "case_ids": str(
                output_root / "case_ids.txt"
            ),
            "trajectory_ids": str(
                output_root / "trajectory_ids.txt"
            ),
        },
    }

    write_json(
        output_root / "prepare_repair_report.json",
        report,
    )

    print("=" * 80)
    print("Unsafe-action repair package prepared")
    print("=" * 80)
    print(f"Semantic cases:       {len(semantic_cases)}")
    print(f"Repair cases:         {len(repair_cases)}")
    print(f"Missing annotations:  {len(missing_annotations)}")
    print(f"Missing normalized:   {len(missing_normalized)}")

    print("\nRecommended actions:")
    for action, count in sorted(
        action_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {action}: {count}")

    print(f"\nOutput root: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
