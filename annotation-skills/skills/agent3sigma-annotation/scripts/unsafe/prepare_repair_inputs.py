#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


KNOWN_ACTIONS = (
    "keep_annotation",
    "rerun_all_annotations",
    "repair_attack_chain",
    "repair_execution_chain",
    "repair_attack_and_execution_chains",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split semantic-validation cases by recommended_action and "
            "prepare annotation, trajectory, and validator-feedback inputs "
            "for subsequent repair."
        )
    )

    parser.add_argument(
        "--validation_root",
        type=Path,
        required=True,
        help=(
            "Semantic validation output directory containing all_cases.jsonl."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Directory in which categorized repair inputs will be created.",
    )
    parser.add_argument(
        "--annotations_root",
        type=Path,
        default=None,
        help=(
            "Optional source annotation directory. This is recorded in the "
            "summary; per-case annotation paths are normally read from the "
            "validator output."
        ),
    )
    parser.add_argument(
        "--trajectories_root",
        type=Path,
        default=None,
        help=(
            "Optional source trajectory directory. This is recorded in the "
            "summary; per-case trajectory paths are normally read from the "
            "validator output."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output_root before creating categorized inputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if an annotation, trajectory, or required field is missing."
        ),
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_num}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_num}"
                )

            cases.append(obj)

    return cases


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_validation_file(
    validation_root: Path,
    case_id: str,
) -> Path | None:
    candidates = [
        validation_root
        / "cases"
        / f"{case_id}.semantic_validation.json",
        validation_root
        / "case_validation"
        / f"{case_id}.semantic_validation.json",
        validation_root / "cases" / f"{case_id}.json",
        validation_root / "case_validation" / f"{case_id}.json",
    ]

    return next((path for path in candidates if path.exists()), None)


def resolve_case_file(
    case: dict[str, Any],
    *,
    key: str,
) -> Path | None:
    files = case.get("files")
    if not isinstance(files, dict):
        return None

    value = files.get(key)
    if not value:
        return None

    return Path(str(value))


def copy_case_file(
    source: Path | None,
    destination_dir: Path,
    *,
    missing_key: str,
    missing: Counter[str],
    strict: bool,
) -> Path | None:
    if source is None:
        missing[missing_key] += 1
        if strict:
            raise FileNotFoundError(
                f"Missing source path for {missing_key}"
            )
        return None

    if not source.exists():
        missing[missing_key] += 1
        if strict:
            raise FileNotFoundError(
                f"Source file does not exist: {source}"
            )
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = parse_args()

    validation_root = args.validation_root.resolve()
    output_root = args.output_root.resolve()

    all_cases_path = validation_root / "all_cases.jsonl"
    if not all_cases_path.is_file():
        raise FileNotFoundError(
            f"Could not find semantic validator output: {all_cases_path}"
        )

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    action_dirs = {
        action: output_root / action
        for action in KNOWN_ACTIONS
    }

    for root in action_dirs.values():
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        (root / "validation").mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(all_cases_path)

    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    copied_counts: Counter[str] = Counter()

    manifest_paths: dict[str, Path] = {
        action: action_dirs[action] / "manifest.jsonl"
        for action in KNOWN_ACTIONS
    }

    manifest_handles = {
        action: path.open("w", encoding="utf-8")
        for action, path in manifest_paths.items()
    }

    try:
        for case in cases:
            action = case.get("recommended_action")

            if action not in action_dirs:
                missing[f"unknown_action:{action}"] += 1
                if args.strict:
                    raise ValueError(
                        f"Unknown recommended_action: {action!r}"
                    )
                continue

            case_id = case.get("case_id")
            trajectory_id = case.get("trajectory_id")

            if not case_id:
                missing["case_id"] += 1
                if args.strict:
                    raise ValueError(
                        "Case is missing required field: case_id"
                    )
                continue

            if not trajectory_id:
                missing["trajectory_id"] += 1
                if args.strict:
                    raise ValueError(
                        f"Case {case_id} is missing trajectory_id"
                    )

            destination = action_dirs[action]

            annotation_path = resolve_case_file(
                case,
                key="annotation",
            )
            trajectory_path = resolve_case_file(
                case,
                key="trajectory",
            )

            copied_annotation = copy_case_file(
                annotation_path,
                destination / "annotations",
                missing_key="annotation",
                missing=missing,
                strict=args.strict,
            )
            if copied_annotation is not None:
                copied_counts["annotation"] += 1

            copied_trajectory = copy_case_file(
                trajectory_path,
                destination / "trajectories",
                missing_key="trajectory",
                missing=missing,
                strict=args.strict,
            )
            if copied_trajectory is not None:
                copied_counts["trajectory"] += 1

            validation_path = find_validation_file(
                validation_root,
                str(case_id),
            )

            if validation_path is not None:
                copied_validation = copy_case_file(
                    validation_path,
                    destination / "validation",
                    missing_key="validation",
                    missing=missing,
                    strict=args.strict,
                )
                if copied_validation is not None:
                    copied_counts["validation"] += 1
            else:
                fallback = (
                    destination
                    / "validation"
                    / f"{case_id}.semantic_validation.json"
                )
                write_json(fallback, case)
                copied_counts["validation_fallback"] += 1

            manifest_record = {
                "case_id": case_id,
                "trajectory_id": trajectory_id,
                "recommended_action": action,
                "semantic_status": case.get("semantic_status"),
                "scores": case.get("scores", {}),
                "weighted_average": case.get("weighted_average"),
                "failed_dimensions": case.get(
                    "failed_dimensions",
                    [],
                ),
                "summary": case.get("summary"),
                "source_annotation": (
                    str(annotation_path)
                    if annotation_path is not None
                    else None
                ),
                "source_trajectory": (
                    str(trajectory_path)
                    if trajectory_path is not None
                    else None
                ),
            }

            manifest_handles[action].write(
                json.dumps(
                    manifest_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            counts[action] += 1

    finally:
        for handle in manifest_handles.values():
            handle.close()

    summary = {
        "source_validation": str(validation_root),
        "source_all_cases": str(all_cases_path),
        "source_annotations": (
            str(args.annotations_root.resolve())
            if args.annotations_root is not None
            else None
        ),
        "source_trajectories": (
            str(args.trajectories_root.resolve())
            if args.trajectories_root is not None
            else None
        ),
        "output_root": str(output_root),
        "total_input_cases": len(cases),
        "categorized_cases": sum(counts.values()),
        "counts": {
            action: counts[action]
            for action in KNOWN_ACTIONS
        },
        "copied_counts": dict(copied_counts),
        "missing": dict(missing),
        "manifests": {
            action: str(manifest_paths[action])
            for action in KNOWN_ACTIONS
        },
    }

    write_json(output_root / "summary.json", summary)

    print("=" * 80)
    print("Semantic repair inputs prepared")
    print("=" * 80)
    print(f"Input cases: {len(cases)}")
    print()

    for action in KNOWN_ACTIONS:
        print(f"{action:36s}: {counts[action]}")

    print()
    print(f"Categorized cases: {sum(counts.values())}")
    print(f"Output root:      {output_root}")
    print(f"Missing:          {dict(missing)}")
    print(f"Summary:          {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()