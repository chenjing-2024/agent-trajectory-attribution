#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SOURCE_NAMES = (
    "target",
    "primary",
    "attack_chain",
    "execution_chain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge target, primary, attack-chain, and execution-chain "
            "annotations by trajectory_id, writing one JSON per trajectory."
        )
    )
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--primary_dir", type=Path, required=True)
    parser.add_argument("--attack_chain_dir", type=Path, required=True)
    parser.add_argument("--execution_chain_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--complete_only",
        action="store_true",
        help="Only write trajectories containing all four annotation types.",
    )
    parser.add_argument(
        "--allow_conflicting_duplicates",
        action="store_true",
        help="Keep the first conflicting duplicate within one source.",
    )
    return parser.parse_args()


def iter_json_records(
    file_path: Path,
) -> Iterable[tuple[dict[str, Any], str]]:
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {file_path}:{line_number}: {exc}"
                    ) from exc

                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected object at {file_path}:{line_number}"
                    )

                yield obj, f"{file_path}:{line_number}"
        return

    with file_path.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON file {file_path}: {exc}"
            ) from exc

    if isinstance(obj, dict):
        yield obj, str(file_path)
        return

    if isinstance(obj, list):
        for index, record in enumerate(obj):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at {file_path}[{index}]"
                )
            yield record, f"{file_path}[{index}]"
        return

    raise ValueError(
        f"Expected object or list in {file_path}, "
        f"got {type(obj).__name__}"
    )


def discover_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input does not exist: {root}")

    if root.is_file():
        return [root]

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )

    if not files:
        raise FileNotFoundError(
            f"No JSON annotation files found under {root}"
        )

    return files


def normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(record))

    metadata = copied.get("_metadata")
    if isinstance(metadata, dict):
        metadata.pop("raw_model_output", None)
        metadata.pop("annotation_path", None)
        metadata.pop("output_path", None)

    return copied


def load_source(
    root: Path,
    source_name: str,
    allow_conflicting_duplicates: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, str],
    dict[str, Any],
]:
    records: dict[str, dict[str, Any]] = {}
    locations: dict[str, str] = {}
    filenames: dict[str, str] = {}

    identical_duplicates = []
    conflicting_duplicates = []
    missing_ids = []

    files = discover_files(root)
    raw_count = 0

    for file_path in files:
        for record, location in iter_json_records(file_path):
            raw_count += 1

            trajectory_id = record.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id.strip():
                missing_ids.append(location)
                continue

            trajectory_id = trajectory_id.strip()

            if trajectory_id not in records:
                records[trajectory_id] = record
                locations[trajectory_id] = location
                filenames[trajectory_id] = file_path.name
                continue

            if normalized_record(records[trajectory_id]) == normalized_record(record):
                identical_duplicates.append(
                    {
                        "trajectory_id": trajectory_id,
                        "kept": locations[trajectory_id],
                        "duplicate": location,
                    }
                )
                continue

            conflicting_duplicates.append(
                {
                    "trajectory_id": trajectory_id,
                    "kept": locations[trajectory_id],
                    "conflicting": location,
                }
            )

            if not allow_conflicting_duplicates:
                raise ValueError(
                    f"Conflicting duplicate in {source_name} for "
                    f"{trajectory_id}\n"
                    f"First: {locations[trajectory_id]}\n"
                    f"Second: {location}"
                )

    stats = {
        "source": source_name,
        "root": str(root),
        "num_files": len(files),
        "num_raw_records": raw_count,
        "num_unique_trajectory_ids": len(records),
        "num_identical_duplicates": len(identical_duplicates),
        "num_conflicting_duplicates": len(conflicting_duplicates),
        "num_missing_trajectory_id": len(missing_ids),
        "identical_duplicates": identical_duplicates,
        "conflicting_duplicates": conflicting_duplicates,
        "missing_trajectory_id_locations": missing_ids,
    }

    return records, locations, filenames, stats


def get_target_component_id(
    annotation: dict[str, Any] | None,
) -> str | None:
    if not isinstance(annotation, dict):
        return None

    target_action = annotation.get("target_action")
    if isinstance(target_action, dict):
        value = target_action.get("component_id")
        if isinstance(value, str):
            return value

    value = annotation.get("target_component_id")
    return value if isinstance(value, str) else None


def get_primary_component_id(
    annotation: dict[str, Any] | None,
) -> str | None:
    if not isinstance(annotation, dict):
        return None

    for key in (
        "primary_component_id",
        "primary_attribution_component_id",
        "primary_root_component_id",
    ):
        value = annotation.get(key)
        if isinstance(value, str):
            return value

    for key in (
        "primary_attribution_component",
        "primary_root_component",
        "primary_component",
    ):
        value = annotation.get(key)
        if isinstance(value, dict):
            component_id = value.get("component_id")
            if isinstance(component_id, str):
                return component_id

    return None


def get_primary_component_object(
    annotation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(annotation, dict):
        return None

    for key in (
        "primary_attribution_component",
        "primary_root_component",
        "primary_component",
    ):
        value = annotation.get(key)
        if isinstance(value, dict):
            return value

    return None


def get_attack_success(
    annotation: dict[str, Any] | None,
) -> bool | None:
    if not isinstance(annotation, dict):
        return None

    value = annotation.get("attack_success")
    return value if isinstance(value, bool) else None


def get_confidence(
    annotation: dict[str, Any] | None,
) -> float | None:
    if not isinstance(annotation, dict):
        return None

    value = annotation.get("confidence")

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    return None


def get_chain(
    annotation: dict[str, Any] | None,
    key: str,
) -> list[Any] | None:
    if not isinstance(annotation, dict):
        return None

    value = annotation.get(key)
    return value if isinstance(value, list) else None


def get_risk_category(
    execution_annotation: dict[str, Any] | None,
    attack_annotation: dict[str, Any] | None,
) -> str | None:
    if isinstance(execution_annotation, dict):
        value = execution_annotation.get("risk_category")
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(attack_annotation, dict):
        value = attack_annotation.get("risk_category")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def consistency_check(
    named_values: dict[str, Any],
) -> dict[str, Any]:
    available = {
        name: value
        for name, value in named_values.items()
        if value is not None
    }

    serialized = {
        json.dumps(value, sort_keys=True, ensure_ascii=False)
        for value in available.values()
    }

    return {
        "consistent": len(serialized) <= 1,
        "available_values": available,
        "num_available": len(available),
    }


def build_confidence(
    annotations: dict[str, dict[str, Any] | None],
) -> dict[str, float | None]:
    values = {
        name: get_confidence(annotation)
        for name, annotation in annotations.items()
    }

    available = [
        value for value in values.values()
        if value is not None
    ]

    return {
        **values,
        "overall_min": min(available) if available else None,
        "overall_mean": (
            statistics.fmean(available)
            if available
            else None
        ),
    }


def canonical_attack_success(
    check: dict[str, Any],
) -> bool | None:
    values = check["available_values"]

    for source_name in SOURCE_NAMES:
        value = values.get(source_name)
        if isinstance(value, bool):
            return value

    return None


def safe_output_filename(
    trajectory_id: str,
    target_filename: str | None,
) -> str:
    if target_filename:
        name = target_filename

        for suffix in (
            ".target.json",
            ".primary.json",
            ".attack_chain.json",
            ".execution_chain.json",
            ".json",
        ):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        if name:
            return f"{name}.annotation.json"

    safe_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in trajectory_id
    )
    return f"{safe_id}.annotation.json"


def build_merged_annotation(
    trajectory_id: str,
    annotations: dict[str, dict[str, Any] | None],
    locations: dict[str, str | None],
) -> dict[str, Any]:
    target = annotations["target"]
    primary = annotations["primary"]
    attack_annotation = annotations["attack_chain"]
    execution_annotation = annotations["execution_chain"]

    attack_success_check = consistency_check(
        {
            name: get_attack_success(annotation)
            for name, annotation in annotations.items()
        }
    )

    target_id_check = consistency_check(
        {
            name: get_target_component_id(annotation)
            for name, annotation in annotations.items()
        }
    )

    primary_id_check = consistency_check(
        {
            "primary": get_primary_component_id(primary),
            "attack_chain": get_primary_component_id(
                attack_annotation
            ),
            "execution_chain": get_primary_component_id(
                execution_annotation
            ),
        }
    )

    conflicts = []

    # Upstream stages use attack_success with different local semantics.
    # Preserve all source values in consistency_checks for audit, but do not
    # turn their disagreement into a merged-annotation conflict. Validators
    # judge the target and causal fields directly.

    if not target_id_check["consistent"]:
        conflicts.append(
            "target_component_id differs across annotation sources"
        )

    if not primary_id_check["consistent"]:
        conflicts.append(
            "primary_component_id differs across annotation sources"
        )

    missing_annotations = [
        name for name in SOURCE_NAMES
        if annotations[name] is None
    ]

    source_review_reasons = []

    for name, annotation in annotations.items():
        if not isinstance(annotation, dict):
            continue

        if annotation.get("needs_review") is True:
            reason = annotation.get("review_reason")
            if isinstance(reason, str) and reason.strip():
                source_review_reasons.append(
                    f"{name}: {reason.strip()}"
                )
            else:
                source_review_reasons.append(
                    f"{name}: source marked needs_review"
                )

    review_reasons = list(
        dict.fromkeys(conflicts + source_review_reasons)
    )

    target_action = None
    target_type = None

    if isinstance(target, dict):
        if isinstance(target.get("target_action"), dict):
            target_action = target["target_action"]

        if isinstance(target.get("target_type"), str):
            target_type = target["target_type"]

    return {
        "trajectory_id": trajectory_id,
        "target_type": target_type,
        "attack_success": canonical_attack_success(
            attack_success_check
        ),
        "risk_category": get_risk_category(
            execution_annotation,
            attack_annotation,
        ),
        "target_component_id": get_target_component_id(target),
        "target_action": target_action,
        "primary_component_id": get_primary_component_id(primary),
        "primary_attribution_component": (
            get_primary_component_object(primary)
        ),
        "attack_chain": get_chain(
            attack_annotation,
            "attack_chain",
        ),
        "execution_chain": get_chain(
            execution_annotation,
            "execution_chain",
        ),
        "confidence": build_confidence(annotations),
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "is_complete": not missing_annotations,
        "missing_annotations": missing_annotations,
        "consistency_checks": {
            "attack_success": {
                **attack_success_check,
                "enforced": False,
            },
            "target_component_id": target_id_check,
            "primary_component_id": primary_id_check,
            "has_conflict": bool(conflicts),
            "conflict_messages": conflicts,
        },
        "chain_status": {
            "attack_chain_annotation_available": (
                attack_annotation is not None
            ),
            "attack_chain_empty": (
                isinstance(
                    get_chain(attack_annotation, "attack_chain"),
                    list,
                )
                and len(
                    get_chain(attack_annotation, "attack_chain")
                ) == 0
            ),
            "execution_chain_annotation_available": (
                execution_annotation is not None
            ),
            "execution_chain_empty": (
                isinstance(
                    get_chain(
                        execution_annotation,
                        "execution_chain",
                    ),
                    list,
                )
                and len(
                    get_chain(
                        execution_annotation,
                        "execution_chain",
                    )
                ) == 0
            ),
        },
        "_metadata": {
            "annotation_paths": locations,
            "source_metadata": {
                name: (
                    annotation.get("_metadata")
                    if isinstance(annotation, dict)
                    and isinstance(
                        annotation.get("_metadata"),
                        dict,
                    )
                    else None
                )
                for name, annotation in annotations.items()
            },
        },
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    args = parse_args()

    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "target": args.target_dir,
        "primary": args.primary_dir,
        "attack_chain": args.attack_chain_dir,
        "execution_chain": args.execution_chain_dir,
    }

    records_by_source = {}
    locations_by_source = {}
    filenames_by_source = {}
    source_stats = {}

    for source_name, root in roots.items():
        records, locations, filenames, stats = load_source(
            root,
            source_name,
            args.allow_conflicting_duplicates,
        )

        records_by_source[source_name] = records
        locations_by_source[source_name] = locations
        filenames_by_source[source_name] = filenames
        source_stats[source_name] = stats

        print(
            f"[{source_name}] "
            f"unique={stats['num_unique_trajectory_ids']} "
            f"duplicates={stats['num_identical_duplicates']} "
            f"conflicts={stats['num_conflicting_duplicates']}"
        )

    all_ids = sorted(
        set().union(
            *(
                set(records_by_source[name])
                for name in SOURCE_NAMES
            )
        )
    )

    written_records = []
    incomplete_records = []
    conflict_records = []

    for trajectory_id in all_ids:
        annotations = {
            name: records_by_source[name].get(trajectory_id)
            for name in SOURCE_NAMES
        }

        locations = {
            name: locations_by_source[name].get(trajectory_id)
            for name in SOURCE_NAMES
        }

        merged = build_merged_annotation(
            trajectory_id,
            annotations,
            locations,
        )

        if args.complete_only and not merged["is_complete"]:
            incomplete_records.append(merged)
            continue

        target_filename = filenames_by_source["target"].get(
            trajectory_id
        )

        output_name = safe_output_filename(
            trajectory_id,
            target_filename,
        )

        output_path = cases_dir / output_name

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                merged,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

        written_records.append(merged)

        if not merged["is_complete"]:
            incomplete_records.append(merged)

        if merged["consistency_checks"]["has_conflict"]:
            conflict_records.append(merged)

    missing_path = args.output_dir / "missing_annotations.jsonl"
    conflict_path = args.output_dir / "conflict_annotations.jsonl"
    summary_path = args.output_dir / "merge_summary.json"

    write_jsonl(missing_path, incomplete_records)
    write_jsonl(conflict_path, conflict_records)

    summary = {
        "inputs": {
            name: str(path)
            for name, path in roots.items()
        },
        "source_stats": source_stats,
        "merge_stats": {
            "num_union_trajectory_ids": len(all_ids),
            "num_json_files_written": len(written_records),
            "num_complete_written": sum(
                record["is_complete"]
                for record in written_records
            ),
            "num_incomplete": len(incomplete_records),
            "num_conflicts": len(conflict_records),
            "complete_only": args.complete_only,
            "risk_category_counts": dict(
                Counter(
                    record["risk_category"] or "<null>"
                    for record in written_records
                )
            ),
        },
        "outputs": {
            "cases_dir": str(cases_dir),
            "missing_annotations": str(missing_path),
            "conflict_annotations": str(conflict_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print()
    print("=" * 80)
    print("Annotation merge finished")
    print("=" * 80)
    print(f"Union trajectory IDs : {len(all_ids)}")
    print(f"JSON files written   : {len(written_records)}")
    print(
        "Complete written     : "
        f"{sum(r['is_complete'] for r in written_records)}"
    )
    print(f"Incomplete           : {len(incomplete_records)}")
    print(f"Conflicts            : {len(conflict_records)}")
    print(f"Cases directory      : {cases_dir}")
    print(f"Summary              : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
