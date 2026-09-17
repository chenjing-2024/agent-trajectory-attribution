#!/usr/bin/env python3
"""
Merge Agent3Sigma target, primary, attack-chain, and execution-chain annotations
into one normalized JSON file per trajectory.

Default expected layout under --run_root:

  02_target_full_normalized_v2/cases/*.target.json
  04_primary_full/cases/*.primary.json
  05_attack_chain_full/cases/*.attack_chain.json
  06_execution_chain_full/cases/*.execution.json

Output:

  <output_root>/cases/<base_name>.combined.json
  <output_root>/merge_summary.json
  <output_root>/missing_or_conflicting_cases.jsonl

The script matches files primarily by trajectory_id, not by filename.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIRS = {
    "target": "02_target_full_normalized_v2/cases",
    "primary": "04_primary_full/cases",
    "attack_chain": "05_attack_chain_full/cases",
    "execution_chain": "06_execution_chain_full/cases",
}


# =============================================================================
# Utilities
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_component_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return f"C{value}" if value >= 0 else None

    if isinstance(value, float):
        if value.is_integer() and value >= 0:
            return f"C{int(value)}"
        return None

    if not isinstance(value, str):
        return None

    text = value.strip()

    if not text:
        return None

    if text[:1].upper() == "C":
        text = text[1:].strip()

    if not text.isdigit():
        return None

    return f"C{int(text)}"


def normalize_chain(
    value: Any,
    chain_name: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"{chain_name} must be a list, got "
            f"{type(value).__name__}"
        )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in value:
        if isinstance(item, str):
            component_id = normalize_component_id(item)

            if component_id is None:
                continue

            normalized_item = {
                "component_id": component_id,
            }

        elif isinstance(item, dict):
            component_id = normalize_component_id(
                item.get("component_id")
            )

            if component_id is None:
                component_id = normalize_component_id(
                    item.get("id")
                )

            if component_id is None:
                continue

            normalized_item = dict(item)
            normalized_item["component_id"] = component_id

        else:
            continue

        if component_id in seen:
            continue

        seen.add(component_id)
        normalized.append(normalized_item)

    return normalized


def deep_merge_metadata(
    destination: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
            continue

        current = destination[key]

        if isinstance(current, dict) and isinstance(value, dict):
            deep_merge_metadata(current, value)
            continue

        if current in (None, "", [], {}):
            destination[key] = value


def derive_base_name(
    source_paths: list[Path],
    trajectory_id: str,
) -> str:
    """
    Prefer the common original case filename prefix.

    Example:
      0002_syn-0252_context_rich_credential_extraction.target.json
    becomes:
      0002_syn-0252_context_rich_credential_extraction
    """
    known_suffixes = (
        ".target.json",
        ".primary.json",
        ".attack_chain.json",
        ".execution.json",
        ".json",
    )

    for path in source_paths:
        name = path.name

        for suffix in known_suffixes:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]

                if trajectory_id in stem:
                    return stem

    return trajectory_id


# =============================================================================
# Source indexing
# =============================================================================

def get_trajectory_id(data: dict[str, Any]) -> str | None:
    metadata = data.get("_metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    return first_nonempty_string(
        data.get("trajectory_id"),
        data.get("data_id"),
        metadata.get("trajectory_id"),
        metadata.get("data_id"),
    )


def index_directory(
    directory: Path,
    source_type: str,
) -> tuple[
    dict[str, list[tuple[Path, dict[str, Any]]]],
    list[dict[str, Any]],
]:
    index: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ] = defaultdict(list)

    errors: list[dict[str, Any]] = []

    if not directory.is_dir():
        errors.append(
            {
                "source_type": source_type,
                "path": str(directory),
                "error": "Source directory does not exist",
            }
        )
        return {}, errors

    for path in sorted(directory.rglob("*.json")):
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append(
                {
                    "source_type": source_type,
                    "path": str(path),
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue

        trajectory_id = get_trajectory_id(data)

        if trajectory_id is None:
            errors.append(
                {
                    "source_type": source_type,
                    "path": str(path),
                    "error": "Missing trajectory_id/data_id",
                }
            )
            continue

        index[trajectory_id].append((path, data))

    return dict(index), errors


# =============================================================================
# Conflict handling
# =============================================================================

def collect_scalar_values(
    records: list[tuple[str, Path, dict[str, Any]]],
    key: str,
    normalizer=None,
) -> list[tuple[str, Path, Any]]:
    values: list[tuple[str, Path, Any]] = []

    for source_type, path, data in records:
        value = data.get(key)

        if normalizer is not None:
            value = normalizer(value)

        if value is not None and value != "":
            values.append((source_type, path, value))

    return values


def choose_scalar(
    records: list[tuple[str, Path, dict[str, Any]]],
    key: str,
    preferred_order: tuple[str, ...],
    normalizer=None,
) -> tuple[Any, list[dict[str, Any]]]:
    values = collect_scalar_values(
        records,
        key,
        normalizer=normalizer,
    )

    conflicts: list[dict[str, Any]] = []

    distinct_values = []

    for _, _, value in values:
        if value not in distinct_values:
            distinct_values.append(value)

    if len(distinct_values) > 1:
        conflicts.append(
            {
                "field": key,
                "values": [
                    {
                        "source_type": source_type,
                        "path": str(path),
                        "value": value,
                    }
                    for source_type, path, value in values
                ],
            }
        )

    for preferred_source in preferred_order:
        for source_type, _, value in values:
            if source_type == preferred_source:
                return value, conflicts

    if values:
        return values[0][2], conflicts

    return None, conflicts


# =============================================================================
# Per-trajectory merge
# =============================================================================

def merge_trajectory(
    trajectory_id: str,
    records_by_type: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ],
    construct_attack_chain: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat_records: list[
        tuple[str, Path, dict[str, Any]]
    ] = []

    for source_type, records in records_by_type.items():
        for path, data in records:
            flat_records.append(
                (source_type, path, data)
            )

    target_component_id, conflicts_target = choose_scalar(
        flat_records,
        "target_component_id",
        preferred_order=(
            "target",
            "primary",
            "attack_chain",
            "execution_chain",
        ),
        normalizer=normalize_component_id,
    )

    primary_component_id, conflicts_primary = choose_scalar(
        flat_records,
        "primary_component_id",
        preferred_order=(
            "primary",
            "attack_chain",
            "execution_chain",
            "target",
        ),
        normalizer=normalize_component_id,
    )

    target_status, conflicts_status = choose_scalar(
        flat_records,
        "target_status",
        preferred_order=(
            "target",
            "primary",
            "attack_chain",
            "execution_chain",
        ),
    )

    attack_success, conflicts_success = choose_scalar(
        flat_records,
        "attack_success",
        preferred_order=(
            "target",
            "primary",
            "attack_chain",
            "execution_chain",
        ),
    )

    risk_category, conflicts_risk = choose_scalar(
        flat_records,
        "risk_category",
        preferred_order=(
            "target",
            "primary",
            "attack_chain",
            "execution_chain",
        ),
    )

    confidence, _ = choose_scalar(
        flat_records,
        "confidence",
        preferred_order=(
            "execution_chain",
            "attack_chain",
            "primary",
            "target",
        ),
    )

    _, _ = choose_scalar(
        flat_records,
        "needs_review",
        preferred_order=(
            "execution_chain",
            "attack_chain",
            "primary",
            "target",
        ),
    )

    review_reason, _ = choose_scalar(
        flat_records,
        "review_reason",
        preferred_order=(
            "execution_chain",
            "attack_chain",
            "primary",
            "target",
        ),
    )
    # Confidence is stage-specific, so differing values are expected. Keep
    # the most downstream value selected above without treating it as a
    # cross-stage identity conflict. Review is cumulative: if any stage
    # requests review, the merged annotation must preserve that signal.
    needs_review = any(
        bool(value)
        for _, _, value in collect_scalar_values(
            flat_records,
            "needs_review",
        )
    )

    attack_chain: list[dict[str, Any]] = []

    for path, data in records_by_type.get(
        "attack_chain",
        [],
    ):
        current = normalize_chain(
            data.get("attack_chain"),
            "attack_chain",
        )

        if current:
            attack_chain = current
            break

    execution_chain: list[dict[str, Any]] = []

    for path, data in records_by_type.get(
        "execution_chain",
        [],
    ):
        current = normalize_chain(
            data.get("execution_chain"),
            "execution_chain",
        )

        if current:
            execution_chain = current
            break

    attack_chain_source = "annotated"

    if (
        construct_attack_chain
        and not attack_chain
        and primary_component_id is not None
    ):
        attack_chain_source = (
            "constructed_primary_plus_execution"
        )

        constructed_items: list[dict[str, Any]] = [
            {
                "component_id": primary_component_id,
                "role_in_attack": "primary_source",
                "summary": (
                    "Primary attribution component."
                ),
            }
        ]

        for item in execution_chain:
            copied = dict(item)

            if "role_in_attack" not in copied:
                copied["role_in_attack"] = (
                    copied.get("role_in_execution")
                    or "execution_step"
                )

            constructed_items.append(copied)

        attack_chain = normalize_chain(
            constructed_items,
            "constructed_attack_chain",
        )

    metadata: dict[str, Any] = {}

    # Merge metadata in a deterministic order.
    for source_type in (
        "target",
        "primary",
        "attack_chain",
        "execution_chain",
    ):
        for _, data in records_by_type.get(
            source_type,
            [],
        ):
            current_meta = data.get("_metadata", {})

            if isinstance(current_meta, dict):
                deep_merge_metadata(
                    metadata,
                    dict(current_meta),
                )

    source_files = {
        source_type: [
            str(path)
            for path, _ in records
        ]
        for source_type, records
        in records_by_type.items()
    }

    metadata["merged_from"] = source_files
    metadata["merge_version"] = "agent3sigma_combined_v1"
    metadata["attack_chain_source"] = attack_chain_source

    combined = {
        "trajectory_id": trajectory_id,
        "target_status": target_status,
        "attack_success": attack_success,
        "risk_category": risk_category,
        "target_component_id": target_component_id,
        "primary_component_id": primary_component_id,
        "attack_chain": attack_chain,
        "execution_chain": execution_chain,
        "confidence": confidence,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "_metadata": metadata,
    }

    all_conflicts = (
        conflicts_target
        + conflicts_primary
        + conflicts_status
        + conflicts_success
        + conflicts_risk
    )

    missing_fields = []

    for field_name in (
        "target_component_id",
        "primary_component_id",
        "attack_success",
    ):
        if combined.get(field_name) is None:
            missing_fields.append(field_name)

    if not attack_chain:
        missing_fields.append("attack_chain")

    if not execution_chain:
        missing_fields.append("execution_chain")

    diagnostic = {
        "trajectory_id": trajectory_id,
        "source_files": source_files,
        "missing_fields": missing_fields,
        "conflicts": all_conflicts,
        "attack_chain_source": attack_chain_source,
    }

    return combined, diagnostic


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Agent3Sigma target, primary, "
            "attack-chain, and execution-chain annotations."
        )
    )

    parser.add_argument(
        "--run_root",
        required=True,
        type=Path,
        help=(
            "Root containing 02_target..., 04_primary..., "
            "05_attack_chain..., and 06_execution_chain..."
        ),
    )

    parser.add_argument(
        "--output_root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--target_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--primary_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--attack_chain_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--execution_chain_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--construct_missing_attack_chain",
        action="store_true",
        help=(
            "When attack_chain is absent, construct it as "
            "primary_component_id + execution_chain."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any case has missing fields "
            "or conflicting scalar annotations."
        ),
    )

    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()

    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Run root not found: {run_root}"
        )

    source_dirs = {
        "target": (
            args.target_dir.resolve()
            if args.target_dir is not None
            else (
                run_root
                / DEFAULT_SOURCE_DIRS["target"]
            ).resolve()
        ),
        "primary": (
            args.primary_dir.resolve()
            if args.primary_dir is not None
            else (
                run_root
                / DEFAULT_SOURCE_DIRS["primary"]
            ).resolve()
        ),
        "attack_chain": (
            args.attack_chain_dir.resolve()
            if args.attack_chain_dir is not None
            else (
                run_root
                / DEFAULT_SOURCE_DIRS[
                    "attack_chain"
                ]
            ).resolve()
        ),
        "execution_chain": (
            args.execution_chain_dir.resolve()
            if args.execution_chain_dir is not None
            else (
                run_root
                / DEFAULT_SOURCE_DIRS[
                    "execution_chain"
                ]
            ).resolve()
        ),
    }

    indexes: dict[
        str,
        dict[str, list[tuple[Path, dict[str, Any]]]],
    ] = {}

    indexing_errors: list[dict[str, Any]] = []

    print("=" * 88)
    print(f"Run root:    {run_root}")
    print(f"Output root: {output_root}")

    for source_type, directory in source_dirs.items():
        index, errors = index_directory(
            directory,
            source_type,
        )

        indexes[source_type] = index
        indexing_errors.extend(errors)

        print(
            f"{source_type:16s}: "
            f"{len(index):4d} trajectory IDs | "
            f"{directory}"
        )

    print("=" * 88)

    trajectory_ids = sorted(
        set().union(
            *[
                set(index.keys())
                for index in indexes.values()
            ]
        )
    )

    output_cases_dir = output_root / "cases"
    output_cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics: list[dict[str, Any]] = []
    written = 0

    source_presence_counts = {
        source_type: 0
        for source_type in source_dirs
    }

    for trajectory_id in trajectory_ids:
        records_by_type = {
            source_type: index.get(
                trajectory_id,
                [],
            )
            for source_type, index in indexes.items()
        }

        for source_type, records in records_by_type.items():
            if records:
                source_presence_counts[source_type] += 1

        combined, diagnostic = merge_trajectory(
            trajectory_id=trajectory_id,
            records_by_type=records_by_type,
            construct_attack_chain=(
                args.construct_missing_attack_chain
            ),
        )

        all_source_paths = [
            path
            for records in records_by_type.values()
            for path, _ in records
        ]

        base_name = derive_base_name(
            all_source_paths,
            trajectory_id,
        )

        output_path = (
            output_cases_dir
            / f"{base_name}.combined.json"
        )

        save_json(output_path, combined)

        diagnostic["output_file"] = str(output_path)
        diagnostics.append(diagnostic)
        written += 1

    problematic = [
        item
        for item in diagnostics
        if item["missing_fields"]
        or item["conflicts"]
    ]

    diagnostics_path = (
        output_root
        / "missing_or_conflicting_cases.jsonl"
    )

    with diagnostics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in problematic:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "configuration": {
            "run_root": str(run_root),
            "output_root": str(output_root),
            "source_directories": {
                key: str(value)
                for key, value in source_dirs.items()
            },
            "construct_missing_attack_chain": (
                args.construct_missing_attack_chain
            ),
        },
        "counts": {
            "unique_trajectory_ids": len(
                trajectory_ids
            ),
            "combined_files_written": written,
            "problematic_cases": len(problematic),
            "indexing_errors": len(
                indexing_errors
            ),
            "source_presence": (
                source_presence_counts
            ),
        },
        "indexing_errors": indexing_errors,
    }

    save_json(
        output_root / "merge_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nCombined annotations: {output_cases_dir}"
    )
    print(f"Summary: {output_root / 'merge_summary.json'}")
    print(f"Diagnostics: {diagnostics_path}")

    if args.strict and (
        problematic or indexing_errors
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
