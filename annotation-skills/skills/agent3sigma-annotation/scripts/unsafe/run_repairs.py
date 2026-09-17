#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KNOWN_ACTIONS = (
    "keep_annotation",
    "rerun_all_annotations",
    "repair_attack_chain",
    "repair_execution_chain",
    "repair_attack_and_execution_chains",
)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

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

            rows.append(obj)

    return rows


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def count_json_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.json"))


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selective semantic annotation repairs from semantic "
            "validator outputs."
        )
    )

    parser.add_argument(
        "--validation_root",
        type=Path,
        required=True,
        help=(
            "Semantic validation output directory containing "
            "all_cases.jsonl."
        ),
    )
    parser.add_argument(
        "--trajectory_root",
        type=Path,
        required=True,
        help=(
            "Directory containing original trajectory JSON files."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Output root for repair runs.",
    )

    parser.add_argument(
        "--category",
        choices=("all",) + KNOWN_ACTIONS,
        default="all",
        help="Run one category or all categories.",
    )
    parser.add_argument(
        "--limit_per_category",
        type=int,
        default=None,
        help=(
            "Only run the first N cases from each selected category. "
            "Useful for testing."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output_root before running.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Prepare inputs and print commands without calling annotators."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to annotator scripts.",
    )

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--target_max_tokens",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--primary_max_tokens",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--attack_max_tokens",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--execution_max_tokens",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--max_component_chars",
        type=int,
        default=3500,
    )

    return parser.parse_args()


# =============================================================================
# Trajectory indexing
# =============================================================================


def extract_trajectory_id(obj: dict[str, Any]) -> str | None:
    value = obj.get("trajectory_id")
    if value:
        return str(value)

    item = obj.get("item")
    if isinstance(item, dict):
        value = item.get("id")
        if value:
            return str(value)

    return None


def build_trajectory_index(
    trajectory_root: Path,
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: defaultdict[str, list[Path]] = defaultdict(list)

    paths = sorted(trajectory_root.glob("*.json"))

    if not paths:
        raise FileNotFoundError(
            f"No JSON files found in trajectory_root: {trajectory_root}"
        )

    for path in paths:
        try:
            obj = load_json(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read trajectory file {path}: {exc}"
            ) from exc

        trajectory_id = extract_trajectory_id(obj)
        if not trajectory_id:
            continue

        if trajectory_id in index:
            duplicates[trajectory_id].append(path)
            continue

        index[trajectory_id] = path

    if duplicates:
        details = "\n".join(
            f"{tid}: {[str(p) for p in paths]}"
            for tid, paths in sorted(duplicates.items())
        )
        raise RuntimeError(
            "Duplicate trajectory IDs found:\n" + details
        )

    return index


def resolve_existing_path(
    value: Any,
    *,
    a3s_root: Path,
) -> Path | None:
    if not value:
        return None

    path = Path(str(value))

    if path.is_file():
        return path.resolve()

    if not path.is_absolute():
        candidate = a3s_root / path
        if candidate.is_file():
            return candidate.resolve()

    return None


def resolve_stage_path(
    combined: dict[str, Any],
    *,
    stage: str,
    a3s_root: Path,
) -> Path | None:
    metadata = combined.get("_metadata")
    if not isinstance(metadata, dict):
        return None

    metadata_key = {
        "target": "target_path",
        "primary": "primary_path",
        "attack_chain": "attack_chain_path",
        "execution_chain": "execution_chain_path",
    }.get(stage)

    # Prefer explicit stage paths. In some combined annotations,
    # merged_from.target points to the trajectory input rather than
    # the actual target annotation.
    if metadata_key:
        path = resolve_existing_path(
            metadata.get(metadata_key),
            a3s_root=a3s_root,
        )
        if path is not None:
            return path

    merged_from = metadata.get("merged_from")
    if not isinstance(merged_from, dict):
        return None

    values = merged_from.get(stage)

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return None

    expected_suffix = {
        "target": ".target.json",
        "primary": ".primary.json",
        "attack_chain": ".attack_chain.json",
        "execution_chain": ".execution.json",
    }.get(stage)

    resolved: list[Path] = []

    for value in values:
        path = resolve_existing_path(
            value,
            a3s_root=a3s_root,
        )
        if path is not None:
            resolved.append(path)

    if expected_suffix:
        for path in resolved:
            if path.name.endswith(expected_suffix):
                return path

    return resolved[0] if resolved else None


# =============================================================================
# Combined annotation extraction
# =============================================================================


def get_annotation_path(
    validation_case: dict[str, Any],
) -> Path:
    files = validation_case.get("files")

    if not isinstance(files, dict):
        raise ValueError(
            f"Validation case has no files object: "
            f"{validation_case.get('case_id')}"
        )

    value = files.get("annotation")
    if not value:
        raise ValueError(
            f"Validation case has no annotation path: "
            f"{validation_case.get('case_id')}"
        )

    return Path(str(value))


def find_first_dict(
    obj: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return None


def extract_target_annotation(
    combined: dict[str, Any],
) -> dict[str, Any]:
    target = find_first_dict(
        combined,
        (
            "target_annotation",
            "target",
        ),
    )

    if target is not None:
        return target

    if (
        "target_status" in combined
        or "target_unsafe_action" in combined
    ):
        return {
            "trajectory_id": combined.get("trajectory_id"),
            "target_status": combined.get("target_status"),
            "attack_success": combined.get("attack_success"),
            "risk_category": combined.get("risk_category"),
            "target_unsafe_action": combined.get(
                "target_unsafe_action"
            ),
            "confidence": combined.get("target_confidence"),
            "needs_review": combined.get("target_needs_review"),
        }

    raise ValueError(
        "Could not extract target annotation from combined annotation"
    )


def extract_primary_annotation(
    combined: dict[str, Any],
) -> dict[str, Any]:
    primary_wrapper = find_first_dict(
        combined,
        (
            "primary_annotation",
            "primary_cause_annotation",
            "primary_attribution_annotation",
        ),
    )

    if primary_wrapper is not None:
        return primary_wrapper

    primary_component = combined.get(
        "primary_attribution_component"
    )

    if isinstance(primary_component, dict):
        return {
            "trajectory_id": combined.get("trajectory_id"),
            "target_component_id": combined.get(
                "target_component_id"
            )
            or (
                combined.get("target_unsafe_action", {})
                if isinstance(
                    combined.get("target_unsafe_action"),
                    dict,
                )
                else {}
            ).get("component_id"),
            "primary_attribution_component": primary_component,
            "confidence": combined.get("primary_confidence"),
            "needs_review": combined.get(
                "primary_needs_review",
                False,
            ),
            "review_reason": combined.get(
                "primary_review_reason"
            ),
        }

    raise ValueError(
        "Could not extract primary annotation from combined annotation"
    )


def extract_attack_annotation(
    combined: dict[str, Any],
) -> dict[str, Any]:
    attack_wrapper = find_first_dict(
        combined,
        (
            "attack_chain_annotation",
            "attack_annotation",
        ),
    )

    if attack_wrapper is not None:
        return attack_wrapper

    for key in (
        "attack_chain",
        "attack_chain_components",
    ):
        value = combined.get(key)
        if isinstance(value, list):
            return {
                "trajectory_id": combined.get("trajectory_id"),
                "attack_chain": value,
                "confidence": combined.get(
                    "attack_chain_confidence"
                ),
                "needs_review": combined.get(
                    "attack_chain_needs_review",
                    False,
                ),
            }

    raise ValueError(
        "Could not extract attack-chain annotation "
        "from combined annotation"
    )


# =============================================================================
# Input preparation
# =============================================================================


def select_cases(
    all_cases: list[dict[str, Any]],
    category: str,
    limit_per_category: int | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )

    for case in all_cases:
        action = case.get("recommended_action")

        if action not in KNOWN_ACTIONS:
            continue

        if category != "all" and action != category:
            continue

        grouped[str(action)].append(case)

    for action in grouped:
        grouped[action].sort(
            key=lambda obj: (
                str(obj.get("trajectory_id", "")),
                str(obj.get("case_id", "")),
            )
        )

        if limit_per_category is not None:
            grouped[action] = grouped[action][
                :limit_per_category
            ]

    return dict(grouped)


def prepare_category_inputs(
    *,
    action: str,
    cases: list[dict[str, Any]],
    trajectory_index: dict[str, Path],
    category_root: Path,
    a3s_root: Path,
) -> dict[str, Path]:
    source_trajectory_cases = (
        category_root / "inputs" / "trajectories" / "cases"
    )
    source_combined_cases = (
        category_root / "inputs" / "combined" / "cases"
    )

    fixed_target_cases = (
        category_root / "fixed" / "target" / "cases"
    )
    fixed_primary_cases = (
        category_root / "fixed" / "primary" / "cases"
    )
    fixed_attack_cases = (
        category_root / "fixed" / "attack_chain" / "cases"
    )

    validation_cases = (
        category_root / "inputs" / "validation"
    )

    for path in (
        source_trajectory_cases,
        source_combined_cases,
        fixed_target_cases,
        fixed_primary_cases,
        fixed_attack_cases,
        validation_cases,
    ):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = category_root / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for case in cases:
            trajectory_id = str(case["trajectory_id"])
            case_id = str(case["case_id"])

            trajectory_path = trajectory_index.get(
                trajectory_id
            )
            if trajectory_path is None:
                raise FileNotFoundError(
                    f"Trajectory not found for {trajectory_id}"
                )

            annotation_path = get_annotation_path(case)
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Combined annotation not found: "
                    f"{annotation_path}"
                )

            combined = load_json(annotation_path)

            trajectory_destination = (
                source_trajectory_cases / trajectory_path.name
            )
            combined_destination = (
                source_combined_cases / annotation_path.name
            )

            copy_file(
                trajectory_path,
                trajectory_destination,
            )
            copy_file(
                annotation_path,
                combined_destination,
            )

            write_json(
                validation_cases
                / f"{case_id}.semantic_validation.json",
                case,
            )

            # Selective-repair categories reuse original stage outputs.
            #
            # keep_annotation:
            #   Keep the combined annotation unchanged.
            #
            # rerun_all_annotations:
            #   Regenerate target, primary, attack, and execution.
            #
            # repair_attack_chain / repair_attack_and_execution_chains:
            #   Keep target and primary, then regenerate attack/execution.
            #
            # repair_execution_chain:
            #   Keep target, primary, and attack, then regenerate execution.
            if action in (
                "repair_attack_chain",
                "repair_execution_chain",
                "repair_attack_and_execution_chains",
            ):
                target_path = resolve_stage_path(
                    combined,
                    stage="target",
                    a3s_root=a3s_root,
                )
                primary_path = resolve_stage_path(
                    combined,
                    stage="primary",
                    a3s_root=a3s_root,
                )

                if target_path is None:
                    raise FileNotFoundError(
                        "Could not resolve original target annotation "
                        f"for case {case_id}"
                    )

                if primary_path is None:
                    raise FileNotFoundError(
                        "Could not resolve original primary annotation "
                        f"for case {case_id}"
                    )

                # Validate that the original stage files contain JSON
                # objects before copying them.
                load_json(target_path)
                load_json(primary_path)

                copy_file(
                    target_path,
                    fixed_target_cases / target_path.name,
                )
                copy_file(
                    primary_path,
                    fixed_primary_cases / primary_path.name,
                )

            if action == "repair_execution_chain":
                attack_path = resolve_stage_path(
                    combined,
                    stage="attack_chain",
                    a3s_root=a3s_root,
                )

                if attack_path is None:
                    raise FileNotFoundError(
                        "Could not resolve original attack-chain "
                        f"annotation for case {case_id}"
                    )

                load_json(attack_path)

                copy_file(
                    attack_path,
                    fixed_attack_cases / attack_path.name,
                )

            manifest.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "trajectory_id": trajectory_id,
                        "recommended_action": action,
                        "trajectory": str(trajectory_path),
                        "combined_annotation": str(
                            annotation_path
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "trajectory_root": source_trajectory_cases.parent,
        "trajectory_cases": source_trajectory_cases,
        "combined_cases": source_combined_cases,
        "fixed_target_root": fixed_target_cases.parent,
        "fixed_target_cases": fixed_target_cases,
        "fixed_primary_root": fixed_primary_cases.parent,
        "fixed_primary_cases": fixed_primary_cases,
        "fixed_attack_root": fixed_attack_cases.parent,
        "fixed_attack_cases": fixed_attack_cases,
        "manifest": manifest_path,
    }



# =============================================================================
# Completed annotation assembly
# =============================================================================


def combined_base_name(path: Path) -> str:
    name = path.name

    if name.endswith(".combined.json"):
        return name[: -len(".combined.json")]

    if name.endswith(".json"):
        return name[:-5]

    return name


def stage_file_path(
    cases_dir: Path,
    *,
    base_name: str,
    suffix: str,
    required: bool = True,
) -> Path | None:
    path = cases_dir / f"{base_name}{suffix}"

    if path.is_file():
        return path

    if required:
        raise FileNotFoundError(
            f"Missing stage annotation: {path}"
        )

    return None


def assert_matching_trajectory_id(
    *,
    expected: str,
    stage_name: str,
    annotation: dict[str, Any],
    path: Path,
) -> None:
    actual = annotation.get("trajectory_id")

    if actual is None:
        raise ValueError(
            f"{stage_name} annotation has no trajectory_id: {path}"
        )

    if str(actual) != str(expected):
        raise ValueError(
            f"{stage_name} trajectory_id mismatch: "
            f"expected={expected!r}, actual={actual!r}, path={path}"
        )


def merge_repaired_stages(
    *,
    original_combined: dict[str, Any],
    trajectory_id: str,
    action: str,
    target_path: Path,
    primary_path: Path | None,
    attack_path: Path | None,
    execution_path: Path | None,
    validation_path: Path | None,
) -> dict[str, Any]:
    result = dict(original_combined)
    if action == "rerun_all_annotations":
        result.pop("_repair_seed", None)

    target = load_json(target_path)
    assert_matching_trajectory_id(
        expected=trajectory_id,
        stage_name="target",
        annotation=target,
        path=target_path,
    )

    primary = load_json(primary_path) if primary_path else None
    attack = load_json(attack_path) if attack_path else None
    execution = load_json(execution_path) if execution_path else None

    if primary is not None and primary_path is not None:
        assert_matching_trajectory_id(
            expected=trajectory_id,
            stage_name="primary",
            annotation=primary,
            path=primary_path,
        )

    if attack is not None and attack_path is not None:
        assert_matching_trajectory_id(
            expected=trajectory_id,
            stage_name="attack_chain",
            annotation=attack,
            path=attack_path,
        )

    if execution is not None and execution_path is not None:
        assert_matching_trajectory_id(
            expected=trajectory_id,
            stage_name="execution_chain",
            annotation=execution,
            path=execution_path,
        )

    # ------------------------------------------------------------------
    # Target-owned fields
    # ------------------------------------------------------------------

    for key in (
        "trajectory_id",
        "target_status",
        "attack_success",
        "risk_category",
        "target_unsafe_action",
    ):
        if key in target:
            result[key] = target[key]

    target_obj = target.get("target_unsafe_action")
    target_component_id = None

    if isinstance(target_obj, dict):
        target_component_id = target_obj.get("component_id")

    # Always overwrite the target component ID so a newly generated
    # no_target annotation cannot inherit a stale ID from the original
    # combined annotation.
    result["target_component_id"] = target_component_id

    # ------------------------------------------------------------------
    # No-target output
    # ------------------------------------------------------------------

    target_status = target.get("target_status")

    if target_status == "no_target" or not isinstance(target_obj, dict):
        # Explicitly clear all target-dependent fields inherited from
        # the original combined annotation.
        result["target_component_id"] = None
        result["target_unsafe_action"] = None

        result.pop("primary_attribution_component", None)
        result.pop("primary_component_id", None)

        result["attack_chain"] = []
        result["execution_chain"] = []

        result["confidence"] = target.get(
            "confidence",
            result.get("confidence", 0.5),
        )
        result["needs_review"] = target.get(
            "needs_review",
            False,
        )
        result["review_reason"] = target.get(
            "review_reason",
        )

    else:
        if primary is None or primary_path is None:
            raise FileNotFoundError(
                f"Missing primary annotation for non-no_target case "
                f"{trajectory_id}"
            )

        primary_component = primary.get(
            "primary_attribution_component"
        )

        if not isinstance(primary_component, dict):
            raise ValueError(
                f"Primary annotation has no valid "
                f"primary_attribution_component: {primary_path}"
            )

        result["primary_attribution_component"] = (
            primary_component
        )

        primary_component_id = primary_component.get(
            "component_id"
        )

        if primary_component_id is not None:
            result["primary_component_id"] = (
                primary_component_id
            )

        if attack is None or attack_path is None:
            raise FileNotFoundError(
                f"Missing attack-chain annotation for "
                f"{trajectory_id}"
            )

        if execution is None or execution_path is None:
            raise FileNotFoundError(
                f"Missing execution-chain annotation for "
                f"{trajectory_id}"
            )

        attack_chain = attack.get("attack_chain")
        execution_chain = execution.get("execution_chain")

        if not isinstance(attack_chain, list):
            raise ValueError(
                f"Invalid attack_chain in {attack_path}"
            )

        if not isinstance(execution_chain, list):
            raise ValueError(
                f"Invalid execution_chain in {execution_path}"
            )

        result["attack_chain"] = attack_chain
        result["execution_chain"] = execution_chain

        # The execution annotation is the final stage and normally
        # supplies the final confidence/review fields.
        final_stage = execution

        result["confidence"] = final_stage.get(
            "confidence",
            attack.get(
                "confidence",
                primary.get(
                    "confidence",
                    target.get(
                        "confidence",
                        result.get("confidence", 0.5),
                    ),
                ),
            ),
        )

        result["needs_review"] = bool(
            target.get("needs_review", False)
            or primary.get("needs_review", False)
            or attack.get("needs_review", False)
            or execution.get("needs_review", False)
        )

        review_reasons = [
            value
            for value in (
                target.get("review_reason"),
                primary.get("review_reason"),
                attack.get("review_reason"),
                execution.get("review_reason"),
            )
            if isinstance(value, str) and value.strip()
        ]

        result["review_reason"] = (
            " | ".join(review_reasons)
            if review_reasons
            else None
        )

    # ------------------------------------------------------------------
    # Repair metadata
    # ------------------------------------------------------------------

    metadata = result.get("_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    # Remove stale or stage-specific metadata inherited from the old
    # combined annotation. These fields may no longer correspond to
    # the repaired completed annotation.
    for stale_key in (
        "raw_model_output",
        "attack_chain_available",
        "attack_chain_used_as_guidance",
    ):
        metadata.pop(stale_key, None)

    merged_from = metadata.get("merged_from")
    if not isinstance(merged_from, dict):
        merged_from = {}
    else:
        merged_from = dict(merged_from)

    merged_from["target"] = [str(target_path)]

    merged_from["primary"] = (
        [str(primary_path)]
        if primary_path is not None
        else []
    )
    merged_from["attack_chain"] = (
        [str(attack_path)]
        if attack_path is not None
        else []
    )
    merged_from["execution_chain"] = (
        [str(execution_path)]
        if execution_path is not None
        else []
    )

    metadata["merged_from"] = merged_from
    metadata["target_path"] = str(target_path)
    metadata["primary_path"] = (
        str(primary_path)
        if primary_path is not None
        else None
    )
    metadata["attack_chain_path"] = (
        str(attack_path)
        if attack_path is not None
        else None
    )
    metadata["execution_chain_path"] = (
        str(execution_path)
        if execution_path is not None
        else None
    )
    metadata["semantic_repair_action"] = action
    metadata["semantic_validation_path"] = (
        str(validation_path)
        if validation_path is not None
        else None
    )
    metadata["merge_version"] = (
        "agent3sigma_semantic_repair_combined_v1"
    )

    result["_metadata"] = metadata
    result["trajectory_id"] = trajectory_id

    return result


def assemble_category_completed_annotations(
    *,
    action: str,
    category_root: Path,
    completed_cases_dir: Path,
    dry_run: bool,
) -> int:
    if dry_run:
        print(
            f"[dry-run] skip completed annotation assembly: "
            f"{action}"
        )
        return 0

    manifest_path = category_root / "manifest.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing category manifest: {manifest_path}"
        )

    records = load_jsonl(manifest_path)
    completed_cases_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_completed = 0

    for record in records:
        trajectory_id = str(record["trajectory_id"])
        case_id = str(record["case_id"])

        original_combined_path = Path(
            str(record["combined_annotation"])
        )

        if not original_combined_path.is_file():
            raise FileNotFoundError(
                f"Original combined annotation not found: "
                f"{original_combined_path}"
            )

        base_name = combined_base_name(
            original_combined_path
        )

        output_path = (
            completed_cases_dir
            / f"{base_name}.combined.json"
        )

        if action == "keep_annotation":
            copy_file(
                original_combined_path,
                output_path,
            )
            num_completed += 1
            print(
                f"[complete keep] {case_id} -> "
                f"{output_path.name}"
            )
            continue

        original_combined = load_json(
            original_combined_path
        )

        validation_path = (
            category_root
            / "inputs"
            / "validation"
            / f"{case_id}.semantic_validation.json"
        )

        generated_root = category_root / "generated"
        fixed_root = category_root / "fixed"

        if action == "rerun_all_annotations":
            target_cases = (
                generated_root / "target" / "cases"
            )
            primary_cases = (
                generated_root / "primary" / "cases"
            )
            attack_cases = (
                generated_root / "attack_chain" / "cases"
            )
            execution_cases = (
                generated_root
                / "execution_chain"
                / "cases"
            )

        elif action in (
            "repair_attack_chain",
            "repair_attack_and_execution_chains",
        ):
            target_cases = (
                fixed_root / "target" / "cases"
            )
            primary_cases = (
                fixed_root / "primary" / "cases"
            )
            attack_cases = (
                generated_root / "attack_chain" / "cases"
            )
            execution_cases = (
                generated_root
                / "execution_chain"
                / "cases"
            )

        elif action == "repair_execution_chain":
            target_cases = (
                fixed_root / "target" / "cases"
            )
            primary_cases = (
                fixed_root / "primary" / "cases"
            )
            attack_cases = (
                fixed_root / "attack_chain" / "cases"
            )
            execution_cases = (
                generated_root
                / "execution_chain"
                / "cases"
            )

        else:
            raise ValueError(
                f"Unsupported repair action: {action}"
            )

        target_path = stage_file_path(
            target_cases,
            base_name=base_name,
            suffix=".target.json",
        )

        target_annotation = load_json(target_path)
        target_status = target_annotation.get(
            "target_status"
        )
        target_obj = target_annotation.get(
            "target_unsafe_action"
        )

        is_no_target = (
            target_status == "no_target"
            or not isinstance(target_obj, dict)
        )

        primary_path = stage_file_path(
            primary_cases,
            base_name=base_name,
            suffix=".primary.json",
            required=not is_no_target,
        )
        attack_path = stage_file_path(
            attack_cases,
            base_name=base_name,
            suffix=".attack_chain.json",
            required=not is_no_target,
        )
        execution_path = stage_file_path(
            execution_cases,
            base_name=base_name,
            suffix=".execution.json",
            required=not is_no_target,
        )

        completed = merge_repaired_stages(
            original_combined=original_combined,
            trajectory_id=trajectory_id,
            action=action,
            target_path=target_path,
            primary_path=primary_path,
            attack_path=attack_path,
            execution_path=execution_path,
            validation_path=(
                validation_path
                if validation_path.is_file()
                else None
            ),
        )

        write_json(output_path, completed)

        num_completed += 1

        print(
            f"[complete repaired] {case_id} -> "
            f"{output_path.name}"
        )

    return num_completed


# =============================================================================
# Annotator invocation
# =============================================================================


def run_command(
    command: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    cwd: Path,
) -> None:
    printable = " \\\n  ".join(command)

    print()
    print("-" * 80)
    print(printable)
    print("-" * 80)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log_path.write_text(
            "[dry-run]\n" + printable + "\n",
            encoding="utf-8",
        )
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. "
            f"See log: {log_path}"
        )


def common_api_args(
    args: argparse.Namespace,
) -> list[str]:
    result = [
        "--model",
        args.model,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.temperature),
    ]

    if args.base_url:
        result.extend(
            [
                "--base_url",
                args.base_url,
            ]
        )

    return result


def force_args(args: argparse.Namespace) -> list[str]:
    return ["--force"] if args.force else []


def run_target(
    *,
    script: Path,
    input_cases: Path,
    output_root: Path,
    log_path: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_dir",
        str(input_cases),
        "--output_root",
        str(output_root),
        *common_api_args(args),
        "--max_tokens",
        str(args.target_max_tokens),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        *force_args(args),
    ]

    run_command(
        command,
        log_path=log_path,
        dry_run=args.dry_run,
        cwd=cwd,
    )


def run_primary(
    *,
    script: Path,
    input_cases: Path,
    target_cases: Path,
    output_root: Path,
    log_path: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_dir",
        str(input_cases),
        "--target_dir",
        str(target_cases),
        "--output_root",
        str(output_root),
        *common_api_args(args),
        "--max_tokens",
        str(args.primary_max_tokens),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--sleep_seconds",
        str(args.sleep_seconds),
        *force_args(args),
    ]

    run_command(
        command,
        log_path=log_path,
        dry_run=args.dry_run,
        cwd=cwd,
    )


def run_attack(
    *,
    script: Path,
    input_cases: Path,
    target_cases: Path,
    primary_cases: Path,
    output_root: Path,
    log_path: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_dir",
        str(input_cases),
        "--target_dir",
        str(target_cases),
        "--primary_dir",
        str(primary_cases),
        "--output_root",
        str(output_root),
        *common_api_args(args),
        "--max_tokens",
        str(args.attack_max_tokens),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--sleep_seconds",
        str(args.sleep_seconds),
        *force_args(args),
    ]

    run_command(
        command,
        log_path=log_path,
        dry_run=args.dry_run,
        cwd=cwd,
    )


def run_execution(
    *,
    script: Path,
    input_root: Path,
    target_root: Path,
    primary_root: Path,
    attack_root: Path,
    output_root: Path,
    log_path: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_root",
        str(input_root),
        "--target_root",
        str(target_root),
        "--primary_root",
        str(primary_root),
        "--attack_chain_root",
        str(attack_root),
        "--output_root",
        str(output_root),
        *common_api_args(args),
        "--max_tokens",
        str(args.execution_max_tokens),
        "--timeout",
        str(args.timeout),
        "--max_retries",
        str(args.retries),
        "--sleep_seconds",
        str(args.sleep_seconds),
        "--max_component_chars",
        str(args.max_component_chars),
        *force_args(args),
    ]

    run_command(
        command,
        log_path=log_path,
        dry_run=args.dry_run,
        cwd=cwd,
    )


# =============================================================================
# Category execution
# =============================================================================


def run_category(
    *,
    action: str,
    paths: dict[str, Path],
    category_root: Path,
    args: argparse.Namespace,
    a3s_root: Path,
) -> None:
    scripts_root = Path(__file__).resolve().parent

    target_script = (
        scripts_root / "annotate_target.py"
    )
    primary_script = (
        scripts_root / "annotate_primary.py"
    )
    attack_script = (
        scripts_root / "annotate_attack_chain.py"
    )
    execution_script = (
        scripts_root / "annotate_execution_chain.py"
    )

    for script in (
        target_script,
        primary_script,
        attack_script,
        execution_script,
    ):
        if not script.is_file():
            raise FileNotFoundError(
                f"Annotator script not found: {script}"
            )

    generated_root = category_root / "generated"
    logs_root = category_root / "logs"

    target_out = generated_root / "target"
    primary_out = generated_root / "primary"
    attack_out = generated_root / "attack_chain"
    execution_out = generated_root / "execution_chain"

    if action == "keep_annotation":
        print(
            f"[keep] {count_json_files(paths['combined_cases'])} "
            f"combined annotations"
        )
        return

    if action == "rerun_all_annotations":
        run_target(
            script=target_script,
            input_cases=paths["trajectory_cases"],
            output_root=target_out,
            log_path=logs_root / "target.log",
            args=args,
            cwd=a3s_root,
        )

        run_primary(
            script=primary_script,
            input_cases=paths["trajectory_cases"],
            target_cases=target_out / "cases",
            output_root=primary_out,
            log_path=logs_root / "primary.log",
            args=args,
            cwd=a3s_root,
        )

        run_attack(
            script=attack_script,
            input_cases=paths["trajectory_cases"],
            target_cases=target_out / "cases",
            primary_cases=primary_out / "cases",
            output_root=attack_out,
            log_path=logs_root / "attack_chain.log",
            args=args,
            cwd=a3s_root,
        )

        run_execution(
            script=execution_script,
            input_root=paths["trajectory_root"],
            target_root=target_out,
            primary_root=primary_out,
            attack_root=attack_out,
            output_root=execution_out,
            log_path=logs_root / "execution_chain.log",
            args=args,
            cwd=a3s_root,
        )
        return

    if action in (
        "repair_attack_chain",
        "repair_attack_and_execution_chains",
    ):
        run_attack(
            script=attack_script,
            input_cases=paths["trajectory_cases"],
            target_cases=paths["fixed_target_cases"],
            primary_cases=paths["fixed_primary_cases"],
            output_root=attack_out,
            log_path=logs_root / "attack_chain.log",
            args=args,
            cwd=a3s_root,
        )

        run_execution(
            script=execution_script,
            input_root=paths["trajectory_root"],
            target_root=paths["fixed_target_root"],
            primary_root=paths["fixed_primary_root"],
            attack_root=attack_out,
            output_root=execution_out,
            log_path=logs_root / "execution_chain.log",
            args=args,
            cwd=a3s_root,
        )
        return

    if action == "repair_execution_chain":
        run_execution(
            script=execution_script,
            input_root=paths["trajectory_root"],
            target_root=paths["fixed_target_root"],
            primary_root=paths["fixed_primary_root"],
            attack_root=paths["fixed_attack_root"],
            output_root=execution_out,
            log_path=logs_root / "execution_chain.log",
            args=args,
            cwd=a3s_root,
        )
        return

    raise ValueError(f"Unsupported action: {action}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    validation_root = args.validation_root.resolve()
    trajectory_root = args.trajectory_root.resolve()
    output_root = args.output_root.resolve()

    a3s_root = Path(__file__).resolve().parents[2]

    all_cases_path = validation_root / "all_cases.jsonl"

    if not all_cases_path.is_file():
        raise FileNotFoundError(all_cases_path)

    if not trajectory_root.is_dir():
        raise NotADirectoryError(trajectory_root)

    if args.limit_per_category is not None:
        if args.limit_per_category <= 0:
            raise ValueError(
                "--limit_per_category must be positive"
            )

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    all_cases = load_jsonl(all_cases_path)
    trajectory_index = build_trajectory_index(
        trajectory_root
    )

    grouped = select_cases(
        all_cases,
        category=args.category,
        limit_per_category=args.limit_per_category,
    )

    summary_counts: Counter[str] = Counter()
    completed_counts: Counter[str] = Counter()

    completed_cases_dir = (
        output_root
        / "completed_annotations"
        / "cases"
    )

    completed_cases_dir.mkdir(parents=True, exist_ok=True)

    for action in KNOWN_ACTIONS:
        cases = grouped.get(action, [])
        if not cases:
            continue

        category_root = output_root / action

        print()
        print("=" * 80)
        print(f"Category: {action}")
        print(f"Cases:    {len(cases)}")
        print("=" * 80)

        paths = prepare_category_inputs(
            action=action,
            cases=cases,
            trajectory_index=trajectory_index,
            category_root=category_root,
            a3s_root=a3s_root,
        )

        run_category(
            action=action,
            paths=paths,
            category_root=category_root,
            args=args,
            a3s_root=a3s_root,
        )

        completed_counts[action] = (
            assemble_category_completed_annotations(
                action=action,
                category_root=category_root,
                completed_cases_dir=completed_cases_dir,
                dry_run=args.dry_run,
            )
        )

        summary_counts[action] = len(cases)

    summary = {
        "validation_root": str(validation_root),
        "trajectory_root": str(trajectory_root),
        "output_root": str(output_root),
        "category": args.category,
        "limit_per_category": args.limit_per_category,
        "dry_run": args.dry_run,
        "counts": {
            action: summary_counts[action]
            for action in KNOWN_ACTIONS
        },
        "completed_counts": {
            action: completed_counts[action]
            for action in KNOWN_ACTIONS
        },
        "completed_annotations_dir": str(
            completed_cases_dir
        ),
        "num_completed_annotations": sum(
            completed_counts.values()
        ),
    }

    write_json(output_root / "run_summary.json", summary)

    print()
    print("=" * 80)
    print("Semantic repair runner finished")
    print("=" * 80)

    for action in KNOWN_ACTIONS:
        print(
            f"{action:36s}: "
            f"selected={summary_counts[action]}, "
            f"completed={completed_counts[action]}"
        )

    print(f"\nOutput root: {output_root}")
    print(
        f"Completed annotations: {completed_cases_dir}"
    )
    print(
        f"Number completed: {sum(completed_counts.values())}"
    )


if __name__ == "__main__":
    main()
