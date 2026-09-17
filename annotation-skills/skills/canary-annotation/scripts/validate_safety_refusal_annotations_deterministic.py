#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic validation for merged safety-refusal annotations.

Expected merged annotation structure:

{
  "trajectory_id": "...",
  "target_type": "safety_refusal",
  "attack_success": false,
  "target_action": {
    "component_id": "C...",
    "role": "...",
    "summary": "...",
    "evidence": "..."
  },
  "primary_attribution_component": {
    "component_id": "C...",
    "role": "...",
    "summary": "...",
    "evidence": "..."
  },
  "confidence": 0.9,
  "needs_review": false,
  "review_reason": null,
  "_metadata": {
    "target_annotation_path": "...",
    "primary_annotation_path": "...",
    ...
  }
}

The validator checks:
- schema validity;
- target_type == safety_refusal;
- target and primary component IDs exist in the normalized trajectory;
- primary occurs strictly before target;
- annotated roles agree with the normalized trajectory;
- merged claims agree with original target/primary annotation files;
- safety-refusal-specific invariants;
- duplicate and file-matching problems.

Outputs:

output_root/
  case_validation/
    <case>.deterministic_validation.json
  pass_cases/
    <case>.annotation.json
  fail_cases/
    <case>.annotation.json
  all_cases.jsonl
  pass_cases.jsonl
  fail_cases.jsonl
  semantic_input_cases.jsonl
  deterministic_validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


VALID_COMPONENT_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
    "memory",
}

COMPONENT_ID_PATTERN = re.compile(r"^C\d+$")


# =============================================================================
# Basic IO
# =============================================================================


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected JSON object at {path}, "
            f"got {type(obj).__name__}."
        )

    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        path.unlink()

    path.touch()


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


def annotation_case_key(path: Path) -> str:
    name = path.name

    for suffix in (
        ".annotation.json",
        ".primary.json",
        ".target.json",
        ".json",
    ):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return path.stem


# =============================================================================
# Normalized trajectory loading
# =============================================================================


def extract_trajectory(
    normalized: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    trajectory_id = normalized.get("trajectory_id")

    annotation_view = normalized.get("annotation_view")

    if isinstance(annotation_view, dict):
        trajectory_id = (
            trajectory_id
            or annotation_view.get("trajectory_id")
        )
        raw_trajectory = annotation_view.get("trajectory")
    else:
        raw_trajectory = None

    if not isinstance(raw_trajectory, list):
        raw_trajectory = normalized.get("trajectory")

    if not isinstance(raw_trajectory, list):
        for key in (
            "components_metadata",
            "components",
            "normalized_components",
        ):
            candidate = normalized.get(key)

            if isinstance(candidate, list):
                raw_trajectory = candidate
                break

    if not isinstance(raw_trajectory, list):
        raise ValueError(
            "Normalized case contains no trajectory or component list."
        )

    trajectory: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_component in raw_trajectory:
        if not isinstance(raw_component, dict):
            continue

        component_id = raw_component.get("component_id")

        if not isinstance(component_id, str):
            continue

        component_id = component_id.strip()

        if not component_id:
            continue

        if component_id in seen_ids:
            raise ValueError(
                f"Duplicate component_id in trajectory: "
                f"{component_id!r}"
            )

        seen_ids.add(component_id)

        role = raw_component.get("role")

        content = raw_component.get("content")

        if content is None:
            content = raw_component.get("content_full")

        if content is None:
            content = raw_component.get("text")

        trajectory.append({
            "component_id": component_id,
            "role": role,
            "content": str(content or ""),
            "_raw": raw_component,
        })

    if not trajectory:
        raise ValueError(
            "No valid trajectory components were extracted."
        )

    return (
        str(trajectory_id) if trajectory_id is not None else None,
        trajectory,
    )


def build_normalized_indices(
    normalized_dir: Path,
) -> dict[str, dict[str, Path]]:
    by_filename: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    by_safe_stem: dict[str, Path] = {}
    by_trajectory_id: dict[str, Path] = {}

    for path in sorted(normalized_dir.glob("*.json")):
        by_filename[path.name] = path
        by_stem[path.stem] = path
        by_safe_stem[safe_name(path.stem)] = path

        try:
            normalized = load_json(path)
            trajectory_id, _ = extract_trajectory(normalized)
        except Exception:
            continue

        if trajectory_id is None:
            continue

        if trajectory_id in by_trajectory_id:
            previous = by_trajectory_id[trajectory_id]

            raise ValueError(
                f"Duplicate normalized trajectory_id "
                f"{trajectory_id!r}: {previous} and {path}"
            )

        by_trajectory_id[trajectory_id] = path

    return {
        "by_filename": by_filename,
        "by_stem": by_stem,
        "by_safe_stem": by_safe_stem,
        "by_trajectory_id": by_trajectory_id,
    }


def find_normalized_path(
    *,
    annotation_path: Path,
    annotation: dict[str, Any],
    indices: dict[str, dict[str, Path]],
) -> Path:
    metadata = annotation.get("_metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    candidate_paths: list[str] = []

    for key in (
        "source_path",
        "normalized_path",
        "normalized_case_path",
    ):
        value = metadata.get(key)

        if value:
            candidate_paths.append(str(value))

    target_metadata = metadata.get("target_metadata")

    if isinstance(target_metadata, dict):
        value = target_metadata.get("source_path")

        if value:
            candidate_paths.append(str(value))

    primary_metadata = metadata.get("primary_metadata")

    if isinstance(primary_metadata, dict):
        value = primary_metadata.get("source_path")

        if value:
            candidate_paths.append(str(value))

    for raw_path in candidate_paths:
        source = Path(raw_path)

        for candidate in (
            indices["by_filename"].get(source.name),
            indices["by_stem"].get(source.stem),
            indices["by_safe_stem"].get(
                safe_name(source.stem)
            ),
        ):
            if candidate is not None:
                return candidate

    trajectory_id = annotation.get("trajectory_id")

    if trajectory_id is not None:
        candidate = indices["by_trajectory_id"].get(
            str(trajectory_id)
        )

        if candidate is not None:
            return candidate

    case_key = annotation_case_key(annotation_path)

    possible_stems = [
        case_key,
        case_key.replace(
            "chain_task_",
            "chain__task_",
            1,
        ),
    ]

    for stem in possible_stems:
        for candidate in (
            indices["by_stem"].get(stem),
            indices["by_safe_stem"].get(
                safe_name(stem)
            ),
            indices["by_filename"].get(
                stem + ".json"
            ),
        ):
            if candidate is not None:
                return candidate

    raise FileNotFoundError(
        "Could not match merged annotation to normalized trajectory: "
        f"{annotation_path.name}, "
        f"trajectory_id={trajectory_id!r}"
    )


# =============================================================================
# Validation helpers
# =============================================================================


def is_valid_component_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and COMPONENT_ID_PATTERN.fullmatch(value) is not None
    )


def normalize_role(value: Any) -> str | None:
    if value is None:
        return None

    role = str(value).strip().lower()

    return role or None


def get_component_map(
    trajectory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        component["component_id"]: component
        for component in trajectory
    }


def get_component_order(
    trajectory: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        component["component_id"]: index
        for index, component in enumerate(trajectory)
    }


def add_error(
    errors: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    errors.append({
        "code": code,
        "message": message,
    })


def add_warning(
    warnings: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    warnings.append({
        "code": code,
        "message": message,
    })


def check_text_field(
    *,
    obj: dict[str, Any],
    key: str,
    prefix: str,
    warnings: list[dict[str, str]],
) -> None:
    value = obj.get(key)

    if not isinstance(value, str) or not value.strip():
        add_warning(
            warnings,
            f"{prefix}_{key}_missing",
            f"{prefix}.{key} is empty or missing.",
        )


def load_optional_source_annotation(
    path_value: Any,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    if not path_value:
        return None, None, None

    path = Path(str(path_value))

    if not path.exists():
        return (
            path,
            None,
            f"Source annotation path does not exist: {path}",
        )

    try:
        return path, load_json(path), None
    except Exception as exc:
        return (
            path,
            None,
            f"Failed to load source annotation {path}: {exc!r}",
        )


# =============================================================================
# Per-case deterministic validation
# =============================================================================


def validate_one(
    *,
    annotation_path: Path,
    normalized_indices: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checks: dict[str, Any] = {}

    try:
        annotation = load_json(annotation_path)
    except Exception as exc:
        return {
            "schema_version": (
                "safety_refusal_deterministic_validation_v1"
            ),
            "case_id": annotation_case_key(annotation_path),
            "annotation_path": str(annotation_path),
            "normalized_path": None,
            "trajectory_id": None,
            "status": "fail",
            "passed": False,
            "errors": [{
                "code": "annotation_load_error",
                "message": repr(exc),
            }],
            "warnings": [],
            "checks": {},
        }

    case_id = annotation_case_key(annotation_path)
    trajectory_id = annotation.get("trajectory_id")

    # ------------------------------------------------------------------
    # Top-level schema
    # ------------------------------------------------------------------

    target_type = annotation.get("target_type")

    checks["target_type"] = target_type

    if target_type != "safety_refusal":
        add_error(
            errors,
            "invalid_target_type",
            "Merged annotation must have "
            "target_type='safety_refusal', "
            f"got {target_type!r}.",
        )

    attack_success = annotation.get("attack_success")

    checks["attack_success"] = attack_success

    if attack_success is not False:
        add_warning(
            warnings,
            "unexpected_attack_success",
            "A safety_refusal annotation normally has "
            f"attack_success=false, got {attack_success!r}.",
        )

    target = annotation.get("target_action")

    if not isinstance(target, dict):
        target = {}
        add_error(
            errors,
            "missing_target_action",
            "Missing or invalid target_action object.",
        )

    primary = annotation.get(
        "primary_attribution_component"
    )

    if not isinstance(primary, dict):
        primary = {}
        add_error(
            errors,
            "missing_primary_attribution_component",
            "Missing or invalid "
            "primary_attribution_component object.",
        )

    target_id = target.get("component_id")
    primary_id = primary.get("component_id")

    checks["target_component_id"] = target_id
    checks["primary_component_id"] = primary_id

    if not is_valid_component_id(target_id):
        add_error(
            errors,
            "invalid_target_component_id",
            f"Invalid target component ID: {target_id!r}.",
        )

    if not is_valid_component_id(primary_id):
        add_error(
            errors,
            "invalid_primary_component_id",
            f"Invalid primary component ID: {primary_id!r}.",
        )

    if (
        is_valid_component_id(target_id)
        and is_valid_component_id(primary_id)
        and target_id == primary_id
    ):
        add_error(
            errors,
            "primary_equals_target",
            "Primary attribution component cannot be "
            "the fixed safety-refusal target itself.",
        )

    check_text_field(
        obj=target,
        key="summary",
        prefix="target",
        warnings=warnings,
    )
    check_text_field(
        obj=target,
        key="evidence",
        prefix="target",
        warnings=warnings,
    )
    check_text_field(
        obj=primary,
        key="summary",
        prefix="primary",
        warnings=warnings,
    )
    check_text_field(
        obj=primary,
        key="evidence",
        prefix="primary",
        warnings=warnings,
    )

    # ------------------------------------------------------------------
    # Match original normalized trajectory
    # ------------------------------------------------------------------

    normalized_path: Path | None = None
    normalized_trajectory_id: str | None = None
    trajectory: list[dict[str, Any]] = []

    try:
        normalized_path = find_normalized_path(
            annotation_path=annotation_path,
            annotation=annotation,
            indices=normalized_indices,
        )

        normalized = load_json(normalized_path)

        (
            normalized_trajectory_id,
            trajectory,
        ) = extract_trajectory(normalized)

    except Exception as exc:
        add_error(
            errors,
            "normalized_case_match_error",
            repr(exc),
        )

    checks["normalized_trajectory_id"] = (
        normalized_trajectory_id
    )
    checks["num_trajectory_components"] = len(trajectory)

    if (
        trajectory_id is not None
        and normalized_trajectory_id is not None
        and str(trajectory_id) != str(normalized_trajectory_id)
    ):
        add_error(
            errors,
            "trajectory_id_mismatch",
            "Merged annotation trajectory_id does not match "
            "normalized trajectory: "
            f"annotation={trajectory_id!r}, "
            f"normalized={normalized_trajectory_id!r}.",
        )

    component_map = get_component_map(trajectory)
    component_order = get_component_order(trajectory)

    # ------------------------------------------------------------------
    # Component existence and order
    # ------------------------------------------------------------------

    if is_valid_component_id(target_id):
        if target_id not in component_map:
            add_error(
                errors,
                "target_component_missing",
                f"Target component {target_id!r} is absent "
                "from the normalized trajectory.",
            )

    if is_valid_component_id(primary_id):
        if primary_id not in component_map:
            add_error(
                errors,
                "primary_component_missing",
                f"Primary component {primary_id!r} is absent "
                "from the normalized trajectory.",
            )

    target_index = component_order.get(target_id)
    primary_index = component_order.get(primary_id)

    checks["target_index"] = target_index
    checks["primary_index"] = primary_index

    if target_index is not None and primary_index is not None:
        checks["distance_primary_to_target"] = (
            target_index - primary_index
        )

        if primary_index >= target_index:
            add_error(
                errors,
                "primary_not_before_target",
                "Primary attribution component must occur "
                "strictly before the safety-refusal target: "
                f"primary={primary_id}@{primary_index}, "
                f"target={target_id}@{target_index}.",
            )

    # ------------------------------------------------------------------
    # Role consistency
    # ------------------------------------------------------------------

    if target_id in component_map:
        actual_target_role = normalize_role(
            component_map[target_id].get("role")
        )
        annotated_target_role = normalize_role(
            target.get("role")
        )

        checks["actual_target_role"] = actual_target_role
        checks["annotated_target_role"] = (
            annotated_target_role
        )

        if (
            annotated_target_role is not None
            and annotated_target_role != actual_target_role
        ):
            add_error(
                errors,
                "target_role_mismatch",
                "Annotated target role does not match "
                "the normalized trajectory: "
                f"annotated={annotated_target_role!r}, "
                f"actual={actual_target_role!r}.",
            )

        if actual_target_role not in {"assistant", "tool"}:
            add_warning(
                warnings,
                "unusual_safety_refusal_target_role",
                "Safety-refusal target is normally represented by an "
                "assistant message or an agent-side tool/action component, "
                f"but actual role is {actual_target_role!r}.",
            )

    if primary_id in component_map:
        actual_primary_role = normalize_role(
            component_map[primary_id].get("role")
        )
        annotated_primary_role = normalize_role(
            primary.get("role")
        )

        checks["actual_primary_role"] = actual_primary_role
        checks["annotated_primary_role"] = (
            annotated_primary_role
        )

        if (
            actual_primary_role is not None
            and actual_primary_role not in VALID_COMPONENT_ROLES
        ):
            add_warning(
                warnings,
                "unusual_primary_role",
                f"Primary component has unusual role "
                f"{actual_primary_role!r}.",
            )

        if (
            annotated_primary_role is not None
            and annotated_primary_role != actual_primary_role
        ):
            add_error(
                errors,
                "primary_role_mismatch",
                "Annotated primary role does not match "
                "the normalized trajectory: "
                f"annotated={annotated_primary_role!r}, "
                f"actual={actual_primary_role!r}.",
            )

    # ------------------------------------------------------------------
    # Source target/primary annotation consistency
    # ------------------------------------------------------------------

    metadata = annotation.get("_metadata")

    if not isinstance(metadata, dict):
        metadata = {}
        add_warning(
            warnings,
            "missing_metadata",
            "Merged annotation has no valid _metadata object.",
        )

    target_source_path, target_source, target_source_error = (
        load_optional_source_annotation(
            metadata.get("target_annotation_path")
        )
    )

    primary_source_path, primary_source, primary_source_error = (
        load_optional_source_annotation(
            metadata.get("primary_annotation_path")
        )
    )

    checks["target_source_path"] = (
        str(target_source_path)
        if target_source_path is not None
        else None
    )
    checks["primary_source_path"] = (
        str(primary_source_path)
        if primary_source_path is not None
        else None
    )

    if target_source_error:
        add_warning(
            warnings,
            "target_source_load_error",
            target_source_error,
        )

    if primary_source_error:
        add_warning(
            warnings,
            "primary_source_load_error",
            primary_source_error,
        )

    if isinstance(target_source, dict):
        source_target_type = target_source.get("target_type")
        source_target = target_source.get("target_action")

        if source_target_type != "safety_refusal":
            add_error(
                errors,
                "source_target_type_mismatch",
                "Original target annotation is not "
                "safety_refusal: "
                f"{source_target_type!r}.",
            )

        if isinstance(source_target, dict):
            source_target_id = source_target.get(
                "component_id"
            )

            if source_target_id != target_id:
                add_error(
                    errors,
                    "source_target_component_mismatch",
                    "Merged target component differs from "
                    "the original target annotation: "
                    f"merged={target_id!r}, "
                    f"source={source_target_id!r}.",
                )
        else:
            add_error(
                errors,
                "source_target_action_missing",
                "Original target annotation has no valid "
                "target_action object.",
            )

        source_trajectory_id = target_source.get(
            "trajectory_id"
        )

        if (
            trajectory_id is not None
            and source_trajectory_id is not None
            and str(trajectory_id) != str(source_trajectory_id)
        ):
            add_error(
                errors,
                "source_target_trajectory_mismatch",
                "Merged trajectory_id differs from target "
                "annotation trajectory_id.",
            )

    if isinstance(primary_source, dict):
        source_primary_type = primary_source.get(
            "target_type"
        )

        if (
            source_primary_type is not None
            and source_primary_type != "safety_refusal"
        ):
            add_error(
                errors,
                "source_primary_target_type_mismatch",
                "Original primary annotation has inconsistent "
                f"target_type={source_primary_type!r}.",
            )

        source_primary = primary_source.get(
            "primary_attribution_component"
        )

        if isinstance(source_primary, dict):
            source_primary_id = source_primary.get(
                "component_id"
            )

            if source_primary_id != primary_id:
                add_error(
                    errors,
                    "source_primary_component_mismatch",
                    "Merged primary component differs from "
                    "the original primary annotation: "
                    f"merged={primary_id!r}, "
                    f"source={source_primary_id!r}.",
                )
        else:
            add_error(
                errors,
                "source_primary_component_missing",
                "Original primary annotation has no valid "
                "primary_attribution_component object.",
            )

        source_fixed_target_id = primary_source.get(
            "target_component_id"
        )

        if (
            source_fixed_target_id is not None
            and source_fixed_target_id != target_id
        ):
            add_error(
                errors,
                "primary_source_fixed_target_mismatch",
                "Primary annotation was generated for a different "
                "fixed target: "
                f"merged_target={target_id!r}, "
                f"primary_source_target="
                f"{source_fixed_target_id!r}.",
            )

        source_trajectory_id = primary_source.get(
            "trajectory_id"
        )

        if (
            trajectory_id is not None
            and source_trajectory_id is not None
            and str(trajectory_id) != str(source_trajectory_id)
        ):
            add_error(
                errors,
                "source_primary_trajectory_mismatch",
                "Merged trajectory_id differs from primary "
                "annotation trajectory_id.",
            )

    review_sources = [annotation, target_source or {}, primary_source or {}]
    requires_review = any(value.get("needs_review") for value in review_sources)
    if requires_review:
        checks["needs_review"] = True
    status = "fail" if errors else "review" if requires_review else "pass"

    return {
        "schema_version": (
            "safety_refusal_deterministic_validation_v1"
        ),
        "case_id": case_id,
        "trajectory_id": (
            str(trajectory_id)
            if trajectory_id is not None
            else normalized_trajectory_id
        ),
        "target_type": target_type,
        "target_component_id": target_id,
        "primary_component_id": primary_id,
        "annotation_path": str(annotation_path),
        "normalized_path": (
            str(normalized_path)
            if normalized_path is not None
            else None
        ),
        "status": status,
        "passed": status == "pass",
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


# =============================================================================
# Output helpers
# =============================================================================


def copy_case(
    *,
    source: Path,
    destination_dir: Path,
    force: bool,
) -> Path:
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = destination_dir / source.name

    if destination.exists():
        if force:
            destination.unlink()
        else:
            return destination

    shutil.copy2(source, destination)

    return destination


def build_semantic_input_case(
    result: dict[str, Any],
    validation_path: Path,
) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "trajectory_id": result.get("trajectory_id"),
        "target_type": result.get("target_type"),
        "target_component_id": result.get(
            "target_component_id"
        ),
        "primary_component_id": result.get(
            "primary_component_id"
        ),
        "deterministic_status": result.get("status"),
        "validation_file": str(validation_path),
        "files": {
            "annotation": result.get(
                "annotation_path"
            ),
            "normalized": result.get(
                "normalized_path"
            ),
        },
    }


# =============================================================================
# Main validation loop
# =============================================================================


def validate_all(args: argparse.Namespace) -> dict[str, Any]:
    annotations_dir = Path(args.annotations_dir)
    normalized_dir = Path(args.normalized_dir)
    output_root = Path(args.output_root)

    if not annotations_dir.is_dir():
        raise NotADirectoryError(
            f"Annotations directory not found: "
            f"{annotations_dir}"
        )

    if not normalized_dir.is_dir():
        raise NotADirectoryError(
            f"Normalized directory not found: "
            f"{normalized_dir}"
        )

    case_validation_dir = output_root / "case_validation"
    pass_cases_dir = output_root / "pass_cases"
    fail_cases_dir = output_root / "fail_cases"
    review_cases_dir = output_root / "review_cases"

    for directory in (
        case_validation_dir,
        pass_cases_dir,
        fail_cases_dir,
        review_cases_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    all_cases_jsonl = output_root / "all_cases.jsonl"
    pass_cases_jsonl = output_root / "pass_cases.jsonl"
    fail_cases_jsonl = output_root / "fail_cases.jsonl"
    review_cases_jsonl = output_root / "review_cases.jsonl"
    semantic_input_jsonl = (
        output_root / "semantic_input_cases.jsonl"
    )

    for path in (
        all_cases_jsonl,
        pass_cases_jsonl,
        fail_cases_jsonl,
        review_cases_jsonl,
        semantic_input_jsonl,
    ):
        reset_file(path)

    annotation_paths = sorted(
        annotations_dir.glob("*.json")
    )

    if args.limit is not None:
        annotation_paths = annotation_paths[:args.limit]

    normalized_indices = build_normalized_indices(
        normalized_dir
    )

    status_counts: Counter[str] = Counter()
    error_code_counts: Counter[str] = Counter()
    warning_code_counts: Counter[str] = Counter()
    target_role_counts: Counter[str] = Counter()
    primary_role_counts: Counter[str] = Counter()

    case_index: list[dict[str, Any]] = []

    for index, annotation_path in enumerate(
        annotation_paths,
        1,
    ):
        result = validate_one(
            annotation_path=annotation_path,
            normalized_indices=normalized_indices,
        )

        case_id = result.get("case_id") or annotation_case_key(
            annotation_path
        )

        validation_path = (
            case_validation_dir
            / f"{safe_name(case_id)}."
              "deterministic_validation.json"
        )

        write_json(validation_path, result)

        status = result.get("status", "fail")
        status_counts[status] += 1

        for error in result.get("errors", []):
            if isinstance(error, dict):
                error_code_counts[
                    str(error.get("code", "unknown"))
                ] += 1

        for warning in result.get("warnings", []):
            if isinstance(warning, dict):
                warning_code_counts[
                    str(warning.get("code", "unknown"))
                ] += 1

        checks = result.get("checks")

        if isinstance(checks, dict):
            target_role_counts[
                str(
                    checks.get("actual_target_role")
                    or "unknown"
                )
            ] += 1

            primary_role_counts[
                str(
                    checks.get("actual_primary_role")
                    or "unknown"
                )
            ] += 1

        append_jsonl(all_cases_jsonl, result)

        if status == "pass":
            copy_case(
                source=annotation_path,
                destination_dir=pass_cases_dir,
                force=args.force,
            )
            append_jsonl(pass_cases_jsonl, result)

            append_jsonl(
                semantic_input_jsonl,
                build_semantic_input_case(
                    result,
                    validation_path,
                ),
            )

        elif status == "review":
            copy_case(source=annotation_path, destination_dir=review_cases_dir, force=args.force)
            append_jsonl(review_cases_jsonl, result)
        else:
            copy_case(
                source=annotation_path,
                destination_dir=fail_cases_dir,
                force=args.force,
            )
            append_jsonl(fail_cases_jsonl, result)

        case_index.append({
            "case_id": result.get("case_id"),
            "trajectory_id": result.get("trajectory_id"),
            "status": status,
            "num_errors": result.get("num_errors"),
            "num_warnings": result.get("num_warnings"),
            "annotation_path": str(annotation_path),
            "normalized_path": result.get(
                "normalized_path"
            ),
            "validation_path": str(validation_path),
        })

        if args.verbose:
            print(
                f"[{status}] {index}/{len(annotation_paths)} "
                f"{annotation_path.name} "
                f"errors={result.get('num_errors')} "
                f"warnings={result.get('num_warnings')}"
            )

    report = {
        "schema_version": (
            "safety_refusal_deterministic_validation_report_v1"
        ),
        "annotations_dir": str(annotations_dir),
        "normalized_dir": str(normalized_dir),
        "output_root": str(output_root),
        "summary": {
            "num_cases": len(annotation_paths),
            "num_pass": status_counts.get("pass", 0),
            "num_fail": status_counts.get("fail", 0),
            "num_review": status_counts.get("review", 0),
            "status_counts": dict(status_counts),
            "error_code_counts": dict(
                sorted(
                    error_code_counts.items(),
                    key=lambda item: (
                        -item[1],
                        item[0],
                    ),
                )
            ),
            "warning_code_counts": dict(
                sorted(
                    warning_code_counts.items(),
                    key=lambda item: (
                        -item[1],
                        item[0],
                    ),
                )
            ),
            "target_role_counts": dict(
                target_role_counts
            ),
            "primary_role_counts": dict(
                primary_role_counts
            ),
        },
        "files": {
            "case_validation_dir": str(
                case_validation_dir
            ),
            "pass_cases_dir": str(pass_cases_dir),
            "fail_cases_dir": str(fail_cases_dir),
            "all_cases_jsonl": str(all_cases_jsonl),
            "pass_cases_jsonl": str(
                pass_cases_jsonl
            ),
            "fail_cases_jsonl": str(
                fail_cases_jsonl
            ),
            "semantic_input_cases_jsonl": str(
                semantic_input_jsonl
            ),
        },
        "case_index": case_index,
    }

    report_path = (
        output_root
        / "deterministic_validation_report.json"
    )

    write_json(report_path, report)

    return report


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically validate merged safety-refusal "
            "target and primary-attribution annotations."
        )
    )

    parser.add_argument(
        "--annotations_dir",
        required=True,
        help=(
            "Directory containing merged "
            "*.annotation.json files."
        ),
    )
    parser.add_argument(
        "--normalized_dir",
        required=True,
        help=(
            "Directory containing original normalized "
            "trajectory JSON files."
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root for validation results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite copied pass/fail case files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_all(args)

    summary = report["summary"]
    files = report["files"]

    print("=" * 80)
    print("Safety-refusal deterministic validation finished")
    print("=" * 80)
    print(f"Cases:  {summary['num_cases']}")
    print(f"Pass:   {summary['num_pass']}")
    print(f"Fail:   {summary['num_fail']}")

    print("\nError codes:")
    if summary["error_code_counts"]:
        for code, count in summary[
            "error_code_counts"
        ].items():
            print(f"  {code}: {count}")
    else:
        print("  none")

    print("\nWarning codes:")
    if summary["warning_code_counts"]:
        for code, count in summary[
            "warning_code_counts"
        ].items():
            print(f"  {code}: {count}")
    else:
        print("  none")

    print("\nOutputs:")
    print(
        "  report:         "
        f"{Path(report['output_root']) / 'deterministic_validation_report.json'}"
    )
    print(
        "  pass cases:     "
        f"{files['pass_cases_dir']}"
    )
    print(
        "  fail cases:     "
        f"{files['fail_cases_dir']}"
    )
    print(
        "  semantic input: "
        f"{files['semantic_input_cases_jsonl']}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
