#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    # Stable field order: start with common important fields, then append extras.
    preferred_fields = [
        "data_id",
        "trajectory_id",
        "source_file",
        "annotation_file",

        "suite_name",
        "pipeline_name",
        "user_task_id",
        "injection_task_id",
        "attack_type",
        "security",
        "utility",

        "num_components",
        "num_message_components",

        "annotation_model",
        "annotation_type",
        "attack_success",

        "critical_component_id",
        "unsafe_action_part",
        "critical_role",
        "critical_function",
        "critical_component_type",
        "critical_group_type",
        "critical_pair_status",
        "critical_has_tool_call",
        "critical_has_tool_result",
        "critical_action_summary",
        "why_attack_related",
        "critical_content_preview",

        "primary_component_id",
        "primary_source_part",
        "primary_annotation_type",
        "primary_confidence",
        "primary_component_type",
        "primary_group_type",
        "primary_function",
        "primary_pair_status",
        "primary_has_tool_call",
        "primary_has_tool_result",
        "primary_contains_injection_marker",
        "primary_content_preview",

        "case_complexity",
        "attack_chain",
        "attack_chain_len",
        "attack_chain_functions",
        "attack_chain_component_types",

        "execution_chain",
        "execution_chain_len",
        "execution_chain_functions",
        "execution_chain_component_types",

        "chain_reason",
        "validation_warnings",
        "num_validation_warnings",
        "needs_review",
    ]

    all_fields = []
    seen = set()

    for field in preferred_fields:
        if any(field in row for row in rows):
            all_fields.append(field)
            seen.add(field)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_fields.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: to_csv_cell(row.get(key))
                for key in all_fields
            })


def to_csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


# ============================================================
# Component helpers
# ============================================================

def clean_component_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)

    return None


def content_preview(text: Any, max_chars: int = 500) -> str:
    if text is None:
        return ""

    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    text = " ".join(text.replace("\r", "\n").split())
    if len(text) > max_chars:
        return text[:max_chars] + " ...[TRUNCATED]"
    return text


def extract_component_items(component_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return merged component records.

    Current expected format:
    {
      "component_scheme": "action_result_pair_schemeA",
      "trajectory": [...],
      "components_metadata": [...]
    }

    The content lives in trajectory, while ids/functions/types usually live in
    components_metadata. This function merges them by index.
    """
    # Explicit component list formats.
    for key in [
        "components",
        "trajectory_components",
        "grouped_components",
        "action_result_components",
    ]:
        value = component_data.get(key)
        if isinstance(value, list) and value:
            return [x for x in value if isinstance(x, dict)]

    trajectory = component_data.get("trajectory")
    metadata = component_data.get("components_metadata")

    if (
        component_data.get("component_scheme") is not None
        and isinstance(trajectory, list)
        and isinstance(metadata, list)
    ):
        merged: list[dict[str, Any]] = []
        n = max(len(trajectory), len(metadata))

        for i in range(n):
            traj_item = (
                trajectory[i]
                if i < len(trajectory) and isinstance(trajectory[i], dict)
                else {}
            )
            meta_item = (
                metadata[i]
                if i < len(metadata) and isinstance(metadata[i], dict)
                else {}
            )

            item: dict[str, Any] = {}
            item.update(traj_item)
            item.update(meta_item)

            if "content" not in item and isinstance(traj_item, dict):
                item["content"] = traj_item.get("content")

            if "component_id" not in item:
                item["component_id"] = i + 1

            merged.append(item)

        return merged

    return []


def build_component_lookup(component_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}

    for idx, comp in enumerate(extract_component_items(component_data), start=1):
        cid = clean_component_id(comp.get("component_id"))
        if cid is None:
            cid = idx

        lookup[cid] = comp

    return lookup


def get_component_field(
    lookup: dict[int, dict[str, Any]],
    component_id: int | None,
    field: str,
    default: Any = None,
) -> Any:
    if component_id is None:
        return default
    comp = lookup.get(component_id)
    if not isinstance(comp, dict):
        return default
    return comp.get(field, default)


def summarize_chain(
    chain_ids: list[int],
    lookup: dict[int, dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    functions = []
    component_types = []

    for cid in chain_ids:
        comp = lookup.get(cid, {})
        functions.append(comp.get("function"))
        component_types.append(comp.get("component_type"))

    return functions, component_types


# ============================================================
# Annotation helpers
# ============================================================

def get_annotation_object(annotation_output: dict[str, Any]) -> dict[str, Any] | None:
    """
    Current C1 wrapper format:
    {
      "annotation": {
        "attack_success": true,
        "annotations": [...]
      }
    }

    Also support direct annotation object.
    """
    ann = annotation_output.get("annotation")
    if isinstance(ann, dict):
        return ann

    if "attack_success" in annotation_output and "annotations" in annotation_output:
        return annotation_output

    return None


def clean_chain(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []

    cleaned: list[int] = []
    for x in value:
        cid = clean_component_id(x)
        if cid is not None and cid not in cleaned:
            cleaned.append(cid)

    return cleaned


def get_first_annotation_item(annotation_obj: dict[str, Any]) -> dict[str, Any] | None:
    annotations = annotation_obj.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return None

    first = annotations[0]
    if isinstance(first, dict):
        return first

    return None


# ============================================================
# Summary builder
# ============================================================

def build_one_summary(
    component_path: Path,
    annotation_path: Path,
    components_root: Path,
    annotations_root: Path,
) -> dict[str, Any]:
    component_data = load_json(component_path)
    annotation_output = load_json(annotation_path)

    component_lookup = build_component_lookup(component_data)

    annotation_obj = get_annotation_object(annotation_output)
    if not isinstance(annotation_obj, dict):
        raise ValueError("Missing annotation object.")

    ann_item = get_first_annotation_item(annotation_obj)

    rel = component_path.relative_to(components_root)

    data_id = component_data.get("data_id")
    trajectory_id = (
        annotation_output.get("trajectory_id")
        or annotation_obj.get("trajectory_id")
        or component_data.get("trajectory_id")
        or data_id
        or str(rel.with_suffix(""))
    )

    row: dict[str, Any] = {
        # Case identity.
        "data_id": data_id,
        "trajectory_id": trajectory_id,
        "source_file": str(component_path),
        "annotation_file": str(annotation_path),
        "relative_path": str(rel),

        # Dataset metadata.
        "suite_name": component_data.get("suite_name"),
        "pipeline_name": component_data.get("pipeline_name"),
        "user_task_id": component_data.get("user_task_id"),
        "injection_task_id": component_data.get("injection_task_id"),
        "attack_type": component_data.get("attack_type"),
        "security": component_data.get("security"),
        "utility": component_data.get("utility"),

        # Component statistics.
        "component_scheme": component_data.get("component_scheme"),
        "num_components": component_data.get("num_components"),
        "num_message_components": component_data.get("num_message_components"),

        # Annotation metadata.
        "annotation_model": annotation_output.get("annotation_model"),
        "annotation_type": annotation_output.get("annotation_type"),
        "component_source_key": annotation_output.get("component_source_key"),
        "attack_success": annotation_obj.get("attack_success"),
    }

    # If attack_success=false or no annotation item, still output a row.
    if not annotation_obj.get("attack_success") or ann_item is None:
        row.update({
            "critical_component_id": None,
            "primary_component_id": None,
            "case_complexity": None,
            "attack_chain": [],
            "attack_chain_len": 0,
            "execution_chain": [],
            "execution_chain_len": 0,
            "needs_review": True,
            "summary_status": "empty_or_attack_success_false",
        })
        return row

    critical_id = clean_component_id(ann_item.get("critical_component_id"))

    primary = ann_item.get("primary_attribution_component")
    if not isinstance(primary, dict):
        primary = {}

    primary_id = clean_component_id(primary.get("component_id"))

    attack_chain = clean_chain(ann_item.get("attack_chain", []))
    execution_chain = clean_chain(ann_item.get("execution_chain", []))

    attack_chain_functions, attack_chain_component_types = summarize_chain(
        attack_chain,
        component_lookup,
    )
    execution_chain_functions, execution_chain_component_types = summarize_chain(
        execution_chain,
        component_lookup,
    )

    critical_comp = component_lookup.get(critical_id, {}) if critical_id is not None else {}
    primary_comp = component_lookup.get(primary_id, {}) if primary_id is not None else {}

    validation_warnings = ann_item.get("validation_warnings", [])
    if not isinstance(validation_warnings, list):
        validation_warnings = [validation_warnings] if validation_warnings else []

    row.update({
        # Critical / target unsafe action.
        "critical_component_id": critical_id,
        "unsafe_action_part": ann_item.get("unsafe_action_part"),
        "critical_role": ann_item.get("critical_role"),
        "critical_function": (
            ann_item.get("critical_function")
            or critical_comp.get("function")
            or critical_comp.get("function_raw")
        ),
        "critical_component_type": critical_comp.get("component_type"),
        "critical_group_type": critical_comp.get("group_type"),
        "critical_pair_status": critical_comp.get("pair_status"),
        "critical_has_tool_call": critical_comp.get("has_tool_call"),
        "critical_has_tool_result": critical_comp.get("has_tool_result"),
        "critical_action_summary": ann_item.get("critical_action_summary", ""),
        "why_attack_related": ann_item.get("why_attack_related", ""),
        "critical_content_preview": (
            critical_comp.get("content_preview")
            or content_preview(critical_comp.get("content"))
        ),

        # Primary attribution.
        "primary_component_id": primary_id,
        "primary_source_part": primary.get("source_part"),
        "primary_annotation_type": primary.get("component_type"),
        "primary_confidence": primary.get("confidence"),
        "primary_component_type": primary_comp.get("component_type"),
        "primary_group_type": primary_comp.get("group_type"),
        "primary_function": primary_comp.get("function") or primary_comp.get("function_raw"),
        "primary_pair_status": primary_comp.get("pair_status"),
        "primary_has_tool_call": primary_comp.get("has_tool_call"),
        "primary_has_tool_result": primary_comp.get("has_tool_result"),
        "primary_contains_injection_marker": primary_comp.get("contains_injection_marker"),
        "primary_content_preview": (
            primary_comp.get("content_preview")
            or content_preview(primary_comp.get("content"))
        ),

        # Chains.
        "case_complexity": ann_item.get("case_complexity"),
        "attack_chain": attack_chain,
        "attack_chain_len": len(attack_chain),
        "attack_chain_functions": attack_chain_functions,
        "attack_chain_component_types": attack_chain_component_types,

        "execution_chain": execution_chain,
        "execution_chain_len": len(execution_chain),
        "execution_chain_functions": execution_chain_functions,
        "execution_chain_component_types": execution_chain_component_types,

        "chain_reason": ann_item.get("chain_reason", ""),

        # Review / warnings.
        "validation_warnings": validation_warnings,
        "num_validation_warnings": len(validation_warnings),
        "needs_review": bool(validation_warnings),
        "summary_status": "ok",
    })

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a flat case-level summary table from componentized trajectories "
            "and C1 annotation outputs."
        )
    )
    parser.add_argument(
        "--components_root",
        required=True,
        type=str,
        help="Root directory containing componentized trajectory JSON files.",
    )
    parser.add_argument(
        "--annotations_root",
        required=True,
        type=str,
        help="Root directory containing C1 annotation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to write case_summary.jsonl and case_summary.csv.",
    )
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help=(
            "If set, write placeholder rows for component files without annotations. "
            "By default, missing annotations are skipped."
        ),
    )

    args = parser.parse_args()

    components_root = Path(args.components_root)
    annotations_root = Path(args.annotations_root)
    output_dir = Path(args.output_dir)

    if not components_root.exists():
        raise FileNotFoundError(f"components_root does not exist: {components_root}")

    if not annotations_root.exists():
        raise FileNotFoundError(f"annotations_root does not exist: {annotations_root}")

    component_files = sorted([
        p for p in components_root.rglob("*.json")
        if not p.name.endswith(".error.json")
    ])

    rows: list[dict[str, Any]] = []

    num_missing = 0
    num_errors = 0

    for component_path in component_files:
        rel = component_path.relative_to(components_root)
        annotation_path = annotations_root / rel

        if not annotation_path.exists():
            num_missing += 1

            if args.include_missing:
                component_data = load_json(component_path)
                rows.append({
                    "data_id": component_data.get("data_id"),
                    "trajectory_id": component_data.get("trajectory_id") or component_data.get("data_id"),
                    "source_file": str(component_path),
                    "annotation_file": str(annotation_path),
                    "relative_path": str(rel),
                    "suite_name": component_data.get("suite_name"),
                    "pipeline_name": component_data.get("pipeline_name"),
                    "user_task_id": component_data.get("user_task_id"),
                    "injection_task_id": component_data.get("injection_task_id"),
                    "attack_type": component_data.get("attack_type"),
                    "security": component_data.get("security"),
                    "utility": component_data.get("utility"),
                    "component_scheme": component_data.get("component_scheme"),
                    "num_components": component_data.get("num_components"),
                    "num_message_components": component_data.get("num_message_components"),
                    "summary_status": "missing_annotation",
                    "needs_review": True,
                })

            continue

        if annotation_path.name.endswith(".error.json"):
            num_errors += 1
            continue

        try:
            row = build_one_summary(
                component_path=component_path,
                annotation_path=annotation_path,
                components_root=components_root,
                annotations_root=annotations_root,
            )
            rows.append(row)
        except Exception as e:
            num_errors += 1
            rows.append({
                "source_file": str(component_path),
                "annotation_file": str(annotation_path),
                "relative_path": str(rel),
                "summary_status": "summary_error",
                "summary_error_type": type(e).__name__,
                "summary_error": str(e),
                "needs_review": True,
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "case_summary.jsonl"
    csv_path = output_dir / "case_summary.csv"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    print("\nCase summary complete.")
    print(f"Components root:  {components_root}")
    print(f"Annotations root: {annotations_root}")
    print(f"Output dir:       {output_dir}")
    print()
    print(f"Component files scanned: {len(component_files)}")
    print(f"Rows written:            {len(rows)}")
    print(f"Missing annotations:     {num_missing}")
    print(f"Summary errors:          {num_errors}")
    print()
    print(f"JSONL saved to: {jsonl_path}")
    print(f"CSV saved to:   {csv_path}")


if __name__ == "__main__":
    main()