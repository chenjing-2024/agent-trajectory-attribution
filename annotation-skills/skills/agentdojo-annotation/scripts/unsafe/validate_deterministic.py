#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.touch()


# ============================================================
# Component helpers
# ============================================================

def clean_component_id(value: Any) -> int | None:
    """
    Convert component id into int if possible.

    Supports:
    - 12
    - "12"
    - "C12"
    - "component_12"
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)

        m = re.fullmatch(r"C(\d+)", s)
        if m:
            return int(m.group(1))

        m = re.fullmatch(r"component[_-]?(\d+)", s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def raw_component_id_display(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def extract_component_list(component_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Support componentized trajectory formats used in this project.

    Expected current format:
    {
      "component_scheme": "action_result_pair_schemeA",
      "trajectory": [...],
      "components_metadata": [...]
    }

    Also supports aliases:
    - components
    - trajectory_components
    - grouped_components
    - action_result_components
    """
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
        and trajectory
        and metadata
    ):
        merged: list[dict[str, Any]] = []

        n = max(len(trajectory), len(metadata))
        for i in range(n):
            comp = trajectory[i] if i < len(trajectory) and isinstance(trajectory[i], dict) else {}
            meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) else {}

            merged_item = {}
            merged_item.update(comp)
            merged_item.update(meta)
            merged.append(merged_item)

        return merged

    return []


def build_component_id_map(component_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    components = extract_component_list(component_data)

    out: dict[int, dict[str, Any]] = {}
    for idx, comp in enumerate(components, start=1):
        cid = clean_component_id(comp.get("component_id"))
        if cid is None:
            cid = idx
        out[cid] = comp

    return out


def get_valid_component_ids(component_data: dict[str, Any]) -> set[int]:
    return set(build_component_id_map(component_data).keys())


# ============================================================
# Annotation helpers
# ============================================================

def get_annotation_object(annotation_output: dict[str, Any]) -> dict[str, Any] | None:
    """
    Current C1 output format:

    {
      "source_file": "...",
      "trajectory_id": "...",
      "annotation": {
        "attack_success": true,
        "annotations": [...]
      },
      ...
    }

    Also tolerate the case where the file itself is directly the annotation object.
    """
    ann = annotation_output.get("annotation")

    if isinstance(ann, dict):
        return ann

    if "attack_success" in annotation_output and "annotations" in annotation_output:
        return annotation_output

    return None


def collect_annotation_ids(ann_item: dict[str, Any]) -> list[tuple[str, Any]]:
    """
    Collect all component IDs that should be valid top-level component IDs.
    """
    ids: list[tuple[str, Any]] = []

    ids.append(("critical_component_id", ann_item.get("critical_component_id")))

    primary = ann_item.get("primary_attribution_component")
    if isinstance(primary, dict):
        ids.append((
            "primary_attribution_component.component_id",
            primary.get("component_id"),
        ))
    else:
        ids.append(("primary_attribution_component.component_id", None))

    attack_chain = ann_item.get("attack_chain", [])
    if isinstance(attack_chain, list):
        for i, cid in enumerate(attack_chain):
            ids.append((f"attack_chain[{i}]", cid))
    else:
        ids.append(("attack_chain", attack_chain))

    execution_chain = ann_item.get("execution_chain", [])
    if isinstance(execution_chain, list):
        for i, cid in enumerate(execution_chain):
            ids.append((f"execution_chain[{i}]", cid))
    else:
        ids.append(("execution_chain", execution_chain))

    return ids


def count_validation_warnings(ann_item: dict[str, Any]) -> int:
    warnings = ann_item.get("validation_warnings", [])
    if isinstance(warnings, list):
        return len(warnings)
    if warnings:
        return 1
    return 0


def get_validation_warnings(ann_item: dict[str, Any]) -> list[str]:
    warnings = ann_item.get("validation_warnings", [])
    if isinstance(warnings, list):
        return [str(x) for x in warnings]
    if warnings:
        return [str(warnings)]
    return []


def add_hard(hard_errors: list[str], name: str) -> None:
    if name not in hard_errors:
        hard_errors.append(name)


def add_soft(soft_warnings: list[str], name: str) -> None:
    if name not in soft_warnings:
        soft_warnings.append(name)


def status_from_errors(hard_errors: list[str], soft_warnings: list[str]) -> str:
    if hard_errors:
        return "hard_error"
    if soft_warnings:
        return "ok_with_warnings"
    return "clean"


def build_validation_decision(hard_errors: list[str]) -> dict[str, Any]:
    deterministic_passed = not hard_errors
    return {
        "deterministic_passed": deterministic_passed,
        "eligible_for_semantic_validation": deterministic_passed,
        "needs_auto_repair": not deterministic_passed,
        "recommended_action": (
            "send_to_semantic_validation"
            if deterministic_passed
            else "rerun_annotation"
        ),
    }


def infer_case_id_from_rel(rel: Path) -> str:
    """
    Stable case id for AgentDojo annotation files.
    Uses full relative path without suffix to avoid collisions.
    """
    return str(rel.with_suffix("")).replace("/", "__")


def parse_agentdojo_path(rel: Path) -> dict[str, Any]:
    """
    Best-effort parse for paths like:
      qwen36/slack/user_task_0/slack_adaptive_template/injection_task_1.json
    """
    parts = rel.parts
    meta: dict[str, Any] = {
        "relative_path": str(rel),
        "case_id": infer_case_id_from_rel(rel),
    }

    if len(parts) >= 1:
        meta["model_name"] = parts[0]
    if len(parts) >= 2:
        meta["suite_name"] = parts[1]
    if len(parts) >= 3:
        meta["user_task_id"] = parts[2]
    if len(parts) >= 4:
        meta["template_name"] = parts[3]
    if len(parts) >= 5:
        meta["injection_task_id"] = Path(parts[4]).stem

    return meta


def case_validation_output_path(output_dir: Path, rel: Path) -> Path:
    return output_dir / "case_validation" / rel.with_suffix(".validation.json")


def ids_from_list(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        cid = clean_component_id(x)
        if cid is not None:
            out.append(cid)
    return out


def has_duplicates(xs: list[int]) -> bool:
    return len(xs) != len(set(xs))


# ============================================================
# Main per-case validation
# ============================================================

def validate_one_annotation(
    component_path: Path,
    annotation_path: Path | None,
    components_root: Path,
    annotations_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Validate one annotation against its corresponding component file.

    Returns:
    - case_result
    - error rows
    """
    rel = component_path.relative_to(components_root)
    meta = parse_agentdojo_path(rel)

    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    info: list[str] = []
    error_rows: list[dict[str, Any]] = []

    files = {
        "component": str(component_path),
        "annotation": str(annotation_path) if annotation_path is not None else None,
    }

    ids_summary: dict[str, Any] = {
        "critical_component_id": None,
        "primary_component_id": None,
        "attack_chain_ids": [],
        "execution_chain_ids": [],
        "valid_component_ids": [],
    }

    # ------------------------------------------------------------
    # Load component file.
    # ------------------------------------------------------------
    try:
        component_data = load_json(component_path)
    except Exception as e:
        add_hard(hard_errors, "component_invalid_json")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path) if annotation_path else None,
            "error_type": "component_invalid_json",
            "error": f"{type(e).__name__}: {e}",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    component_id_map = build_component_id_map(component_data)
    valid_component_ids = set(component_id_map.keys())
    ids_summary["valid_component_ids"] = sorted(valid_component_ids)

    if not valid_component_ids:
        add_hard(hard_errors, "component_missing_ids")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path) if annotation_path else None,
            "error_type": "component_missing_ids",
            "error": "No valid component IDs found in componentized trajectory.",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    # ------------------------------------------------------------
    # Missing annotation.
    # ------------------------------------------------------------
    if annotation_path is None or not annotation_path.exists():
        add_hard(hard_errors, "missing_annotation")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "expected_annotation_file": str(annotation_path) if annotation_path else None,
            "error_type": "missing_annotation",
            "error": "No corresponding annotation file found for component file.",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    # ------------------------------------------------------------
    # Load annotation file.
    # ------------------------------------------------------------
    try:
        annotation_output = load_json(annotation_path)
    except Exception as e:
        add_hard(hard_errors, "annotation_invalid_json")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path),
            "error_type": "annotation_invalid_json",
            "error": f"{type(e).__name__}: {e}",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    ann = get_annotation_object(annotation_output)
    if not isinstance(ann, dict):
        add_hard(hard_errors, "missing_annotation_object")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path),
            "error_type": "missing_annotation_object",
            "error": "Missing top-level 'annotation' object or direct annotation fields.",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    if "attack_success" not in ann:
        add_hard(hard_errors, "missing_attack_success")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path),
            "error_type": "missing_attack_success",
            "error": "Missing required field: annotation.attack_success.",
        })
    elif ann.get("attack_success") is not True:
        add_soft(soft_warnings, "attack_success_false")

    annotations = ann.get("annotations")
    if not isinstance(annotations, list):
        add_hard(hard_errors, "annotations_not_list")
        error_rows.append({
            **meta,
            "component_file": str(component_path),
            "annotation_file": str(annotation_path),
            "error_type": "annotations_not_list",
            "error": "Missing or invalid field: annotation.annotations must be a list.",
        })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    if len(annotations) == 0:
        add_soft(soft_warnings, "empty_annotations")
        if ann.get("attack_success") is True:
            add_hard(hard_errors, "empty_annotations_for_success")
            error_rows.append({
                **meta,
                "component_file": str(component_path),
                "annotation_file": str(annotation_path),
                "error_type": "empty_annotations_for_success",
                "error": "annotation.attack_success=true but annotation.annotations is empty.",
            })

        return finalize_case_result(
            meta=meta,
            files=files,
            ids_summary=ids_summary,
            hard_errors=hard_errors,
            soft_warnings=soft_warnings,
            info=info,
            error_rows=error_rows,
        )

    if len(annotations) > 1:
        add_soft(soft_warnings, f"multiple_annotation_items:{len(annotations)}")

    # ------------------------------------------------------------
    # Validate each annotation item.
    # ------------------------------------------------------------
    for ann_idx, ann_item in enumerate(annotations):
        if not isinstance(ann_item, dict):
            add_hard(hard_errors, f"annotation_item_not_object:{ann_idx}")
            error_rows.append({
                **meta,
                "component_file": str(component_path),
                "annotation_file": str(annotation_path),
                "error_type": "annotation_item_not_object",
                "annotation_index": ann_idx,
                "error": "Each item in annotation.annotations must be an object.",
            })
            continue

        required_item_fields = [
            "critical_component_id",
            "primary_attribution_component",
            "case_complexity",
            "attack_chain",
            "execution_chain",
        ]

        for field in required_item_fields:
            if field not in ann_item:
                add_hard(hard_errors, f"missing_annotation_item_field:{field}")
                error_rows.append({
                    **meta,
                    "component_file": str(component_path),
                    "annotation_file": str(annotation_path),
                    "error_type": "missing_annotation_item_field",
                    "annotation_index": ann_idx,
                    "field": field,
                    "error": f"Missing required field: annotations[{ann_idx}].{field}",
                })

        critical_id = clean_component_id(ann_item.get("critical_component_id"))
        ids_summary["critical_component_id"] = critical_id

        primary = ann_item.get("primary_attribution_component")
        primary_id: int | None = None
        if not isinstance(primary, dict):
            add_hard(hard_errors, "invalid_primary_attribution_component")
            error_rows.append({
                **meta,
                "component_file": str(component_path),
                "annotation_file": str(annotation_path),
                "error_type": "invalid_primary_attribution_component",
                "annotation_index": ann_idx,
                "error": "primary_attribution_component must be an object.",
            })
        else:
            if "component_id" not in primary:
                add_hard(hard_errors, "missing_primary_component_id")
                error_rows.append({
                    **meta,
                    "component_file": str(component_path),
                    "annotation_file": str(annotation_path),
                    "error_type": "missing_primary_component_id",
                    "annotation_index": ann_idx,
                    "error": "Missing primary_attribution_component.component_id.",
                })
            primary_id = clean_component_id(primary.get("component_id"))

        ids_summary["primary_component_id"] = primary_id

        attack_chain_ids = ids_from_list(ann_item.get("attack_chain", []))
        execution_chain_ids = ids_from_list(ann_item.get("execution_chain", []))
        ids_summary["attack_chain_ids"] = attack_chain_ids
        ids_summary["execution_chain_ids"] = execution_chain_ids

        # Validate all referenced IDs.
        for label, raw_cid in collect_annotation_ids(ann_item):
            cid = clean_component_id(raw_cid)
            if cid is None or cid not in valid_component_ids:
                add_hard(hard_errors, f"invalid_component_id:{label}:{raw_component_id_display(raw_cid)}")
                error_rows.append({
                    **meta,
                    "component_file": str(component_path),
                    "annotation_file": str(annotation_path),
                    "error_type": "invalid_component_id",
                    "annotation_index": ann_idx,
                    "field": label,
                    "value": raw_cid,
                    "valid_component_ids": sorted(valid_component_ids),
                    "error": (
                        f"Invalid component id for {label}: {raw_cid}. "
                        f"Valid IDs: {sorted(valid_component_ids)}"
                    ),
                })

        # Soft structural checks.
        if attack_chain_ids:
            info.append(f"attack_chain_length:{len(attack_chain_ids)}")
        else:
            info.append("attack_chain_empty")

        if execution_chain_ids:
            info.append(f"execution_chain_length:{len(execution_chain_ids)}")
        else:
            info.append("execution_chain_empty")

        if has_duplicates(attack_chain_ids):
            add_soft(soft_warnings, "duplicate_attack_chain_component")

        if has_duplicates(execution_chain_ids):
            add_soft(soft_warnings, "duplicate_execution_chain_component")

        # Protocol-style warnings: do not hard-fail yet.
        # If your final protocol says these are invalid, promote to hard errors later.
        if primary_id is not None and primary_id in attack_chain_ids:
            add_soft(soft_warnings, "attack_chain_contains_primary")

        if primary_id is not None and primary_id in execution_chain_ids:
            add_soft(soft_warnings, "execution_chain_contains_primary")

        if critical_id is not None and critical_id in execution_chain_ids:
            add_soft(soft_warnings, "execution_chain_contains_critical_component")

        if critical_id is not None and primary_id is not None:
            if primary_id > critical_id:
                add_soft(soft_warnings, f"primary_after_critical:{primary_id}>{critical_id}")

        for cid in attack_chain_ids:
            if critical_id is not None and cid > critical_id:
                add_soft(soft_warnings, f"attack_chain_after_critical:{cid}>{critical_id}")

        # Existing model-side validation warnings.
        for warning in get_validation_warnings(ann_item):
            add_soft(soft_warnings, f"model_validation_warning:{warning}")

    return finalize_case_result(
        meta=meta,
        files=files,
        ids_summary=ids_summary,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
        info=info,
        error_rows=error_rows,
    )


def finalize_case_result(
    *,
    meta: dict[str, Any],
    files: dict[str, Any],
    ids_summary: dict[str, Any],
    hard_errors: list[str],
    soft_warnings: list[str],
    info: list[str],
    error_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status = status_from_errors(hard_errors, soft_warnings)
    decision = build_validation_decision(hard_errors)

    case_result = {
        "schema_version": "agentdojo_c1_deterministic_validation_case_v1",
        **meta,
        "status": status,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "info": sorted(set(info)),
        "ids": ids_summary,
        "validation_decision": decision,
        "files": files,
    }

    return case_result, error_rows


# ============================================================
# Summary helpers
# ============================================================

def print_human_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})

    def pct(n: int, d: int) -> str:
        if d <= 0:
            return "0.0%"
        return f"{100.0 * n / d:.1f}%"

    total = int(summary.get("num_component_files", 0) or 0)

    print("\n" + "=" * 80)
    print("AgentDojo C1 deterministic validation summary")
    print("=" * 80)
    print(f"Component files:   {summary.get('num_component_files', 0)}")
    print(f"Annotation files:  {summary.get('num_annotation_files', 0)}")
    print(f"Error files:       {summary.get('num_error_files', 0)}")

    print("\nStatus:")
    for key in ["num_clean", "num_ok_with_warnings", "num_hard_error"]:
        n = int(summary.get(key, 0) or 0)
        print(f"  {key.replace('num_', ''):<18} {n:>6}  ({pct(n, total)})")

    print("\nRouting:")
    for key in ["num_eligible_for_semantic_validation", "num_needs_auto_repair"]:
        n = int(summary.get(key, 0) or 0)
        print(f"  {key.replace('num_', ''):<34} {n:>6}  ({pct(n, total)})")

    def show_counter(title: str, obj: Any, limit: int = 12) -> None:
        print(f"\n{title}:")
        if not isinstance(obj, dict) or not obj:
            print("  none")
            return
        items = sorted(obj.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        for k, v in items[:limit]:
            print(f"  {k}: {v}")
        if len(items) > limit:
            print(f"  ... ({len(items) - limit} more)")

    show_counter("Top hard errors", summary.get("hard_error_counts"))
    show_counter("Top soft warnings", summary.get("soft_warning_counts"))
    show_counter("Recommended actions", summary.get("recommended_action_counts"))

    print("\nOutput files:")
    for k, v in report.get("files", {}).items():
        print(f"  {k}: {v}")

    print("=" * 80)


# ============================================================
# Main validation
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic gate for AgentDojo C1 component-level annotation outputs."
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
        help="Directory to write validation outputs.",
    )
    parser.add_argument(
        "--strict_missing",
        action="store_true",
        help=(
            "If set, exit 2 when missing annotations exist. "
            "Default keeps exit code 0 for partial runs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N component files for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case status.",
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

    if args.limit is not None:
        component_files = component_files[:args.limit]

    annotation_files = sorted([
        p for p in annotations_root.rglob("*.json")
        if not p.name.endswith(".error.json")
    ])

    error_files = sorted([
        p for p in annotations_root.rglob("*.error.json")
    ])

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "validation_report_json": output_dir / "validation_report.json",
        "validation_errors_jsonl": output_dir / "validation_errors.jsonl",
        "clean_cases_jsonl": output_dir / "clean_cases.jsonl",
        "warning_cases_jsonl": output_dir / "warning_cases.jsonl",
        "bad_cases_jsonl": output_dir / "bad_cases.jsonl",
        "repair_cases_jsonl": output_dir / "repair_cases.jsonl",
        "semantic_input_cases_jsonl": output_dir / "semantic_input_cases.jsonl",
        "accepted_by_deterministic_jsonl": output_dir / "accepted_by_deterministic.jsonl",
        "case_validation_dir": output_dir / "case_validation",
    }

    for key, p in paths.items():
        if key.endswith("_jsonl"):
            reset_file(p)

    paths["case_validation_dir"].mkdir(parents=True, exist_ok=True)

    counts = Counter()
    hard_error_counts = Counter()
    soft_warning_counts = Counter()
    recommended_action_counts = Counter()
    info_counts = Counter()

    counts["num_component_files"] = len(component_files)
    counts["num_annotation_files"] = len(annotation_files)
    counts["num_error_files"] = len(error_files)

    all_error_rows: list[dict[str, Any]] = []
    case_index: list[dict[str, Any]] = []

    # Treat explicit .error.json files as hard errors.
    # They may not correspond to component files by exact relative path, so record globally.
    for error_path in error_files:
        rel = error_path.relative_to(annotations_root)
        row = {
            "relative_path": str(rel),
            "case_id": infer_case_id_from_rel(rel),
            "annotation_file": str(error_path),
            "error_type": "annotation_error_file",
            "error": "C1 annotator produced an .error.json file.",
        }
        all_error_rows.append(row)

    for component_path in component_files:
        rel = component_path.relative_to(components_root)
        annotation_path = annotations_root / rel

        case_result, error_rows = validate_one_annotation(
            component_path=component_path,
            annotation_path=annotation_path if annotation_path.exists() else None,
            components_root=components_root,
            annotations_root=annotations_root,
        )

        validation_path = case_validation_output_path(output_dir, rel)
        write_json(validation_path, case_result)

        case_result["validation_file"] = str(validation_path)

        status = case_result["status"]
        decision = case_result["validation_decision"]
        action = decision["recommended_action"]

        counts[f"num_{status}"] += 1
        recommended_action_counts[action] += 1

        if decision["eligible_for_semantic_validation"]:
            counts["num_eligible_for_semantic_validation"] += 1
            append_jsonl(paths["semantic_input_cases_jsonl"], case_result)
            append_jsonl(paths["accepted_by_deterministic_jsonl"], case_result)

        if decision["needs_auto_repair"]:
            counts["num_needs_auto_repair"] += 1
            append_jsonl(paths["repair_cases_jsonl"], case_result)

        if status == "clean":
            append_jsonl(paths["clean_cases_jsonl"], case_result)
        elif status == "ok_with_warnings":
            append_jsonl(paths["warning_cases_jsonl"], case_result)
        else:
            append_jsonl(paths["bad_cases_jsonl"], case_result)

        for e in case_result["hard_errors"]:
            hard_error_counts[e] += 1

        for w in case_result["soft_warnings"]:
            soft_warning_counts[w] += 1

        for item in case_result["info"]:
            info_counts[item] += 1

        all_error_rows.extend(error_rows)

        case_index.append({
            "relative_path": case_result["relative_path"],
            "case_id": case_result["case_id"],
            "status": status,
            "recommended_action": action,
            "validation_file": str(validation_path),
        })

        if args.verbose:
            print(f"[{status}] {case_result['relative_path']} -> {action}")

    # Detect extra annotation files that do not correspond to a component file.
    component_rel_set = {
        p.relative_to(components_root)
        for p in component_files
    }

    for annotation_path in annotation_files:
        rel = annotation_path.relative_to(annotations_root)
        if rel not in component_rel_set:
            counts["num_extra_annotations"] += 1
            all_error_rows.append({
                "relative_path": str(rel),
                "case_id": infer_case_id_from_rel(rel),
                "annotation_file": str(annotation_path),
                "error_type": "extra_annotation",
                "error": "Annotation file has no corresponding component file.",
            })

    # Write errors jsonl.
    reset_file(paths["validation_errors_jsonl"])
    for row in all_error_rows:
        append_jsonl(paths["validation_errors_jsonl"], row)

    report = {
        "schema_version": "agentdojo_c1_deterministic_validation_report_v1",
        "components_root": str(components_root),
        "annotations_root": str(annotations_root),
        "output_dir": str(output_dir),
        "summary": {
            "num_component_files": counts["num_component_files"],
            "num_annotation_files": counts["num_annotation_files"],
            "num_error_files": counts["num_error_files"],

            "num_clean": counts["num_clean"],
            "num_ok_with_warnings": counts["num_ok_with_warnings"],
            "num_hard_error": counts["num_hard_error"],

            "num_eligible_for_semantic_validation": counts["num_eligible_for_semantic_validation"],
            "num_needs_auto_repair": counts["num_needs_auto_repair"],

            # Compatibility with old report fields.
            "num_ok_annotations": counts["num_clean"] + counts["num_ok_with_warnings"],
            "num_missing_annotations": hard_error_counts["missing_annotation"],
            "num_invalid_json": (
                hard_error_counts["component_invalid_json"]
                + hard_error_counts["annotation_invalid_json"]
            ),
            "num_missing_required_fields": sum(
                v for k, v in hard_error_counts.items()
                if k.startswith("missing_") or k in {
                    "annotations_not_list",
                    "missing_annotation_object",
                    "invalid_primary_attribution_component",
                    "empty_annotations_for_success",
                }
            ),
            "num_invalid_component_ids": sum(
                v for k, v in hard_error_counts.items()
                if k.startswith("invalid_component_id")
            ),
            "num_attack_success_false": soft_warning_counts["attack_success_false"],
            "num_empty_annotations": soft_warning_counts["empty_annotations"],
            "num_warnings": sum(soft_warning_counts.values()),
            "num_extra_annotations": counts["num_extra_annotations"],
            "num_validation_error_rows": len(all_error_rows),

            "hard_error_counts": dict(sorted(hard_error_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "soft_warning_counts": dict(sorted(soft_warning_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "recommended_action_counts": dict(recommended_action_counts),
            "info_counts": dict(sorted(info_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "files": {k: str(v) for k, v in paths.items()},
        "case_index": case_index,
        "notes": {
            "clean": "No deterministic hard errors or soft warnings.",
            "ok_with_warnings": "Structurally valid and eligible for semantic validation, but has soft warnings.",
            "hard_error": "Not eligible for semantic validation; should be repaired or re-annotated.",
            "semantic_input_cases_jsonl": "Cases that passed deterministic validation and should be sent to semantic validation.",
            "repair_cases_jsonl": "Cases with deterministic hard errors that should be re-annotated.",
            "accepted_by_deterministic_jsonl": "Cases accepted by deterministic gate; semantic validation may still reject them.",
        },
    }

    write_json(paths["validation_report_json"], report)
    print_human_summary(report)

    if args.strict_missing and hard_error_counts["missing_annotation"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
