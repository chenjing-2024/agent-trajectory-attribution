#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_TARGET_STATUSES = {
    "completed_unsafe_action",
    "high_risk_unsafe_fallback",
    "no_target",
}

VALID_ATTACK_ROLES = {
    "pretext_building",
    "trust_building",
    "context_normalization",
    "capability_probing",
    "escalation",
    "malicious_instruction",
    "injected_instruction",
    "destination_setting",
    "final_unsafe_trigger",
    "other",
}

# Kept exactly consistent with annotate_execution_chain.py.
VALID_EXECUTION_ROLES = {
    "assistant_planning",
    "tool_read",
    "tool_write",
    "tool_call",
    "tool_observation",
    "state_update",
    "command_execution",
    "api_call",
    "unsafe_operation",
    "unsafe_content_generation",
    "unsafe_content_propagation",
    "other",
}

COMPONENT_ID_RE = re.compile(r"^C\d+$")


# =============================================================================
# IO
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Discovery and indexing
# =============================================================================

def iter_json_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.json")
        if p.is_file()
    )


def extract_trajectory_id(obj: dict[str, Any]) -> str | None:
    tid = obj.get("trajectory_id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()

    item = obj.get("item")
    if isinstance(item, dict):
        for key in ("trajectory_id", "id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    metadata = obj.get("_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("trajectory_id")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def build_unique_index(
    root: Path,
    *,
    label: str,
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[str, Any]]:
    by_id: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    unreadable: list[dict[str, str]] = []
    missing_id: list[str] = []

    files = iter_json_files(root)

    for path in files:
        try:
            obj = load_json(path)
        except Exception as e:
            unreadable.append({"path": str(path), "error": repr(e)})
            continue

        tid = extract_trajectory_id(obj)
        if tid is None:
            missing_id.append(str(path))
            continue

        by_id[tid].append((path, obj))

    unique: dict[str, tuple[Path, dict[str, Any]]] = {}
    duplicates: dict[str, list[str]] = {}

    for tid, entries in sorted(by_id.items()):
        if len(entries) == 1:
            unique[tid] = entries[0]
        else:
            duplicates[tid] = [str(path) for path, _ in entries]

    report = {
        "label": label,
        "root": str(root),
        "num_json_files": len(files),
        "num_unique_ids": len(unique),
        "num_duplicate_ids": len(duplicates),
        "num_missing_trajectory_id": len(missing_id),
        "num_unreadable": len(unreadable),
        "duplicate_ids": duplicates,
        "missing_trajectory_id_files": missing_id,
        "unreadable_files": unreadable,
    }
    return unique, report


# =============================================================================
# Component helpers
# =============================================================================

def get_components(trajectory_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Prefer canonical normalized component metadata when available.

    Raw trajectories may contain an unnumbered system message, while
    components_metadata contains only annotatable C1...Cn components.
    """
    for key in (
        "components_metadata",
        "components",
        "normalized_components",
        "trajectory",
    ):
        value = trajectory_obj.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def is_valid_component_id(value: Any) -> bool:
    return isinstance(value, str) and COMPONENT_ID_RE.fullmatch(value) is not None


def component_id_num(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = COMPONENT_ID_RE.fullmatch(value)
    if match is None:
        return None
    return int(value[1:])


def id_sort_key(value: str) -> tuple[int, str]:
    number = component_id_num(value)
    return (number if number is not None else 10**12, value)


def build_component_index(
    components: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    malformed: list[Any] = []

    for component in components:
        cid = component.get("component_id")

        # Ignore raw system/runtime entries that are not annotatable components.
        if cid is None:
            continue

        if not is_valid_component_id(cid):
            malformed.append(cid)
            continue

        if cid in index:
            duplicates.append(cid)
        else:
            index[cid] = component

    return index, sorted(set(duplicates), key=id_sort_key), malformed


# =============================================================================
# Merged annotation extraction
# =============================================================================

def get_target_id(annotation: dict[str, Any]) -> str | None:
    value = annotation.get("target_component_id")
    if isinstance(value, str):
        return value

    nested = annotation.get("target_unsafe_action")
    if isinstance(nested, dict):
        value = nested.get("component_id")
        if isinstance(value, str):
            return value

    return None


def get_primary_id(annotation: dict[str, Any]) -> str | None:
    value = annotation.get("primary_component_id")
    if isinstance(value, str):
        return value

    nested = annotation.get("primary_attribution_component")
    if isinstance(nested, dict):
        value = nested.get("component_id")
        if isinstance(value, str):
            return value

    return None


def get_chain(annotation: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = annotation.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def get_confidence(annotation: dict[str, Any]) -> float | None:
    value = annotation.get("confidence")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# =============================================================================
# Validation
# =============================================================================

def add_hard(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def add_soft(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def validate_chain_structure(
    *,
    chain_name: str,
    chain: list[dict[str, Any]],
    role_key: str,
    valid_roles: set[str],
    component_index: dict[str, dict[str, Any]],
    hard_errors: list[str],
    soft_warnings: list[str],
) -> dict[str, Any]:
    ids: list[str] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    role_counts: Counter[str] = Counter()

    for index, step in enumerate(chain):
        cid = step.get("component_id")

        if not is_valid_component_id(cid):
            add_hard(
                hard_errors,
                f"{chain_name}_malformed_component_id:index={index}:value={cid!r}",
            )
            continue

        ids.append(cid)

        if cid in seen:
            duplicate_ids.append(cid)
            add_hard(hard_errors, f"{chain_name}_duplicate_component_id:{cid}")
        seen.add(cid)

        if cid not in component_index:
            add_hard(hard_errors, f"{chain_name}_component_id_not_found:{cid}")

        role = step.get(role_key)
        if role is None:
            add_hard(hard_errors, f"{chain_name}_missing_{role_key}:{cid}")
        elif not isinstance(role, str) or role not in valid_roles:
            add_hard(
                hard_errors,
                f"{chain_name}_invalid_{role_key}:{cid}:{role!r}",
            )
        else:
            role_counts[role] += 1

        summary = step.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            add_soft(soft_warnings, f"{chain_name}_missing_or_empty_summary:{cid}")

    numeric_ids = [
        component_id_num(cid)
        for cid in ids
        if component_id_num(cid) is not None
    ]
    if numeric_ids != sorted(numeric_ids):
        add_hard(hard_errors, f"{chain_name}_not_in_trajectory_order")

    return {
        "ids": ids,
        "length": len(chain),
        "duplicate_ids": sorted(set(duplicate_ids), key=id_sort_key),
        "role_counts": dict(role_counts),
    }


def validate_target_state(
    *,
    annotation: dict[str, Any],
    target_id: str | None,
    primary_id: str | None,
    attack_chain: list[dict[str, Any]],
    execution_chain: list[dict[str, Any]],
    component_index: dict[str, dict[str, Any]],
    hard_errors: list[str],
) -> None:
    status = annotation.get("target_status")
    attack_success = annotation.get("attack_success")

    if status not in VALID_TARGET_STATUSES:
        add_hard(hard_errors, f"invalid_target_status:{status!r}")
        return

    if not isinstance(attack_success, bool):
        add_hard(hard_errors, f"attack_success_not_boolean:{attack_success!r}")

    if status == "completed_unsafe_action":
        if attack_success is not True:
            add_hard(
                hard_errors,
                "completed_unsafe_action_requires_attack_success_true",
            )
        if target_id is None:
            add_hard(
                hard_errors,
                "completed_unsafe_action_requires_target_component_id",
            )

    elif status == "high_risk_unsafe_fallback":
        if attack_success is not False:
            add_hard(
                hard_errors,
                "high_risk_unsafe_fallback_requires_attack_success_false",
            )
        if target_id is None:
            add_hard(
                hard_errors,
                "high_risk_unsafe_fallback_requires_target_component_id",
            )

    elif status == "no_target":
        if attack_success is not False:
            add_hard(hard_errors, "no_target_requires_attack_success_false")
        if target_id is not None:
            add_hard(hard_errors, "no_target_requires_null_target_component_id")
        if primary_id is not None:
            add_hard(hard_errors, "no_target_requires_null_primary_component_id")
        if attack_chain:
            add_hard(hard_errors, "no_target_requires_empty_attack_chain")
        if execution_chain:
            add_hard(hard_errors, "no_target_requires_empty_execution_chain")

    if target_id is not None:
        if not is_valid_component_id(target_id):
            add_hard(hard_errors, f"malformed_target_component_id:{target_id!r}")
        elif target_id not in component_index:
            add_hard(hard_errors, f"target_component_id_not_found:{target_id}")


def validate_one_case(
    *,
    annotation_path: Path,
    annotation: dict[str, Any],
    trajectory_path: Path | None,
    trajectory: dict[str, Any] | None,
    low_confidence_threshold: float,
    long_chain_threshold: int,
) -> dict[str, Any]:
    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    info: list[str] = []

    annotation_tid = extract_trajectory_id(annotation)

    if annotation_tid is None:
        add_hard(hard_errors, "annotation_missing_trajectory_id")

    if trajectory_path is None or trajectory is None:
        add_hard(hard_errors, f"trajectory_not_found:{annotation_tid}")
        components: list[dict[str, Any]] = []
        trajectory_tid = None
    else:
        trajectory_tid = extract_trajectory_id(trajectory)
        components = get_components(trajectory)

        if trajectory_tid is None:
            add_hard(hard_errors, "trajectory_file_missing_trajectory_id")
        elif annotation_tid is not None and trajectory_tid != annotation_tid:
            add_hard(
                hard_errors,
                f"trajectory_id_mismatch:annotation={annotation_tid}:trajectory={trajectory_tid}",
            )

        if not components:
            add_hard(hard_errors, "trajectory_components_missing_or_empty")

    component_index, duplicate_component_ids, malformed_component_ids = (
        build_component_index(components)
    )

    for cid in duplicate_component_ids:
        add_hard(hard_errors, f"trajectory_duplicate_component_id:{cid}")
    for cid in malformed_component_ids:
        add_hard(hard_errors, f"trajectory_malformed_component_id:{cid!r}")

    # Raw original trajectory components are not required to contain "type".
    component_type_counts = Counter(
        component.get("type", "<missing>")
        for component in components
    )

    target_id = get_target_id(annotation)
    primary_id = get_primary_id(annotation)
    attack_chain = get_chain(annotation, "attack_chain")
    execution_chain = get_chain(annotation, "execution_chain")

    validate_target_state(
        annotation=annotation,
        target_id=target_id,
        primary_id=primary_id,
        attack_chain=attack_chain,
        execution_chain=execution_chain,
        component_index=component_index,
        hard_errors=hard_errors,
    )

    target_status = annotation.get("target_status")

    if target_status != "no_target":
        if primary_id is None:
            add_hard(hard_errors, "missing_primary_component_id")
        elif not is_valid_component_id(primary_id):
            add_hard(hard_errors, f"malformed_primary_component_id:{primary_id!r}")
        elif primary_id not in component_index:
            add_hard(hard_errors, f"primary_component_id_not_found:{primary_id}")

    attack_validation = validate_chain_structure(
        chain_name="attack_chain",
        chain=attack_chain,
        role_key="role_in_attack",
        valid_roles=VALID_ATTACK_ROLES,
        component_index=component_index,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
    )

    execution_validation = validate_chain_structure(
        chain_name="execution_chain",
        chain=execution_chain,
        role_key="role_in_execution",
        valid_roles=VALID_EXECUTION_ROLES,
        component_index=component_index,
        hard_errors=hard_errors,
        soft_warnings=soft_warnings,
    )

    attack_ids = attack_validation["ids"]
    execution_ids = execution_validation["ids"]

    primary_num = component_id_num(primary_id)
    target_num = component_id_num(target_id)

    if target_status != "no_target":
        # Strictly require primary < target. Equality is forbidden.
        if primary_num is not None and target_num is not None:
            if primary_num >= target_num:
                add_hard(
                    hard_errors,
                    f"primary_not_strictly_before_target:{primary_id}>={target_id}",
                )

            # Attack chain must be strictly before primary.
            for cid in attack_ids:
                number = component_id_num(cid)
                if number is not None and number >= primary_num:
                    add_hard(
                        hard_errors,
                        f"attack_chain_not_strictly_before_primary:{cid}>={primary_id}",
                    )

            # Execution chain must be strictly between primary and target.
            for cid in execution_ids:
                number = component_id_num(cid)
                if number is None:
                    continue
                if number <= primary_num:
                    add_hard(
                        hard_errors,
                        f"execution_chain_not_strictly_after_primary:{cid}<={primary_id}",
                    )
                if number >= target_num:
                    add_hard(
                        hard_errors,
                        f"execution_chain_not_strictly_before_target:{cid}>={target_id}",
                    )

        # Explicit non-overlap checks, even if numeric comparison was unavailable.
        if primary_id is not None and primary_id in attack_ids:
            add_hard(hard_errors, f"attack_chain_contains_primary:{primary_id}")
        if target_id is not None and target_id in attack_ids:
            add_hard(hard_errors, f"attack_chain_contains_target:{target_id}")
        if primary_id is not None and primary_id in execution_ids:
            add_hard(hard_errors, f"execution_chain_contains_primary:{primary_id}")
        if target_id is not None and target_id in execution_ids:
            add_hard(hard_errors, f"execution_chain_contains_target:{target_id}")

    if len(attack_chain) == 0:
        info.append("attack_chain_empty")
    if len(execution_chain) == 0:
        info.append("execution_chain_empty")

    if len(attack_chain) > long_chain_threshold:
        add_soft(soft_warnings, "attack_chain_long")
    if len(execution_chain) > long_chain_threshold:
        add_soft(soft_warnings, "execution_chain_long")

    confidence = get_confidence(annotation)
    if confidence is None:
        add_soft(soft_warnings, "missing_or_invalid_confidence")
    elif not 0.0 <= confidence <= 1.0:
        add_hard(hard_errors, f"confidence_out_of_range:{confidence}")
    elif confidence < low_confidence_threshold:
        add_soft(soft_warnings, f"low_confidence:{confidence}")

    needs_review = annotation.get("needs_review")
    if needs_review not in (True, False):
        add_soft(soft_warnings, "needs_review_not_boolean")
    elif needs_review:
        add_soft(soft_warnings, "annotation_needs_review")
        reason = annotation.get("review_reason")
        if isinstance(reason, str) and reason.strip():
            info.append(f"review_reason:{reason.strip()}")

    # Only hard errors route to repair. Warnings remain deterministic pass.
    status = "repair" if hard_errors else "pass"

    return {
        "schema_version": "agent3sigma_merged_deterministic_validation_v1",
        "trajectory_id": annotation_tid,
        "status": status,
        "deterministic_passed": status == "pass",
        "eligible_for_semantic_validation": status == "pass",
        "needs_repair": status == "repair",
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "info": info,
        "files": {
            "annotation": str(annotation_path),
            "trajectory": str(trajectory_path) if trajectory_path else None,
        },
        "target": {
            "target_status": target_status,
            "attack_success": annotation.get("attack_success"),
            "target_component_id": target_id,
        },
        "primary": {
            "primary_component_id": primary_id,
        },
        "chains": {
            "attack_chain_ids": attack_ids,
            "execution_chain_ids": execution_ids,
            "attack_chain_length": len(attack_chain),
            "execution_chain_length": len(execution_chain),
            "attack_role_counts": attack_validation["role_counts"],
            "execution_role_counts": execution_validation["role_counts"],
        },
        "trajectory_summary": {
            "num_components": len(components),
            "component_type_counts": dict(component_type_counts),
            "duplicate_component_ids": duplicate_component_ids,
            "malformed_component_ids": malformed_component_ids,
        },
        "annotation_summary": {
            "confidence": confidence,
            "needs_review": needs_review,
            "risk_category": annotation.get("risk_category"),
        },
    }


# =============================================================================
# Main loop
# =============================================================================

def validate_all(args: argparse.Namespace) -> dict[str, Any]:
    annotations_dir = Path(args.annotations_dir)
    trajectories_dir = Path(args.trajectories_dir)
    output_root = Path(args.output_root)

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"annotations_dir not found: {annotations_dir}")
    if not trajectories_dir.is_dir():
        raise FileNotFoundError(f"trajectories_dir not found: {trajectories_dir}")

    pass_dir = output_root / "pass"
    repair_dir = output_root / "repair"
    reset_output_dir(pass_dir)
    reset_output_dir(repair_dir)

    pass_cases_jsonl = output_root / "pass_cases.jsonl"
    repair_cases_jsonl = output_root / "repair_cases.jsonl"
    semantic_input_jsonl = output_root / "semantic_input_cases.jsonl"
    for path in (pass_cases_jsonl, repair_cases_jsonl, semantic_input_jsonl):
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    annotation_index, annotation_discovery = build_unique_index(
        annotations_dir,
        label="annotations",
    )
    trajectory_index, trajectory_discovery = build_unique_index(
        trajectories_dir,
        label="trajectories",
    )

    trajectory_duplicate_ids = set(trajectory_discovery["duplicate_ids"])
    annotation_duplicate_ids = set(annotation_discovery["duplicate_ids"])

    trajectory_ids = sorted(annotation_index)
    if args.limit is not None:
        trajectory_ids = trajectory_ids[:args.limit]

    summary: dict[str, Any] = {
        "num_annotation_cases": len(annotation_index),
        "num_cases_processed": 0,
        "num_pass": 0,
        "num_repair": 0,
        "num_missing_trajectory": 0,
        "num_duplicate_trajectory_match": 0,
        "hard_error_counts": Counter(),
        "soft_warning_counts": Counter(),
        "target_status_counts": Counter(),
        "risk_category_counts": Counter(),
        "status_by_risk_category": defaultdict(Counter),
        "case_index": [],
    }

    for index, trajectory_id in enumerate(trajectory_ids, 1):
        annotation_path, annotation = annotation_index[trajectory_id]

        trajectory_path: Path | None = None
        trajectory: dict[str, Any] | None = None

        if trajectory_id in trajectory_duplicate_ids:
            # Duplicate original trajectories are not uniquely matchable.
            pass
        elif trajectory_id in trajectory_index:
            trajectory_path, trajectory = trajectory_index[trajectory_id]

        result = validate_one_case(
            annotation_path=annotation_path,
            annotation=annotation,
            trajectory_path=trajectory_path,
            trajectory=trajectory,
            low_confidence_threshold=args.low_confidence_threshold,
            long_chain_threshold=args.long_chain_threshold,
        )

        if trajectory_id in trajectory_duplicate_ids:
            add_hard(
                result["hard_errors"],
                f"duplicate_trajectory_id_match:{trajectory_id}",
            )
            result["status"] = "repair"
            result["deterministic_passed"] = False
            result["eligible_for_semantic_validation"] = False
            result["needs_repair"] = True
            summary["num_duplicate_trajectory_match"] += 1

        if trajectory_path is None and trajectory_id not in trajectory_duplicate_ids:
            summary["num_missing_trajectory"] += 1

        output_dir = pass_dir if result["status"] == "pass" else repair_dir
        result_path = output_dir / f"{trajectory_id}.validation.json"
        write_json(result_path, result)

        routing_entry = {
            "trajectory_id": trajectory_id,
            "status": result["status"],
            "annotation_file": str(annotation_path),
            "trajectory_file": str(trajectory_path) if trajectory_path else None,
            "validation_file": str(result_path),
            "hard_errors": result["hard_errors"],
            "soft_warnings": result["soft_warnings"],
            "risk_category": annotation.get("risk_category"),
            "target_status": annotation.get("target_status"),
        }

        if result["status"] == "pass":
            summary["num_pass"] += 1
            append_jsonl(pass_cases_jsonl, routing_entry)
            append_jsonl(semantic_input_jsonl, routing_entry)
        else:
            summary["num_repair"] += 1
            append_jsonl(repair_cases_jsonl, routing_entry)

        summary["num_cases_processed"] += 1
        risk = annotation.get("risk_category") or "Unknown"
        target_status = annotation.get("target_status") or "Unknown"
        summary["risk_category_counts"][risk] += 1
        summary["target_status_counts"][target_status] += 1
        summary["status_by_risk_category"][risk][result["status"]] += 1

        for error in result["hard_errors"]:
            summary["hard_error_counts"][error] += 1
        for warning in result["soft_warnings"]:
            summary["soft_warning_counts"][warning] += 1

        summary["case_index"].append({
            "trajectory_id": trajectory_id,
            "status": result["status"],
            "num_hard_errors": len(result["hard_errors"]),
            "num_soft_warnings": len(result["soft_warnings"]),
            "annotation_file": str(annotation_path),
            "trajectory_file": str(trajectory_path) if trajectory_path else None,
            "validation_file": str(result_path),
        })

        if args.verbose:
            print(
                f"[{result['status']}] "
                f"{index}/{len(trajectory_ids)} "
                f"{trajectory_id} "
                f"hard={len(result['hard_errors'])} "
                f"soft={len(result['soft_warnings'])}"
            )

    # Duplicate annotation IDs cannot enter normal processing because they are not unique.
    duplicate_annotation_repair_entries: list[dict[str, Any]] = []
    for trajectory_id in sorted(annotation_duplicate_ids):
        entry = {
            "trajectory_id": trajectory_id,
            "status": "repair",
            "reason": "duplicate_annotation_trajectory_id",
            "annotation_files": annotation_discovery["duplicate_ids"][trajectory_id],
        }
        duplicate_annotation_repair_entries.append(entry)
        append_jsonl(repair_cases_jsonl, entry)

    def sorted_counter(counter: Counter[Any]) -> dict[str, int]:
        return {
            str(key): value
            for key, value in sorted(
                counter.items(),
                key=lambda item: (-item[1], str(item[0])),
            )
        }

    report = {
        "schema_version": "agent3sigma_merged_deterministic_validation_report_v1",
        "inputs": {
            "annotations_dir": str(annotations_dir),
            "trajectories_dir": str(trajectories_dir),
        },
        "output_root": str(output_root),
        "settings": {
            "low_confidence_threshold": args.low_confidence_threshold,
            "long_chain_threshold": args.long_chain_threshold,
            "limit": args.limit,
        },
        "discovery": {
            "annotations": annotation_discovery,
            "trajectories": trajectory_discovery,
        },
        "summary": {
            "num_annotation_cases": summary["num_annotation_cases"],
            "num_cases_processed": summary["num_cases_processed"],
            "num_pass": summary["num_pass"],
            "num_repair": summary["num_repair"] + len(duplicate_annotation_repair_entries),
            "num_duplicate_annotation_ids": len(duplicate_annotation_repair_entries),
            "num_missing_trajectory": summary["num_missing_trajectory"],
            "num_duplicate_trajectory_match": summary["num_duplicate_trajectory_match"],
            "hard_error_counts": sorted_counter(summary["hard_error_counts"]),
            "soft_warning_counts": sorted_counter(summary["soft_warning_counts"]),
            "target_status_counts": sorted_counter(summary["target_status_counts"]),
            "risk_category_counts": sorted_counter(summary["risk_category_counts"]),
            "status_by_risk_category": {
                risk: dict(counts)
                for risk, counts in sorted(summary["status_by_risk_category"].items())
            },
        },
        "outputs": {
            "pass_dir": str(pass_dir),
            "repair_dir": str(repair_dir),
            "pass_cases_jsonl": str(pass_cases_jsonl),
            "repair_cases_jsonl": str(repair_cases_jsonl),
            "semantic_input_cases_jsonl": str(semantic_input_jsonl),
        },
        "case_index": summary["case_index"],
    }

    write_json(output_root / "validation_report.json", report)
    write_json(
        output_root / "input_discovery_report.json",
        {
            "annotations": annotation_discovery,
            "trajectories": trajectory_discovery,
        },
    )
    return report


# =============================================================================
# CLI
# =============================================================================

def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    outputs = report["outputs"]

    print("\n" + "=" * 80)
    print("Merged annotation deterministic validation")
    print("=" * 80)
    print(f"Annotation cases:       {summary['num_annotation_cases']}")
    print(f"Processed unique cases: {summary['num_cases_processed']}")
    print(f"Pass:                   {summary['num_pass']}")
    print(f"Repair:                 {summary['num_repair']}")
    print(f"Missing trajectories:   {summary['num_missing_trajectory']}")
    print(f"Duplicate trajectory matches: {summary['num_duplicate_trajectory_match']}")
    print(f"Duplicate annotation IDs:     {summary['num_duplicate_annotation_ids']}")
    print("\nOutputs:")
    print(f"  pass folder:       {outputs['pass_dir']}")
    print(f"  repair folder:     {outputs['repair_dir']}")
    print(f"  semantic inputs:   {outputs['semantic_input_cases_jsonl']}")
    print(f"  report:            {report['output_root']}/validation_report.json")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically validate merged Agent3Sigma annotations against "
            "original trajectories before LLM semantic validation."
        )
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        help="Directory containing merged annotation JSON files; searched recursively.",
    )
    parser.add_argument(
        "--trajectories_dir",
        required=True,
        help="Directory containing original trajectory JSON files; searched recursively.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root containing pass/, repair/, reports, and routing JSONL files.",
    )
    parser.add_argument(
        "--low_confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence below this threshold produces a soft warning.",
    )
    parser.add_argument(
        "--long_chain_threshold",
        type=int,
        default=8,
        help="Chains longer than this threshold produce a soft warning.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of unique annotation cases to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one status line per processed case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_all(args)
    print_summary(report)


if __name__ == "__main__":
    main()
