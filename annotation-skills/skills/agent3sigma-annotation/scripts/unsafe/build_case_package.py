#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# IO helpers
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def as_cases_dir(root: Path) -> Path:
    if (root / "cases").is_dir():
        return root / "cases"
    return root


def safe_name(s: Any) -> str:
    s = str(s or "unknown")
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "unknown"


def short_text(x: Any, n: int = 800) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:
            s = str(x)
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."


def id_sort_key(cid: str) -> tuple[int, str]:
    m = re.fullmatch(r"C(\d+)", str(cid))
    if m:
        return int(m.group(1)), str(cid)
    return 10**9, str(cid)


# =============================================================================
# Annotation path matching
# =============================================================================

ANNOTATION_SUFFIXES = {
    "target": ".target.json",
    "primary": ".primary.json",
    "attack_chain": ".attack_chain.json",
    "execution_chain": ".execution_chain.json",
}


def case_id_from_normalized(path: Path) -> str:
    return path.stem


def find_annotation_path(annotation_root: Path, case_id: str, kind: str) -> Path | None:
    canonical = annotation_root / f"{case_id}{ANNOTATION_SUFFIXES[kind]}"
    if canonical.exists():
        return canonical

    fallback_names: list[str] = []
    if kind == "target":
        fallback_names = [
            f"{case_id}.target_annotation.json",
            f"{case_id}.target_component.json",
        ]
    elif kind == "primary":
        fallback_names = [
            f"{case_id}.primary_annotation.json",
            f"{case_id}.primary_cause.json",
        ]
    elif kind == "attack_chain":
        fallback_names = [
            f"{case_id}.attack.json",
            f"{case_id}.chain.json",
        ]
    elif kind == "execution_chain":
        fallback_names = [
            f"{case_id}.execution.json",
            f"{case_id}.execution_path.json",
        ]

    for name in fallback_names:
        p = annotation_root / name
        if p.exists():
            return p

    keyword = kind.replace("_", "*")
    matches = sorted(annotation_root.glob(f"{case_id}*{keyword}*.json"))
    if matches:
        return matches[0]

    return None


# =============================================================================
# Normalized component helpers
# =============================================================================

def get_components(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("components_metadata", "components", "normalized_components"):
        v = normalized.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def component_id_map(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in components:
        cid = c.get("component_id")
        if isinstance(cid, str):
            out[cid] = c
    return out


def extract_component_text(component: dict[str, Any]) -> str:
    preferred_keys = [
        "content",
        "text",
        "message",
        "user",
        "assistant",
        "observation",
        "tool_result",
        "result",
        "summary",
        "raw",
    ]

    parts: list[str] = []
    for key in preferred_keys:
        if key in component and component.get(key) is not None:
            parts.append(f"{key}: {short_text(component.get(key), 3000)}")

    if parts:
        return "\n".join(parts)

    return short_text(component, 3000)


def make_component_compact(component: dict[str, Any], preview_chars: int) -> dict[str, Any]:
    return {
        "component_id": component.get("component_id"),
        "type": component.get("type"),
        "turn": component.get("turn"),
        "role": component.get("role"),
        "tool_name": component.get("tool_name"),
        "summary": component.get("summary"),
        "preview": short_text(extract_component_text(component), preview_chars),
    }


def make_component_full(component: dict[str, Any]) -> dict[str, Any]:
    return dict(component)


def get_annotated_component_ids(
    *,
    target_id: str | None,
    primary_id: str | None,
    attack_chain: list[dict[str, Any]],
    execution_chain: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []

    for cid in [target_id, primary_id]:
        if isinstance(cid, str) and cid not in ids:
            ids.append(cid)

    for chain in [attack_chain, execution_chain]:
        for step in chain:
            cid = step.get("component_id")
            if isinstance(cid, str) and cid not in ids:
                ids.append(cid)

    return sorted(ids, key=id_sort_key)


def case_meta(normalized: dict[str, Any]) -> dict[str, Any]:
    item = normalized.get("item")
    if not isinstance(item, dict):
        item = {}

    return {
        "trajectory_id": normalized.get("trajectory_id") or item.get("id"),
        "case_name": item.get("name") or normalized.get("case_name"),
        "risk_category": item.get("risk_category") or normalized.get("risk_category"),
        "scenario": item.get("scenario"),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "technique": item.get("technique"),
        "source_path": normalized.get("source_path") or item.get("source_path"),
        "item": item,
    }


# =============================================================================
# Annotation extraction
# =============================================================================

def get_target_component_id(target_ann: dict[str, Any]) -> str | None:
    """
    Current target schema:
      target_unsafe_action.component_id
    """
    target_unsafe_action = target_ann.get("target_unsafe_action")
    if isinstance(target_unsafe_action, dict):
        v = target_unsafe_action.get("component_id")
        if isinstance(v, str):
            return v

    for key in (
        "target_component_id",
        "target_id",
        "unsafe_component_id",
        "unsafe_target_component_id",
        "component_id",
    ):
        v = target_ann.get(key)
        if isinstance(v, str):
            return v

    target = target_ann.get("target")
    if isinstance(target, dict):
        for key in ("target_component_id", "component_id", "id"):
            v = target.get(key)
            if isinstance(v, str):
                return v

    return None


def get_primary_component_id(primary_ann: dict[str, Any]) -> str | None:
    """
    Current primary schema:
      primary_attribution_component.component_id
    """
    primary_attribution_component = primary_ann.get("primary_attribution_component")
    if isinstance(primary_attribution_component, dict):
        v = primary_attribution_component.get("component_id")
        if isinstance(v, str):
            return v

    for key in (
        "primary_component_id",
        "primary_id",
        "primary_cause_component_id",
        "primary_cause_id",
        "component_id",
    ):
        v = primary_ann.get(key)
        if isinstance(v, str):
            return v

    primary = primary_ann.get("primary")
    if isinstance(primary, dict):
        for key in ("primary_component_id", "component_id", "id"):
            v = primary.get(key)
            if isinstance(v, str):
                return v

    return None


def get_attack_chain(attack_ann: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("attack_chain", "chain", "components"):
        v = attack_ann.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def get_execution_chain(exec_ann: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("execution_chain", "chain", "execution_path", "operational_path"):
        v = exec_ann.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


# =============================================================================
# Validation report integration
# =============================================================================

def load_validation_map(validation_report_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Load output from validate_annotations.py.

    Expected report structure:
      validation_report.json
      {
        "summary": {...},
        "case_index": [
          {
            "case_id": "...",
            "status": "...",
            "validation_file": "..."
          }
        ]
      }

    For each case_id, this tries to load the corresponding full
    case_validation/*.validation.json file if available.
    """
    if validation_report_path is None:
        return {}, {}

    if not validation_report_path.exists():
        raise FileNotFoundError(f"validation_report not found: {validation_report_path}")

    report = load_json(validation_report_path)
    report_dir = validation_report_path.parent

    validation_map: dict[str, dict[str, Any]] = {}

    for row in report.get("case_index", []):
        if not isinstance(row, dict):
            continue

        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            continue

        case_validation: dict[str, Any] = {
            "status": row.get("status"),
            "hard_errors": [],
            "soft_warnings": [],
            "info": [],
            "review": {
                "needs_review": row.get("needs_review"),
            },
            "ids": {
                "target_component_id": row.get("target_component_id"),
                "primary_component_id": row.get("primary_component_id"),
            },
            "chain_lengths": {
                "attack_chain": row.get("attack_chain_length"),
                "execution_chain": row.get("execution_chain_length"),
            },
            "validation_file": row.get("validation_file"),
            "case_index_row": row,
        }

        validation_file = row.get("validation_file")
        if isinstance(validation_file, str) and validation_file:
            vp = Path(validation_file)
            if not vp.is_absolute():
                # Most runs store validation_file as a project-relative path.
                # Try as-is first, then relative to report_dir.
                if not vp.exists():
                    vp2 = report_dir / vp
                    if vp2.exists():
                        vp = vp2

            if vp.exists():
                try:
                    full = load_json(vp)
                    case_validation = {
                        "status": full.get("status"),
                        "hard_errors": full.get("hard_errors", []),
                        "soft_warnings": full.get("soft_warnings", []),
                        "info": full.get("info", []),
                        "review": full.get("review", {}),
                        "ids": full.get("ids", {}),
                        "chain_lengths": full.get("chain_lengths", {}),
                        "role_counts": full.get("role_counts", {}),
                        "component_summary": full.get("component_summary", {}),
                        "validation_file": str(vp),
                    }
                except Exception as e:
                    case_validation["validation_load_error"] = repr(e)

        validation_map[case_id] = case_validation

    return validation_map, report.get("summary", {})


def default_validation() -> dict[str, Any]:
    return {
        "status": "not_provided",
        "hard_errors": [],
        "soft_warnings": [],
        "info": [],
        "review": {
            "needs_review": False,
            "needs_review_sources": [],
        },
        "ids": {},
        "chain_lengths": {},
        "validation_file": None,
    }


def is_hard_error(validation: dict[str, Any] | None) -> bool:
    if not validation:
        return False
    if validation.get("status") == "hard_error":
        return True
    hard_errors = validation.get("hard_errors")
    return isinstance(hard_errors, list) and len(hard_errors) > 0


# =============================================================================
# Packet / view / skill input builders
# =============================================================================

def build_case_packet(
    *,
    case_id: str,
    normalized: dict[str, Any],
    normalized_path: Path,
    target_ann: dict[str, Any],
    primary_ann: dict[str, Any],
    attack_ann: dict[str, Any],
    exec_ann: dict[str, Any],
    target_path: Path | None,
    primary_path: Path | None,
    attack_chain_path: Path | None,
    execution_chain_path: Path | None,
    annotation_validation: dict[str, Any],
    target_version: str | None,
    primary_version: str | None,
    attack_chain_version: str | None,
    execution_chain_version: str | None,
    preview_chars: int,
    include_all_components_full: bool,
) -> dict[str, Any]:
    meta = case_meta(normalized)
    components = get_components(normalized)
    components_by_id = component_id_map(components)

    target_id = get_target_component_id(target_ann)
    primary_id = get_primary_component_id(primary_ann)
    attack_chain = get_attack_chain(attack_ann)
    execution_chain = get_execution_chain(exec_ann)

    attack_ids = [
        step.get("component_id")
        for step in attack_chain
        if isinstance(step.get("component_id"), str)
    ]
    execution_ids = [
        step.get("component_id")
        for step in execution_chain
        if isinstance(step.get("component_id"), str)
    ]

    annotated_ids = get_annotated_component_ids(
        target_id=target_id,
        primary_id=primary_id,
        attack_chain=attack_chain,
        execution_chain=execution_chain,
    )

    annotated_components_full = {}
    for cid in annotated_ids:
        c = components_by_id.get(cid)
        if c is not None:
            annotated_components_full[cid] = make_component_full(c)

    all_components_compact = [
        make_component_compact(c, preview_chars=preview_chars)
        for c in components
    ]

    components_payload: dict[str, Any] = {
        "num_components": len(components),
        "all_components_compact": all_components_compact,
        "annotated_components_full": annotated_components_full,
    }

    if include_all_components_full:
        components_payload["all_components_full"] = [
            make_component_full(c)
            for c in components
        ]

    packet = {
        "schema_version": "case_packet_v2",
        "case_id": case_id,
        "trajectory_id": meta.get("trajectory_id"),
        "case_name": meta.get("case_name"),
        "risk_category": meta.get("risk_category"),
        "scenario": meta.get("scenario"),
        "category": meta.get("category"),
        "subcategory": meta.get("subcategory"),
        "technique": meta.get("technique"),

        "task": {
            "item": meta.get("item", {}),
            "source_path": meta.get("source_path"),
        },

        "components": components_payload,

        "annotations": {
            "target": {
                "version": target_version,
                "target_component_id": target_id,
                "raw": target_ann,
            },
            "primary": {
                "version": primary_version,
                "primary_component_id": primary_id,
                "raw": primary_ann,
            },
            "attack_chain": {
                "version": attack_chain_version,
                "chain": attack_chain,
                "raw": attack_ann,
            },
            "execution_chain": {
                "version": execution_chain_version,
                "chain": execution_chain,
                "raw": exec_ann,
            },
        },

        "annotation_validation": annotation_validation,

        "derived_views": {
            "component_id_sets": {
                "target": [target_id] if isinstance(target_id, str) else [],
                "primary": [primary_id] if isinstance(primary_id, str) else [],
                "attack_chain": attack_ids,
                "execution_chain": execution_ids,
                "annotated_union": annotated_ids,
            },
            "chain_lengths": {
                "attack_chain": len(attack_chain),
                "execution_chain": len(execution_chain),
            },
            "component_type_counts": dict(Counter(c.get("type") for c in components)),
        },

        "provenance": {
            "normalized_file": str(normalized_path),
            "target_file": str(target_path) if target_path else None,
            "primary_file": str(primary_path) if primary_path else None,
            "attack_chain_file": str(attack_chain_path) if attack_chain_path else None,
            "execution_chain_file": str(execution_chain_path) if execution_chain_path else None,
            "validation_file": annotation_validation.get("validation_file"),
        },
    }

    return packet


def make_case_view(packet: dict[str, Any], view_excerpt_chars: int) -> dict[str, Any]:
    annotations = packet.get("annotations", {})
    target = annotations.get("target", {})
    primary = annotations.get("primary", {})
    attack = annotations.get("attack_chain", {})
    execution = annotations.get("execution_chain", {})

    target_raw = target.get("raw", {})
    primary_raw = primary.get("raw", {})

    target_action = target_raw.get("target_unsafe_action")
    if not isinstance(target_action, dict):
        target_action = {}

    primary_comp = primary_raw.get("primary_attribution_component")
    if not isinstance(primary_comp, dict):
        primary_comp = {}

    annotated_full = packet.get("components", {}).get("annotated_components_full", {})

    evidence_excerpts = {}
    for cid, comp in annotated_full.items():
        evidence_excerpts[cid] = {
            "component_id": cid,
            "type": comp.get("type"),
            "excerpt": short_text(extract_component_text(comp), view_excerpt_chars),
        }

    view = {
        "schema_version": "case_view_v2",
        "case_id": packet.get("case_id"),
        "trajectory_id": packet.get("trajectory_id"),
        "case_name": packet.get("case_name"),
        "risk_category": packet.get("risk_category"),

        "annotation_validation": {
            "status": packet.get("annotation_validation", {}).get("status"),
            "hard_errors": packet.get("annotation_validation", {}).get("hard_errors", []),
            "soft_warnings": packet.get("annotation_validation", {}).get("soft_warnings", []),
            "info": packet.get("annotation_validation", {}).get("info", []),
            "needs_review": packet.get("annotation_validation", {}).get("review", {}).get("needs_review"),
        },

        "target": {
            "component_id": target.get("target_component_id"),
            "type": target_action.get("type"),
            "target_action_type": target_action.get("target_action_type"),
            "summary": target_action.get("summary"),
            "evidence": short_text(target_action.get("evidence"), view_excerpt_chars),
        },

        "primary": {
            "component_id": primary.get("primary_component_id"),
            "type": primary_comp.get("type"),
            "summary": primary_comp.get("summary"),
            "evidence": short_text(primary_comp.get("evidence"), view_excerpt_chars),
        },

        "attack_chain": attack.get("chain", []),
        "execution_chain": execution.get("chain", []),

        "component_id_sets": packet.get("derived_views", {}).get("component_id_sets", {}),
        "chain_lengths": packet.get("derived_views", {}).get("chain_lengths", {}),
        "evidence_excerpts": evidence_excerpts,
        "provenance": packet.get("provenance", {}),
    }

    return view


def make_skill_input(packet: dict[str, Any], skill_excerpt_chars: int) -> dict[str, Any]:
    annotations = packet.get("annotations", {})
    target = annotations.get("target", {})
    primary = annotations.get("primary", {})
    attack = annotations.get("attack_chain", {})
    execution = annotations.get("execution_chain", {})
    validation = packet.get("annotation_validation", default_validation())

    annotated_full = packet.get("components", {}).get("annotated_components_full", {})

    evidence_components = []
    for cid in packet.get("derived_views", {}).get("component_id_sets", {}).get("annotated_union", []):
        comp = annotated_full.get(cid)
        if not isinstance(comp, dict):
            continue
        evidence_components.append({
            "component_id": cid,
            "type": comp.get("type"),
            "excerpt": short_text(extract_component_text(comp), skill_excerpt_chars),
        })

    return {
        "schema_version": "skill_input_case_v2",
        "case_id": packet.get("case_id"),
        "trajectory_id": packet.get("trajectory_id"),
        "case_name": packet.get("case_name"),
        "risk_category": packet.get("risk_category"),
        "category": packet.get("category"),
        "subcategory": packet.get("subcategory"),
        "technique": packet.get("technique"),

        "annotation_validation": {
            "status": validation.get("status"),
            "hard_errors": validation.get("hard_errors", []),
            "soft_warnings": validation.get("soft_warnings", []),
            "info": validation.get("info", []),
            "needs_review": validation.get("review", {}).get("needs_review"),
        },

        "target_component_id": target.get("target_component_id"),
        "primary_component_id": primary.get("primary_component_id"),
        "attack_chain": attack.get("chain", []),
        "execution_chain": execution.get("chain", []),
        "evidence_components": evidence_components,
        "provenance": packet.get("provenance", {}),
    }


# =============================================================================
# Main build loop
# =============================================================================

def build_all(args: argparse.Namespace) -> dict[str, Any]:
    normalized_root = as_cases_dir(Path(args.normalized_root))
    target_root = as_cases_dir(Path(args.target_root))
    primary_root = as_cases_dir(Path(args.primary_root))
    attack_chain_root = as_cases_dir(Path(args.attack_chain_root))
    execution_chain_root = as_cases_dir(Path(args.execution_chain_root))
    output_root = Path(args.output_root)

    for name, root in [
        ("normalized_root", normalized_root),
        ("target_root", target_root),
        ("primary_root", primary_root),
        ("attack_chain_root", attack_chain_root),
        ("execution_chain_root", execution_chain_root),
    ]:
        if not root.is_dir():
            raise FileNotFoundError(f"{name} does not exist or is not a directory: {root}")

    validation_report_path = Path(args.validation_report) if args.validation_report else None
    validation_map, validation_summary = load_validation_map(validation_report_path)

    packets_dir = output_root / "packets"
    views_dir = output_root / "views"
    skill_inputs_dir = output_root / "skill_inputs"
    by_risk_dir = skill_inputs_dir / "by_risk_category"

    for d in [packets_dir, views_dir, skill_inputs_dir, by_risk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cases_jsonl = skill_inputs_dir / "cases.jsonl"
    reset_file(cases_jsonl)

    # Reset old by-risk jsonl files to avoid appending stale runs.
    for old in by_risk_dir.glob("*.jsonl"):
        old.unlink()

    normalized_files = sorted(normalized_root.glob("*.json"))
    if args.limit is not None:
        normalized_files = normalized_files[:args.limit]

    summary: dict[str, Any] = {
        "num_cases_seen": 0,
        "num_packets_written": 0,
        "num_views_written": 0,
        "num_skill_inputs_written": 0,
        "num_skipped_existing": 0,
        "num_skipped_missing_annotations": 0,
        "num_skipped_hard_error": 0,
        "num_failed": 0,

        "risk_category_counts": Counter(),
        "component_type_counts": Counter(),
        "attack_chain_length_counts": Counter(),
        "execution_chain_length_counts": Counter(),
        "attack_role_counts": Counter(),
        "execution_role_counts": Counter(),

        "validation_status_counts": Counter(),
        "validation_hard_error_counts": Counter(),
        "validation_soft_warning_counts": Counter(),
        "validation_info_counts": Counter(),

        "failures": [],
        "skipped": [],
    }

    manifest_cases = []

    for i, normalized_path in enumerate(normalized_files, 1):
        case_id = case_id_from_normalized(normalized_path)
        summary["num_cases_seen"] += 1

        try:
            annotation_validation = validation_map.get(case_id, default_validation())

            if args.exclude_hard_error and is_hard_error(annotation_validation):
                summary["num_skipped_hard_error"] += 1
                summary["skipped"].append({
                    "case_id": case_id,
                    "reason": "hard_error",
                    "hard_errors": annotation_validation.get("hard_errors", []),
                })
                if args.verbose:
                    print(f"[skip-hard-error] {i}/{len(normalized_files)} {case_id}")
                continue

            target_path = find_annotation_path(target_root, case_id, "target")
            primary_path = find_annotation_path(primary_root, case_id, "primary")
            attack_chain_path = find_annotation_path(attack_chain_root, case_id, "attack_chain")
            execution_chain_path = find_annotation_path(execution_chain_root, case_id, "execution_chain")

            missing = []
            if target_path is None:
                missing.append("target")
            if primary_path is None:
                missing.append("primary")
            if attack_chain_path is None:
                missing.append("attack_chain")
            if execution_chain_path is None:
                missing.append("execution_chain")

            if missing and args.skip_missing_annotations:
                summary["num_skipped_missing_annotations"] += 1
                summary["skipped"].append({
                    "case_id": case_id,
                    "reason": "missing_annotations",
                    "missing": missing,
                })
                if args.verbose:
                    print(f"[skip-missing] {i}/{len(normalized_files)} {case_id} missing={missing}")
                continue

            packet_path = packets_dir / f"{case_id}.case_packet.json"
            view_path = views_dir / f"{case_id}.case_view.json"

            if packet_path.exists() and view_path.exists() and not args.force:
                summary["num_skipped_existing"] += 1
                if args.verbose:
                    print(f"[skip-existing] {i}/{len(normalized_files)} {case_id}")
                continue

            normalized = load_json(normalized_path)
            target_ann = load_json(target_path) if target_path else {}
            primary_ann = load_json(primary_path) if primary_path else {}
            attack_ann = load_json(attack_chain_path) if attack_chain_path else {}
            exec_ann = load_json(execution_chain_path) if execution_chain_path else {}

            packet = build_case_packet(
                case_id=case_id,
                normalized=normalized,
                normalized_path=normalized_path,
                target_ann=target_ann,
                primary_ann=primary_ann,
                attack_ann=attack_ann,
                exec_ann=exec_ann,
                target_path=target_path,
                primary_path=primary_path,
                attack_chain_path=attack_chain_path,
                execution_chain_path=execution_chain_path,
                annotation_validation=annotation_validation,
                target_version=args.target_version,
                primary_version=args.primary_version,
                attack_chain_version=args.attack_chain_version,
                execution_chain_version=args.execution_chain_version,
                preview_chars=args.preview_chars,
                include_all_components_full=args.include_all_components_full,
            )

            view = make_case_view(packet, view_excerpt_chars=args.view_excerpt_chars)
            skill_input = make_skill_input(packet, skill_excerpt_chars=args.skill_excerpt_chars)

            write_json(packet_path, packet)
            write_json(view_path, view)
            append_jsonl(cases_jsonl, skill_input)

            risk = packet.get("risk_category") or "Unknown"
            by_risk_path = by_risk_dir / f"{safe_name(risk)}.jsonl"
            append_jsonl(by_risk_path, skill_input)

            summary["num_packets_written"] += 1
            summary["num_views_written"] += 1
            summary["num_skill_inputs_written"] += 1

            summary["risk_category_counts"][risk] += 1

            for typ, n in packet.get("derived_views", {}).get("component_type_counts", {}).items():
                summary["component_type_counts"][typ] += n

            acl = packet.get("derived_views", {}).get("chain_lengths", {}).get("attack_chain")
            ecl = packet.get("derived_views", {}).get("chain_lengths", {}).get("execution_chain")
            summary["attack_chain_length_counts"][acl] += 1
            summary["execution_chain_length_counts"][ecl] += 1

            for step in packet.get("annotations", {}).get("attack_chain", {}).get("chain", []):
                role = step.get("role_in_attack")
                if isinstance(role, str):
                    summary["attack_role_counts"][role] += 1

            for step in packet.get("annotations", {}).get("execution_chain", {}).get("chain", []):
                role = step.get("role_in_execution")
                if isinstance(role, str):
                    summary["execution_role_counts"][role] += 1

            v_status = annotation_validation.get("status", "not_provided")
            summary["validation_status_counts"][v_status] += 1

            for e in annotation_validation.get("hard_errors", []) or []:
                summary["validation_hard_error_counts"][e] += 1

            for w in annotation_validation.get("soft_warnings", []) or []:
                summary["validation_soft_warning_counts"][w] += 1

            for info in annotation_validation.get("info", []) or []:
                summary["validation_info_counts"][info] += 1

            manifest_cases.append({
                "case_id": case_id,
                "trajectory_id": packet.get("trajectory_id"),
                "case_name": packet.get("case_name"),
                "risk_category": risk,
                "validation_status": v_status,
                "validation_needs_review": annotation_validation.get("review", {}).get("needs_review"),
                "num_validation_hard_errors": len(annotation_validation.get("hard_errors", []) or []),
                "num_validation_soft_warnings": len(annotation_validation.get("soft_warnings", []) or []),
                "target_component_id": packet.get("annotations", {}).get("target", {}).get("target_component_id"),
                "primary_component_id": packet.get("annotations", {}).get("primary", {}).get("primary_component_id"),
                "attack_chain_length": acl,
                "execution_chain_length": ecl,
                "packet_file": str(packet_path),
                "view_file": str(view_path),
            })

            if args.verbose:
                print(f"[ok] {i}/{len(normalized_files)} {case_id}")

        except Exception as e:
            summary["num_failed"] += 1
            failure = {
                "case_id": case_id,
                "normalized_file": str(normalized_path),
                "error": repr(e),
            }
            summary["failures"].append(failure)
            if args.verbose:
                print(f"[fail] {i}/{len(normalized_files)} {case_id}: {repr(e)}")

    report_summary = {
        "num_cases_seen": summary["num_cases_seen"],
        "num_packets_written": summary["num_packets_written"],
        "num_views_written": summary["num_views_written"],
        "num_skill_inputs_written": summary["num_skill_inputs_written"],
        "num_skipped_existing": summary["num_skipped_existing"],
        "num_skipped_missing_annotations": summary["num_skipped_missing_annotations"],
        "num_skipped_hard_error": summary["num_skipped_hard_error"],
        "num_failed": summary["num_failed"],

        "risk_category_counts": dict(summary["risk_category_counts"]),
        "component_type_counts": dict(summary["component_type_counts"]),

        "attack_chain_length_counts": {
            str(k): v
            for k, v in sorted(summary["attack_chain_length_counts"].items(), key=lambda kv: (kv[0] is None, kv[0]))
        },
        "execution_chain_length_counts": {
            str(k): v
            for k, v in sorted(summary["execution_chain_length_counts"].items(), key=lambda kv: (kv[0] is None, kv[0]))
        },

        "attack_role_counts": dict(summary["attack_role_counts"]),
        "execution_role_counts": dict(summary["execution_role_counts"]),

        "validation_status_counts": dict(summary["validation_status_counts"]),
        "validation_hard_error_counts": dict(summary["validation_hard_error_counts"]),
        "validation_soft_warning_counts": dict(summary["validation_soft_warning_counts"]),
        "validation_info_counts": dict(summary["validation_info_counts"]),

        "failures": summary["failures"],
        "skipped": summary["skipped"],
    }

    manifest = {
        "schema_version": "case_package_manifest_v2",
        "input_roots": {
            "normalized_root": str(normalized_root),
            "target_root": str(target_root),
            "primary_root": str(primary_root),
            "attack_chain_root": str(attack_chain_root),
            "execution_chain_root": str(execution_chain_root),
            "validation_report": str(validation_report_path) if validation_report_path else None,
        },
        "output_root": str(output_root),
        "settings": {
            "target_version": args.target_version,
            "primary_version": args.primary_version,
            "attack_chain_version": args.attack_chain_version,
            "execution_chain_version": args.execution_chain_version,
            "skip_missing_annotations": args.skip_missing_annotations,
            "exclude_hard_error": args.exclude_hard_error,
            "include_all_components_full": args.include_all_components_full,
            "preview_chars": args.preview_chars,
            "view_excerpt_chars": args.view_excerpt_chars,
            "skill_excerpt_chars": args.skill_excerpt_chars,
            "limit": args.limit,
        },
        "annotation_validation_summary_from_report": validation_summary,
        "summary": report_summary,
        "files": {
            "packets_dir": str(packets_dir),
            "views_dir": str(views_dir),
            "skill_inputs_jsonl": str(cases_jsonl),
            "skill_inputs_by_risk_dir": str(by_risk_dir),
        },
        "cases": manifest_cases,
    }

    write_json(output_root / "manifest.json", manifest)
    return manifest


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable case packages from normalized components and staged annotations."
    )

    parser.add_argument(
        "--normalized_root",
        required=True,
        help="Root containing normalized component JSON files, either directly or under cases/.",
    )
    parser.add_argument(
        "--target_root",
        required=True,
        help="Root containing target annotations, either directly or under cases/.",
    )
    parser.add_argument(
        "--primary_root",
        required=True,
        help="Root containing primary annotations, either directly or under cases/.",
    )
    parser.add_argument(
        "--attack_chain_root",
        required=True,
        help="Root containing attack-chain annotations, either directly or under cases/.",
    )
    parser.add_argument(
        "--execution_chain_root",
        required=True,
        help="Root containing execution-chain annotations, either directly or under cases/.",
    )
    parser.add_argument(
        "--validation_report",
        default=None,
        help="Optional validation_report.json from validate_annotations.py.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root for packets/, views/, skill_inputs/, and manifest.json.",
    )

    parser.add_argument("--target_version", default=None)
    parser.add_argument("--primary_version", default=None)
    parser.add_argument("--attack_chain_version", default=None)
    parser.add_argument("--execution_chain_version", default=None)

    parser.add_argument(
        "--preview_chars",
        type=int,
        default=600,
        help="Preview length for all_components_compact.",
    )
    parser.add_argument(
        "--view_excerpt_chars",
        type=int,
        default=1600,
        help="Excerpt length for case_view evidence.",
    )
    parser.add_argument(
        "--skill_excerpt_chars",
        type=int,
        default=1200,
        help="Excerpt length for skill_inputs evidence.",
    )

    parser.add_argument(
        "--include_all_components_full",
        action="store_true",
        help="Include full content for all components in each packet. Default only includes full annotated components.",
    )
    parser.add_argument(
        "--skip_missing_annotations",
        action="store_true",
        help="Skip cases missing target/primary/attack_chain/execution_chain files.",
    )
    parser.add_argument(
        "--exclude_hard_error",
        action="store_true",
        help="Skip cases whose annotation_validation.status is hard_error.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing packet/view outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of normalized cases for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case progress.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_all(args)

    print("\n[done]")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"\nOutput root: {args.output_root}")
    print(f"Manifest: {Path(args.output_root) / 'manifest.json'}")


if __name__ == "__main__":
    main()
