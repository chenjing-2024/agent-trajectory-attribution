#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CID_RE = re.compile(r"^C(\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def discover_json_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def get_components(case: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("trajectory", "components_metadata", "components", "normalized_components"):
        value = case.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    view = case.get("annotation_view")
    if isinstance(view, dict) and isinstance(view.get("trajectory"), list):
        return [x for x in view["trajectory"] if isinstance(x, dict)]
    return []


def component_index(components: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, component in enumerate(components):
        cid = component.get("component_id")
        if isinstance(cid, str):
            out[cid] = idx
    return out


def get_target_id(ann: dict[str, Any]) -> str | None:
    value = ann.get("target_component_id")
    if isinstance(value, str):
        return value
    target = ann.get("target_action")
    if isinstance(target, dict) and isinstance(target.get("component_id"), str):
        return target["component_id"]
    return None


def get_primary_id(ann: dict[str, Any]) -> str | None:
    value = ann.get("primary_component_id")
    if isinstance(value, str):
        return value
    primary = ann.get("primary_attribution_component")
    if isinstance(primary, dict) and isinstance(primary.get("component_id"), str):
        return primary["component_id"]
    return None


def chain_ids(value: Any) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    errors: list[str] = []
    if not isinstance(value, list):
        return ids, ["chain_not_list"]
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"chain_item_{i}_not_object")
            continue
        cid = item.get("component_id")
        if not isinstance(cid, str) or not CID_RE.fullmatch(cid):
            errors.append(f"chain_item_{i}_invalid_component_id")
            continue
        ids.append(cid)
    return ids, errors


def find_normalized_case(normalized_dir: Path, ann: dict[str, Any], ann_path: Path) -> Path | None:
    meta = ann.get("_metadata")
    if isinstance(meta, dict):
        source_paths = meta.get("annotation_paths")
        if isinstance(source_paths, dict):
            for key in ("normalized", "source"):
                p = source_paths.get(key)
                if isinstance(p, str) and Path(p).exists():
                    return Path(p)
        source_meta = meta.get("source_metadata")
        if isinstance(source_meta, dict):
            for source in ("target", "primary", "attack_chain", "execution_chain"):
                m = source_meta.get(source)
                if isinstance(m, dict):
                    p = m.get("source_path")
                    if isinstance(p, str) and Path(p).exists():
                        return Path(p)

    tid = ann.get("trajectory_id")
    if not isinstance(tid, str):
        return None
    matches: list[Path] = []
    for p in normalized_dir.rglob("*.json"):
        if tid in p.name:
            matches.append(p)
    if len(matches) == 1:
        return matches[0]
    for p in matches:
        try:
            obj = load_json(p)
            if obj.get("trajectory_id") == tid:
                return p
        except Exception:
            pass
    return None


def check_summary_object(name: str, obj: Any, expected_id: str | None, hard: list[str], soft: list[str]) -> None:
    if not isinstance(obj, dict):
        hard.append(f"missing_or_invalid_{name}")
        return
    cid = obj.get("component_id")
    if cid != expected_id:
        hard.append(f"{name}_component_id_mismatch:{cid}!={expected_id}")
    if not isinstance(obj.get("summary"), str) or not obj.get("summary", "").strip():
        soft.append(f"{name}_missing_summary")
    if not isinstance(obj.get("evidence"), str) or not obj.get("evidence", "").strip():
        soft.append(f"{name}_missing_evidence")


def validate_case(ann_path: Path, normalized_dir: Path) -> dict[str, Any]:
    ann = load_json(ann_path)
    hard: list[str] = []
    soft: list[str] = []

    tid = ann.get("trajectory_id")
    if not isinstance(tid, str) or not tid:
        hard.append("missing_trajectory_id")

    if ann.get("target_type") != "unsafe_action":
        hard.append(f"target_type_not_unsafe_action:{ann.get('target_type')}")

    # attack_success may use different semantics across upstream annotation
    # files. For this merged unsafe-action validation, do not fail or warn on
    # its canonical value or cross-source disagreement. Each annotation field
    # is validated from its designated source instead.

    normalized_path = find_normalized_case(normalized_dir, ann, ann_path)
    if normalized_path is None:
        hard.append("normalized_case_not_found")
        components: list[dict[str, Any]] = []
    else:
        normalized = load_json(normalized_path)
        components = get_components(normalized)
        if not components:
            hard.append("normalized_trajectory_empty")
        normalized_tid = normalized.get("trajectory_id")
        if isinstance(tid, str) and normalized_tid and normalized_tid != tid:
            hard.append(f"trajectory_id_mismatch:{normalized_tid}!={tid}")

    order = component_index(components)
    target_id = get_target_id(ann)
    primary_id = get_primary_id(ann)

    if not target_id:
        hard.append("missing_target_component_id")
    elif target_id not in order:
        hard.append(f"target_component_not_found:{target_id}")

    if not primary_id:
        hard.append("missing_primary_component_id")
    elif primary_id not in order:
        hard.append(f"primary_component_not_found:{primary_id}")

    if target_id and primary_id and target_id == primary_id:
        hard.append("primary_equals_target")
    if target_id in order and primary_id in order and order[primary_id] >= order[target_id]:
        hard.append("primary_not_strictly_before_target")

    check_summary_object("target_action", ann.get("target_action"), target_id, hard, soft)
    check_summary_object("primary_attribution_component", ann.get("primary_attribution_component"), primary_id, hard, soft)

    attack_ids, attack_errors = chain_ids(ann.get("attack_chain"))
    execution_ids, execution_errors = chain_ids(ann.get("execution_chain"))
    hard.extend(f"attack_{x}" for x in attack_errors)
    hard.extend(f"execution_{x}" for x in execution_errors)

    for label, ids in (("attack", attack_ids), ("execution", execution_ids)):
        duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
        if duplicates:
            hard.append(f"{label}_chain_duplicate_ids:{','.join(duplicates)}")
        unknown = [cid for cid in ids if cid not in order]
        if unknown:
            hard.append(f"{label}_chain_unknown_ids:{','.join(unknown)}")
        known_positions = [order[cid] for cid in ids if cid in order]
        if known_positions != sorted(known_positions):
            hard.append(f"{label}_chain_not_in_trajectory_order")

    # Current attack-chain prompt: strictly before primary, excludes primary and target.
    review_details: list[dict[str, Any]] = []
    if primary_id in attack_ids:
        soft.append("review_primary_in_attack_chain:" + str(primary_id))
        review_details.append({
            "review_code": "primary_in_attack_chain",
            "message": "The attack chain contains the selected primary component, but attack_chain must exclude primary.",
            "problematic_component_ids": [primary_id],
            "primary_component_id": primary_id,
            "target_component_id": target_id,
            "attack_chain_component_ids": attack_ids,
            "suggested_check": "Remove the primary from attack_chain if the primary attribution is correct; otherwise review the primary selection.",
        })
    if target_id in attack_ids:
        hard.append("target_in_attack_chain")
    if primary_id in order:
        after_or_equal_primary = [cid for cid in attack_ids if cid in order and order[cid] >= order[primary_id]]
        if after_or_equal_primary:
            soft.append(
                "review_attack_chain_not_strictly_before_primary:"
                + ",".join(after_or_equal_primary)
            )
            review_details.append({
                "review_code": "attack_chain_not_strictly_before_primary",
                "message": "One or more attack-chain components occur at or after the selected primary component in trajectory order.",
                "problematic_component_ids": after_or_equal_primary,
                "primary_component_id": primary_id,
                "primary_component_position": order.get(primary_id),
                "target_component_id": target_id,
                "target_component_position": order.get(target_id),
                "attack_chain_component_ids": attack_ids,
                "attack_chain_component_positions": {cid: order.get(cid) for cid in attack_ids if cid in order},
                "suggested_check": "Check whether the problematic component should be removed from attack_chain, moved to execution_chain, or whether primary attribution is incorrect.",
            })

    # Current execution-chain prompt: strictly after primary and strictly before target.
    # It explicitly excludes both primary and target.
    if primary_id in execution_ids:
        hard.append("primary_in_execution_chain")
    if target_id in execution_ids:
        hard.append("target_in_execution_chain")
    if primary_id in order and target_id in order:
        outside = [
            cid for cid in execution_ids
            if cid in order and not (order[primary_id] < order[cid] < order[target_id])
        ]
        if outside:
            hard.append("execution_chain_not_strictly_between_primary_target:" + ",".join(outside))

    overlap = sorted(set(attack_ids) & set(execution_ids))
    if overlap:
        hard.append("attack_execution_overlap:" + ",".join(overlap))

    # Do not use the merged object's generic cross-source conflict flag as a
    # validation failure. Canonical fields are taken from their designated
    # annotation files: target, primary, attack_chain, and execution_chain.
    # In particular, attack_success disagreement is intentionally ignored.

    confidence = ann.get("confidence")
    if isinstance(confidence, dict):
        for name in ("target", "primary", "attack_chain", "execution_chain"):
            value = confidence.get(name)
            if value is None:
                soft.append(f"missing_{name}_confidence")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 <= float(value) <= 1):
                hard.append(f"invalid_{name}_confidence:{value}")
    else:
        soft.append("missing_or_invalid_confidence_object")

    # Structural errors fail. Review-only findings and ordinary warnings are
    # routed to review without being counted as annotation failures.
    review_warnings = [
        item for item in soft
        if item.startswith("review_")
    ]
    status = "fail" if hard else ("review" if review_warnings else ("warning" if soft else "pass"))
    return {
        "schema_version": "unsafe_action_deterministic_validation_v3",
        "case_id": ann_path.stem,
        "trajectory_id": tid,
        "status": status,
        "needs_review": status == "review",
        "hard_errors": hard,
        "soft_warnings": soft,
        "review_details": review_details,
        "target_component_id": target_id,
        "primary_component_id": primary_id,
        "attack_chain_component_ids": attack_ids,
        "execution_chain_component_ids": execution_ids,
        "files": {
            "annotation": str(ann_path),
            "normalized": str(normalized_path) if normalized_path else None,
        },
        "annotation_snapshot": {
            "target_type": ann.get("target_type"),
            "attack_success": ann.get("attack_success"),
            "risk_category": ann.get("risk_category"),
        },
    }



def copy_case_bundle(result: dict[str, Any], validation_path: Path, bundle_root: Path) -> dict[str, str | None]:
    case_id = str(result.get("case_id") or result.get("trajectory_id") or "unknown_case")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id)
    case_dir = bundle_root / safe_name
    case_dir.mkdir(parents=True, exist_ok=True)

    files = result.get("files", {})
    if not isinstance(files, dict):
        files = {}

    copied = {"case_dir": str(case_dir), "annotation": None, "normalized": None, "validation": None}

    for key, out_name in (("annotation", "merged_annotation.json"), ("normalized", "normalized_trajectory.json")):
        value = files.get(key)
        if isinstance(value, str) and Path(value).is_file():
            dst = case_dir / out_name
            shutil.copy2(value, dst)
            copied[key] = str(dst)

    if validation_path.is_file():
        dst = case_dir / "deterministic_validation.json"
        shutil.copy2(validation_path, dst)
        copied["validation"] = str(dst)

    review_lines = [
        f"# {case_id}",
        "",
        f"- Trajectory ID: `{result.get('trajectory_id')}`",
        f"- Status: `{result.get('status')}`",
        f"- Target: `{result.get('target_component_id')}`",
        f"- Primary: `{result.get('primary_component_id')}`",
        f"- Attack chain: `{', '.join(result.get('attack_chain_component_ids', []))}`",
        f"- Execution chain: `{', '.join(result.get('execution_chain_component_ids', []))}`",
        "",
    ]
    if result.get("hard_errors"):
        review_lines += ["## Hard errors", ""] + [f"- `{x}`" for x in result["hard_errors"]] + [""]
    if result.get("soft_warnings"):
        review_lines += ["## Warnings", ""] + [f"- `{x}`" for x in result["soft_warnings"]] + [""]
    if result.get("review_details"):
        review_lines += ["## Review details", ""]
        for i, detail in enumerate(result["review_details"], 1):
            review_lines += [
                f"### Issue {i}: {detail.get('review_code')}",
                "",
                str(detail.get("message", "")),
                "",
                f"- Problematic components: `{', '.join(detail.get('problematic_component_ids', []))}`",
                f"- Suggested check: {detail.get('suggested_check', '')}",
                "",
            ]
    (case_dir / "REVIEW.md").write_text("\n".join(review_lines).rstrip() + "\n", encoding="utf-8")
    copied["review_readme"] = str(case_dir / "REVIEW.md")
    return copied


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic validation of merged unsafe-action annotations.")
    p.add_argument("--annotations_dir", required=True, type=Path, help="Directory containing one merged annotation JSON per trajectory.")
    p.add_argument("--normalized_dir", required=True, type=Path, help="Directory containing normalized trajectory case JSON files.")
    p.add_argument("--output_root", required=True, type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_json_files(args.annotations_dir)
    if args.limit is not None:
        files = files[:args.limit]

    case_dir = args.output_root / "case_validation"
    case_dir.mkdir(parents=True, exist_ok=True)
    semantic_input = args.output_root / "semantic_input_cases.jsonl"
    all_jsonl = args.output_root / "all_cases.jsonl"
    fail_jsonl = args.output_root / "fail_cases.jsonl"
    review_jsonl = args.output_root / "review_cases.jsonl"
    warning_jsonl = args.output_root / "warning_cases.jsonl"
    review_dir = args.output_root / "review_cases"
    fail_dir = args.output_root / "fail_cases"
    warning_dir = args.output_root / "warning_cases"
    validator_error_dir = args.output_root / "validator_error_cases"
    for d in (review_dir, fail_dir, warning_dir, validator_error_dir):
        d.mkdir(parents=True, exist_ok=True)
    for p in (semantic_input, all_jsonl, fail_jsonl, review_jsonl, warning_jsonl):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")

    counts = Counter()
    hard_counts = Counter()
    soft_counts = Counter()
    index: list[dict[str, Any]] = []

    for i, ann_path in enumerate(files, 1):
        out_path = case_dir / f"{ann_path.stem}.deterministic_validation.json"
        if out_path.exists() and not args.force:
            result = load_json(out_path)
        else:
            try:
                result = validate_case(ann_path, args.normalized_dir)
            except Exception as exc:
                result = {
                    "schema_version": "unsafe_action_deterministic_validation_v3",
                    "case_id": ann_path.stem,
                    "trajectory_id": None,
                    "status": "validator_error",
                    "hard_errors": [repr(exc)],
                    "soft_warnings": [],
                    "files": {"annotation": str(ann_path), "normalized": None},
                }
            write_json(out_path, result)

        status = str(result.get("status"))
        counts[status] += 1
        append_jsonl(all_jsonl, result)
        bundle = None
        if status == "fail":
            append_jsonl(fail_jsonl, result)
            bundle = copy_case_bundle(result, out_path, fail_dir)
        elif status == "review":
            append_jsonl(review_jsonl, result)
            bundle = copy_case_bundle(result, out_path, review_dir)
        elif status == "warning":
            append_jsonl(warning_jsonl, result)
            bundle = copy_case_bundle(result, out_path, warning_dir)
        elif status == "validator_error":
            append_jsonl(fail_jsonl, result)
            bundle = copy_case_bundle(result, out_path, validator_error_dir)

        for x in result.get("hard_errors", []):
            hard_counts[str(x).split(":", 1)[0]] += 1
        for x in result.get("soft_warnings", []):
            soft_counts[str(x).split(":", 1)[0]] += 1

        # Semantic validator should see every structurally loadable case, including deterministic failures.
        if result.get("files", {}).get("normalized"):
            semantic_case = {
                "case_id": result.get("case_id"),
                "trajectory_id": result.get("trajectory_id"),
                "deterministic_status": status,
                "deterministic_hard_errors": result.get("hard_errors", []),
                "deterministic_soft_warnings": result.get("soft_warnings", []),
                "deterministic_review_details": result.get("review_details", []),
                "validation_file": str(out_path),
                "files": result.get("files", {}),
            }
            append_jsonl(semantic_input, semantic_case)

        index.append({
            "case_id": result.get("case_id"),
            "trajectory_id": result.get("trajectory_id"),
            "status": status,
            "hard_errors": result.get("hard_errors", []),
            "soft_warnings": result.get("soft_warnings", []),
            "review_details": result.get("review_details", []),
            "validation_file": str(out_path),
            "case_bundle": bundle,
        })
        if args.verbose:
            suffix = ""
            if status == "review":
                problematic = []
                for detail in result.get("review_details", []):
                    if isinstance(detail, dict):
                        problematic.extend(detail.get("problematic_component_ids", []))
                problematic = list(dict.fromkeys(problematic))
                if problematic:
                    suffix = " problematic=" + ",".join(problematic)
            print(f"[{status}] {i}/{len(files)} {ann_path.name}{suffix}")

    report = {
        "schema_version": "unsafe_action_deterministic_validation_report_v3",
        "settings": {
            "annotations_dir": str(args.annotations_dir),
            "normalized_dir": str(args.normalized_dir),
            "limit": args.limit,
        },
        "summary": {
            "num_cases": len(files),
            "status_counts": dict(counts),
            "hard_error_counts": dict(hard_counts.most_common()),
            "soft_warning_counts": dict(soft_counts.most_common()),
        },
        "files": {
            "case_validation_dir": str(case_dir),
            "semantic_input_cases_jsonl": str(semantic_input),
            "all_cases_jsonl": str(all_jsonl),
            "fail_cases_jsonl": str(fail_jsonl),
            "review_cases_jsonl": str(review_jsonl),
            "warning_cases_jsonl": str(warning_jsonl),
            "review_cases_dir": str(review_dir),
            "fail_cases_dir": str(fail_dir),
            "warning_cases_dir": str(warning_dir),
            "validator_error_cases_dir": str(validator_error_dir),
        },
        "case_index": index,
    }
    write_json(args.output_root / "deterministic_validation_report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nReview/failure bundles:")
    print(f"  review: {review_dir}")
    print(f"  fail:   {fail_dir}")
    print(f"  warning:{warning_dir}")
    print(f"  errors: {validator_error_dir}")


if __name__ == "__main__":
    main()
