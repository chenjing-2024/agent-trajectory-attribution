#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize Agent3Sigma single trajectory.json files into:

{
  "item": {...},
  "trajectory": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "...", "component_id": "C1"},
    {"role": "tool", "content": "...", "component_id": "C2"},
    {"role": "assistant", "content": "...", "component_id": "C3"}
  ],
  "components_metadata": [
    {
      "component_id": "C1",
      "turn": 1,
      "type": "user_message",
      "role": "user",
      ...
    }
  ],
  "judgment": {...},
  "execution": {...}
}

Component types:
- user_message
- assistant_message
- action_result_pair

Usage for one file:
python scripts/common/normalize_components.py \
  --input prepared_cases/cases/example/trajectory.json \
  --output annotation_runs/example/normalized_role_content.jsonl \
  --summary annotation_runs/example/summary.json

Usage for a cases directory:
python scripts/common/normalize_components.py \
  --input prepared_cases/cases \
  --output annotation_runs/unsafe/normalized_role_content.jsonl \
  --summary annotation_runs/unsafe/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Basic utilities
# =============================================================================


def json_dumps(obj: Any, indent: Optional[int] = 2) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    except TypeError:
        return str(obj)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json_dumps(value, indent=2)


def maybe_truncate(text: str, max_chars: Optional[int]) -> Tuple[str, bool]:
    if max_chars is None:
        return text, False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[TRUNCATED]", True


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_single_trajectory(obj: Any) -> bool:
    return isinstance(obj, dict) and "item" in obj and "turns" in obj


def discover_inputs(input_path: Path) -> List[Path]:
    """
    Supports:
    - a single trajectory.json file
    - a directory containing */trajectory.json files
    - a directory containing trajectory.json recursively
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*/trajectory.json"))
        if files:
            return files

        files = sorted(input_path.rglob("trajectory.json"))
        if files:
            return files

        raise FileNotFoundError(f"No trajectory.json found under directory: {input_path}")

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


# =============================================================================
# Tool log grouping
# =============================================================================


def group_logs_by_tool_call_id(
    logs: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Group logs by tool_call_id while preserving original order.

    Normal pattern:
        phase=start
        phase=result

    But this function also handles:
        - missing start
        - missing result
        - missing tool_call_id
        - multiple result logs
    """
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    for idx, log in enumerate(logs):
        tool_call_id = log.get("tool_call_id") or f"missing_tool_call_id_{idx}"
        grouped.setdefault(tool_call_id, []).append(log)

    return list(grouped.items())


def infer_tool_status(result_log: Dict[str, Any], result: Any) -> str:
    if not result_log:
        return "missing_result"

    if result_log.get("phase") == "error":
        return "error"

    if "error" in result_log:
        return "error"

    if isinstance(result, dict):
        if result.get("status") == "error":
            return "error"
        if "error" in result:
            return "error"

    return "success"


def extract_path_from_args(args: Any) -> Optional[str]:
    if not isinstance(args, dict):
        return None

    path_keys = [
        "path",
        "file_path",
        "filepath",
        "target_path",
        "source_path",
        "output_path",
        "input_path",
        "filename",
    ]

    for key in path_keys:
        value = args.get(key)
        if isinstance(value, str):
            return value

    return None


def build_tool_content(
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    max_content_chars: Optional[int],
) -> Tuple[str, str, bool]:
    """
    Build content for trajectory role=tool.

    Returns:
        content, content_full, is_truncated
    """
    content_full = (
        f"Tool: {tool_name}\n"
        f"Arguments:\n{json_dumps(args, indent=2)}\n"
        f"Result:\n{stringify(result)}"
    )

    content, is_truncated = maybe_truncate(content_full, max_content_chars)
    return content, content_full, is_truncated


def normalize_tool_call(
    component_id: str,
    turn_id: int,
    tool_call_id: str,
    logs: List[Dict[str, Any]],
    max_content_chars: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert one grouped tool call into:
    - one trajectory message with role=tool
    - one component metadata entry with type=action_result_pair
    """
    starts = [x for x in logs if x.get("phase") == "start"]
    results = [x for x in logs if x.get("phase") in {"result", "error"}]

    start_log = starts[0] if starts else {}
    result_log = results[-1] if results else {}

    tool_name = (
        start_log.get("tool_name")
        or result_log.get("tool_name")
        or logs[0].get("tool_name")
        or "unknown_tool"
    )

    args = start_log.get("args", {})
    result = result_log.get("result", None)

    if result is None and "error" in result_log:
        result = {"error": result_log.get("error")}

    status = infer_tool_status(result_log, result)
    path = extract_path_from_args(args)
    timestamps = [x.get("timestamp") for x in logs if x.get("timestamp")]

    content, content_full, is_truncated = build_tool_content(
        tool_name=tool_name,
        args=args,
        result=result,
        max_content_chars=max_content_chars,
    )

    trajectory_message = {
        "role": "tool",
        "content": content,
        "component_id": component_id,
    }

    metadata = {
        "component_id": component_id,
        "turn": turn_id,
        "type": "action_result_pair",
        "role": "tool",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "status": status,
        "path": path,
        "args": args,
        "result": result,
        "content_full": content_full,
        "is_truncated": is_truncated,
        "phase_start": bool(starts),
        "phase_result": bool(results),
        "num_logs": len(logs),
        "timestamps": timestamps,
        "run_id": logs[0].get("run_id") if logs else None,
        "session_id": logs[0].get("session_id") if logs else None,
        "session_key": logs[0].get("session_key") if logs else None,
    }

    return trajectory_message, metadata


# =============================================================================
# Main normalization
# =============================================================================


def normalize_single_trajectory(
    case: Dict[str, Any],
    source_path: str,
    max_content_chars: Optional[int] = None,
    include_system: bool = True,
) -> Dict[str, Any]:
    """
    Normalize one trajectory.json object.
    """
    if not is_single_trajectory(case):
        raise ValueError("Input JSON is not a single trajectory. Expected keys: item and turns.")

    item = case.get("item", {})
    turns = case.get("turns", [])
    turns_meta = item.get("turns_meta", [])

    trajectory: List[Dict[str, Any]] = []
    components_metadata: List[Dict[str, Any]] = []

    if include_system:
        trajectory.append(
            {
                "role": "system",
                "content": "",
            }
        )

    component_counter = 1

    def next_component_id() -> str:
        nonlocal component_counter
        cid = f"C{component_counter}"
        component_counter += 1
        return cid

    for turn_index, turn in enumerate(turns):
        turn_id = turn.get("turn", turn_index + 1)
        turn_meta = turns_meta[turn_index] if turn_index < len(turns_meta) else {}

        # ------------------------------------------------------------------
        # 1. user message
        # ------------------------------------------------------------------
        if turn.get("user") is not None:
            component_id = next_component_id()
            content_full = str(turn.get("user", ""))
            content, is_truncated = maybe_truncate(content_full, max_content_chars)

            trajectory.append(
                {
                    "role": "user",
                    "content": content,
                    "component_id": component_id,
                }
            )

            components_metadata.append(
                {
                    "component_id": component_id,
                    "turn": turn_id,
                    "type": "user_message",
                    "role": "user",
                    "content_full": content_full,
                    "is_truncated": is_truncated,
                    "injection_in_user": turn_meta.get("injection_in_user"),
                    "injection_in_tool": turn_meta.get("injection_in_tool"),
                    "injection_type": turn_meta.get("injection_type"),
                }
            )

        # ------------------------------------------------------------------
        # 2. action-result pairs from new_logs
        # ------------------------------------------------------------------
        logs = turn.get("new_logs") or []
        grouped_logs = group_logs_by_tool_call_id(logs)

        for tool_call_id, log_group in grouped_logs:
            component_id = next_component_id()

            trajectory_message, metadata = normalize_tool_call(
                component_id=component_id,
                turn_id=turn_id,
                tool_call_id=tool_call_id,
                logs=log_group,
                max_content_chars=max_content_chars,
            )

            trajectory.append(trajectory_message)
            components_metadata.append(metadata)

        # ------------------------------------------------------------------
        # 3. assistant response
        # ------------------------------------------------------------------
        if turn.get("agent") is not None:
            component_id = next_component_id()
            content_full = str(turn.get("agent", ""))
            content, is_truncated = maybe_truncate(content_full, max_content_chars)

            trajectory.append(
                {
                    "role": "assistant",
                    "content": content,
                    "component_id": component_id,
                }
            )

            components_metadata.append(
                {
                    "component_id": component_id,
                    "turn": turn_id,
                    "type": "assistant_message",
                    "role": "assistant",
                    "content_full": content_full,
                    "is_truncated": is_truncated,
                    "turn_success": turn.get("success"),
                    "turn_error": turn.get("error"),
                    "turn_duration": turn.get("duration"),
                }
            )

    type_counts = Counter(x["type"] for x in components_metadata)

    return {
        "trajectory_id": item.get("id"),
        "source_path": source_path,
        "item": item,
        "trajectory": trajectory,
        "components_metadata": components_metadata,
        "judgment": case.get("judgment"),
        "execution": case.get("execution"),
        "normalization_metadata": {
            "num_turns": len(turns),
            "num_components": len(components_metadata),
            "component_type_counts": dict(type_counts),
            "has_system_message": include_system,
            "max_content_chars": max_content_chars,
        },
    }



def safe_name(text: str) -> str:
    """
    Make a string safe for use as a filename.
    """
    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "unknown"


def make_case_filename(norm: Dict[str, Any], source_path: str) -> str:
    """
    Create filename like:
        0003_syn-0253_gradual_system_prompt_extraction.json
    """
    item = norm.get("item") or {}

    index = item.get("index")
    case_id = item.get("id") or norm.get("trajectory_id") or "unknown_id"
    name = item.get("name")

    if not name:
        name = Path(source_path).parent.name

    if index is not None:
        prefix = f"{int(index):04d}_{case_id}_{name}"
    else:
        prefix = f"{case_id}_{name}"

    return safe_name(prefix) + ".json"


def write_per_case_json_files(
    normalized_items: List[Dict[str, Any]],
    cases_output_dir: Path,
) -> Dict[str, str]:
    """
    Write one pretty JSON file per normalized trajectory.

    Returns:
        mapping from trajectory_id to output path
    """
    cases_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, str] = {}

    for norm in normalized_items:
        source_path = norm.get("source_path") or ""
        filename = make_case_filename(norm, source_path)
        out_path = cases_output_dir / filename

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(norm, f, ensure_ascii=False, indent=2)

        tid = norm.get("trajectory_id") or filename
        output_paths[str(tid)] = str(out_path)

    return output_paths



# =============================================================================
# Output helpers
# =============================================================================


def write_jsonl(items: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    return n


def write_summary(
    normalized_items: List[Dict[str, Any]],
    output_path: Path,
    input_files: List[Path],
    failed: List[Dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_components = 0
    component_type_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    risk_category_counts: Counter[str] = Counter()
    response_category_counts: Counter[str] = Counter()

    for obj in normalized_items:
        meta = obj.get("normalization_metadata", {})
        total_components += meta.get("num_components", 0)
        component_type_counts.update(meta.get("component_type_counts", {}))

        item = obj.get("item") or {}
        category_counts.update([str(item.get("category", "MISSING"))])
        risk_category_counts.update([str(item.get("risk_category", "MISSING"))])

        judgment = obj.get("judgment") or {}
        response_category_counts.update([str(judgment.get("response_category", "MISSING"))])

    summary = {
        "num_discovered": len(input_files),
        "num_processed": len(normalized_items),
        "num_failed": len(failed),
        "num_components": total_components,
        "avg_components_per_trajectory": (
            total_components / len(normalized_items) if normalized_items else 0
        ),
        "component_type_counts": dict(component_type_counts),
        "category_counts": dict(category_counts),
        "risk_category_counts": dict(risk_category_counts),
        "response_category_counts": dict(response_category_counts),
        "failed": failed,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Agent3Sigma trajectory.json into role/content trajectory + components_metadata."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input trajectory.json file or cases directory.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON path.",
    )

    parser.add_argument(
        "--max_content_chars",
        type=int,
        default=None,
        help="Optional max chars for trajectory[].content. Full content is preserved in components_metadata[].content_full.",
    )

    parser.add_argument(
        "--no_system",
        action="store_true",
        help="Do not add initial empty system message.",
    )

    parser.add_argument(
        "--cases_output_dir",
        default=None,
        help=(
            "Optional directory for per-case pretty JSON files. "
            "If omitted, defaults to <output_parent>/cases."
        ),
    )

    parser.add_argument(
        "--no_per_case_json",
        action="store_true",
        help="Do not write one pretty JSON file per case.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    input_files = discover_inputs(input_path)

    normalized_items: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for path in input_files:
        try:
            case = load_json(path)
            norm = normalize_single_trajectory(
                case=case,
                source_path=str(path),
                max_content_chars=args.max_content_chars,
                include_system=not args.no_system,
            )
            normalized_items.append(norm)
        except Exception as e:
            failed.append(
                {
                    "source_path": str(path),
                    "error": repr(e),
                }
            )

    # Write output.
    # If --output ends with .json, write a pretty JSON file.
    #   - single input trajectory -> one JSON object
    #   - directory input -> list of JSON objects
    # Otherwise, write JSONL: one normalized trajectory per line.
    if output_path.suffix == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(normalized_items) == 1:
            obj_to_write = normalized_items[0]
        else:
            obj_to_write = normalized_items

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(obj_to_write, f, ensure_ascii=False, indent=2)

        n = len(normalized_items)
    else:
        n = write_jsonl(normalized_items, output_path)

    # Write one pretty JSON file per case.
    per_case_output_paths = {}
    if not args.no_per_case_json:
        if args.cases_output_dir:
            cases_output_dir = Path(args.cases_output_dir)
        else:
            cases_output_dir = output_path.parent / "cases"

        per_case_output_paths = write_per_case_json_files(
            normalized_items=normalized_items,
            cases_output_dir=cases_output_dir,
        )

    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")

    write_summary(
        normalized_items=normalized_items,
        output_path=summary_path,
        input_files=input_files,
        failed=failed,
    )

    print("=" * 80)
    print("Normalize components finished")
    print("=" * 80)
    print(f"Input:       {input_path}")
    print(f"Discovered:  {len(input_files)}")
    print(f"Processed:   {n}")
    print(f"Failed:      {len(failed)}")
    print(f"Output:      {output_path}")
    print(f"Summary:     {summary_path}")
    if not args.no_per_case_json:
        print(f"Case JSONs:  {cases_output_dir}")
    print("=" * 80)

    if failed:
        print("Failed files:")
        for x in failed[:20]:
            print(f"- {x['source_path']}: {x['error']}")
        if len(failed) > 20:
            print(f"... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
