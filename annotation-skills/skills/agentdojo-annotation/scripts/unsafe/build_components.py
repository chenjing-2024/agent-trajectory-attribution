import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepared_input import is_prepared, validate_prepared


def extract_text_content(content):
    """
    Convert AgentDojo message['content'] into a plain string.

    AgentDojo content often looks like:
    [
        {
            "type": "text",
            "content": "..."
        }
    ]
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict):
                if block.get("content") is not None:
                    pieces.append(str(block["content"]))
                elif block.get("text") is not None:
                    pieces.append(str(block["text"]))
            elif block is not None:
                pieces.append(str(block))
        return "\n".join(pieces)

    return json.dumps(content, ensure_ascii=False)


def clean_function_name(function_name):
    """
    Normalize slightly dirty function names.

    Examples:
    - function=read_channel_messages -> read_channel_messages
    - function_name="get_day_calendar_events" -> get_day_calendar_events

    The raw function name is still preserved separately as function_raw.
    """
    if function_name is None:
        return None

    s = str(function_name).strip()

    if s.startswith("function="):
        return s.replace("function=", "", 1).strip().strip("\"'")

    if s.startswith("function_name="):
        return s.replace("function_name=", "", 1).strip().strip("\"'")

    return s


def get_tool_call_from_message(msg):
    """
    Extract tool call information from either:
    - assistant messages with msg['tool_calls']
    - tool result messages with msg['tool_call']

    Return:
    - raw_function
    - clean_function
    - args
    """
    raw_function = None
    args = None

    # Tool result step
    if isinstance(msg.get("tool_call"), dict):
        tool_call = msg["tool_call"]
        raw_function = tool_call.get("function")
        args = tool_call.get("args")

    # Assistant tool-call step
    elif isinstance(msg.get("tool_calls"), list) and len(msg["tool_calls"]) > 0:
        # AgentDojo examples usually contain one tool call per assistant message.
        # If multiple calls appear, keep the first one for metadata.
        tool_call = msg["tool_calls"][0]
        raw_function = tool_call.get("function")
        args = tool_call.get("args")

    clean_function = clean_function_name(raw_function)

    return raw_function, clean_function, args


def normalize_message_scheme_a(msg):
    """
    Message-level Scheme A normalization.

    Return:
    {
        "role": ...,
        "content": ...
    }

    For tool calls and tool results, function/args/error are written into content.
    """
    role = msg.get("role", "user")
    raw_content = extract_text_content(msg.get("content"))

    raw_function, clean_function, args = get_tool_call_from_message(msg)
    error = msg.get("error")

    args_text = json.dumps(args, ensure_ascii=False) if args is not None else "null"

    if role == "tool":
        normalized_content = (
            "[TOOL RESULT]\n"
            f"function_raw: {raw_function}\n"
            f"function: {clean_function}\n"
            f"args: {args_text}\n"
            f"error: {error}\n"
            "content:\n"
            f"{raw_content}"
        )

    elif role == "assistant" and (msg.get("tool_calls") or msg.get("tool_call")):
        normalized_content = (
            "[ASSISTANT TOOL CALL]\n"
            f"function_raw: {raw_function}\n"
            f"function: {clean_function}\n"
            f"args: {args_text}\n"
            "raw_content:\n"
            f"{raw_content}"
        )

    else:
        normalized_content = raw_content

    return {
        "role": role,
        "content": normalized_content
    }


def build_message_component_metadata(msg, component_id):
    """
    Build metadata for the original message-level component.

    This preserves the old component indexing:
    1 message = 1 component.

    This is useful for mapping old annotations such as:
    critical_component_id -> critical_group_id.
    """
    role = msg.get("role", "user")
    raw_content = extract_text_content(msg.get("content"))

    raw_function, clean_function, args = get_tool_call_from_message(msg)
    error = msg.get("error")

    if role == "tool":
        component_type = "tool_result"
    elif role == "assistant" and (msg.get("tool_calls") or msg.get("tool_call")):
        component_type = "assistant_tool_call"
    elif role == "assistant":
        component_type = "assistant_message"
    elif role == "user":
        component_type = "user_message"
    elif role == "system":
        component_type = "system_message"
    else:
        component_type = "unknown"

    return {
        "component_id": component_id,
        "original_step_id": component_id,
        "role": role,
        "component_type": component_type,
        "function_raw": raw_function,
        "function": clean_function,
        "args": args,
        "error": error,
        "content_preview": raw_content[:300].replace("\n", " ")
    }


def build_action_result_components(message_trajectory, message_metadata):
    """
    Build grouped action-result components.

    Grouping rules:
    - assistant_tool_call + immediately following tool_result
      -> one action_result_pair component
    - system/user/assistant normal message
      -> standalone component
    - assistant_tool_call without following tool_result
      -> standalone_assistant_tool_call
    - tool_result without paired previous assistant_tool_call
      -> standalone_tool_result

    Role policy in grouped trajectory:
    - system_message -> role = system
    - user_message -> role = user
    - assistant_message -> role = assistant
    - action_result_pair -> role = assistant
    - standalone_assistant_tool_call -> role = assistant
    - standalone_tool_result -> role = assistant

    Metadata still preserves the original roles, e.g. roles = ["assistant", "tool"].
    """
    grouped_trajectory = []
    grouped_metadata = []
    component_to_group_id = {}
    group_to_component_ids = {}

    i = 0
    group_id = 1
    n = len(message_trajectory)

    while i < n:
        msg = message_trajectory[i]
        meta = message_metadata[i]

        component_type = meta.get("component_type")
        component_id = meta.get("component_id")

        # Case 1: assistant tool call + immediately following tool result
        if (
            component_type == "assistant_tool_call"
            and i + 1 < n
            and message_metadata[i + 1].get("component_type") == "tool_result"
        ):
            action_msg = message_trajectory[i]
            result_msg = message_trajectory[i + 1]

            action_meta = message_metadata[i]
            result_meta = message_metadata[i + 1]

            original_component_ids = [
                action_meta["component_id"],
                result_meta["component_id"],
            ]

            function_raw = action_meta.get("function_raw") or result_meta.get("function_raw")
            function = action_meta.get("function") or result_meta.get("function")

            args = action_meta.get("args")
            if args is None:
                args = result_meta.get("args")

            tool_error = result_meta.get("error")
            pair_status = "error" if tool_error else "ok"

            args_text = json.dumps(args, ensure_ascii=False) if args is not None else "null"

            grouped_content = (
                "[ACTION-RESULT COMPONENT]\n"
                f"group_id: {group_id}\n"
                f"group_type: action_result_pair\n"
                f"pair_status: {pair_status}\n"
                f"original_component_ids: {original_component_ids}\n"
                f"function_raw: {function_raw}\n"
                f"function: {function}\n"
                f"args: {args_text}\n"
                f"tool_error: {tool_error}\n\n"
                "[ACTION / ASSISTANT TOOL CALL]\n"
                f"{action_msg.get('content', '')}\n\n"
                "[RESULT / TOOL OUTPUT]\n"
                f"{result_msg.get('content', '')}"
            )

            # For an action-result pair, the original roles are mixed:
            # assistant + tool. In the grouped trajectory, use assistant as
            # the closest semantic role, while preserving the exact original
            # roles in metadata["roles"].
            grouped_trajectory.append({
                "role": "assistant",
                "content": grouped_content
            })

            grouped_metadata.append({
                "component_id": group_id,
                "group_id": group_id,
                "original_step_id": original_component_ids[0],
                "original_component_ids": original_component_ids,

                # Metadata role is descriptive, not necessarily the chat role.
                "role": "grouped",
                "roles": [
                    action_meta.get("role"),
                    result_meta.get("role")
                ],

                "component_type": "action_result_pair",
                "group_type": "action_result_pair",
                "component_types": [
                    action_meta.get("component_type"),
                    result_meta.get("component_type")
                ],

                "pair_status": pair_status,
                "function_raw": function_raw,
                "function": function,
                "args": args,
                "error": tool_error,
                "tool_error": tool_error,

                "has_tool_call": True,
                "has_tool_result": True,

                # Useful for inspection only. Do not use this in attribution scoring.
                "contains_injection_marker": "<INFORMATION>" in grouped_content,

                "content_preview": grouped_content[:300].replace("\n", " ")
            })

            for cid in original_component_ids:
                component_to_group_id[str(cid)] = group_id

            group_to_component_ids[str(group_id)] = original_component_ids

            group_id += 1
            i += 2
            continue

        # Case 2: standalone component
        original_component_ids = [component_id]

        standalone_type = component_type
        if component_type == "assistant_tool_call":
            standalone_type = "standalone_assistant_tool_call"
        elif component_type == "tool_result":
            standalone_type = "standalone_tool_result"

        args_text = (
            json.dumps(meta.get("args"), ensure_ascii=False)
            if meta.get("args") is not None
            else "null"
        )

        grouped_content = (
            "[STANDALONE COMPONENT]\n"
            f"group_id: {group_id}\n"
            f"group_type: {standalone_type}\n"
            f"original_component_ids: {original_component_ids}\n"
            f"component_type: {component_type}\n"
            f"function_raw: {meta.get('function_raw')}\n"
            f"function: {meta.get('function')}\n"
            f"args: {args_text}\n"
            f"error: {meta.get('error')}\n\n"
            f"{msg.get('content', '')}"
        )

        # Output chat role for grouped trajectory.
        # Keep user as user, system as system, and map assistant/tool-like
        # standalone components to assistant.
        out_role = msg.get("role", "user")

        if standalone_type in {
            "assistant_message",
            "standalone_assistant_tool_call",
            "standalone_tool_result",
        }:
            out_role = "assistant"

        grouped_trajectory.append({
            "role": out_role,
            "content": grouped_content
        })

        grouped_metadata.append({
            "component_id": group_id,
            "group_id": group_id,
            "original_step_id": component_id,
            "original_component_ids": original_component_ids,

            # This is the output role used in grouped_trajectory.
            "role": out_role,

            # This preserves the original message role.
            "roles": [meta.get("role")],

            "component_type": standalone_type,
            "group_type": standalone_type,
            "component_types": [component_type],

            "pair_status": None,
            "function_raw": meta.get("function_raw"),
            "function": meta.get("function"),
            "args": meta.get("args"),
            "error": meta.get("error"),
            "tool_error": meta.get("error") if component_type == "tool_result" else None,

            "has_tool_call": component_type == "assistant_tool_call",
            "has_tool_result": component_type == "tool_result",

            # Useful for inspection only. Do not use this in attribution scoring.
            "contains_injection_marker": "<INFORMATION>" in grouped_content,

            "content_preview": grouped_content[:300].replace("\n", " ")
        })

        component_to_group_id[str(component_id)] = group_id
        group_to_component_ids[str(group_id)] = original_component_ids

        group_id += 1
        i += 1

    return grouped_trajectory, grouped_metadata, component_to_group_id, group_to_component_ids

def make_data_id(path, src_root, data):
    """
    Example output:
    qwen36_slack_user_task_1_slack_adaptive_template_injection_task_3
    """
    pipeline_name = data.get("pipeline_name")
    suite_name = data.get("suite_name")
    user_task_id = data.get("user_task_id")
    attack_type = data.get("attack_type")
    injection_task_id = data.get("injection_task_id")

    pieces = [
        pipeline_name,
        suite_name,
        user_task_id,
        attack_type,
        injection_task_id
    ]

    if all(pieces):
        return "_".join(pieces)

    # fallback: derive from relative path
    rel = path.relative_to(src_root).with_suffix("")
    return "_".join(rel.parts)


def normalize_file(path, src_root, keep_only_security_true=True):
    """
    Normalize one AgentDojo JSON file into action-result component format.

    Main output fields:
    - trajectory: grouped action-result trajectory
    - components_metadata: grouped action-result metadata

    Preserved message-level fields:
    - message_trajectory
    - message_components_metadata

    Mapping fields:
    - component_to_group_id
    - group_to_component_ids
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if keep_only_security_true and data.get("security") is not True:
        return None

    if is_prepared(data):
        validate_prepared(data, "unsafe")
        print(f"[REUSE] {path}：已校验，跳过标准化和组件拆分，保留原组件 ID。")
        return data

    messages = data.get("messages")
    if not isinstance(messages, list):
        return None

    message_trajectory = []
    message_metadata = []

    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            msg = {
                "role": "user",
                "content": str(msg)
            }

        message_trajectory.append(normalize_message_scheme_a(msg))
        message_metadata.append(build_message_component_metadata(msg, i))

    (
        grouped_trajectory,
        grouped_metadata,
        component_to_group_id,
        group_to_component_ids,
    ) = build_action_result_components(message_trajectory, message_metadata)

    out_data = {
        "data_id": make_data_id(path, src_root, data),
        "source_path": str(path),

        "suite_name": data.get("suite_name"),
        "pipeline_name": data.get("pipeline_name"),
        "user_task_id": data.get("user_task_id"),
        "injection_task_id": data.get("injection_task_id"),
        "attack_type": data.get("attack_type"),

        "utility": data.get("utility"),
        "security": data.get("security"),

        "benchmark_version": data.get("benchmark_version"),
        "evaluation_timestamp": data.get("evaluation_timestamp"),
        "agentdojo_package_version": data.get("agentdojo_package_version"),
        "duration": data.get("duration"),
        "error": data.get("error"),

        # Keep original injection dictionary for inspection only.
        # Do not use this for attribution scoring.
        "injections": data.get("injections"),

        # Main representation for the new attribution experiments.
        # This replaces message-level componentization with action-result grouping.
        "component_scheme": "action_result_pair_schemeA",
        "num_components": len(grouped_trajectory),
        "trajectory": grouped_trajectory,
        "components_metadata": grouped_metadata,

        # Original message-level representation, kept for inspection and
        # mapping old annotations to the new group-level components.
        "num_message_components": len(message_trajectory),
        "message_trajectory": message_trajectory,
        "message_components_metadata": message_metadata,

        # Mapping from old message-level component ids to new group ids.
        "component_to_group_id": component_to_group_id,
        "group_to_component_ids": group_to_component_ids,
    }

    return out_data


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select AgentDojo trajectories and normalize them into "
            "action-result grouped components. Each assistant tool call "
            "and its immediately following tool result are grouped as one component."
        )
    )

    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Original AgentDojo run directory."
    )

    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output directory for action-result grouped trajectories."
    )

    parser.add_argument(
        "--include_non_security_true",
        action="store_true",
        help=(
            "If set, include all trajectories with messages. "
            "By default, only security=true trajectories are selected."
        )
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    keep_only_security_true = not args.include_non_security_true

    total = 0
    selected = 0
    skipped = 0
    errors = 0
    case_results = []

    for path in src_root.rglob("*.json"):
        total += 1

        try:
            normalized = normalize_file(
                path=path,
                src_root=src_root,
                keep_only_security_true=keep_only_security_true
            )

            if normalized is None:
                skipped += 1
                continue

            rel_path = path.relative_to(src_root)
            out_path = out_root / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)

            case_results.append({"source_path": str(path), "status": "reused" if is_prepared(json.loads(path.read_text())) else "normalized"})
            selected += 1
            print(f"[OK] {out_path}")

        except Exception as e:
            case_results.append({"source_path": str(path), "status": "failed", "reason": str(e)})
            errors += 1
            print(f"[ERROR] {path}: {e}")

    (out_root.parent / (out_root.name + "_outcomes.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in case_results))

    print("\nDone.")
    print(f"Total json files scanned: {total}")
    print(f"Selected files: {selected}")
    print(f"Skipped files: {skipped}")
    print(f"Errors: {errors}")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()