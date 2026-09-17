import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepared_input import is_prepared, validate_prepared


def extract_text_content(content):
    """
    AgentDojo message['content'] often looks like:
    [
        {
            "type": "text",
            "content": "..."
        }
    ]

    This function converts it into a plain string.
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
    Some trajectories contain function names like:
    function=read_channel_messages

    We keep the raw information in content, but normalize metadata slightly.
    """
    if function_name is None:
        return None

    function_name = str(function_name)

    if function_name.startswith("function="):
        return function_name.replace("function=", "", 1)

    return function_name


def get_tool_call_from_message(msg):
    """
    Assistant messages may use 'tool_calls'.
    Tool result messages may use 'tool_call'.

    Return:
    - raw_function
    - clean_function
    - args
    """
    raw_function = None
    args = None

    # For tool result step
    if isinstance(msg.get("tool_call"), dict):
        tool_call = msg["tool_call"]
        raw_function = tool_call.get("function")
        args = tool_call.get("args")

    # For assistant tool-call step
    elif isinstance(msg.get("tool_calls"), list) and len(msg["tool_calls"]) > 0:
        tool_call = msg["tool_calls"][0]
        raw_function = tool_call.get("function")
        args = tool_call.get("args")

    clean_function = clean_function_name(raw_function)

    return raw_function, clean_function, args


def normalize_message_scheme_a(msg):
    """
    Scheme A:
    Return only {"role": ..., "content": ...}
    but write tool function/args/error back into content.
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



def is_assistant_tool_call(msg):
    """
    Return True when an assistant message contains a structured tool call.
    """
    if not isinstance(msg, dict):
        return False

    if msg.get("role") != "assistant":
        return False

    if isinstance(msg.get("tool_call"), dict):
        return True

    tool_calls = msg.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def is_tool_result(msg):
    """
    Return True when the message is a tool-result message.
    """
    return isinstance(msg, dict) and msg.get("role") == "tool"


def normalize_action_result_pair(action_msg, result_msg):
    """
    Merge an assistant tool call and its immediately following tool result
    into one action-result component.
    """
    action_raw_content = extract_text_content(action_msg.get("content"))
    result_raw_content = extract_text_content(result_msg.get("content"))

    (
        action_raw_function,
        action_clean_function,
        action_args,
    ) = get_tool_call_from_message(action_msg)

    (
        result_raw_function,
        result_clean_function,
        result_args,
    ) = get_tool_call_from_message(result_msg)

    result_error = result_msg.get("error", "unknown")

    action_args_text = (
        json.dumps(action_args, ensure_ascii=False)
        if action_args is not None
        else "null"
    )
    result_args_text = (
        json.dumps(result_args, ensure_ascii=False)
        if result_args is not None
        else "null"
    )

    normalized_content = (
        "[ACTION RESULT PAIR]\n"
        "\n"
        "[ACTION]\n"
        f"function_raw: {action_raw_function}\n"
        f"function: {action_clean_function}\n"
        f"args: {action_args_text}\n"
        "raw_content:\n"
        f"{action_raw_content}\n"
        "\n"
        "[RESULT]\n"
        f"function_raw: {result_raw_function}\n"
        f"function: {result_clean_function}\n"
        f"args: {result_args_text}\n"
        f"error: {result_error}\n"
        "content:\n"
        f"{result_raw_content}"
    )

    return {
        "role": "assistant",
        "content": normalized_content,
    }


def build_action_result_pair_metadata(
    action_msg,
    result_msg,
    component_id,
    action_step_id,
    result_step_id,
):
    """
    Build metadata for a merged action-result pair.
    """
    action_raw_content = extract_text_content(action_msg.get("content"))
    result_raw_content = extract_text_content(result_msg.get("content"))

    (
        action_raw_function,
        action_clean_function,
        action_args,
    ) = get_tool_call_from_message(action_msg)

    (
        result_raw_function,
        result_clean_function,
        result_args,
    ) = get_tool_call_from_message(result_msg)

    return {
        "component_id": component_id,

        # Keep compatibility with code expecting a single original_step_id.
        "original_step_id": action_step_id,

        # Full provenance of the merged component.
        "original_step_ids": [action_step_id, result_step_id],

        "role": "assistant",
        "component_type": "action_result_pair",

        "function_raw": action_raw_function,
        "function": action_clean_function,
        "args": action_args,

        "result_function_raw": result_raw_function,
        "result_function": result_clean_function,
        "result_args": result_args,
        "error": result_msg.get("error"),

        "action_content_preview": (
            action_raw_content[:300].replace("\n", " ")
        ),
        "result_content_preview": (
            result_raw_content[:300].replace("\n", " ")
        ),
    }

def build_component_metadata(msg, component_id):
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


def normalize_file(path, src_root):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只保留顶层明确为 security == false 的轨迹
    if data.get("security") is not False:
        return None

    if is_prepared(data):
        validate_prepared(data, "task_alignment")
        print(f"[REUSE] {path}：已校验，跳过标准化和组件拆分，保留原组件 ID。")
        return data

    messages = data.get("messages")
    if not isinstance(messages, list):
        return None

    trajectory = []
    metadata = []

    message_index = 0
    component_id = 1

    while message_index < len(messages):
        msg = messages[message_index]

        if not isinstance(msg, dict):
            msg = {
                "role": "user",
                "content": str(msg)
            }

        next_msg = None
        if message_index + 1 < len(messages):
            next_msg = messages[message_index + 1]

            if not isinstance(next_msg, dict):
                next_msg = {
                    "role": "user",
                    "content": str(next_msg)
                }

        # Merge only when an assistant tool call is immediately followed
        # by a tool-result message.
        if (
            is_assistant_tool_call(msg)
            and next_msg is not None
            and is_tool_result(next_msg)
        ):
            action_step_id = message_index + 1
            result_step_id = message_index + 2

            trajectory.append(
                normalize_action_result_pair(msg, next_msg)
            )
            metadata.append(
                build_action_result_pair_metadata(
                    action_msg=msg,
                    result_msg=next_msg,
                    component_id=component_id,
                    action_step_id=action_step_id,
                    result_step_id=result_step_id,
                )
            )

            message_index += 2
            component_id += 1
            continue

        # Messages without an immediately adjacent action-result pair
        # remain standalone components.
        trajectory.append(normalize_message_scheme_a(msg))
        metadata.append(
            build_component_metadata(
                msg=msg,
                component_id=component_id,
            )
        )

        message_index += 1
        component_id += 1

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
        "num_components": len(trajectory),

        # 这个 trajectory 可以直接喂给你现有 attribution 代码
        "trajectory": trajectory,

        # 这个用于人工检查、DeepSeek 标注、summary table
        "components_metadata": metadata
    }

    return out_data


def main():
    parser = argparse.ArgumentParser(
        description=("Select security=false AgentDojo trajectories, normalize them, ""and merge adjacent tool-call/tool-result messages into pairs.")
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
        help="Output directory for normalized security=false trajectories."
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    selected = 0
    skipped_no_messages = 0
    errors = 0
    case_results = []

    for path in src_root.rglob("*.json"):
        total += 1

        try:
            normalized = normalize_file(path, src_root)

            if normalized is None:
                # Either security is not true, or messages missing
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
    print(f"Selected security=false files: {selected}")
    print(f"Errors: {errors}")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()