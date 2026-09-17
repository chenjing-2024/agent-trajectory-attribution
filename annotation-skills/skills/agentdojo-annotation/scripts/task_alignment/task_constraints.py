"""Conservative exact task constraints, supplied separately from trajectory content."""

import hashlib
import json
import re


EMAIL = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


def expected_actions(prompt, specification=None):
    if specification is not None:
        if not isinstance(specification, dict) or specification.get("prompt") != prompt:
            raise ValueError("Task specification must match the initiating prompt exactly")
        actions = specification.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Task specification requires nonempty actions")
        for action in actions:
            if not isinstance(action, dict) or not isinstance(action.get("function"), str) or not isinstance(action.get("args"), dict) or not action["args"]:
                raise ValueError("Each task action requires a function and nonempty exact args")
        for action in actions:
            requirements = action.get("requires", [])
            if not isinstance(requirements, list):
                raise ValueError("requires must be a list of reviewed prerequisite evidence")
            for requirement in requirements:
                if not isinstance(requirement, dict) or not isinstance(requirement.get("function"), str) or not isinstance(requirement.get("args"), dict):
                    raise ValueError("Each prerequisite requires function and exact args")
                if not re.fullmatch(r"[0-9a-f]{64}", str(requirement.get("result_sha256", ""))):
                    raise ValueError("Each prerequisite requires a result_sha256 binding")
                if any(not isinstance(requirement.get(k), str) or not requirement[k].strip() for k in ("evidence_quote", "review_reason")):
                    raise ValueError("Prerequisites require evidence_quote and review_reason")
        return [{**action, "matching": "exact_args"} for action in actions]

    # Full-prompt templates avoid extracting positive intent from a negated or
    # multi-clause instruction. Other tasks need an externally reviewed spec.
    explicit = re.fullmatch(
        rf'Send an email to ({EMAIL}) with subject "([^"\n]+)" and body "([^"\n]+)"\.?',
        prompt.strip(), re.IGNORECASE,
    )
    if explicit:
        recipient, subject, body = explicit.groups()
        return [{"function": "send_email", "args": {"to": recipient, "subject": subject, "body": body}, "matching": "email"}]
    saying = re.fullmatch(
        rf'Send an email to ({EMAIL}) saying (?:"([^"\n]+)"|([A-Za-z0-9 ]+))\.?',
        prompt.strip(), re.IGNORECASE,
    )
    if saying:
        recipient, quoted, plain = saying.groups()
        return [{"function": "send_email", "args": {"to": recipient, "body": quoted if quoted is not None else plain.strip()}, "matching": "email"}]
    return []


def matches_action(function, args, expected):
    if function != expected["function"] or not isinstance(args, dict):
        return False
    wanted = expected["args"]
    if expected["matching"] == "exact_args":
        # Extra recipients/attachments/other parameters must not silently pass.
        return args == wanted
    addresses = [args[key] for key in ("to", "recipient", "recipients") if key in args]
    if len(addresses) != 1:
        return False
    actual = addresses[0]
    if isinstance(actual, str):
        actual = [actual]
    if actual != [wanted["to"]]:
        return False
    for key, value in args.items():
        if key not in {"to", "recipient", "recipients", "subject", "body"} and value not in (None, "", []):
            return False
    if args.get("body") != wanted["body"]:
        return False
    if "subject" in wanted and args.get("subject") != wanted["subject"]:
        return False
    return True


def execution_status(component):
    content = component.get("content", "")
    if not isinstance(content, str) or not content.lstrip().startswith("[ACTION RESULT PAIR]") or "[RESULT]" not in content:
        return "unknown"
    result = content.split("[RESULT]", 1)[1]
    error = re.search(r"(?m)^error:[ \t]*(.*)$", result)
    if not error or error.group(1).strip() == "unknown":
        return "unknown"
    if error.group(1).strip() not in {"None", "none", "null", "NULL"}:
        return "failed"
    # Require observed result content as well as the explicit no-error marker.
    match = re.search(r"(?ms)^content:[ \t]*\n?(.*)\Z", result)
    if not match or not match.group(1).strip():
        return "unknown"
    try:
        payload = json.loads(match.group(1).strip())
    except ValueError:
        payload = None
    if payload is False:
        return "failed"
    if isinstance(payload, dict) and (
        payload.get("success") is False or payload.get("ok") is False
        or payload.get("error") not in (None, "", False)
        or payload.get("status") in ("failed", "error", "failure")
    ):
        return "failed"
    return "succeeded"


def result_content(component):
    """Observed RESULT body only; never use the agent's rationale as evidence."""
    content = component.get("content", "")
    if not isinstance(content, str) or "[RESULT]" not in content:
        return None
    result = content.split("[RESULT]", 1)[1]
    match = re.search(r"(?ms)^content:[ \t]*\n?(.*)\Z", result)
    return match.group(1).strip() if match else None


def prerequisite_evidence(candidate, expected, candidates, root_index):
    """None means unverified; a list contains the bound preceding evidence."""
    evidence = []
    for requirement in expected.get("requires", []):
        matches = []
        for previous in candidates:
            if not root_index < previous["index"] < candidate["index"]:
                continue
            if previous["function"] != requirement["function"] or previous["args"] != requirement["args"]:
                continue
            if execution_status(previous["component"]) != "succeeded":
                continue
            body = result_content(previous["component"])
            if body is None or hashlib.sha256(body.encode("utf-8")).hexdigest() != requirement["result_sha256"]:
                continue
            if requirement["evidence_quote"] not in body:
                continue
            matches.append({"component_index": previous["index"],
                            "function": previous["function"],
                            "result_sha256": requirement["result_sha256"],
                            "evidence_quote": requirement["evidence_quote"],
                            "review_reason": requirement["review_reason"]})
        if not matches:
            return None
        evidence.append(matches[-1])
    return evidence
