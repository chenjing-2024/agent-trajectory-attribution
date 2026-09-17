import json
import re
import argparse
from pathlib import Path
from collections import Counter

from task_constraints import expected_actions, matches_action, execution_status, prerequisite_evidence


ACTION_RESULT_MARKER = "[ACTION RESULT PAIR]"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_function_name(content):
    """
    Extract a function name from an action-like assistant component.

    Supported formats:
    1. [ACTION RESULT PAIR]
       [ACTION]
       function: send_money

    2. [ASSISTANT TOOL CALL]
       function: send_money

    3. Raw textual tool call:
       <function=send_money>{...}</function>
    """
    if not isinstance(content, str):
        return None

    # For merged pairs, inspect only the ACTION section.
    action_section = content
    if "[RESULT]" in action_section:
        action_section = action_section.split("[RESULT]", 1)[0]

    # Prefer the normalized "function:" metadata line.
    match = re.search(
        r"(?m)^function:\s*(.+?)\s*$",
        action_section,
    )

    if match:
        function_name = match.group(1).strip()

        if function_name not in {"", "None", "null"}:
            return function_name

    # Fall back to raw AgentDojo textual tool-call syntax.
    match = re.search(
        r"<function=([^>\s]+)>",
        action_section,
    )

    if not match:
        return None

    function_name = match.group(1).strip()

    # Handle malformed names such as function=search_calendar_events.
    if function_name.startswith("function="):
        function_name = function_name.replace("function=", "", 1)

    if function_name in {"", "None", "null"}:
        return None

    return function_name

def extract_action_args(content):
    """
    Extract the ACTION args from an action-result pair.

    Returns:
        dict/list/value when valid JSON;
        raw string when parsing fails;
        None when unavailable.
    """
    if not isinstance(content, str):
        return None

    action_section = content

    if "[RESULT]" in content:
        action_section = content.split("[RESULT]", 1)[0]

    match = re.search(
        r"(?m)^args:\s*(.+?)\s*$",
        action_section,
    )

    if not match:
        return None

    args_text = match.group(1).strip()

    if args_text in {"", "None", "null"}:
        return None

    try:
        return json.loads(args_text)
    except json.JSONDecodeError:
        return args_text


def is_action_result_pair(component):
    """
    Detect a merged action-result pair.

    Primary detection:
        content starts with [ACTION RESULT PAIR]

    The role remains assistant after normalization.
    """
    if not isinstance(component, dict):
        return False

    content = component.get("content")

    return (
        component.get("role") == "assistant"
        and isinstance(content, str)
        and content.lstrip().startswith(ACTION_RESULT_MARKER)
    )



def get_action_component_type(component):
    """
    Return the action component type, or None if the component does not
    contain a recognizable tool action.

    Recognized forms:
    - action_result_pair
    - assistant_tool_call
    - textual_tool_call
    """
    if not isinstance(component, dict):
        return None

    if component.get("role") != "assistant":
        return None

    content = component.get("content")
    if not isinstance(content, str):
        return None

    stripped = content.lstrip()

    if stripped.startswith("[ACTION RESULT PAIR]"):
        return "action_result_pair"

    if stripped.startswith("[ASSISTANT TOOL CALL]"):
        return "assistant_tool_call"

    if re.search(r"<function=[^>\s]+>", content):
        return "textual_tool_call"

    return None

def make_component_id(index):
    """
    trajectory index is zero-based;
    component IDs are C1, C2, ...
    """
    return f"C{index + 1}"


def find_root_user_prompt(trajectory, target_index):
    """
    Return the first user component before the target action.

    This implements prompt-level root-trigger attribution.
    """
    for index, component in enumerate(trajectory[:target_index]):
        if (
            isinstance(component, dict)
            and component.get("role") == "user"
        ):
            return index, component

    return None, None



TASK_FUNCTION_PRIORITIES = {
    "banking": {
        "payment": {
            "core": {
                "send_money",
                "make_payment",
                "create_transaction",
                "pay_bill",
                "schedule_transaction",
                "schedule_payment",
                "update_scheduled_transaction",
            },
            "support": {
                "read_file",
                "get_iban",
                "get_most_recent_transactions",
                "search_transactions",
                "get_transaction",
                "get_scheduled_transactions",
            },
        },
        "transaction_lookup": {
            "core": {
                "get_most_recent_transactions",
                "search_transactions",
                "get_transaction",
                "get_scheduled_transactions",
            },
            "support": {
                "get_iban",
                "read_file",
            },
        },
        "account_lookup": {
            "core": {
                "get_iban",
                "get_balance",
                "get_account_balance",
            },
            "support": {
                "get_most_recent_transactions",
                "search_transactions",
            },
        },
    },

    "workspace": {
        "calendar_create": {
            "core": {
                "create_calendar_event",
                "add_calendar_event",
                "schedule_calendar_event",
            },
            "support": {
                "search_calendar_events",
                "get_day_calendar_events",
                "get_calendar_event",
                "get_current_date",
                "search_contacts",
                "get_contacts",
            },
        },
        "calendar_update": {
            "core": {
                "update_calendar_event",
                "modify_calendar_event",
                "delete_calendar_event",
            },
            "support": {
                "search_calendar_events",
                "get_day_calendar_events",
                "get_calendar_event",
            },
        },
        "calendar_lookup": {
            "core": {
                "search_calendar_events",
                "get_day_calendar_events",
                "get_calendar_event",
            },
            "support": {
                "get_current_date",
            },
        },
        "email_send": {
            "core": {
                "send_email",
                "reply_email",
                "forward_email",
            },
            "support": {
                "search_emails",
                "get_email",
                "get_received_emails",
                "get_unread_emails",
                "get_contacts",
                "search_contacts",
            },
        },
        "email_lookup": {
            "core": {
                "search_emails",
                "get_email",
                "get_received_emails",
                "get_unread_emails",
            },
            "support": {
                "get_contacts",
                "search_contacts",
            },
        },
        "file_create": {
            "core": {
                "create_file",
            },
            "support": {
                "search_files",
                "search_files_by_filename",
                "list_files",
                "read_file",
            },
        },
        "file_update": {
            "core": {
                "append_to_file",
                "update_file",
                "edit_file",
                "write_file",
            },
            "support": {
                "read_file",
                "search_files",
                "search_files_by_filename",
                "list_files",
            },
        },
        "file_share": {
            "core": {
                "share_file",
            },
            "support": {
                "search_files",
                "search_files_by_filename",
                "list_files",
                "read_file",
                "get_contacts",
                "search_contacts",
            },
        },
        "file_lookup": {
            "core": {
                "read_file",
                "search_files",
                "search_files_by_filename",
                "list_files",
            },
            "support": set(),
        },
        "file_delete": {
            "core": {
                "delete_file",
            },
            "support": {
                "search_files",
                "search_files_by_filename",
                "list_files",
            },
        },
    },

    "slack": {
        "message_send": {
            "core": {
                "send_message",
                "send_channel_message",
                "send_direct_message",
                "post_message",
            },
            "support": {
                "search_messages",
                "get_channel_history",
                "get_user_profile",
                "get_users",
                "get_channels",
            },
        },
        "message_lookup": {
            "core": {
                "search_messages",
                "get_channel_history",
                "read_message",
            },
            "support": {
                "get_user_profile",
                "get_users",
                "get_channels",
            },
        },
        "channel_membership_update": {
            "core": {
                "add_user_to_channel",
                "invite_user_to_slack",
            },
            "support": {
                "get_user_profile",
                "get_users",
                "get_channels",
            },
        },
    },

    "travel": {
        "flight_lookup": {
            "core": {
                "search_flights",
                "get_flight_price",
                "get_rating_reviews_for_airlines",
            },
            "support": {
                "get_user_profile",
            },
        },
        "hotel_lookup": {
            "core": {
                "search_hotels",
                "get_hotels_address",
                "get_hotels_price",
                "get_rating_reviews_for_hotels",
            },
            "support": {
                "get_user_profile",
            },
        },
        "restaurant_lookup": {
            "core": {
                "search_restaurants",
                "get_restaurants_address",
                "get_price_for_restaurants",
                "get_rating_reviews_for_restaurants",
            },
            "support": {
                "get_user_profile",
            },
        },
        "car_rental_lookup": {
            "core": {
                "search_cars",
                "get_car_price_per_day",
                "get_rating_reviews_for_car_rental",
                "get_car_types",
                "get_car_rental_companies",
            },
            "support": {
                "get_user_profile",
            },
        },
        "booking": {
            "core": {
                "book_flight",
                "book_hotel",
                "reserve_hotel",
                "book_car",
                "purchase_ticket",
            },
            "support": {
                "search_flights",
                "search_hotels",
                "search_cars",
                "get_user_profile",
            },
        },
        "booking_lookup": {
            "core": {
                "get_booking",
            },
            "support": {
                "get_user_profile",
            },
        },
        "calendar_create": {
            "core": {
                "create_calendar_event",
                "add_calendar_event",
                "schedule_calendar_event",
            },
            "support": set(),
        },
        "web_lookup": {
            "core": {
                "get_webpage",
            },
            "support": set(),
        },
        "web_publish": {
            "core": {
                "post_webpage",
            },
            "support": {
                "get_webpage",
            },
        },
    },
}


def normalize_suite_name(value):
    if not isinstance(value, str):
        return ""

    value = value.strip().lower()

    aliases = {
        "workspace": "workspace",
        "banking": "banking",
        "slack": "slack",
        "travel": "travel",
    }

    return aliases.get(value, value)


def infer_prompt_intents(prompt, suite_name):
    """
    Infer all explicit task intents from the initiating user prompt.

    Detection is deterministic. Intents are returned according to the
    earliest matching phrase in the prompt. Specific intents are preferred
    over broad domain-level categories.
    """
    if not isinstance(prompt, str):
        prompt = str(prompt or "")

    text = prompt.lower()
    suite_name = normalize_suite_name(suite_name)

    if suite_name == "banking":
        intent_patterns = [
            (
                "payment",
                (
                    "schedule a transaction",
                    "scheduled transaction",
                    "recurring transaction",
                    "send money",
                    "send a transaction",
                    "transfer money",
                    "make a payment",
                    "pay the bill",
                    "pay ",
                    "payment",
                    "bill",
                    "wire ",
                    "remit",
                    "refund",
                    "reimburse",
                    "pay back",
                    "send back",
                    "return the money",
                ),
            ),
            (
                "transaction_lookup",
                (
                    "transaction history",
                    "recent transactions",
                    "most recent transactions",
                    "search transactions",
                    "find the transaction",
                    "find a transaction",
                    "what did i spend",
                    "how much did i spend",
                    "total spending",
                    "usually pay",
                    "previous payment",
                    "past payment",
                    "what they've sent me",
                    "what they sent me",
                    "received from",
                    "transactions in",
                ),
            ),
            (
                "account_lookup",
                (
                    "what is my iban",
                    "get my iban",
                    "account balance",
                    "what is my balance",
                    "check my balance",
                    "account number",
                ),
            ),
        ]

    elif suite_name == "workspace":
        intent_patterns = [
            (
                "calendar_create",
                (
                    "create an event",
                    "create event",
                    "add an event",
                    "add a reminder to my calendar",
                    "add a reminder",
                    "add to my calendar",
                    "schedule a meeting",
                    "schedule an event",
                    "put on my calendar",
                    "calendar reminder",
                ),
            ),
            (
                "calendar_update",
                (
                    "update the event",
                    "change the event",
                    "reschedule",
                    "move the meeting",
                    "delete the event",
                    "cancel the event",
                    "modify the event",
                ),
            ),
            (
                "calendar_lookup",
                (
                    "am i free",
                    "check my availability",
                    "how many appointments",
                    "appointments do i have",
                    "calendar events",
                    "search my calendar",
                    "where is the",
                    "when is the",
                    "what time is the",
                    "meeting on",
                    "event on",
                ),
            ),
            (
                "email_send",
                (
                    "send an email",
                    "send email",
                    "reply to the email",
                    "reply to",
                    "forward the email",
                    "forward this email",
                    "email them",
                    "mail them",
                ),
            ),
            (
                "email_lookup",
                (
                    "find the email",
                    "search email",
                    "search my email",
                    "look for the email",
                    "read the email",
                    "received recently",
                    "email i received",
                    "email received",
                    "in an email",
                    "password reset email",
                    "reset link",
                    "get me the link",
                    "unread emails",
                    "received emails",
                ),
            ),
            (
                "file_create",
                (
                    "create a file",
                    "create the file",
                    "make a file",
                    "new file",
                ),
            ),
            (
                "file_update",
                (
                    "add them to the file",
                    "add to the file",
                    "append to the file",
                    "append them",
                    "update the file",
                    "edit the file",
                    "modify the file",
                    "write to the file",
                    "put them in the file",
                    "save them to the file",
                    "add this to the file",
                ),
            ),
            (
                "file_share",
                (
                    "share the file",
                    "share this file",
                    "give access to the file",
                    "grant access",
                ),
            ),
            (
                "file_delete",
                (
                    "delete the file",
                    "remove the file",
                ),
            ),
            (
                "file_lookup",
                (
                    "take a look at the file",
                    "find the file",
                    "search files",
                    "read the file",
                    "open the file",
                    "which file",
                    "look at the document",
                    "find the document",
                    "read the document",
                ),
            ),
        ]

    elif suite_name == "slack":
        intent_patterns = [
            (
                "channel_membership_update",
                (
                    "add user to channel",
                    "add them to the channel",
                    "invite user",
                    "invite them to slack",
                    "invite to slack",
                ),
            ),
            (
                "message_send",
                (
                    "send a message",
                    "send message",
                    "message ",
                    "tell ",
                    "notify",
                    "post in",
                    "post to",
                    "write to",
                ),
            ),
            (
                "message_lookup",
                (
                    "search messages",
                    "find the message",
                    "what did",
                    "who said",
                    "channel history",
                    "read the message",
                ),
            ),
        ]

    elif suite_name == "travel":
        intent_patterns = [
            (
                "calendar_create",
                (
                    "add an event to my calendar",
                    "add a reminder to my calendar",
                    "add a calendar event",
                    "create a calendar event",
                    "calendar reminder",
                    "event title should be",
                    "remind me to book",
                ),
            ),
            (
                "car_rental_lookup",
                (
                    "car rental",
                    "rental company",
                    "rent a car",
                    "rent two cars",
                    "electric car",
                    "electric cars",
                    "suv",
                    "suvs",
                    "car price",
                    "price per day",
                    "top-rated car rental",
                    "car rental company",
                ),
            ),
            (
                "restaurant_lookup",
                (
                    "restaurant",
                    "restaurants",
                    "cuisine",
                    "vegan options",
                    "restaurant name",
                    "restaurant's name",
                    "restaurant address",
                    "restaurant's address",
                    "best-rated restaurant",
                    "highest-rated restaurant",
                ),
            ),
            (
                "hotel_lookup",
                (
                    "hotel",
                    "hotels",
                    "hotel address",
                    "hotel price",
                    "best-rated hotel",
                    "highest-rated hotel",
                ),
            ),
            (
                "flight_lookup",
                (
                    "find a flight",
                    "search flights",
                    "flight options",
                    "flight price",
                    "flights to",
                    "flights from",
                ),
            ),
            (
                "booking_lookup",
                (
                    "my booking",
                    "existing booking",
                    "booking details",
                    "check my booking",
                ),
            ),
            (
                "booking",
                (
                    "book a flight",
                    "book the flight",
                    "purchase a ticket",
                    "buy a ticket",
                    "book a hotel",
                    "book the hotel",
                    "reserve a hotel",
                    "reserve the hotel",
                    "book a car",
                    "rent the car now",
                ),
            ),
            (
                "web_publish",
                (
                    "publish the webpage",
                    "post the webpage",
                    "create a webpage",
                    "publish online",
                ),
            ),
            (
                "web_lookup",
                (
                    "open the webpage",
                    "read the webpage",
                    "get the webpage",
                    "look at the website",
                ),
            ),
        ]

    else:
        intent_patterns = []

    matches = []

    for intent_order, (intent, terms) in enumerate(intent_patterns):
        positions = [
            text.find(term)
            for term in terms
            if term in text
        ]

        if positions:
            matches.append(
                (
                    min(positions),
                    intent_order,
                    intent,
                )
            )

    detected = []

    for _, _, intent in sorted(matches):
        if intent not in detected:
            detected.append(intent)

    return detected


def infer_prompt_intent(prompt, suite_name):
    """
    Backward-compatible single-intent wrapper.
    """
    intents = infer_prompt_intents(
        prompt=prompt,
        suite_name=suite_name,
    )

    return intents[0] if intents else None


def extract_action_error(content):
    """
    Return the normalized tool error from the RESULT section.

    None means either:
    - no RESULT section exists, or
    - the result explicitly reports no error.
    """
    if not isinstance(content, str):
        return None

    if "[RESULT]" not in content:
        return None

    result_section = content.split("[RESULT]", 1)[1]

    match = re.search(
        r"(?m)^error:\s*(.*?)\s*$",
        result_section,
    )

    if not match:
        return None

    value = match.group(1).strip()

    if value in {"", "None", "none", "null", "NULL"}:
        return None

    return value


def collect_action_candidates(
    trajectory,
    allowed_functions=None,
    excluded_functions=None,
):
    """
    Collect all recognizable action-like components in trajectory order.
    """
    candidates = []

    for index, component in enumerate(trajectory):
        component_type = get_action_component_type(component)

        if component_type is None:
            continue

        content = component.get("content", "")
        function_name = extract_function_name(content)

        if function_name is None:
            continue

        if (
            allowed_functions is not None
            and function_name not in allowed_functions
        ):
            continue

        if (
            excluded_functions is not None
            and function_name in excluded_functions
        ):
            continue

        candidates.append(
            {
                "index": index,
                "component": component,
                "component_type": component_type,
                "function": function_name,
                "args": extract_action_args(content),
                "error": extract_action_error(content),
            }
        )

    return candidates


def find_initial_user_prompt(trajectory):
    """
    Return the initiating user component.

    AgentDojo normalized trajectories in this dataset contain one root
    user task near the beginning of the trajectory.
    """
    for index, component in enumerate(trajectory):
        if (
            isinstance(component, dict)
            and component.get("role") == "user"
        ):
            return index, component

    return None, None


def find_critical_action(
    trajectory,
    suite_name,
    root_prompt,
    allowed_functions=None,
    excluded_functions=None,
    task_spec=None,
):
    """
    Select the latest successfully executed core action that directly serves
    an explicit task requested in the initiating user prompt.

    Mandatory conditions:
    1. explicit task constraints must be available;
    2. the action function and arguments must match these constraints;
    3. the action must report successful execution;
    4. the action must occur after the initiating user prompt.

    Recency is used only after relevance, task criticality, and execution
    success have been established.

    There is intentionally no fallback to:
    - support actions;
    - failed actions;
    - the latest arbitrary action.

    If no successful requested core action is observed, the case is skipped.
    """
    candidates = collect_action_candidates(
        trajectory=trajectory,
        allowed_functions=allowed_functions,
        excluded_functions=excluded_functions,
    )

    normalized_suite = normalize_suite_name(
        suite_name
    )

    detected_intents = infer_prompt_intents(
        prompt=root_prompt,
        suite_name=normalized_suite,
    )

    strategy = (
        "prompt_conditioned_strict_"
        "latest_requested_core_action"
    )

    if not candidates:
        return None, None, None, {
            "strategy": strategy,
            "detected_intents": detected_intents,
            "intent": (
                detected_intents[0]
                if len(detected_intents) == 1
                else None
            ),
            "selection_type": "no_action",
            "candidate_count": 0,
            "core_functions": [],
            "support_functions": [],
            "matched_core_actions": [],
            "matched_successful_core_actions": [],
            "matched_support_actions": [],
        }

    expected = expected_actions(root_prompt, task_spec)
    if not expected:
        return None, None, None, {
            "strategy": strategy,
            "detected_intents": [],
            "intent": None,
            "selection_type": "unverified_task_constraints",
            "candidate_count": len(candidates),
            "core_functions": [],
            "support_functions": [],
            "matched_core_actions": [],
            "matched_successful_core_actions": [],
            "matched_support_actions": [],
        }

    suite_taxonomy = TASK_FUNCTION_PRIORITIES.get(
        normalized_suite,
        {},
    )

    function_to_core_intents = {}
    function_to_support_intents = {}

    all_core_functions = set()
    all_support_functions = set()

    intent_taxonomy = {}

    for intent in detected_intents:
        config = suite_taxonomy.get(intent, {})

        core_functions = set(
            config.get("core", set())
        )
        support_functions = set(
            config.get("support", set())
        )

        intent_taxonomy[intent] = {
            "core": sorted(core_functions),
            "support": sorted(support_functions),
        }

        all_core_functions.update(
            core_functions
        )
        all_support_functions.update(
            support_functions
        )

        for function_name in core_functions:
            function_to_core_intents.setdefault(
                function_name,
                [],
            ).append(intent)

        for function_name in support_functions:
            function_to_support_intents.setdefault(
                function_name,
                [],
            ).append(intent)

    matched_core = []
    matched_support = []

    for candidate in candidates:
        function_name = candidate["function"]

        if any(matches_action(function_name, candidate["args"], action) for action in expected):
            matched_core.append(
                {
                    **candidate,
                    "matched_intents": list(
                        function_to_core_intents.get(function_name, ["explicit_task_constraints"])
                    ),
                }
            )

        if function_name in function_to_support_intents:
            matched_support.append(
                {
                    **candidate,
                    "matched_intents": list(
                        function_to_support_intents[
                            function_name
                        ]
                    ),
                }
            )

    successful_core = []
    unverified_preconditions = False
    root_index, _ = find_initial_user_prompt(trajectory)
    for candidate in matched_core:
        if execution_status(candidate["component"]) != "succeeded" or root_index is None or candidate["index"] <= root_index:
            continue
        for action in expected:
            if not matches_action(candidate["function"], candidate["args"], action):
                continue
            evidence = prerequisite_evidence(candidate, action, candidates, root_index)
            if evidence is None:
                unverified_preconditions = True
                continue
            candidate["prerequisite_evidence"] = evidence
            successful_core.append(candidate)
            break

    def serialize_match(candidate):
        return {
            "component_id": make_component_id(
                candidate["index"]
            ),
            "component_index": candidate["index"],
            "prerequisite_evidence": candidate.get("prerequisite_evidence", []),
            "function": candidate["function"],
            "args": candidate["args"],
            "error": candidate["error"],
            "matched_intents": candidate.get(
                "matched_intents",
                [],
            ),
        }

    base_metadata = {
        "strategy": strategy,
        "detected_intents": detected_intents,
        "intent": (
            detected_intents[0]
            if len(detected_intents) == 1
            else None
        ),
        "candidate_count": len(candidates),
        "task_constraint_source": "external_specification" if task_spec is not None else "exact_prompt_template",
        "verified_task_actions": expected,
        "execution_status_counts": dict(Counter(execution_status(c["component"]) for c in candidates)),
        "core_functions": sorted(
            all_core_functions
        ),
        "support_functions": sorted(
            all_support_functions
        ),
        "intent_taxonomy": intent_taxonomy,
        "matched_core_actions": [
            serialize_match(candidate)
            for candidate in matched_core
        ],
        "matched_successful_core_actions": [
            serialize_match(candidate)
            for candidate in successful_core
        ],
        "matched_support_actions": [
            serialize_match(candidate)
            for candidate in matched_support
        ],
    }

    if not matched_core:
        return None, None, None, {
            **base_metadata,
            "selection_type": (
                "no_requested_core_action_match"
            ),
            "selected_matched_intents": [],
            "selected_error": None,
        }

    if not successful_core:
        return None, None, None, {
            **base_metadata,
            "selection_type": (
                "unverified_task_preconditions" if unverified_preconditions else "no_successful_requested_core_action"
            ),
            "selected_matched_intents": [],
            "selected_error": None,
        }

    selected = max(
        successful_core,
        key=lambda candidate: candidate["index"],
    )

    selection_metadata = {
        **base_metadata,
        "selection_type": (
            "latest_successful_requested_core_action"
        ),
        "selected_matched_intents": (
            selected.get("matched_intents", [])
        ),
        "selected_error": selected["error"],
    }

    return (
        selected["index"],
        selected["component"],
        selected["function"],
        selection_metadata,
    )


def find_last_action_result_pair(
    trajectory,
    allowed_functions=None,
    excluded_functions=None,
):
    """
    Select the latest recognizable action-like assistant component.

    Search is performed from the end of the trajectory toward the
    beginning, so the selected target is always the action nearest to
    the end of the trajectory.

    Despite the legacy function name, this supports:
    - merged action-result pairs;
    - standalone assistant tool calls;
    - raw textual <function=...> calls.
    """
    for index in range(len(trajectory) - 1, -1, -1):
        component = trajectory[index]
        component_type = get_action_component_type(component)

        if component_type is None:
            continue

        function_name = extract_function_name(
            component.get("content", "")
        )

        # An action without an identifiable function is not usable as
        # a deterministic target.
        if function_name is None:
            continue

        if allowed_functions is not None:
            if function_name not in allowed_functions:
                continue

        if excluded_functions is not None:
            if function_name in excluded_functions:
                continue

        return index, component, function_name

    return None, None, None

def annotate_trajectory(
    data,
    source_path,
    allowed_functions=None,
    excluded_functions=None,
    task_spec=None,
):
    trajectory = data.get("trajectory")

    if not isinstance(trajectory, list):
        return None, "missing_trajectory"

    root_index, root_component = find_initial_user_prompt(
        trajectory
    )

    if root_component is None:
        return None, "no_user_prompt"

    root_content = root_component.get("content", "")

    (
        target_index,
        target_component,
        target_function,
        target_selection,
    ) = find_critical_action(
        trajectory=trajectory,
        suite_name=data.get("suite_name"),
        root_prompt=root_content,
        allowed_functions=allowed_functions,
        excluded_functions=excluded_functions,
        task_spec=task_spec,
    )

    if target_component is None:
        return (
            None,
            target_selection.get(
                "selection_type",
                "no_target_action",
            ),
        )

    if target_index <= root_index:
        return None, "target_not_after_user_prompt"

    target_content = target_component.get("content", "")

    annotation = {
        "data_id": data.get("data_id"),
        "source_path": str(source_path),

        "suite_name": data.get("suite_name"),
        "pipeline_name": data.get("pipeline_name"),
        "user_task_id": data.get("user_task_id"),
        "injection_task_id": data.get("injection_task_id"),
        "attack_type": data.get("attack_type"),
        "utility": data.get("utility"),
        "security": data.get("security"),

        "num_components": len(trajectory),

        "target": {
            "component_id": make_component_id(target_index),
            "component_index": target_index,
            "role": target_component.get("role"),
            "component_type": (
                get_action_component_type(target_component)
                or "unknown_action"
            ),
            "action": target_function,
            "args": extract_action_args(target_content),
            "content_preview": target_content[:500],
        },

        "target_selection": target_selection,

        "attribution": {
            "component_id": make_component_id(root_index),
            "component_index": root_index,
            "role": root_component.get("role"),
            "content": root_content,
            "content_preview": root_content[:500],
        },

        "attribution_type": "prompt_level_root_trigger",
        "annotation_method": "deterministic_rule",
        "rule_version": "prompt_root_trigger_verified_constraints_v7",

        "rule_description": (
            "Select the latest observed no-error action matching explicit task constraints. "
            "Constraints come from a supported full-prompt template or an externally reviewed "
            "exact-argument specification. Unknown task constraints or execution outcomes are skipped. "
            "Attribution is fixed to the initiating user prompt by convention."
        ),
    }

    return annotation, None


def read_function_set(path):
    """
    Read one function name per line.

    Blank lines and lines beginning with # are ignored.
    """
    if path is None:
        return None

    path = Path(path)

    functions = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()

            if not value or value.startswith("#"):
                continue

            functions.add(value)

    return functions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic prompt-level root-trigger annotations "
            "with strict prompt-conditioned core-action targets for "
            "normalized AgentDojo trajectories."
        )
    )

    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help=(
            "Input root containing normalized trajectory JSON files."
        ),
    )

    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help=(
            "Output root for deterministic annotation JSON files."
        ),
    )

    parser.add_argument(
        "--allowed_functions_file",
        type=str,
        default=None,
        help=(
            "Optional text file containing allowed target functions, "
            "one function name per line."
        ),
    )

    parser.add_argument(
        "--excluded_functions_file",
        type=str,
        default=None,
        help=(
            "Optional text file containing excluded target functions, "
            "one function name per line."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory first.",
    )

    parser.add_argument(
        "--write_skipped",
        action="store_true",
        help=(
            "Write skipped trajectory records under "
            "_skipped_annotations.jsonl."
        ),
    )

    parser.add_argument("--task_constraints", help="JSON mapping relative case paths to reviewed prompt/actions specifications.")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(
            f"Source directory not found: {src_root}"
        )

    if not src_root.is_dir():
        raise NotADirectoryError(
            f"Source path is not a directory: {src_root}"
        )

    if src_root == out_root:
        raise ValueError(
            "--src_root and --out_root must be different"
        )

    if args.overwrite and out_root.exists():
        import shutil

        print(f"[REMOVE OLD OUTPUT] {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    allowed_functions = read_function_set(
        args.allowed_functions_file
    )
    excluded_functions = read_function_set(
        args.excluded_functions_file
    )

    specifications = load_json(Path(args.task_constraints)) if args.task_constraints else {}
    if not isinstance(specifications, dict):
        raise ValueError("task_constraints must be a JSON object keyed by relative case path")

    json_files = sorted(src_root.rglob("*.json"))

    total = 0
    annotated = 0
    skipped = 0
    errors = 0

    skip_reasons = Counter()
    target_actions = Counter()
    target_selection_types = Counter()
    inferred_intents = Counter()
    suite_counts = Counter()
    skipped_records = []
    failed_records = []

    for path in json_files:
        total += 1

        try:
            data = load_json(path)

            annotation, skip_reason = annotate_trajectory(
                data=data,
                source_path=path,
                allowed_functions=allowed_functions,
                excluded_functions=excluded_functions,
                task_spec=specifications.get(path.relative_to(src_root).as_posix()),
            )

            if annotation is None:
                skipped += 1
                skip_reasons[skip_reason] += 1

                print(
                    f"[SKIP] reason={skip_reason:<28} "
                    f"{path.relative_to(src_root)}"
                )

                if args.write_skipped:
                    skipped_records.append(
                        {
                            "source_path": str(path),
                            "data_id": data.get("data_id"),
                            "reason": skip_reason,
                        }
                    )

                continue

            rel_path = path.relative_to(src_root)
            out_path = out_root / rel_path
            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with out_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    annotation,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            annotated += 1

            target_action = (
                annotation["target"].get("action")
                or "<unknown>"
            )
            target_actions[target_action] += 1

            selection_info = annotation.get(
                "target_selection", {}
            )
            selection_type = selection_info.get(
                "selection_type"
            ) or "<unknown>"
            detected_intents = selection_info.get(
                "detected_intents", []
            )
            target_selection_types[selection_type] += 1
            if detected_intents:
                for intent in detected_intents:
                    inferred_intents[intent] += 1
            else:
                inferred_intents["<unknown>"] += 1

            suite_name = (
                annotation.get("suite_name")
                or rel_path.parts[0]
                if rel_path.parts
                else "<unknown>"
            )
            suite_counts[suite_name] += 1

            print(
                f"[OK] "
                f"target={annotation['target']['component_id']:<5} "
                f"action={target_action:<30} "
                f"type={annotation['target_selection']['selection_type']:<42} "
                f"intents={str(annotation['target_selection'].get('detected_intents', [])):<35} "
                f"attribution="
                f"{annotation['attribution']['component_id']:<5} "
                f"{rel_path}"
            )

        except Exception as e:
            failed_records.append({"source_path": str(path), "reason": str(e)})
            errors += 1
            print(f"[ERROR] {path}: {e}")

    if args.write_skipped and skipped_records:
        skipped_path = (
            out_root / "_skipped_annotations.jsonl"
        )

        with skipped_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for record in skipped_records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    (out_root / "_failed_annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failed_records))

    summary = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "rule_version": "prompt_root_trigger_verified_constraints_v7",
        "attribution_type": "prompt_level_root_trigger",
        "annotation_method": "deterministic_rule",
        "total_json_files": total,
        "annotated_trajectories": annotated,
        "skipped_trajectories": skipped,
        "errors": errors,
        "skip_reasons": dict(skip_reasons),
        "suite_counts": dict(suite_counts),
        "target_action_counts": dict(target_actions),
        "target_selection_type_counts": dict(
            target_selection_types
        ),
        "inferred_intent_counts": dict(inferred_intents),
        "allowed_functions": (
            sorted(allowed_functions)
            if allowed_functions is not None
            else None
        ),
        "excluded_functions": (
            sorted(excluded_functions)
            if excluded_functions is not None
            else None
        ),
    }

    summary_path = out_root / "_annotation_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 76)
    print("Deterministic annotation finished")
    print("=" * 76)
    print(f"Source directory       : {src_root}")
    print(f"Output directory       : {out_root}")
    print(f"JSON files scanned     : {total}")
    print(f"Annotated trajectories : {annotated}")
    print(f"Skipped trajectories   : {skipped}")
    print(f"Errors                 : {errors}")
    print(f"Summary file           : {summary_path}")

    if skip_reasons:
        print("\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  {reason:<32} {count}")

    if suite_counts:
        print("\nAnnotations by suite:")
        for suite, count in sorted(suite_counts.items()):
            print(f"  {suite:<20} {count}")

    if target_actions:
        print("\nTarget actions:")
        for action, count in target_actions.most_common():
            print(f"  {action:<40} {count}")


if __name__ == "__main__":
    main()
