"""Aggregate refusal quality without dropping upstream review or failure counts."""


def count(report, key):
    value = report[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative count: {key}")
    return value


def summarize(target, primary=None, validation=None):
    refused = count(target["label_counts"], "safety_refusal")
    no_target = count(target["label_counts"], "no_target")
    failed = count(target, "failed_this_run")
    target_review = count(target, "needs_review")
    saved = count(target, "saved_annotations")
    if saved != refused + no_target or target_review > saved:
        raise ValueError("Inconsistent target summary")
    unresolved = {"target_failed": failed, "target_review": target_review,
                  "primary_failed": 0, "primary_review": 0, "validation_review": 0,
                  "missing_primary": 0}
    if refused:
        if primary is None or validation is None:
            raise ValueError("Refusal cases require primary and validation summaries")
        if validation.get("schema_version") != "safety_refusal_primary_validation_summary_v1":
            raise ValueError("Unexpected refusal validation summary schema")
        processed = count(primary, "processed")
        selected = count(primary, "selected_safety_refusal_targets")
        checked = count(validation, "input_cases")
        passed = count(validation, "pass")
        review = count(validation, "review")
        if selected != refused or processed > selected or checked != processed or passed + review != checked:
            raise ValueError("Inconsistent primary/validation case counts")
        unresolved.update(primary_failed=count(primary, "failed"),
                          primary_review=count(primary, "needs_review"),
                          validation_review=review, missing_primary=refused - processed)
    if failed or unresolved["primary_failed"] or unresolved["missing_primary"]:
        status, code = "failed", 2
    elif any(unresolved.values()):
        status, code = "completed_with_review", 1
    elif refused == 0:
        status, code = "no_refusal_targets", 0
    else:
        status, code = "ok", 0
    return status, code, {"target_summary": target, "primary_summary": primary,
                          "validation_summary": validation, "unresolved": unresolved}
