# Annotation schemas

## Unsafe annotation

An unsafe case records the executed unsafe target and its causal structure.
Important fields include:

- `critical_component_id` or the compatible target-component field;
- `target_unsafe_action`;
- `primary_attribution_component`;
- `attack_chain`;
- `execution_chain`;
- `risk_category`, `case_complexity`, and confidence/review metadata.

The primary component is the strongest direct causal source of attacker-
controlled content. Attack and execution chains exclude the primary and
critical components and may be empty.

## Task-alignment annotation

A task-alignment case records:

- the initiating prompt intent;
- the selected requested core action;
- its target component ID or number;
- deterministic rule metadata;
- skip metadata when no eligible successful action exists.

## Path contract

Component, annotation, validation, and repair trees use matching relative JSON
paths. Preserve those paths across stages. Do not flatten case files: repeated
task and injection names can otherwise collide.

Use JSON for per-case records and JSONL for queues, manifests, and aggregate
case lists. Treat `.error.json` and skipped-case records as diagnostics rather
than valid annotations.
