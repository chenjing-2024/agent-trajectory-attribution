# Annotation schemas

## Unsafe action

Each final record contains:

- `trajectory_id`
- `target_type: unsafe_action`
- `target_action`
- `primary_attribution_component`
- `attack_chain`
- `execution_chain`
- `confidence`
- `needs_review`
- `_metadata`

The primary component is strictly before the target. Attack-chain components
are before and exclude the primary and target. Execution-chain components are
strictly between primary and target and exclude both.

## Safety refusal

Each final record contains:

- `trajectory_id`
- `target_type: safety_refusal`
- `target_action`
- `primary_attribution_component`
- `confidence`
- `needs_review`
- `_metadata`

Preserve `no_target` separately. A refused harmful request is not an
unsafe-action success.
