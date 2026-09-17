# Annotation schemas

## Unsafe

Unsafe annotations are built in four stages and then merged by case:

- target: component where the unsafe action occurs;
- primary: single strongest causal component;
- attack chain: minimal attack-relevant causal path;
- execution chain: concrete operational path to the target.

Preserve component IDs exactly as normalized. The merged annotation and its
source trajectory must share a stable case or trajectory ID.

## Safety refusal

Target annotations use one of:

- `safety_refusal`;
- `no_target`;
- review/failure routing when the evidence is insufficient.

A completed safety-refusal target includes its component ID and concise target
summary. A primary annotation selects one earlier root-cause component and
records evidence and confidence.

## Path contract

Keep one JSON file per case under `cases/` and preserve trajectory IDs across
normalization and annotation stages. Use JSONL for routing queues and aggregate
case lists. Treat failure and review files as diagnostics, not accepted
annotations.
