---
name: agent3sigma-annotation
description: Build, validate, repair, and finalize Agent3Sigma component-level annotations. Use for unsafe successful-attack trajectories or safety-refusal trajectories, including case preparation, normalization, unsafe target and causal-chain annotation, refusal target and primary-root-cause annotation, deterministic and semantic quality gates, and final annotation dataset preparation.
---

# Agent3Sigma Annotation

Use this skill for annotation only. Do not run downstream component
attribution or attribution evaluation as part of either workflow.

## Inspect input readiness

Read [references/prepared_input.md](references/prepared_input.md) when the input
may already be standardized or componentized. Inspect its schema and select the
prepared-input route when compatible, preserving IDs and skipping repeated
conversion. Partial or incompatible preprocessing must not silently bypass
validation. Continue annotation and quality checks after reuse.

## Select one workflow

- Use `unsafe` for successful harmful trajectories. Read
  `references/unsafe.md`.
- Use `safety_refusal` when the agent refused an unsafe request and the goal is
  to identify the refusal target and its primary root cause. Read
  `references/safety_refusal.md`.
  Use `--raw_input` for raw item/turns cases or detailed.json; the runner
  normalizes them before model stages. Existing `--data_dir` inputs are checked
  as normalized cases without renumbering their components.
- Read `references/schemas.md` when validating or transforming records.

Keep unsafe and safety-refusal annotations in separate output roots. A
safety-refusal case is not an unsafe-success case.

## General procedure

1. Inspect the input format and select exactly one workflow.
2. Use a new output root. Never mix artifacts from different runs.
3. Start with `--dry_run`, then a small `--limit`.
4. Keep API keys in the environment variable named by `--api_key_env`.
5. Inspect failures, review queues, validation reports, and manifests before
   scaling.
6. Treat deterministic hard errors and semantic failures as unresolved.
7. Keep raw trajectories, model responses, and run outputs outside this skill
   folder.

## Unsafe completion rule

Use `scripts/unsafe/run_full_pipeline.py` for a complete run. Do not call the
initial combined annotations final. Use the finalized annotations only after
repair and final deterministic/semantic validation. Read `quality_summary.json`
and `case_outcomes.jsonl`: every screened and limited input must have a recorded
outcome. Only `ok` (exit 0) accepts the run; `completed_with_review` (1) and
`failed` (2) require attention. Use the joint deterministic/semantic repair queue;
a semantic pass must never cancel deterministic hard errors.

## Safety-refusal completion rule

Use `scripts/safety_refusal/run_pipeline.py`. Preserve `no_target` and
`needs_review` outcomes; do not force every trajectory into a refusal label.
Call primary-root-cause annotations usable only after deterministic validation.
Target and primary review flags both remain unresolved until reviewed. Read
`run_manifest.json` final_quality, not just individual stage exit codes.
The runner returns 1 for `completed_with_review` and 2 for `failed`; neither
is an accepted final annotation set. `no_eligible_cases` means classification
produced no refusal targets, not that refusal annotations were validated.

## Reporting

Report the selected workflow, input and output roots, case counts, unresolved
or failed cases, validation status, and final annotation path. Do not claim a
dry run or syntax check produced annotations.


## Unified result reporting

Read [references/output.md](references/output.md) for the output contract.
For complete runs, use `quality_summary.json` and `case_outcomes.jsonl`, and show
the Chinese case-count line and run result. Do not infer completion from a saved
annotation file or a stage exit code. Keep every selected input accounted for;
report explicit exclusions, review cases and failures separately. Dry runs must
say that no annotations were produced.
