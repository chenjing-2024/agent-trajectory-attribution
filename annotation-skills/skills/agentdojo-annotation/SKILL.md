---
name: agentdojo-annotation
description: Build, validate, repair, and summarize AgentDojo component-level annotations. Use for unsafe trajectory causal annotation or for deterministic task-alignment annotation of trajectories without successful unsafe behavior, including trajectory normalization, component construction, quality gates, repair selection, and final annotation dataset preparation.
---

# AgentDojo Annotation

Use this skill only for annotation preparation. Do not run attribution or
evaluation as part of the skill workflow.

## Inspect input readiness

Read [references/prepared_input.md](references/prepared_input.md) when the input
may already be standardized or componentized. Inspect its schema and select the
prepared-input route when compatible, preserving IDs and skipping repeated
conversion. Partial or incompatible preprocessing must not silently bypass
validation. Continue annotation and quality checks after reuse.

## Select the workflow

- Use `unsafe` when the input contains successful unsafe behavior and the goal
  is to identify the critical unsafe action, primary causal component, attack
  chain, and execution chain. Read `references/unsafe.md`.
- Use `task_alignment` when the attack did not produce unsafe behavior and
  the goal is to label the successful action that directly serves the
  initiating user's task. Read `references/task_alignment.md`.
- Read `references/schemas.md` when inspecting, transforming, or validating
  annotation records from either workflow.

Keep the two annotation datasets separate. Never infer that a task-alignment
trajectory is an unsafe annotation case merely because it contains injected
content.

## General procedure

1. Inspect the source layout and select one workflow.
2. Start with a small output directory and a case limit where the script
   supports one.
3. Run deterministic stages before model-based stages.
4. Inspect manifests, skipped cases, errors, and review queues before scaling.
5. Write every new run to a new output root unless an explicit resume or
   overwrite option is intended.
6. Keep credentials in environment variables. Never write keys into commands,
   documentation, manifests, or committed configuration.
7. Keep raw trajectories, model responses, and run outputs outside the skill
   folder.

For unsafe work, use `scripts/unsafe/run_initial_pipeline.py` only when the
requested result is explicitly an initial annotation. Use
`scripts/unsafe/run_full_pipeline.py` when the user asks for repaired, reviewed,
final, or complete annotations.

Before model-based unsafe stages, read
[references/model_parameters.md](references/model_parameters.md) for supported
model/provider profiles and explicit configuration for other endpoints.

## Unsafe quality gates

Require deterministic validation before semantic validation. Send
deterministic hard errors to re-annotation or repair; do not send them directly
to semantic validation. Treat semantic `uncertain`, `fail`, and validator-error
cases as review cases. Merge repairs into a new tree, then validate the merged
tree again before calling it final.

## Task-alignment quality gates

Normalize raw cases or validate and reuse prepared components, filter by the configured minimum component count, then create
deterministic annotations. Preserve skipped-case reasons. Do not add an LLM
annotation stage unless the user explicitly changes the workflow definition.
Select a target only when its function and arguments match explicit task
constraints and an observed result explicitly reports no tool error. Missing
results are unknown, not success. Use the supported exact email templates or
a separately reviewed `--task_constraints` file, optionally binding preceding
successful query evidence for conditional tasks, described in
`references/task_alignment.md`. Unknown task constraints must be recorded as needs_review;
do not claim general natural-language task understanding from these rules.

## Outputs

Report the selected workflow, input root, output root, case counts, skipped or
review counts, and validation status. Do not claim a dataset is final when only
syntax checks or dry runs have completed.


## Unified result reporting

Read [references/output.md](references/output.md) for the output contract.
For complete runs, use `quality_summary.json` and `case_outcomes.jsonl`, and show
the Chinese case-count line and run result. Do not infer completion from a saved
annotation file or a stage exit code. Keep every selected input accounted for;
report explicit exclusions, review cases and failures separately. Dry runs must
say that no annotations were produced.
