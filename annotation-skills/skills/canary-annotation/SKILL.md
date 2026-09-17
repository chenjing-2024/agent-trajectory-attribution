---
name: canary-annotation
description: Build, validate, repair, and finalize Agent3Sigma-Canary component annotations. Use for Canary normalized trajectories that require unsafe-action target, primary root cause, attack chain, and execution chain annotations, or safety-refusal target and primary-root-cause annotations, including deterministic and semantic quality gates and repair-set reannotation.
---

# Canary Annotation

Use this skill only for annotation. Keep downstream attribution scoring and
evaluation outside this workflow.

## Inspect input readiness

Read [references/prepared_input.md](references/prepared_input.md) when the input
may already be standardized or componentized. Inspect its schema and select the
prepared-input route when compatible, preserving IDs and skipping repeated
conversion. Partial or incompatible preprocessing must not silently bypass
validation. Continue annotation and quality checks after reuse.

## Configure

1. Copy `config/annotation.example.env` to a run-specific file outside the
   skill. Set `PROJECT_ROOT` to an absolute run directory and set `MODEL_ID`
   to a model supported by your API provider. Adjust the input and output paths.
2. Set `ANNOTATION_CONFIG` to the absolute path of this config file.
   Set `INPUT_FORMAT=raw` and `RAW_INPUT` for item/turns cases, prepared-case
   directories, or detailed.json. Complete unsafe and refusal workflows
   normalize these into `NORMALIZED_ROOT/cases` before target annotation.
   The default `INPUT_FORMAT=normalized` validates existing normalized input
   without renumbering IDs. Other raw formats require an explicit adapter.
3. Put the API key in the environment variable named by
   `OPENAI_API_KEY_ENV`. Never write the key into a config or command.
4. Start with small input directories before running the full dataset.

Run commands through `bin/annotation_skill.sh`.

Start with `unsafe-all --dry_run --plan /path/to/new-unsafe-plan.json` or
`safety-refusal-all --dry_run --plan /path/to/new-refusal-plan.json`.
Dry runs require a configured model name and an existing selected input
path, but no API key. They record commands without executing normalization, annotation,
merge, or validation scripts. Missing intermediate artifacts are recorded as
deferred dependencies, not validated inputs. Use a new plan path for each run.

## Select a workflow

- Run `initial` for initial unsafe-action annotation and merge.
- Run `validate-unsafe` for initial deterministic and semantic validation.
- Run `repair-unsafe` to extract non-pass cases and reannotate them.
- Run `finalize-unsafe` to merge previous semantic passes with repairs.
- Run `validate-final-unsafe` before treating repaired unsafe annotations as
  final.
- Run `safety-refusal` to annotate safety-refusal primary causes, merge them
  with fixed targets, and deterministically validate the result.
- Run `safety-refusal-all` to first generate targets from normalized trajectories,
  then run the safety-refusal workflow without depending on an earlier unsafe run.
- Run `finalize-refusal` after unsafe repair reclassifies cases as
  safety-refusal.
- Run `validate-final-refusal` before treating the augmented refusal set as
  final.

Use `unsafe-all`, `safety-refusal`, or `finalize-all` only after inspecting
the corresponding individual stages.

Read `references/workflows.md` for commands and output paths. Read
`references/schemas.md` when reviewing or transforming annotations.

## Invariants

- Require target, primary, attack chain, and execution chain for
  `unsafe_action`.
- Require target and primary for `safety_refusal`.
- Keep `no_target` separate; do not invent a target.
- Require primary to occur strictly before target.
- Let repaired annotations override previous annotations by `trajectory_id`.
- Remove cases reclassified during repair from the unsafe final set.
- Treat expected-count settings as optional dataset assertions, not universal
  constants.
- Validate final merged outputs; an initial merge or repair run is not final.
- Final validation commands write `final_quality.json` beside the deterministic
  report. Accept only `status=ok` (exit 0); review/fail/uncertain findings or an
  empty final set return 1, while validator/report errors return 2. Initial
  unsafe validation remains nonblocking so its findings can reach repair.

## Report

Report the workflow, config path, normalized input, model, output roots, case
counts, validation failures, review cases, reclassifications, and final
validated cases directory. Do not claim a syntax check or dry command produced
annotations.


## Unified result reporting

Read [references/output.md](references/output.md) for the output contract.
For complete runs, use `quality_summary.json` and `case_outcomes.jsonl`, and show
the Chinese case-count line and run result. Do not infer completion from a saved
annotation file or a stage exit code. Keep every selected input accounted for;
report explicit exclusions, review cases and failures separately. Dry runs must
say that no annotations were produced.
