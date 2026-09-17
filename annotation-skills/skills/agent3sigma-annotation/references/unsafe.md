# Unsafe annotation workflow

The unsafe workflow identifies:

- target unsafe action;
- primary causal component;
- minimal attack chain;
- operational execution chain.

Run from the `agent3sigma-annotation` directory.

## Complete runner

Input may be an Agent3Sigma `detailed.json` file or a prepared case directory.
When given `detailed.json`, only successful attack cases are prepared.

```bash
python scripts/unsafe/run_full_pipeline.py \
  --input <detailed-json-or-case-directory> \
  --output_root <new-output-root> \
  --model <model-id> \
  --provider compatible \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY \
  --limit 10 \
  --force
```

For the official OpenAI API, use `--provider openai` and omit `--base_url`.
Use `--dry_run` before model calls.

The runner executes:

1. freeze the screened and limited case set and raw snapshots;
2. normalize components;
3. annotate target;
4. annotate primary cause;
5. annotate attack chain;
6. annotate execution chain;
7. merge staged annotations;
8. deterministic validation;
9. semantic validation;
10. join deterministic and semantic findings into one repair task per selected ID;
11. run one repair round, then finalize candidate unsafe annotations;
12. validate all completed annotations, including potential `no_target` exclusions;
13. audit every selected case and set the process exit code from the result.

These steps correspond to 15 subprocess stages (finalization and each final
validator have their own stage). `--limit` applies only during input selection.
Detailed exports use the attack-success filter; prepared directories are already
selected, and all their `trajectory.json` files are eligible.

The primary downstream path is recorded as `final_annotations` in
`run_manifest.json`. Initial annotations under `06_combined/` are not final.

Use each stage script's `--help` when manually rerunning a failed stage.

Resume a failed run without repeating completed model stages:

```bash
python scripts/unsafe/run_full_pipeline.py \
  --input <same-input> \
  --output_root <existing-failed-output-root> \
  --model <same-model-id> \
  --provider compatible \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY \
  --limit 10 \
  --force \
  --resume
```

The initial merge deliberately permits incomplete chains so validators and
repair stages can handle them. Use `merge_annotations.py --strict` only as a
standalone final-quality check, not before repair.


## Accounting and repair artifacts

- `00_selection/selected_cases.json`: fixed cohort, case keys, trajectory IDs,
  original source locations, raw snapshot paths and input errors.
- `08_repair_queue/repair_tasks.jsonl`: one task per input, both validators'
  findings, merged action and any blocked disposition. `all_cases.jsonl` contains
  executable tasks in the existing repair runner's format.
- `09_repairs/completed_annotations/cases`: includes `no_target` results for final
  validation. `final_annotations/cases` contains candidate unsafe results.
- `case_outcomes.jsonl`: one row per selected input, with outcome, reason and
  normalization/repair/final validation evidence.
- `coverage_summary.json`: counts, report consistency problems, final quality and
  aggregate status. `run_manifest.json` records the same status and exit code.

`finalized` requires a final file and both validators passing. `excluded` requires
an explicitly validated `no_target`, not just absence from the final folder.
Review flags or unresolved validation findings produce `needs_review`. Input or
runtime errors produce `failed`; unexplained missing output is also reported as
`failed` in the common ledger. A wholly validated excluded cohort produces
`no_eligible_cases` (exit 0), with no unsafe annotations.

Hard errors take precedence over semantic keep recommendations. Only recognized
chain-specific errors allow selective chain reruns; all other annotation errors
require full reruns. Missing annotations receive seeds used only for rerunning,
never accepted as outputs. Input trajectory errors are blocked for input repair.
Semantic validator errors without annotation hard errors are blocked for review.
The runner uses existing annotators for one repair round and validates the result;
it does not guarantee repair success or retry until a passing judgment appears.

Exit codes: `ok` = 0, `completed_with_review` = 1, `failed` = 2. Check the status
before consuming candidate final files. A dry run produces planned commands only.
Resume requires the same input, model and limit and uses existing snapshots.
After a selection failure, fix the source and choose a new output root. A final
audit failure resumes at the audit, not at another automatic model repair round.

For whole-run consumption, use [output.md](output.md) and `quality_summary.json`.
