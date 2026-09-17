# Synthetic quick start

Run from the repository root after installing dependencies. These examples
contain synthetic data only. No API key, local model or GPU is needed.
Choose a fresh output directory outside the skill folders for each run.

## Deterministic task alignment

```bash
python skills/agentdojo-annotation/scripts/task_alignment/run_pipeline.py \
  --raw_root examples/task-alignment/raw \
  --task_constraints examples/task-alignment/constraints.example.json \
  --output_root /tmp/annotation-skills-task-demo \
  --min_components 1 \
  --write_skipped
```

The example asks to send `hello` to alice@example.com and includes a synthetic
successful tool result. It should finalize one case. Inspect
`/tmp/annotation-skills-task-demo/quality_summary.json` and `case_outcomes.jsonl`.
Use the supplied constraints only for this synthetic example; real constraints
must be reviewed against the initiating task.

## Unsafe workflow planning

```bash
python skills/agent3sigma-annotation/scripts/unsafe/run_full_pipeline.py \
  --input examples/raw-turns.json \
  --input_format raw \
  --output_root /tmp/annotation-skills-unsafe-plan \
  --model dry-run-placeholder \
  --base_url https://example.invalid/v1 \
  --dry_run
```

The placeholder endpoint is never contacted during a dry run. This checks stage
planning only and does not produce annotations. Other workflows' dry-run entry
points are listed in the main README.
