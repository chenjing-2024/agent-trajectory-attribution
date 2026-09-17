# Unsafe annotation workflow

Run from the `agentdojo-annotation` skill directory.

## Initial annotation only

```bash
python scripts/unsafe/run_initial_pipeline.py \
  --raw_root <raw-unsafe-root> \
  --output_root <run-root> \
  --suite_name <suite> \
  --model <model-id> \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY \
  --force
```

This builds components, creates initial C1 annotations, runs deterministic
validation, and builds summaries. Use `--limit` for debugging and `--dry_run`
to inspect commands without model calls.

## Complete annotation and repair

Use the full runner to produce final annotations:

```bash
python scripts/unsafe/run_full_pipeline.py \
  --raw_root <raw-unsafe-root> \
  --output_root <new-full-run-root> \
  --suite_name <suite> \
  --model <model-id> \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY \
  --use_max_completion_tokens \
  --force
```

The output root must not already exist. The runner performs initial annotation,
initial semantic validation, repair, merge, final deterministic and semantic
validation, and final summary construction. It reports `ok` only when the final
validation has no hard errors, uncertain cases, failed cases, or validator
errors; otherwise it reports `completed_with_review`.

Use `--limit 10` for a full debug flow and `--dry_run` to inspect every planned
stage without calling a model.

## Manual stage execution

Run the stages separately when semantic review and repair are required:

1. `scripts/unsafe/validate_deterministic.py`
2. `scripts/unsafe/validate_semantic.py`
3. `scripts/unsafe/repair.py`
4. `scripts/unsafe/merge.py`
5. Repeat deterministic and semantic validation against the merged tree.
6. `scripts/unsafe/build_summary.py`

The deterministic validator writes:

- `semantic_input_cases.jsonl` for structurally valid cases;
- `repair_cases.jsonl` for hard failures;
- `validation_report.json` and per-case validation records.

The semantic validator writes pass, uncertain, fail, validator-error, and
combined review JSONL files. Supply `review_cases.jsonl` to the repair script.

The merge script refuses to overwrite an existing output root, requires every
repair to match an original relative path or, when `--components_root` is supplied,
a known component file. It writes a replacement manifest.

Use each script's `--help` for the complete CLI. Keep the normalized,
annotation, validation, repair, and final roots distinct.

The complete runner applies the limit once to screened raw inputs and writes
the shared [output contract](output.md), including nonzero review/failure exits.

## Model compatibility

Read [model_parameters.md](model_parameters.md) before choosing a model or provider.
All unsafe model calls share the same capability-based request builder.
