# Safety-refusal annotation workflow

This workflow labels the unsafe request the agent refused and the earliest
component that most directly caused the refusal.

## Start from normalized cases

For raw `item` + `turns` cases, a prepared-case directory, or a `detailed.json`
with a `results` list, use `--raw_input` instead of `--data_dir`:

```bash
python scripts/safety_refusal/run_pipeline.py \
  --raw_input /path/to/detailed.json \
  --output_root /path/to/new-refusal-run \
  --model your-model-id --dry_run
```

Normalization writes `00_normalized/cases` before target annotation in a real
run. It does not filter by attack-success labels; target annotation determines
refusal versus no-target. Invalid inputs and duplicate IDs fail preparation.
Already-normalized inputs are validated without changing component IDs.

To execute only offline preparation, without running the model stages:

```bash
python scripts/common/prepare_annotation_input.py \
  --input /path/to/detailed.json --input_format raw \
  --output_root /path/to/new-normalized-directory
```

Other raw formats are not inferred. Use a new output directory for preparation.

```bash
python scripts/safety_refusal/run_pipeline.py \
  --data_dir <normalized-case-directory> \
  --output_root <new-output-root> \
  --model <model-id> \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY \
  --limit 10
```

## Start from a no-target review file

```bash
python scripts/safety_refusal/run_pipeline.py \
  --reviews_file <no_target_reviews.json> \
  --cases_dir <all-normalized-cases> \
  --output_root <new-output-root> \
  --model <model-id> \
  --base_url <openai-compatible-base-url> \
  --api_key_env OPENAI_API_KEY
```

The runner optionally collects review cases, annotates refusal targets,
annotates primary root causes only for completed refusal targets, and validates
the primary annotations.

Important outputs:

- `01_target/safety_refusal_annotations.json`;
- `01_target/no_target_annotations.json`;
- `01_target/needs_review.json`;
- `02_primary/cases/`;
- `03_validation/`;
- `run_manifest.json`.

Preserve `no_target` rather than inventing a refusal target when the trajectory
does not contain a clear refusal.

## Review and completion states

Primary output combines target and primary review flags using logical OR and
records each reason in `_metadata.review_sources`. A structurally valid record
with `needs_review=true` is routed to the validator's review output.

The runner aggregates all three stage summaries. Exit 0 is `ok` or
`no_eligible_cases`; exit 1 is `completed_with_review`; exit 2 is `failed`.
`no_eligible_cases` skips primary annotation/validation, and retains target
review or failure findings if any. Counts in `final_quality.unresolved` describe
separate stages and may refer to the same case; do not sum them as unique cases.

The validator's `--report_only` option is used internally so the runner can
aggregate review results. It does not suppress missing-input or runtime errors.
Standalone validation still exits 1 when review cases exist.

The common report counts unique selected cases; see [output.md](output.md).
