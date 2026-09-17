# Reuse prepared components

Inspect the case contents before choosing an input mode. Standardization already
builds the component trajectory; it is not necessary to split it a second time.
A directory named `cases` or `normalized` is not evidence of a compatible schema.

- Raw item/turns or detailed exports: use the existing raw entrypoints.
- Already normalized unsafe cases: pass `--input_format normalized --input /path/to/cases`
  to `scripts/unsafe/run_full_pipeline.py`. Selection freezes the provided cohort,
  applies `--limit` once, validates IDs and component structure, and copies the
  validated records into the new run without changing their content or IDs.
  This replaces `normalize_components` with `validate_reuse_components`.
- Already normalized refusal cases: use `--data_dir /path/to/cases` rather than
  `--raw_input`. The existing validation stage preserves component IDs.

Use the per-case directory, not a parent containing reports, summaries or old
annotations. Unsafe normalized inputs represent an already selected unsafe
candidate cohort: unlike raw detailed exports, no judgment-label filtering is
applied to them. All supplied cases up to the limit are tracked, including
invalid/duplicate IDs as failures. A no-target model result is still permitted.

Normalized records require `trajectory_id`, a nonempty `trajectory` with string
content and valid roles, and unique C1-style component IDs (system entries may
have no ID). If present, `components_metadata` must match the component IDs and
order. See `schemas.md`. Message-only standardization without component IDs is
not this format. Do not guess or reconstruct lost provenance.

Reuse only skips input conversion. It never skips target annotation, quality
validation, or repairs. Existing annotations require their own validation flow;
they are not normalized trajectory input. Dry runs plan this branch but do not
validate input contents or produce annotations.
