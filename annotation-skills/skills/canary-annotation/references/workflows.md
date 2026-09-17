# Canary annotation workflows

## Input preparation

For raw item/turns trajectories or a detailed.json results list, configure
`INPUT_FORMAT=raw` and `RAW_INPUT=/absolute/path/to/input`. `unsafe-all`,
`safety-refusal-all`, and `initial` normalize before generating targets.
Use a new `PROJECT_ROOT` for each raw run; preparation refuses an existing
`NORMALIZED_ROOT`. Switch to `INPUT_FORMAT=normalized` when reusing previously
prepared cases. The normalized-input check preserves IDs and rejects malformed
components, metadata ID mismatches, and duplicate trajectory IDs.

Preparation can also run alone, without an API key:

```bash
python3 scripts/common/prepare_annotation_input.py \
  --input /path/to/raw.json --input_format raw \
  --output_root /path/to/new-normalized-root
```

This adapter supports the existing Agent3Sigma item/turns format only, plus
already-normalized cases using `--input_format normalized`. It does not guess
how unrelated messages-based exports should be converted.

Run from the skill directory:

```bash
export ANNOTATION_CONFIG=/path/to/canary-annotation.env
export OPENAI_API_KEY='...'

bin/annotation_skill.sh show-config
bin/annotation_skill.sh check-config
```

## Unsafe action

Preview the complete unsafe workflow without a key or model calls:

```bash
bin/annotation_skill.sh unsafe-all --dry_run --plan /path/to/new-unsafe-plan.json
```

Set `PROJECT_ROOT`, `MODEL_ID`, and `ANNOTATION_CONFIG` as described in
`SKILL.md`. The selected input path must exist. An empty input directory
can check command planning, but cannot validate the data format. Use a fresh plan
filename; existing plans are never overwritten.

All workflow commands accept `--dry_run --plan PATH`. Plans include stage
commands and dependency presence checks, with absent intermediate files marked
`deferred`. They do not produce annotation outputs. Repair reannotation is
recorded as one shell stage; its internal commands are not expanded. Successful
planning is recorded as `dry_run_complete`, and a planning error as `failed`.

Run stages separately for the first small test:

```bash
bin/annotation_skill.sh initial
bin/annotation_skill.sh validate-unsafe
bin/annotation_skill.sh repair-unsafe
bin/annotation_skill.sh finalize-unsafe
bin/annotation_skill.sh validate-final-unsafe
```

After verifying the configuration and outputs:

```bash
bin/annotation_skill.sh unsafe-all
```

The initial merged set is under `UNSAFE_MERGED_ROOT/cases`. The repaired,
merged set is under `UNSAFE_FINAL_ROOT/cases`. Only use the latter after
`UNSAFE_FINAL_VALIDATION_ROOT` reports no unresolved hard failures.

## Safety refusal

The target stage shared with unsafe annotation creates
`TARGET_ROOT/by_target_type/safety_refusal`.

To start directly from normalized trajectories, including target generation:

```bash
bin/annotation_skill.sh safety-refusal-all --dry_run --plan /path/to/new-refusal-plan.json
```

After reviewing the plan, omit `--dry_run --plan ...` to execute the workflow
with a configured API key. This runs target annotation, refusal primary-cause
annotation, merge, and deterministic validation. Unsafe-repair reclassifications
remain a separate, conditional finalization workflow below.

If target annotations already exist, use the original fixed-target entry point:

```bash
bin/annotation_skill.sh safety-refusal
```

This annotates fixed-target primary causes, merges target and primary records,
and runs deterministic validation.

After unsafe repair has reclassified cases:

```bash
bin/annotation_skill.sh finalize-refusal
bin/annotation_skill.sh validate-final-refusal
```

The final cases are under `SAFETY_REFUSAL_FINAL_ROOT/cases`.

## Final quality exit codes

`validate-final-unsafe`, `validate-final-refusal`, and the completed
`safety-refusal`/`safety-refusal-all` workflows assess report contents, not just
validator process exit codes. The assessment writes `final_quality.json` in the
deterministic validation directory. Unsafe assessment requires both final
deterministic and semantic reports with matching case counts.

Exit 0 means `ok` or `no_eligible_cases`; exit 1 means `completed_with_review`;
exit 2 means `failed` due to validator errors or missing, malformed, inconsistent
reports. Ordinary deterministic warnings remain nonblocking. Other pipeline
runtime failures also propagate nonzero, possibly before an assessment is written.
Dry runs plan the assessment without reading or producing validation reports.

## Count assertions

Leave `EXPECTED_*` values empty for a new dataset. Set them only after counts
are independently known; the merge scripts then use them as assertions.


## Whole-run reports

Read [output.md](output.md). Complete workflows and final-validation commands
print a Chinese case summary and save common reports in a new directory under
`$RESULTS_ROOT/skill_runs/`. Full runs require fresh annotation output paths and
snapshot their inputs. Dry-run reports are beside the requested plan in
`<plan-stem>.outputs/`.

`final_quality.json` still describes the final validation subset. Use the common
`quality_summary.json` to decide whole-run acceptance, including missing outputs
and explicit target exclusions. For standalone final validation without an
upstream target cohort, the manifest says `provided_annotations_only`.

Unsafe repair now reads both deterministic and semantic reports. The retained
pass set comes from `$UNSAFE_REPAIR_ROOT/accepted_cases`, after checking both;
`repair_tasks.jsonl` records executable, kept and blocked dispositions. Refusal
review flags from both target and primary are preserved and validated.
