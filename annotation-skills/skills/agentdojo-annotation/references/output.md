# Unified run outputs

Complete workflow entrypoints print the same Chinese summary, derived from the
same per-case records as the machine-readable report:

```text
本轮 10 条：完成 7 条，明确排除 1 条，待复核 2 条，失败 0 条。
运行结果：需要复核。
结果目录：……
逐案例结果及原因：……/case_outcomes.jsonl
完整运行报告：……/run_manifest.json
```

## Files and acceptance

- `quality_summary.json` (`annotation_quality_v1`) is the authoritative whole-run
  result: `skill`, `workflow`, `status`, `status_zh`, `returncode`, `selected`,
  `counts`, `problems`, and `case_outcomes`.
- `case_outcomes.jsonl` has exactly one row per selected input. Required fields
  are `case_key`, `source`, `outcome`, and `reason`; when available, it also records
  `trajectory_id`, `annotation_path`, and stage evidence. Case keys identify input
  occurrences, so duplicate or invalid trajectory IDs cannot disappear.
- `run_manifest.json` records the workflow, stage information and report path.
  Existing stage-specific reports remain available for diagnosis. They may cover
  a subset of the cohort; do not use their success as whole-run acceptance.

For a real run, `selected` equals the sum of these four counts:

| Outcome | Chinese label | Meaning |
| --- | --- | --- |
| `finalized` | 完成 | Annotation meets this workflow's acceptance checks |
| `excluded` | 明确排除 | An explicit workflow rule excludes the case; reason retained |
| `needs_review` | 待复核 | Output exists but has unresolved review or validation findings |
| `failed` | 失败 | Processing/evidence failure or unexplained missing output |

Do not equate a saved JSON file with a completed annotation. Do not convert parse
errors or missing output into exclusions. Unknown disposition becomes `failed`
with a reason such as `unaccounted_case`. A whole-run/report error can fail the
run even when some or all individual annotations passed; inspect `problems`.

| Status | 中文结果 | Exit code |
| --- | --- | --- |
| `ok` | 完成 | 0 |
| `no_eligible_cases` | 无符合条件的案例 | 0 |
| `completed_with_review` | 需要复核 | 1 |
| `failed` | 存在失败 | 2 |
| `dry_run_complete` | 演练完成，未生成标注 | 0 |

Failure takes precedence over review. With neither, at least one finalized case
produces `ok`; an empty or wholly excluded cohort produces `no_eligible_cases`.
A dry run has `selected: null`, zero outcome counts and no outcome rows; it has
not evaluated data or generated accepted annotations. Exit code 0 alone does not
mean a dataset was produced. For actual data consumption, inspect both status
and counts.

## Selection and repairs

Freeze the input cohort before annotation and apply `--limit` once when supported.
Keep snapshots and original source locations. Existing benchmark selection rules
still apply; files screened out before selection are outside the run count.
Post-selection component filtering, no-target classification and reclassification
must have explicit case outcomes. Preserve both repair findings and final checks.
Initial validation findings feed repair; only final evidence decides acceptance.
One repair round does not guarantee that a model fixes an error.

These helpers are bundled independently in each Skill. No other Skill needs to
be installed. Offline tests check the common contract and real local stages;
model outputs/semantic judgments are fixtures, not evidence of model quality.
