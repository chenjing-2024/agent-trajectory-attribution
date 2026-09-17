# Reuse prepared components

Both normalizers build components while standardizing the raw messages. Inspect
input contents; a familiar filename alone does not establish readiness.

The unsafe and task-alignment runners retain the `--raw_root` argument for
compatibility. This directory may contain raw cases, compatible prepared
cases, or a mixture. Each selected case is checked independently:

- Raw `messages`: normalize and construct components as before.
- `trajectory` / `components_metadata` / `component_scheme` present: validate the
  prepared structure, then copy it unchanged to the new run. Print `[REUSE]` and
  record `status: reused` in the normalization outcomes ledger. No second
  component split and no ID reassignment occur.
- Incomplete or incompatible prepared data: record a failure. Do not fall back
  to raw messages and silently produce different component IDs.

Point `--raw_root` at the previous run's `initial/components` (unsafe) or
`normalized` (task alignment), not the whole run directory. The source remains
unchanged. Keep the required security labels: unsafe selects `security=true`
unless `--include_non_security_true`; task alignment selects `security=false`.
Screening and unsafe `--limit` still define this run's cohort.

Reusable records require a nonempty `data_id`, string component content, valid
roles, matching `num_components`, and metadata IDs ordered 1..N. Unsafe requires
`action_result_pair_schemeA` plus consistent message-to-group mappings. Task
alignment requires its own component metadata, including adjacent original step
IDs for action/result pairs. The two workflow formats are not interchangeable.
See `schemas.md`; already standardized chat messages without compatible
components still require component construction or an explicit input adapter.

The normalizer process still runs to check and copy prepared records; its
conversion and grouping operations are bypassed. Subsequent filtering,
annotation, final validation, repair and per-case accounting remain active.
Dry runs do not inspect or certify prepared data.
