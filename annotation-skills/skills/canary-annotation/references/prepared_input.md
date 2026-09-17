# Reuse prepared components

Raw normalization constructs the component trajectory at the same time. For
already normalized component cases, use the existing configuration:

```bash
export INPUT_FORMAT=normalized
export NORMALIZED_CASES=/absolute/path/to/normalized/cases
```

Check that the copied config does not override these values. Use the case
directory, not a whole run with reports and annotations. Both `unsafe-all` and
`safety-refusal-all` run `Validate normalized input (preserve IDs)` and then
continue annotation without raw conversion or component renumbering.

Inspect the actual contents before routing: compatible records have a nonempty
`trajectory_id`, string component content, valid roles and unique C0/C1-style nonnegative
component IDs. System entries can omit an ID. Metadata, when supplied, must match
the component IDs/order. See `schemas.md`. Raw messages, partial preprocessing,
other benchmarks' grouped formats, and final annotation objects are not this
normalized format. Invalid prepared inputs must fail validation, not be silently
reconstructed.

Reuse skips input conversion only. Annotation, deterministic/semantic checks,
repair and final accounting still run. Dry runs only record the intended branch;
they do not prove that the supplied data passes validation.

Existing C0 IDs are preserved. Negative IDs, leading-zero forms such as C00,
duplicates and metadata-order mismatches are rejected. Raw normalization keeps
its existing numbering; this change does not renumber previously prepared data.
