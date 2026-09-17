#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_file \
  "$PIPELINE_DIR/merge_reclassified_safety_refusal.py"

require_dir \
  "$SAFETY_REFUSAL_MERGED_ROOT/cases"

require_dir \
  "$UNSAFE_REANNOTATION_ROOT/target"

require_dir \
  "$UNSAFE_REANNOTATION_ROOT/primary"

args=(
  python
  "$PIPELINE_DIR/merge_reclassified_safety_refusal.py" \
  --base_dir "$SAFETY_REFUSAL_MERGED_ROOT/cases" \
  --target_dir "$UNSAFE_REANNOTATION_ROOT/target" \
  --primary_dir "$UNSAFE_REANNOTATION_ROOT/primary" \
  --output_dir "$SAFETY_REFUSAL_FINAL_ROOT"
)
[[ -n "${EXPECTED_SAFETY_REFUSAL_BASE:-}" ]] && args+=(--expected_base "$EXPECTED_SAFETY_REFUSAL_BASE")
[[ -n "${EXPECTED_SAFETY_REFUSAL_FINAL:-}" ]] && args+=(--expected_final "$EXPECTED_SAFETY_REFUSAL_FINAL")
args+=(
  --force
)

run_step "Finalize safety-refusal annotations" "${args[@]}"

echo
echo "Final safety-refusal output:"
echo "  $SAFETY_REFUSAL_FINAL_ROOT/cases"
