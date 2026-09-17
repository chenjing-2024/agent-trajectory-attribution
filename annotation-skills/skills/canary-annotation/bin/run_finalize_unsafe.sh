#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_dir "$UNSAFE_REPAIR_ROOT/accepted_cases"
require_dir "$UNSAFE_REANNOTATION_ROOT/merged/cases"

args=(
  python
  "$PIPELINE_DIR/merge_previous_pass_and_repaired.py" \
  --pass_dir "$UNSAFE_REPAIR_ROOT/accepted_cases" \
  --repaired_dir "$UNSAFE_REANNOTATION_ROOT/merged/cases" \
  --output_dir "$UNSAFE_FINAL_ROOT" \
  --required_target_type unsafe_action
)
[[ -n "${EXPECTED_UNSAFE_PASS:-}" ]] && args+=(--expected_pass "$EXPECTED_UNSAFE_PASS")
[[ -n "${EXPECTED_UNSAFE_REPAIRED:-}" ]] && args+=(--expected_repaired "$EXPECTED_UNSAFE_REPAIRED")
[[ -n "${EXPECTED_UNSAFE_FINAL:-}" ]] && args+=(--expected_final "$EXPECTED_UNSAFE_FINAL")
args+=(
  --force
)

run_step "Finalize unsafe-action annotations" "${args[@]}"
