#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_dir "$SAFETY_REFUSAL_FINAL_ROOT/cases"
require_dir "$NORMALIZED_CASES"

run_step \
  "Validate final safety-refusal annotations" \
  python \
  "$PIPELINE_DIR/validate_safety_refusal_annotations_deterministic.py" \
  --annotations_dir "$SAFETY_REFUSAL_FINAL_ROOT/cases" \
  --normalized_dir "$NORMALIZED_CASES" \
  --output_root "$SAFETY_REFUSAL_FINAL_VALIDATION_ROOT" \
  --force \
  --verbose

run_step "Assess final refusal quality" \
  python3 "$BIN_DIR/check_final_quality.py" \
  --kind refusal \
  --deterministic "$SAFETY_REFUSAL_FINAL_VALIDATION_ROOT/deterministic_validation_report.json" \
  --output "$SAFETY_REFUSAL_FINAL_VALIDATION_ROOT/final_quality.json"
